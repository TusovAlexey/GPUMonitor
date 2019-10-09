from agents.BaseAgent import BaseAgent
from networks.networks import ActorCritic
from utils.RolloutStorage import RolloutStorage
from GPUMonitor import GPUMonitor, GPUMonitorWrapper
from utils.plot import plot_reward, plot_gpu
import gym
gym.logger.set_level(40)
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from timeit import default_timer as timer
from datetime import timedelta
import os
import glob
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from utils.hyperparameters import Config
from utils.plot import plot
from utils.wrappers import make_env_a2c_atari
import matplotlib
matplotlib.use('Agg')


class Model(BaseAgent):
    def __init__(self, log_dir, static_policy=False, env=None, config=None):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device

        self.noisy = config.USE_NOISY_NETS
        self.priority_replay = config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.learn_start = config.LEARN_START
        self.sigma_init = config.SIGMA_INIT
        self.num_agents = config.num_agents
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.rollout = config.rollout
        self.grad_norm_max = config.grad_norm_max

        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_feats = (self.num_feats[0] * 4, *self.num_feats[1:])
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-5)

        # move to correct device
        self.model = self.model.to(self.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()

        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
                                       self.num_feats, self.env.action_space, self.device, config.USE_GAE,
                                       config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []

    def declare_networks(self):
        self.model = ActorCritic(self.num_feats, self.num_actions)

    def get_action(self, s, deterministic=False):
        logits, values = self.model(s)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample().view(-1, 1)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        return values, actions, action_log_probs

    def evaluate_actions(self, s, actions):
        logits, values = self.model(s)

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy

    def get_values(self, s):
        _, values = self.model(s)

        return values

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        loss = action_loss + self.value_loss_weight * value_loss - self.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout):
        loss, action_loss, value_loss, dist_entropy = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optimizer.step()

        # self.save_loss(loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item())
        # self.save_sigma_param_magnitudes()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    '''def save_loss(self, loss, policy_loss, value_loss, entropy_loss):
        super(Model, self).save_td(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)'''


def A2C_experiment(env, batch_size, max_frames, log_dir):
    log_dir = log_dir+"A2C/"

    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    config = Config()

    # a2c control
    config.num_agents = 16
    config.rollout = 5

    # misc agent variables
    config.GAMMA = 0.99
    config.LR = 7e-4
    config.entropy_loss_weight = 0.01
    config.value_loss_weight = 0.5

    # batch size
    config.BATCH_SIZE = batch_size

    # Number of updates in 10000000 frames
    # config.MAX_FRAMES = int(1e7 / config.num_agents / config.rollout)
    config.MAX_FRAMES = int(max_frames / config.num_agents / config.rollout)

    # training loop
    seed = 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.set_num_threads(1)

    env_id = env
    envs = [make_env_a2c_atari(env_id, seed, i, log_dir) for i in range(config.num_agents)]
    envs = SubprocVecEnv(envs) if config.num_agents > 1 else DummyVecEnv(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * 4, *obs_shape[1:])

    # GPU Monitor
    monitor = GPUMonitor()
    #envs = [GPUMonitorWrapper(monitor, env, os.path.join(log_dir, env_id)) for env in envs]  # for all envs
    envs[0] = GPUMonitorWrapper(monitor, envs[0], os.path.join(log_dir, env_id))

    model = Model(log_dir, env=envs, config=config)

    current_obs = torch.zeros(config.num_agents, *obs_shape,
                              device=config.device, dtype=torch.float)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    model.rollouts.observations[0].copy_(current_obs)

    episode_rewards = np.zeros(config.num_agents, dtype=np.float)
    final_rewards = np.zeros(config.num_agents, dtype=np.float)

    start = timer()

    print_step = 1
    print_threshold = 10

    for frame_idx in range(1, config.MAX_FRAMES + 1):
        for step in range(config.rollout):
            with torch.no_grad():
                values, actions, action_log_prob = model.get_action(model.rollouts.observations[step])
            cpu_actions = actions.view(-1).cpu().numpy()

            obs, reward, done, _ = envs.step(cpu_actions)

            episode_rewards += reward
            masks = 1. - done.astype(np.float32)
            final_rewards *= masks
            final_rewards += (1. - masks) * episode_rewards
            episode_rewards *= masks

            rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(config.device)
            masks = torch.from_numpy(masks).to(config.device).view(-1, 1)

            current_obs *= masks.view(-1, 1, 1, 1)
            update_current_obs(obs)

            model.rollouts.insert(current_obs, actions.view(-1, 1), action_log_prob, values, rewards, masks)

        with torch.no_grad():
            next_value = model.get_values(model.rollouts.observations[-1])

        model.rollouts.compute_returns(next_value, config.GAMMA)

        value_loss, action_loss, dist_entropy = model.update(model.rollouts)

        model.rollouts.after_update()

        if frame_idx % 100 == 0:
            try:
                clear_output()
                end = timer()
                total_num_steps = (frame_idx + 1) * config.num_agents * config.rollout
                print(
                    "Updates {}, Num Timesteps {}, FPS {},\nMean/Median Reward {:.1f}/{:.1f}, Min/Max Reward {:.1f}/{:.1f},\nEntropy {:.5f}, Value Loss {:.5f}, Policy Loss {:.5f}".
                    format(frame_idx, total_num_steps,
                           int(total_num_steps / (end - start)),
                           np.mean(final_rewards),
                           np.median(final_rewards),
                           np.min(final_rewards),
                           np.max(final_rewards), dist_entropy,
                           value_loss, action_loss))
                plot(log_dir, env_id, 'a2c',
                     config.MAX_FRAMES * config.num_agents * config.rollout)

                dtime = int(timer() - start)
                plot_gpu(log_dir, env_id, 'A2C', config.MAX_FRAMES * config.num_agents * config.rollout, bin_size=10,
                         smooth=1, time=timedelta(seconds=dtime))
            except IOError:
                pass

    model.save_w()
    envs.close()
    return
