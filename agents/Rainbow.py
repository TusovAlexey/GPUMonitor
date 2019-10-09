import torch
from agents.DQN import Model as DQN_Agent
from networks.networks import CategoricalDuelingDQN
from utils.ReplayMemory import PrioritizedReplayMemory
from GPUMonitor import GPUMonitor, GPUMonitorWrapper
from utils.plot import plot_gpu

from IPython.display import clear_output
from matplotlib import pyplot as plt
#%matplotlib inline
from timeit import default_timer as timer
from datetime import timedelta
from utils.wrappers import *
from utils.hyperparameters import Config
import matplotlib
matplotlib.use('Agg')


class Model(DQN_Agent):
    def __init__(self, log_dir, static_policy=False, env=None, config=None):
        self.atoms=config.ATOMS
        self.v_max=config.V_MAX
        self.v_min=config.V_MIN
        self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms).to(config.device)
        self.delta = (self.v_max - self.v_min) / (self.atoms - 1)

        super(Model, self).__init__(static_policy, env, config, log_dir=log_dir)

        self.nsteps=max(self.nsteps,3)
    
    def declare_networks(self):
        self.model = CategoricalDuelingDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=True, sigma_init=self.sigma_init, atoms=self.atoms)
        self.target_model = CategoricalDuelingDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=True, sigma_init=self.sigma_init, atoms=self.atoms)

    def declare_memory(self):
        self.memory = PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batch_size, 1, self.atoms), device=self.device, dtype=torch.float) + 1./self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                self.target_model.sample_noise()
                max_next_dist[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
                max_next_dist = max_next_dist.squeeze()


            Tz = batch_reward.view(-1, 1) + (self.gamma**self.nsteps)*self.supports.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            

            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batch_size, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        #estimate
        self.model.sample_noise()
        current_dist = self.model(batch_state).gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)
          
        loss = -(target_prob * current_dist.log()).sum(-1)
        self.memory.update_priorities(indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
        loss = loss * weights
        loss = loss.mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            self.model.sample_noise()
            a = self.model(X) * self.supports
            a = a.sum(dim=2).max(1)[1].view(1, 1)
            return a.item()

    def get_max_next_state_action(self, next_states):
        next_dist = self.model(next_states) * self.supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)


def plot(folder, frame_idx, rewards, losses, sigma, elapsed_time, ipynb=False, save_filename = "RainbowReward"):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    if losses:
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
    if sigma:
        plt.subplot(133)
        plt.title('noisy param magnitude')
        plt.plot(sigma)
    if ipynb:
        plt.show()
    else:
        plt.savefig(folder + save_filename)
    plt.clf()
    plt.close()


def Rainbow_experiment(env, batch_size, max_frames, log_dir):
    monitor = GPUMonitor()
    log_dir = log_dir + "Rainbow/"

    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-step returns
    config.N_STEPS = 3

    # misc agent variables
    config.GAMMA = 0.99
    config.LR = 1e-4

    # memory
    config.TARGET_NET_UPDATE_FREQ = 1000
    config.EXP_REPLAY_SIZE = 100000
    config.BATCH_SIZE = batch_size  # 32
    config.PRIORITY_ALPHA = 0.6
    config.PRIORITY_BETA_START = 0.4
    config.PRIORITY_BETA_FRAMES = 100000

    # epsilon variables
    config.SIGMA_INIT = 0.5

    # Learning control variables
    config.LEARN_START = 10000
    config.MAX_FRAMES = max_frames # 700000

    # Categorical Params
    config.ATOMS = 51
    config.V_MAX = 10
    config.V_MIN = -10

    # Training loop
    start = timer()

    env_id = env
    env = make_atari(env_id)
    env = GPUMonitorWrapper(monitor, env, os.path.join(log_dir, env_id))
    env = wrap_deepmind(env, frame_stack=False)
    env = wrap_pytorch(env)
    model = Model(log_dir, env=env, config=config)

    episode_reward = 0

    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        action = model.get_action(observation)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        observation = None if done else observation

        model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward

        if done:
            model.finish_nstep()
            model.reset_hx()
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0

            if np.mean(model.rewards[-10:]) > 19:
                plot(log_dir, frame_idx, model.rewards, model.losses, model.sigma_parameter_mag,
                     timedelta(seconds=int(timer() - start)))
                break

        if frame_idx % 10000 == 0:
            plot(log_dir, frame_idx, model.rewards, model.losses, model.sigma_parameter_mag,
                 timedelta(seconds=int(timer() - start)))

            dtime = int(timer() - start)
            plot_gpu(log_dir, env_id, 'Rainbow', config.MAX_FRAMES, bin_size=10, smooth=1,
                     time=timedelta(seconds=dtime))

    model.save_w()
    env.close()
    return


