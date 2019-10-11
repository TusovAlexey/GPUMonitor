import numpy as np

# switch backend in driver file
import matplotlib
import matplotlib.pyplot as plt

import os
import glob
from scipy.signal import medfilt


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
            np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                    (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_reward_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


def load_partial_gpu_data(indir, smooth, bin_size, idx):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.GPUMonitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                # t_time = float(tmp[6])
                tmp = [float(tmp[0]), int(tmp[1]), float(tmp[idx])]
                datas.append(tmp)
    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


def load_gpu_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.GPUmonitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[6])
                episode_length = int(tmp[7])
                # memory, power, fan_speed, temperature, gpu_utilization, mem_utilization, time, episode_length
                # time[0], episode_length[1], memory[2], power[3], fan_speed[4], temperature[5], gpu_utilization[6], mem_utilization[7]
                temp = [t_time, episode_length, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]]
                datas.append(temp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    memory = []
    power = []
    fan_speed = []
    temperature = []
    gpu_utilization = []
    mem_utilization = []
    timesteps = 0
    for i in range(len(datas)):
        memory.append([timesteps, datas[i][2]])
        power.append([timesteps, datas[i][3]])
        fan_speed.append([timesteps, datas[i][4]])
        temperature.append([timesteps, datas[i][5]])
        gpu_utilization.append([timesteps, datas[i][6]])
        mem_utilization.append([timesteps, datas[i][7]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x_memory, y_memory = np.array(memory)[:, 0], np.array(memory)[:, 1]
    x_power, y_power = np.array(power)[:, 0], np.array(power)[:, 1]
    x_fan_speed, y_fan_speed = np.array(fan_speed)[:, 0], np.array(fan_speed)[:, 1]
    x_temperature, y_temperature = np.array(temperature)[:, 0], np.array(temperature)[:, 1]
    x_gpu_utilization, y_gpu_utilization = np.array(gpu_utilization)[:, 0], np.array(gpu_utilization)[:, 1]
    x_memory_utilization, y_memory_utilization = np.array(mem_utilization)[:, 0], np.array(mem_utilization)[:, 1]

    if smooth == 1:
        x_memory, y_memory = smooth_reward_curve(x_memory, y_memory)
        x_power, y_power = smooth_reward_curve(x_power, y_power)
        x_fan_speed, y_fan_speed = smooth_reward_curve(x_fan_speed, y_fan_speed)
        x_temperature, y_temperature = smooth_reward_curve(x_temperature, y_temperature)
        x_gpu_utilization, y_gpu_utilization = smooth_reward_curve(x_gpu_utilization, y_gpu_utilization)
        x_memory_utilization, y_memory_utilization = smooth_reward_curve(x_memory_utilization, y_memory_utilization)

    if smooth == 2:
        y_memory = medfilt(y_memory, kernel_size=9)
        y_power = medfilt(y_power, kernel_size=9)
        y_fan_speed = medfilt(y_fan_speed, kernel_size=9)
        y_temperature = medfilt(y_temperature, kernel_size=9)
        y_gpu_utilization = medfilt(y_gpu_utilization, kernel_size=9)
        y_memory_utilization = medfilt(y_memory_utilization, kernel_size=9)

    x_memory, y_memory = fix_point(x_memory, y_memory, bin_size)
    x_power, y_power = fix_point(x_power, y_power, bin_size)
    x_fan_speed, y_fan_speed = fix_point(x_fan_speed, y_fan_speed, bin_size)
    x_temperature, y_temperature = fix_point(x_temperature, y_temperature, bin_size)
    x_gpu_utilization, y_gpu_utilization = fix_point(x_gpu_utilization, y_gpu_utilization, bin_size)
    x_memory_utilization, y_memory_utilization = fix_point(x_memory_utilization, y_memory_utilization, bin_size)
    ret = (
    x_memory, y_memory, x_power, y_power, x_fan_speed, y_fan_speed, x_temperature, y_temperature, x_gpu_utilization,
    y_gpu_utilization, x_memory_utilization, y_memory_utilization)
    return ret


# TODO: only works for Experience Replay style training for now
def load_custom_data(indir, stat_file, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, stat_file))

    for inf in infiles:  # should be 1
        with open(inf, 'r') as f:
            for line in f:
                tmp = line.split(',')
                tmp = [int(tmp[0]), float(tmp[1])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    for i in range(len(datas)):
        result.append([datas[i][0], datas[i][1]])

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


# TODO: only works for Experience Replay style training for now
def load_action_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, 'action_log.csv'))

    for inf in infiles:  # should be 1
        with open(inf, 'r') as f:
            for line in f:
                tmp = line.split(',')
                tmp = [int(tmp[0])] + [float(tmp[i]) for i in range(1, len(tmp))]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = datas
    # for i in range(len(datas)):
    #    result.append([datas[i][0], datas[i][1]])

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1:]

    '''if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)'''
    return [x, np.transpose(y)]


def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty = load_reward_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))

    return viz.image(image, win=win)


def plot(folder, game, name, num_steps, bin_size=100, smooth=1,
         ipynb=False, save_filename="A2CReward"):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20, 5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(folder + save_filename)
    plt.clf()
    plt.close()


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_all_data(folder, game, name, num_steps, bin_size=(10, 100, 100, 1), smooth=1, time=None,
                  save_filename='results.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    params = {
        'xtick.labelsize': 20,
        'ytick.labelsize': 15,
        'legend.fontsize': 15
    }
    plt.rcParams.update(params)

    tx, ty = load_reward_data(folder, smooth, bin_size[0])

    if tx is None or ty is None:
        return

    if time is not None:
        title = 'Avg. Last 10 Rewards: ' + str(np.round(np.mean(ty[-10]))) + ' || ' + game + ' || Elapsed Time: ' + str(
            time)
    else:
        title = 'Avg. Last 10 Rewards: ' + str(np.round(np.mean(ty[-10]))) + ' || ' + game

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15),
                                        subplot_kw=dict(xticks=ticks, xlim=(0, num_steps * 1.15), xlabel='Timestep',
                                                        title=title))
    ax1.set_xticklabels(tick_names)
    ax2.set_xticklabels(tick_names)
    ax3.set_xticklabels(tick_names)

    ax1.set_ylabel('Reward')

    p1, = ax1.plot(tx, ty, label="Reward")
    # lines = [p1]

    ax1.yaxis.label.set_color(p1.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax1.legend([p1], [p1.get_label()], loc=4)

    # Load td data if it exists
    tx, ty = load_custom_data(folder, 'td.csv', smooth, bin_size[1])

    ax2.set_title('Loss vs Timestep')

    if tx is not None or ty is not None:
        ax2.set_ylabel('Avg .Temporal Difference')
        p2, = ax2.plot(tx, ty, 'r-', label='Avg. TD')
        g2_lines = [p2]

        ax2.yaxis.label.set_color(p2.get_color())
        ax2.tick_params(axis='y', colors=p2.get_color())

        ax2.legend(g2_lines, [l.get_label() for l in g2_lines], loc=4)

    # Load Sigma Parameter Data if it exists
    tx, ty = load_custom_data(folder, 'sig_param_mag.csv', smooth, bin_size[2])

    if tx is not None or ty is not None:
        # need to update g2 title if sig data will be included
        ax2.set_title('Loss/Avg. Sigma Parameter Magnitude vs Timestep')

        ax4 = ax2.twinx()

        ax4.set_ylabel('Avg. Sigma Parameter Mag.')
        p4, = ax4.plot(tx, ty, 'g-', label='Avg. Sigma Mag.')
        g2_lines += [p4]

        ax4.yaxis.label.set_color(p4.get_color())
        ax4.tick_params(axis='y', colors=p4.get_color())

        # ax4.spines["right"].set_position(("axes", 1.05))
        # make_patch_spines_invisible(ax4)
        # ax4.spines["right"].set_visible(True)

        ax2.legend(g2_lines, [l.get_label() for l in g2_lines], loc=4)  # remake g2 legend because we have a new line

    # Load action selection data if it exists
    tx, ty = load_action_data(folder, smooth, bin_size[3])

    ax3.set_title('Action Selection Frequency(%) vs Timestep')

    if tx is not None or ty is not None:
        ax3.set_ylabel('Action Selection Frequency(%)')
        labels = ['Action {}'.format(i) for i in range(ty.shape[0])]
        p3 = ax3.stackplot(tx, ty, labels=labels)

        base = 0.0
        for percent, index in zip(ty, range(ty.shape[0])):
            offset = base + percent[-1] / 3.0
            ax3.annotate(str('{:.2f}'.format(ty[index][-1])), xy=(tx[-1], offset),
                         color=p3[index].get_facecolor().ravel())
            base += percent[-1]

        # ax3.yaxis.label.set_color(p3.get_color())
        # ax3.tick_params(axis='y', colors=p3.get_color())

        ax3.legend(loc=4)  # remake g2 legend because we have a new line

    plt.tight_layout()  # prevent label cutoff

    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()

    # return np.round(np.mean(ty[-10:]))


def plot_reward(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='results.png',
                ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20, 5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(folder + save_filename)
    plt.clf()
    plt.close()

    return np.round(np.mean(ty[-10]))


def plot_gpu_aux(folder, game, name, num_steps, time, save_filename, x, y, y_label, ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(20, 5))
    plt.plot(x, y, label="{}".format(name))
    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)
    plt.xlabel('Number of Timesteps')
    plt.ylabel(y_label)
    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(y[-10]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(y[-10]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(folder + save_filename)
    plt.clf()
    plt.close()
    return np.round(np.mean(y[-10]))


def plot_gpu(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='results.png',
             ipynb=False):
    x_memory, y_memory = load_partial_gpu_data(folder, smooth, bin_size, 2)
    if x_memory is None or y_memory is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU memory usage", num_steps, time, "gpu_memory_usage_results.png", x_memory,
                 y_memory, 'GPU memory usage [Mb]', False)

    x_power, y_power = load_partial_gpu_data(folder, smooth, bin_size, 3)
    if x_power is None or y_power is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU power usage", num_steps, time, "gpu_power_usage_results.png", x_power,
                 y_power, 'GPU power usage [W]', False)

    x_fan_speed, y_fan_speed = load_partial_gpu_data(folder, smooth, bin_size, 4)
    if x_fan_speed is None or y_fan_speed is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU fan speed", num_steps, time, "gpu_fan_speed_results.png", x_fan_speed,
                 y_fan_speed, 'GPU fan speed [percent]', False)

    x_temperature, y_temperature = load_partial_gpu_data(folder, smooth, bin_size, 5)
    if x_temperature is None or y_temperature is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU temperature", num_steps, time, "gpu_temperature_results.png",
                 x_temperature, y_temperature, 'GPU temperature [celsius]', False)

    x_gpu_utilization, y_gpu_utilization = load_partial_gpu_data(folder, smooth, bin_size, 6)
    if x_gpu_utilization is None or y_gpu_utilization is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU utilization", num_steps, time, "gpu_utilization_results.png",
                 x_gpu_utilization, y_gpu_utilization, 'GPU utilization [percent]', False)

    x_memory_utilization, y_memory_utilization = load_partial_gpu_data(folder, smooth, bin_size, 7)
    if x_memory_utilization is None or y_memory_utilization is None:
        return
    plot_gpu_aux(folder, game, name + " - GPU memory utilization", num_steps, time, "gpu_memory_utilization_results.png",
                 x_memory_utilization, y_memory_utilization, 'GPU memory utilization [percent]', False)
    return


'''def plot_td(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='td.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_custom_data(folder, 'td.csv', smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()

    return np.round(np.mean(ty[-1]))

def plot_sig(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='sig.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_custom_data(folder, 'sig_param_mag.csv', smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()

    return np.round(np.mean(ty[-1]))'''