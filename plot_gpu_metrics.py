import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_multiple_curves(game, metric, algorithms=None, ipynb=None):
    subdirs = algorithms
    if subdirs == None:
        subdirs = next(os.walk('log/'))[1]

    for subdir in subdirs:
        df = pd.read_csv('log/' + subdir + '/' + game + '.GPUMonitor.csv', header=1)
        x_index = df.columns.get_loc("time")
        y_index = df.columns.get_loc(metric)
        x = df.iloc[:, x_index:x_index + 1]
        y = df.iloc[:, y_index:y_index + 1]
        plt.plot(x, y, label="{}".format(subdir))

    plt.title(game)
    plt.xlabel('Time')
    plt.ylabel(metric)
    #plt.xlim(0, num_steps * 1.01)
    plt.legend()

    save_filename = 'log/' + game + '.' + metric + '.png'
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    game = 'PongNoFrameskip-v4'
    metric = "power_mean"
    plot_multiple_curves(game, metric) #,algorithms=["A2C"])

