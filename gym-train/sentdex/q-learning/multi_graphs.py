import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import os


RUN = ('16', '17', '18', '19')
DIM = (2, 3)

EPISODES = 50000
style.use('ggplot')

fig = plt.figure(figsize=(16, 10))


def save_graph(run_nums, dim):

    for i, curr_run in enumerate(run_nums, 1):
        ax = fig.add_subplot(int(f"{dim[0]}{dim[1]}{i}"))
        plot_graph(ax, curr_run)

    title = '-'.join(RUN)
    plt.savefig(f'Compare-{title}.png')


def plot_graph(ax, run_num):
    aggr = np.load(f"qtables_{run_num}/aggregated.npy", allow_pickle=True).item()
    rewads = np.load(f"qtables_{run_num}/rewards.npy")

    x = aggr['ep']
    y_max = aggr['max']
    y_avg = aggr['avg']
    y_min = aggr['min']

    plt.plot(x, y_max, label='Max score')
    plt.plot(x, y_avg, label='Avg score')
    plt.plot(x, y_min, label='Min score')
    plt.scatter(range(EPISODES), rewads, c='m', marker='o', s=5, label='Reward', alpha=0.1)
    # plt.grid()

    ax.title.set_text("Move Left")
    ax.title.set_text("Do nothing")
    ax.title.set_text("Move right")
    ax.title.set_text(f"Run #{run_num}")
    ax.set_ylim([-205, -80])

    ax.legend(loc=2)
    # ax.set_ylabel('Reward')
    # ax.set_xlabel('Episodes')


save_graph(RUN, DIM)





