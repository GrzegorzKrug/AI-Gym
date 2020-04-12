import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import matplotlib
import os


RUN_NUM = 24
EPISODES = 70000
PLOT_EPS = True
HIGH_RES = True


plots = np.linspace(0, EPISODES-10, 30)


aggr = np.load(f"qtables_{RUN_NUM}/aggregated.npy", allow_pickle=True).item()
rewards = np.load(f"qtables_{RUN_NUM}/rewards.npy")

os.makedirs(f"qtables_{RUN_NUM}_charts", exist_ok=True)


def get_q_color(value, vals):
    if value == max(vals):
        return 'green', 1.0
    else:
        return 'red', 0.4


if HIGH_RES:
    fig = plt.figure(figsize=(32, 18))
    style.use('fivethirtyeight')
    matplotlib.rcParams.update({'font.size': 20})
else:
    fig = plt.figure(figsize=(16, 9))
    style.use('bmh')


def save_chart(n):
    q_table = np.load(f"qtables_{RUN_NUM}/{n}-qtable.npy")
    for _x in q_table:
        for _y in _x:
            _y[:] = _y == _y.max()

    i = n // 10
    interp = 'nearest'

    ax1 = fig.add_subplot(221)
    plt.imshow(np.flip(q_table[:, :, 0], axis=1), cmap='GnBu', interpolation=interp)

    ax3 = fig.add_subplot(223)
    plt.imshow(np.flip(q_table[:, :, 2], axis=1), cmap='GnBu', interpolation=interp)

    x = aggr['ep'][0:i]
    y_max = aggr['max'][0:i]
    y_avg = aggr['avg'][0:i]
    y_min = aggr['min'][0:i]
    eps = aggr['eps'][0:i]

    ax2 = fig.add_subplot(222)
    if not PLOT_EPS:
        plt.imshow(np.flip(q_table[:, :, 1], axis=1), cmap='GnBu', interpolation=interp)
        ax2.title.set_text("Do nothing")
    else:
        ax2.title.set_text("Exploration chance")
        ax2.set_ylabel("Epsilon")
        plt.plot(x, eps, color='k', label='Epsilon')
        plt.legend(loc=0)

    ax4 = fig.add_subplot(224)

    plt.plot(x, y_avg, label='Avg score')
    plt.plot(x, y_min, label='Min score')
    plt.plot(x, y_max, label='Max score')
    plt.scatter(range(n), rewards[:n], c='m', marker='o', s=5, label='Reward', alpha=0.1)
    # plt.grid()

    ax1.title.set_text("Move Left")

    ax3.title.set_text("Move right")
    ax4.title.set_text(f"Episode #{n}")

    ax1.grid()
    ax3.grid()
    ax4.legend(loc=2)
    ax4.set_ylabel('Reward')
    ax4.set_xlabel('Episodes')

    plt.savefig(f"qtables_{RUN_NUM}_charts/{n}.png")
    plt.clf()


save_chart(n=EPISODES-10)

for index in plots:
    n = int(round(index, -1))
    print(f"Started[{RUN_NUM}]: {n}")
    save_chart(n=n)

