import matplotlib.pyplot as plt
from matplotlib.collections import CircleCollection, PatchCollection
from matplotlib.patches import Circle, Polygon, Patch

import numpy as np
import cv2
import os

dir_go = os.path.join("frames", "go")
dir_stop = os.path.join("frames", "stop")


def check_end_condition(board):
    board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    mini_board = cut_board(board_hsv)

    "Mini Board, Left side"
    hsv_mean = mini_board.mean(axis=0).mean(axis=0)
    hsv_max = mini_board.max(axis=0).max(axis=0)
    hue_max, sat_max, val_max = hsv_max
    hue_mean, sat_mean, val_mean = hsv_mean

    # if 150 < sat_max < 210:
    #     return False

    if 70 < sat_max < 174:
        return False
    if 214 <= sat_max < 224:
        return True
    if val_max < 137:
        return True
    if sat_mean < 8:
        return True

    if hue_mean < 40 and val_mean > 100:
        return True

    if 60 < sat_mean < 75 and 45 < val_mean <= 67:
        return True

    # if hue_mean < 10 and sat_mean < 10:
    #     return True

    # bwhite = board.sum(axis=-1).max()
    # if bwhite >= 750:
    #     return True

    # if sat_mean
    # if sat_mean < 90

    "Full board"
    hsv_mean = board_hsv.mean(axis=0).mean(axis=0)
    hsv_max = board_hsv.max(axis=0).max(axis=0)
    hue_mean, sat_mean, val_mean = hsv_mean
    hue_max, sat_max, val_max = hsv_max

    if hue_mean > 100:
        return True
    if 34 < hue_mean < 66 and 25 < val_mean < 57:
        return True
    elif 43 < hue_mean < 60 and 55 < val_mean < 67:
        return True

    if 170 < sat_max < 199 and val_max > 200:
        return False
    if 215 < val_max < 235 and 80 < val_mean < 90:
        return True

    if val_max < 206:
        return True

    return False


def cut_board(bord):
    # bord = bord[:, 19:35, :]
    bord = bord[:, :35, :]  # left side
    return bord


fails = 0
figsize = (8, 8)

fig1 = plt.figure(figsize=figsize)
ax1 = plt.gca()
fig2 = plt.figure(figsize=figsize)
ax2 = plt.gca()
fig3 = plt.figure(figsize=figsize)
ax3 = plt.gca()
fig4 = plt.figure(figsize=figsize)
ax4 = plt.gca()
fig5 = plt.figure(figsize=figsize)
ax5 = plt.gca()
fig6 = plt.figure(figsize=figsize)
ax6 = plt.gca()

# patches1 = []
# patches2 = []
# patches3 = []
# patches4 = []

for file in os.listdir(dir_go):
    im_path = os.path.join(dir_go, file)
    board = cv2.imread(im_path, cv2.IMREAD_COLOR)

    board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    # board_hsv = cut_board(board_hsv)
    hsv_mean = board_hsv.mean(axis=0).mean(axis=0)
    hsv_max = board_hsv.max(axis=0).max(axis=0)
    hue_mean, sat_mean, val_mean = hsv_mean
    hue_max, sat_max, val_max = hsv_max

    mini_board = cut_board(board_hsv)
    hsv_mean = mini_board.mean(axis=0).mean(axis=0)
    hsv_max = mini_board.max(axis=0).max(axis=0)
    mini_hue_max, mini_sat_max, mini_val_max = hsv_max
    mini_hue_mean, mini_sat_mean, mini_val_mean = hsv_mean

    "Check if screen is dark to end"
    if check_end_condition(board):
        print(f"Fail go at: {im_path}")
        fails += 1
        size = 2
    else:
        size = 0.5

    color = [0, 1, 0]
    circ1 = Circle((sat_mean, val_mean), size, color=color)
    circ2 = Circle((sat_max, val_max), radius=size, color=color)
    circ3 = Circle((mini_sat_mean, mini_val_mean), radius=size, color=color)
    circ4 = Circle((mini_sat_max, mini_val_max), radius=size, color=color)
    circ5 = Circle((hue_mean, val_mean), radius=size, color=color)
    circ6 = Circle((mini_hue_mean, mini_val_mean), radius=size, color=color)

    ax1.add_patch(circ1)
    ax2.add_patch(circ2)
    ax3.add_patch(circ3)
    ax4.add_patch(circ4)
    ax5.add_patch(circ5)
    ax6.add_patch(circ6)

print()

for file in os.listdir(dir_stop):
    im_path = os.path.join(dir_stop, file)
    board = cv2.imread(im_path, cv2.IMREAD_COLOR)

    board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    # board_hsv = cut_board(board_hsv)
    hsv_mean = board_hsv.mean(axis=0).mean(axis=0)
    hsv_max = board_hsv.max(axis=0).max(axis=0)
    hue_mean, sat_mean, val_mean = hsv_mean
    hue_max, sat_max, val_max = hsv_max

    mini_board = cut_board(board_hsv)
    hsv_mean = mini_board.mean(axis=0).mean(axis=0)
    hsv_max = mini_board.max(axis=0).max(axis=0)
    mini_hue_max, mini_sat_max, mini_val_max = hsv_max
    mini_hue_mean, mini_sat_mean, mini_val_mean = hsv_mean

    "Check if screen is dark to end"
    if not check_end_condition(board):
        print(f"Fail stop at: {im_path}")
        fails += 1
        size = 2
    else:
        size = 0.5

    color = [1, 0, 0]
    circ1 = Circle((sat_mean, val_mean), size, color=color)
    circ2 = Circle((sat_max, val_max), radius=size, color=color)
    circ3 = Circle((mini_sat_mean, mini_val_mean), radius=size, color=color)
    circ4 = Circle((mini_sat_max, mini_val_max), radius=size, color=color)
    circ5 = Circle((hue_mean, val_mean), radius=size, color=color)
    circ6 = Circle((mini_hue_mean, mini_val_mean), radius=size, color=color)

    ax1.add_patch(circ1)
    ax2.add_patch(circ2)
    ax3.add_patch(circ3)
    ax4.add_patch(circ4)
    ax5.add_patch(circ5)
    ax6.add_patch(circ6)

print(f"\nFailed {fails} times!")

ax1.set_title("Mean")
ax2.set_title("Max")
ax3.set_title("Mini mean")
ax4.set_title("Mini max")
ax5.set_title("Hue mean")
ax6.set_title("Mini_ Hue mean")

for _ax in (ax1, ax2, ax3, ax4):
    _ax.set_xlabel("Saturation")
    _ax.set_ylabel("Value")
    _ax.set_xlim(-1, 260)
    _ax.set_ylim(-1, 260)

for _ax in (ax5, ax6):
    _ax.set_xlabel("Hue")
    _ax.set_ylabel("Value")
    _ax.set_xlim(-1, 260)
    _ax.set_ylim(-1, 260)

plt.show()
