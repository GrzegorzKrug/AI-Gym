from PIL import ImageGrab, ImageFilter

from model import agent

from mouse_control import Mouse

import win32api as wapi
import pyautogui
import ctypes
import mouse
import numpy as np
import time
import cv2
import os

TEMPLATES = [
        None,
        os.path.abspath(os.path.join("patterns", "red.png")),
        os.path.abspath(os.path.join("patterns", "green.png")),
        os.path.abspath(os.path.join("patterns", "green_bonus.png")),
        os.path.abspath(os.path.join("patterns", "blue.png")),
        os.path.abspath(os.path.join("patterns", "star.png")),
        os.path.abspath(os.path.join("patterns", "stone.png")),
]
TEMPLATES = [cv2.imread(path) if path else None for path in TEMPLATES]
INTERVAL = 1
FALL_DELAY = 0.1
INITIAL_MOUSE_POS = pyautogui.position()


# TEMPLATES = [cv2.blur(img, (15, 15)) if img is not None else None for img in TEMPLATES]


def click():
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.005)
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTUP)
    # time.sleep(0.005)  # Click speed


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()

        out = func(*args, **kwargs)

        tend = time.time()
        dur = tend - t0
        name = func.__qualname__
        if dur < 1:
            print(f"Execution: {name:<30} was {dur * 1000:>4.3f} ms")
        elif dur > 60:
            print(f"Execution: {name:<30} was {dur / 60:>4.3f} m")
        else:
            print(f"Execution: {name:<30} was {dur:>4.3f} s")

        return out

    return wrapper


def locate_click():
    winds = pyautogui.getWindowsWithTitle("Clickr")
    for wnd in winds:
        if wnd.title == "Clickr":
            return wnd
    return None


# @timeit
def grab_frame(left, top, width, height):
    # left, upper, right, lowe
    screen = ImageGrab.grab([left, top, left + width, top + height], all_screens=True)
    return screen
    # screen.save(fp="screen.png",format='png')


def conv(arr):
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def get_action(xticks, yticks):
    while True:
        for x in xticks:
            for y in yticks:
                yield x, y


class Colors:
    empty = 0
    black = 0
    red = 1
    green = 2
    blue = 3
    yellow = 4
    star = 4
    stone = 5
    gray = 5
    block = 5


@timeit
def main():
    window = locate_click()
    window.activate()
    left = window.left  # + 3
    top = window.top  # + 1

    height = 797 - 10
    width = 1030 - 10

    bbox = [220, 585, 625, 990]
    board_size = bbox[1] - bbox[0], bbox[3] - bbox[2]
    size = 15

    def get_board():
        window_image = grab_frame(left, top, width, height)
        window_image = np.array(window_image)

        arr_rgb = conv(window_image)
        bord = arr_rgb[220:585, 625:990, :]
        return bord

    yticks = np.linspace(bbox[0] + size + 14, bbox[1] - size - 5, 8, dtype=int) - bbox[0]
    xticks = np.linspace(bbox[2] + size + 14, bbox[3] - size - 5, 8, dtype=int) - bbox[2]

    screen_yticks = np.linspace(bbox[0] + 20, bbox[1] - 15, 8) + top
    screen_xticks = np.linspace(bbox[2] + 20, bbox[3] - 15, 8) + left

    run = 0
    games_count = 10
    model = agent
    new_state = get_board()

    while True:

        board = new_state.copy()
        state = board.copy()
        vision = board.copy()
        board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
        hsv_mean = board_hsv.mean(axis=0).mean(axis=0)

        state = cv2.resize(state, (100, 100))
        cv2.imshow("Small", state)

        if 100 > hsv_mean[1] > 40 and hsv_mean[2] < 41 or \
                100 > hsv_mean[1] > 70 and hsv_mean[2] < 61 or \
                70 > hsv_mean[1] > 55 and 30 < hsv_mean[2] < 60:
            games_count -= 1
            if games_count <= 0:
                print(f"Game over")
                break
            else:
                time.sleep(3)
                mouse.move(left + 500, top + 700)
                click()
                print(f"Games left: {games_count}")
                time.sleep(2.5)
                new_state = get_board()
                continue

        board_int = np.zeros((8, 8), dtype=int)
        flood_mat_group = np.zeros((8, 8), dtype=int)
        flood_group_members = dict()

        "Create board colors"
        "Flood same colors"
        for indx, x in enumerate(xticks):
            x = int(x)
            for indy, y in enumerate(yticks):
                y = int(y)
                size = 10
                frag = board_hsv[y - size:y + size, x - size:x + size, :]
                hue = frag[:, :, 0].mean()
                val = frag[:, :, 2].mean()
                # frag = board[y - size:y + size, x - size:x + size, :]
                # frag = cv2.resize(frag, (100, 100))
                # cv2.imwrite(f"hue_{hue:4.0f}_val_{val:4.0f}.png", frag)
                if hue < 5 and val < 20:
                    "black"
                    board_int[indy, indx] = Colors.black
                elif hue < 10:
                    "gray"
                    board_int[indy, indx] = Colors.gray
                elif hue < 25:
                    "blue"
                    board_int[indy, indx] = Colors.blue
                elif val > 200:
                    "yellow"
                    board_int[indy, indx] = Colors.yellow
                elif hue < 85:
                    "green"
                    board_int[indy, indx] = Colors.green
                else:
                    "red"
                    board_int[indy, indx] = Colors.red

                left_col = board_int[indy, indx - 1]
                up_col = board_int[indy - 1, indx]
                my_col = board_int[indy, indx]

                if indy > 0 and indx > 0 and my_col == left_col == up_col:
                    group1 = flood_mat_group[indy - 1, indx]
                    group2 = flood_mat_group[indy, indx - 1]

                    if group1 != group2:
                        "Merge and fuze groups"
                        members_to_fuze = flood_group_members.pop(group2)
                        members1 = flood_group_members.get(group1)
                        for memy, memx in members_to_fuze:
                            flood_mat_group[memy, memx] = group1
                            members1.append((memy, memx))

                        flood_group_members[group2] = None
                    join_this_group = group1

                elif indx > 0 and my_col == left_col:
                    "Fuze with left"
                    join_this_group = flood_mat_group[indy, indx - 1]

                elif indy > 0 and my_col == up_col:
                    "Fuze with up"
                    join_this_group = flood_mat_group[indy - 1, indx]
                else:
                    "New group"
                    join_this_group = len(flood_group_members) + 1

                # print(indy, indx, join_this_group)
                flood_mat_group[indy, indx] = join_this_group
                group = flood_group_members.get(join_this_group, None)
                if group is None:
                    group = [(indy, indx)]
                    flood_group_members[join_this_group] = group
                else:
                    group.append((indy, indx))

        fuzzy_board = cv2.medianBlur(board, 3)

        highlight = []
        for ind, template in enumerate(TEMPLATES):
            if template is None:
                continue
            # matches = cv2.matchTemplate(board, template, method=cv2.TM_CCOEFF_NORMED)
            matches = cv2.matchTemplate(fuzzy_board, template, method=cv2.TM_CCOEFF_NORMED)
            loc = np.where(matches >= 0.7)  # median
            # loc = np.where(matches >= 0.8)  # board

            Y, X = loc
            for y, x in zip(Y, X):
                highlight.append((y, x, ind))

        floods = set()
        for y, x, ind in highlight:
            # print(ind)
            if ind == Colors.blue:
                col = (255, 0, 0)
            elif ind == Colors.green:
                col = (0, 255, 0)
            elif ind == Colors.red:
                col = (0, 0, 255)
            elif ind == Colors.yellow:
                col = (0, 200, 200)
            else:
                col = (30, 60, 130)

            pt1 = x, y
            pt2 = x + 80, y + 80

            xdist = np.absolute(xticks - x).min()
            ydist = np.absolute(yticks - y).min()
            try:
                indx = int(np.where(xticks - x == xdist)[0])
                indy = int(np.where(yticks - y == ydist)[0])
                pos = indy, indx
                floods.add(pos)
            except Exception:
                pass

            vision = cv2.rectangle(vision, pt1, pt2, color=col, thickness=5)

        # hi_score_boxes = []
        # for indy, indx in floods:
        # print(floods)
        "Reduce values"
        group_scores = {key: len(members) if members else 0 for key, members in flood_group_members.items()}
        for indy, indx in floods:
            group = flood_mat_group[indy, indx]
            score = group_scores[group]
            group_scores[group] = score - 3
        "Add 3 fields for each big box"
        temp_floods = floods
        floods = set()
        for indy, indx in temp_floods:
            floods.add((indy, indx))
            floods.add((indy + 1, indx))
            floods.add((indy, indx + 1))
            floods.add((indy + 1, indx + 1))

        while True:
            click_pos = np.random.randint(0, 8, 2)
            best_click = -1
            if board_int[click_pos[0], click_pos[1]] in (Colors.red, Colors.green, Colors.blue):
                break

        for indy, y in enumerate(yticks):
            y = y + 5
            for indx, x in enumerate(xticks):
                x -= 25
                color = board_int[indy, indx]

                if color == Colors.black:
                    reward = -2
                elif (indy, indx) in floods:
                    "Big block if"
                    if color == Colors.yellow:
                        reward = 1
                    elif color == Colors.gray:
                        reward = 2
                    else:
                        group = flood_mat_group[indy, indx]
                        reward = group_scores[group] - 1
                elif color == Colors.stone:
                    "Small stone"
                    reward = -2
                elif color == Colors.yellow:
                    "Yellow does not reset combo"
                    reward = 0
                else:
                    "CLick cost"
                    reward = -1

                if reward > best_click:
                    click_pos = indy, indx
                    best_click = reward
                elif best_click == -1:
                    if color in (Colors.green, Colors.red, Colors.blue):
                        grp = flood_mat_group[indy, indx]
                        members = flood_group_members[grp]
                        if len(members) == 1:
                            click_pos = indy, indx

                vision = cv2.putText(vision, f"{reward:2.0f}", org=(x, y),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN,
                                     fontScale=2, color=(255, 255, 255))

        # if prev_click_pos[0] == click_pos[0] and prev_click_pos[1] == click_pos[1]:
        #     click_pos = np.random.randint(0, 8, 2)

        y = int(yticks[click_pos[0]])
        x = int(xticks[click_pos[1]])

        pt1 = x - 20, y - 20
        pt2 = x + 20, y + 20

        col = (255, 0, 255)
        vision = cv2.rectangle(vision, pt1, pt2, color=col, thickness=3)
        board = cv2.rectangle(board, pt1, pt2, color=col, thickness=3)
        cv2.imshow("Game", board)
        cv2.imshow("Vision", vision)

        y = screen_yticks[click_pos[0]]
        x = screen_xticks[click_pos[1]]
        mouse.move(x, y)
        click()

        action = click_pos[0] * 8 + click_pos[1]
        if action < 64:
            # reward = 1
            pass
        else:
            best_click = 0

        key = cv2.waitKey(INTERVAL) & 0xFF
        time.sleep(FALL_DELAY)  # fall pieces

        mouse.move(0, 0)
        new_state = get_board()
        next_state = cv2.resize(new_state, (100, 100))
        model.add_memory((state, action, next_state, best_click))

        if key == ord("q") or wapi.GetAsyncKeyState(ord("Q")):
            break

        if run >= 5000:
            break
        else:
            run += 1

    cv2.destroyAllWindows()
    agent.save()


main()

mouse.move(*INITIAL_MOUSE_POS)
