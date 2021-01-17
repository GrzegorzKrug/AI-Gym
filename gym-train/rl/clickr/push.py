from PIL import ImageGrab, ImageFilter

import tensorflow
import keras

import win32api as wapi
import keyboard
import numpy as np
import time
import cv2

from scipy import signal

import pyautogui
import ctypes
import mouse



class Mouse:
    """It simulates the mouse"""
    MOUSEEVENTF_MOVE = 0x0001  # mouse move
    MOUSEEVENTF_LEFTDOWN = 0x0002  # left button down
    MOUSEEVENTF_LEFTUP = 0x0004  # left button up
    MOUSEEVENTF_RIGHTDOWN = 0x0008  # right button down
    MOUSEEVENTF_RIGHTUP = 0x0010  # right button up
    MOUSEEVENTF_MIDDLEDOWN = 0x0020  # middle button down
    MOUSEEVENTF_MIDDLEUP = 0x0040  # middle button up
    MOUSEEVENTF_WHEEL = 0x0800  # wheel button rolled
    MOUSEEVENTF_ABSOLUTE = 0x8000  # absolute move
    SM_CXSCREEN = 0
    SM_CYSCREEN = 1

    def _do_event(self, flags, x_pos, y_pos, data, extra_info):
        """generate a mouse event"""
        x_calc = 65536 * x_pos / ctypes.windll.user32.GetSystemMetrics(self.SM_CXSCREEN) + 1
        y_calc = 65536 * y_pos / ctypes.windll.user32.GetSystemMetrics(self.SM_CYSCREEN) + 1
        return ctypes.windll.user32.mouse_event(flags, x_calc, y_calc, data, extra_info)

    def _get_button_value(self, button_name, button_up=False):
        """convert the name of the button into the corresponding value"""
        buttons = 0
        if button_name.find("right") >= 0:
            buttons = self.MOUSEEVENTF_RIGHTDOWN
        if button_name.find("left") >= 0:
            buttons = buttons + self.MOUSEEVENTF_LEFTDOWN
        if button_name.find("middle") >= 0:
            buttons = buttons + self.MOUSEEVENTF_MIDDLEDOWN
        if button_up:
            buttons = buttons << 1
        return buttons

    def move_mouse(self, pos):
        """move the mouse to the specified coordinates"""
        (x, y) = pos
        old_pos = self.get_position()
        x = x if (x != -1) else old_pos[0]
        y = y if (y != -1) else old_pos[1]
        self._do_event(self.MOUSEEVENTF_MOVE + self.MOUSEEVENTF_ABSOLUTE, x, y, 0, 0)

    def click(self, pos=(-1, -1), button_name="left"):
        """Click at the specified placed"""
        # self.move_mouse(pos)
        self._do_event(self._get_button_value(button_name, False) \
                       + self._get_button_value(button_name, True),
                       0, 0, 0, 0)


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


def click():
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.08)
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTUP)
    time.sleep(0.01)  # Click speed


@timeit
def main():
    window = locate_click()
    left = window.left  # + 3
    top = window.top  # + 1

    height = 797 - 10
    width = 1030 - 10

    bbox = [220, 585, 625, 990]
    size = 15

    yticks = np.linspace(bbox[0] + size + 14, bbox[1] - size - 5, 8, dtype=int) - bbox[0]
    xticks = np.linspace(bbox[2] + size + 14, bbox[3] - size - 5, 8, dtype=int) - bbox[2]

    screen_yticks = np.linspace(bbox[0] + 20, bbox[1] - 15, 8) + top
    screen_xticks = np.linspace(bbox[2] + 20, bbox[3] - 15, 8) + left

    run = 0
    acts = get_action(screen_xticks, screen_yticks)
    new_state = grab_frame(left, top, width, height)
    while True:
        # print(run)
        state = new_state
        arr = np.array(state)
        arr = conv(arr)
        board = arr[220:585, 625:990, :]
        board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
        hsv_mean = board_hsv.mean(axis=0).mean(axis=0)

        if hsv_mean[2] < 60:
            break

        board_int = np.zeros((8, 8), dtype=int)
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

                # board = cv2.putText(board, str(board_int[indy, indx]), org=(x, y),
                #                     fontFace=cv2.FONT_HERSHEY_PLAIN,
                #                     fontScale=2, color=(255, 255, 255))

        board_reward = np.zeros((8, 8)) - 1

        for yind, row in enumerate(board_int):
            y = yticks[yind] + 9
            for xind, val in enumerate(row):
                x = xticks[xind] - 30

                if yind < 7 and xind < 7:
                    a1 = board_int[yind + 1, xind]
                    a2 = board_int[yind, xind + 1]
                    a3 = board_int[yind + 1, xind + 1]
                    if val == a1 and val == a2 and val == a3:
                        rew = 0
                        board_reward[yind + 1, xind] = rew
                        board_reward[yind, xind + 1] = rew
                        board_reward[yind + 1, xind + 1] = rew
                    else:
                        rew = -1
                else:
                    rew = -1

                if board_reward[yind, xind] >= 0:
                    rew = board_reward[yind, xind]
                elif rew < 0 and val == Colors.gray or val == Colors.empty:
                    rew = -2
                    board_reward[yind, xind] = rew

                board = cv2.putText(board, f"{rew:2.0f}", org=(x, y),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=2, color=(255, 255, 255))

        # cv2.imwrite("board.png", board)

        act = next(acts)
        x, y = act
        # xi, yi = np.random.randint(0, 8, 2)
        mouse.move(x, y)
        click()
        mouse.move(0, 0)
        time.sleep(0.1)  # fall pieces
        new_state = grab_frame(left, top, width, height)

        cv2.imshow("Game", board)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or wapi.GetAsyncKeyState(ord("Q")):
            break

        if run >= 5000:
            break
        else:
            run += 1

    cv2.destroyAllWindows()


main()
