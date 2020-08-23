# import matplotlib.pyplot as plt
import cvlib as cv
import cv2
import os
from cvlib.object_detection import draw_bbox

from tkinter import (
    Frame, Grid, Canvas,
    Button, Label)
from PIL import Image, ImageTk
from tkinter import messagebox
from collections import deque
import tkinter as tk
import time
import cv2

LARGE_FONT = ("Verdana", 12)


class CameraWindow(Frame):
    def __init__(self, root=0, video_source=None):
        Frame.__init__(self)
        self.root = root
        self.cam = video_source
        self.pic = Canvas(root, width=1000, height=800)
        self.pic.pack(side='top', fill='both', expand=True)
        self.memory = deque(maxlen=30 * 1)

        self.config = dict.fromkeys(["gray", "detect_object", "detect_face"], False)

        button_frame = Frame(root)
        button_frame.pack(fill='both', expand=True)

        add_button(button_frame, f'GrayScale', 'black', row=0, col=0, sticky="wse").configure(
                command=lambda: self.toggle_config('gray'))
        add_button(button_frame, f'Object', 'black', row=0, col=1, sticky="wse").configure(
                command=lambda: self.toggle_config('detect_object'))
        add_button(button_frame, f'Face', 'black', row=0, col=2, sticky="wse").configure(
                command=lambda: self.toggle_config('detect_face'))
        add_button(button_frame, f'{3 + 1}', 'black', row=0, col=3, sticky="wse").configure(
                command=lambda: print(3 + 1))
        add_button(button_frame, f'{4 + 1}', 'black', row=0, col=4, sticky="wse").configure(
                command=lambda: print(4 + 1))
        add_button(button_frame, f'{5 + 1}', 'black', row=0, col=5, sticky="wse").configure(
                command=lambda: print(5 + 1))
        add_button(button_frame, f'{6 + 1}', 'black', row=0, col=6, sticky="wse").configure(
                command=lambda: print(6 + 1))
        add_button(button_frame, f'{7 + 1}', 'black', row=0, col=7, sticky="wse").configure(
                command=lambda: print(7 + 1))
        add_button(button_frame, f'{8 + 1}', 'black', row=0, col=8, sticky="wse").configure(
                command=lambda: print(8 + 1))
        add_button(button_frame, f'{9 + 1}', 'black', row=0, col=9, sticky="wse").configure(
                command=lambda: print(9 + 1))

        for _col in range(10):
            button_frame.grid_columnconfigure(_col, weight=1)

        button_frame.grid_rowconfigure(0, weight=1)

        self.update()

    def toggle_config(self, param):
        if param == 'gray':
            self.config.update({'gray': self.config['gray'] ^ True})
            print(f"Grayscale is now: {self.config['gray']}")
        elif param == 'detect_object':
            self.config.update({'detect_object': self.config['detect_object'] ^ True})
            print(f"Detecting objects: {self.config['detect_object']}")
        elif param == 'detect_face':
            self.config.update({'detect_face': self.config['detect_face'] ^ True})
            print(f"Detecting faces: {self.config['detect_face']}")
        else:
            print(f"Param unrecognized: {param}")

    def set_size(self, width, height):
        self.pic.configure(width=width, height=height)

    def update(self):

        ret, frame = self.cam.get_frame()
        photo = self.process_image(frame)
        self.memory.append(photo)
        self.pic.create_image(0, 0, image=photo, anchor='nw')

        if self.config['detect_object']:
            self.toggle_config('detect_object')
            self.root.after(2000, self.update)
        else:
            self.root.after(20, self.update)

    def process_image(self, image):
        """Returns Tkinter Canvas photo object"""
        if self.config['gray']:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # output_image = image
        if self.config['detect_object']:
            bbox, label, conf = cv.detect_common_objects(image)
            image = draw_bbox(image, bbox, label, conf)
            image = Image.fromarray(image)
            image.save("last_frame.jpg")
        else:
            image = Image.fromarray(image)

        # if self.config['detect_face']:
        #     bbox, label, conf = cv.detect_face(image)
        #     output_image = draw_bbox(image, bbox, label, conf)

        photo = ImageTk.PhotoImage(image=image)
        return photo


def add_button(_frame, text, fg, side=None, fill=None,
               row=0, col=0, sticky=None):
    but = Button(_frame, text=text, fg=fg)
    but.grid(row=row, column=col, sticky=sticky)
    # but.pack(side=side, fill='both')
    # but.pack(expand=1)
    return but


def hello_world(text=None):
    if text:
        messagebox.showinfo('Msg', "Hello " + str(text))
    else:
        messagebox.showinfo('Msg', "Hello")


class MyCameraCapture:
    def __init__(self, video_source=0):
        print(f"Opened camera src")

        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, frame
            else:
                return ret, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        print(f"deleting")
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Main window")

    root2 = tk.Toplevel()
    root2.title("Camera")

    frame = Frame(root)
    frame.pack(expand=True, fill='both', side='top')

    line = Frame(root)
    line.pack(fill='both', expand=True)

    buttons = {(2, 1), (2, 2), (2, 3), (3, 2)}
    for r in range(8):
        for c in range(10):
            if (r, c) in buttons:
                continue
            elif 0 <= r < 5 and 4 < c < 8:
                continue
            padx = 5
            pady = 2
            Label(frame, text=f"R_{r} C_{c}", relief='solid', borderwidth=2).grid(row=r, column=c, sticky='nwse',
                                                                                  ipadx=padx, ipady=pady)

    for x in range(8):
        Grid.columnconfigure(frame, x, weight=1)

    for y in range(10):
        Grid.rowconfigure(frame, y, weight=1)

    add_button(frame, 'Blue', 'blue', row=2, col=1, sticky="nwse").grid(rowspan=2)
    add_button(frame, 'Green', 'green', row=2, col=2, sticky="nwse").configure(command=hello_world)
    add_button(frame, 'Black', 'black', row=2, col=3, sticky="nwse").grid(rowspan=2)
    add_button(frame, 'Red', 'red', row=3, col=2, sticky="nwse")

    im = Image.open('cat.jpeg')
    render = ImageTk.PhotoImage(im)
    img = Label(frame, image=render)
    img.grid(row=0, column=5, rowspan=5, columnspan=3)

    with MyCameraCapture(2) as c1:
        cam_win = CameraWindow(root2, c1)
        width, height = c1.width, c1.height
        cam_win.set_size(width=width, height=height)
        cam_win.cam = c1
        # cam_win.update()
        root2.mainloop()

    # for x in range(3):
    #     ret, frame = c1.get_frame()
    #     im = Image.fromarray(frame)
    #     im.show()
    #     time.sleep(0.1)
    #     im.terminate()

    # frame.mainloop()
