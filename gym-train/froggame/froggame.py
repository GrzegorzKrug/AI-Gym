import matplotlib.pyplot as plt
import numpy as np

import cv2
import pygame

from matplotlib.patches import Rectangle

from collections import deque


class FrogGame:
    def __init__(self, size=10, resx=400, resy=400):
        self.size = int(size)

        self.frog = np.array((0, 0), dtype=int)
        self.food = np.array((size - 1, size - 1), dtype=int)
        self.steps = 0
        self.history = deque(maxlen=self.size ** 2 + 100)

        self.light = (190, 190, 190)
        self.dark = (25, 25, 25)
        self.resx, self.resy = resx, resy

        self.board = self.draw_board()
        self.image_board = self.make_canvas_board()
        self.display = pygame.display.set_mode((400, 400))

    def restart(self):
        self.frog = (0, 0)
        self.food = (self.size - 1, self.size - 1)
        self.steps = 0
        self.history = deque(maxlen=self.size ** 2 + 100)

    def make_canvas_board(self):
        # dimy = self.resy / self.size
        # dimx = self.resx / self.size
        #
        # image = np.array((self.resy, self.resx, 3))
        bd = self.board.astype(np.uint8)
        return cv2.resize(bd, (self.resy, self.resx), interpolation=0)

    def plot(self):
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        # plt.plot([0, 0, sz, sz, 0], [0, sz, sz, 0, 0])# Border

        board = self.board
        plt.imshow(board)

        off = 0.5
        frog = Rectangle(self.frog - off, 1, 1, color=(0.05, 0.7, 0))
        food = Rectangle(self.food - off, 1, 1, color=(0.4, 0.1, 0))

        ax.add_patch(frog)
        ax.add_patch(food)

        plt.tight_layout()
        plt.show()

    def draw_board(self):
        light = self.light
        dark = self.dark
        sz = self.size

        pair = np.array([light, dark], dtype=int).reshape(1, 2, 3)
        line = np.tile(pair, (np.round((self.size ** 2 + self.size) / 2).astype(int), 1))
        inds_to_remove = [i * sz + i + sz for i in range(sz)]

        line = np.delete(line, inds_to_remove, axis=1)
        bd = line.reshape(self.size, self.size, 3)

        # print(f"par: {pair.shape}")
        # print(f"Line: {line.shape}")
        # print(f"Board: {bd.shape}")

        return bd

    def render(self):
        display = self.display
        display.fill((0, 0, 0))

        # board = self.board
        board = self.image_board
        board = pygame.surfarray.make_surface(board)
        # board = pygame.surface.Surface(board, (self.resy, self.resx))
        display.blit(board, (0, 0))
        # pygame.display.update()

        # off = 0.5
        # frog = Rectangle(self.frog - off, 1, 1, color=(0.05, 0.7, 0))
        # food = Rectangle(self.food - off, 1, 1, color=(0.4, 0.1, 0))

        pygame.display.update()


if __name__ == "__main__":
    pygame.display.init()

    game = FrogGame()
    game.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False
