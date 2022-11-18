import matplotlib.pyplot as plt
import numpy as np

import cv2
import pygame
import time

from matplotlib.patches import Rectangle

from collections import deque


# import trainer.trainer


class FrogGame:
    WRONG_MOVE_PENALTY = -20
    MOVE_PENALTY = -1
    WIN_REWARD = 1

    def __init__(self, size=10, resx=400, resy=400):
        self.size = int(size)

        self.frog = np.array((0, 0), dtype=int)  # X, Y
        self.food = np.array((size - 1, size - 1), dtype=int)  # X, Y
        self.steps = 0
        self.history = deque(maxlen=self.size ** 2 + 100)

        self.light = (190, 190, 190)
        self.dark = (25, 25, 25)
        self.resx, self.resy = resx, resy
        self.end = False

        self.board = self.draw_board()
        self.image_board = self.make_canvas_board()

    def reset(self):
        self.frog = np.array((0, 0), dtype=int)  # X, Y
        self.food = np.array((self.size - 1, self.size - 1), dtype=int)  # X, Y
        self.steps = 0
        self.history = deque(maxlen=self.size ** 2 + 100)
        self.end = False

        state = np.vstack([self.frog, self.food]).ravel() / self.size
        return state

    def make_canvas_board(self):
        # dimy = self.resy / self.size
        # dimx = self.resx / self.size
        #
        # image = np.array((self.resy, self.resx, 3))
        bd = self.board.astype(np.uint8)
        return cv2.resize(bd, (self.resy, self.resx), interpolation=0)

    def plot(self, new_figure=True):
        if new_figure:
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
        # display = self.display
        display = pygame.display.set_mode((400, 400))
        display.fill((0, 0, 0))

        "Copy pixels"
        board = self.board.astype(float)
        board[self.frog[0], self.frog[1]] = (0, 0.7, 0)
        board[self.food[0], self.food[1]] = (0.8, 0.1, 0)

        "Resize to bigger"
        board = np.round(board * 255).astype(np.uint8)
        board = cv2.resize(board, (self.resy, self.resx), interpolation=0)
        # board[0,0]=()
        board = pygame.surfarray.make_surface(board)
        # board = pygame.surface.Surface(board, (self.resy, self.resx))
        display.blit(board, (0, 0))
        # pygame.display.update()

        # off = 0.5
        # frog = Rectangle(self.frog - off, 1, 1, color=(0.05, 0.7, 0))
        # food = Rectangle(self.food - off, 1, 1, color=(0.4, 0.1, 0))

        pygame.display.update()

    def step(self, action):
        """

        Args:
            action:
                integer <0, 3>
                0 - right
                1 - down
                2 - left
                3 - up

        Returns:
            state - np array 2x2
                axis 0 : frog, food
                axis 1 : x, y position

            reward - integer
            end - boolean
            info - None

        """
        if self.end:
            print("Game has ended, no more moves!")
            return (self.frog, self.food), 0, True

        newx, newy = self.frog
        if action == 0:
            newx += 1
        elif action == 1:
            newy += 1
        elif action == 2:
            newx -= 1
        else:
            newx -= 1

        if newx >= self.size or newy >= self.size:
            reward = self.WRONG_MOVE_PENALTY

        elif newx < 0 or newy < 0:
            reward = self.WRONG_MOVE_PENALTY
        else:
            if ((newx, newy) == self.food).all():
                # print("Win")
                self.end = True
                reward = self.WIN_REWARD
            else:
                reward = self.MOVE_PENALTY
            self.frog = (newx, newy)

        end = self.end
        state = np.vstack([self.frog, self.food]).ravel() / self.size
        # print(state)
        # print(state)
        return state, reward, end, None



if __name__ == "__main__":
    pygame.display.init()
    game = FrogGame()
    game.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False
        else:
            time.sleep(0.10)

        st, re, ed,_ = game.step(np.random.randint(0, 2))
        # print(f"State: {st}, rew: {re}")
        game.render()
        if ed:
            break
