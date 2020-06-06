import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import settings
import datetime
import random
import time
import gym
import cv2
import os

# from matplotlib import style
from lunar_phil import Agent


def record_game():
    render = True
    Games = []  # Close screen
    States = []
    for loop_ind in range(1):
        game = gym.make('LunarLander-v2')
        state = game.reset()
        Games.append(game)
        States.append(state)

    Scores = [0] * len(Games)
    step = 0
    All_score = []
    All_steps = []

    while len(Games):
        step += 1
        Old_states = np.array(States)
        Actions = agent.choose_action_list(Old_states)
        Dones = []
        Rewards = []
        States = []

        for g_index, game in enumerate(Games):
            # print(Actions[g_index])
            state, reward, done, info = game.step(action=Actions[g_index])
            Rewards.append(reward)
            Scores[g_index] += reward
            Dones.append(done)
            States.append(state)

        if render:
            Games[0].render()
            array = Games[0].viewer.get_array()
            cv2.imwrite(f"{settings.MODEL_NAME}/game-{episode_offset}/{step}.png", array[:, :, [2, 1, 0]])

        for ind_d in range(len(Games) - 1, -1, -1):
            if Dones[ind_d]:
                if ind_d == 0 and render:
                    render = False
                    Games[0].close()

                All_score.append(Scores[ind_d])
                All_steps.append(step)

                Scores.pop(ind_d)
                Games.pop(ind_d)
                States.pop(ind_d)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.compat.v1.Session(config=config)

    try:
        if settings.LOAD_MODEL:
            episode_offset = np.load(f"{settings.MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
        else:
            episode_offset = 0
    except FileNotFoundError:
        episode_offset = 0

    os.makedirs(f"{settings.MODEL_NAME}/game-{episode_offset}", exist_ok=True)

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "flighttime": []}

    agent = Agent(alpha=1e-5, beta=3e-6, gamma=0.99,
                  input_shape=settings.INPUT_SHAPE,
                  action_space=settings.ACTION_SPACE,
                  dense1=settings.DENSE1,
                  dense2=settings.DENSE2,
                  episode_offset=episode_offset,
                  record_game=True)
    record_game()
