import numpy as np
import sys
import os
import time
import gym


discrete_obs_win_size = 40


def get_discrete_state(state):
    dc_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(dc_state.astype(np.int))


if __name__ == "__main__":
    numbers = np.linspace(0, 49990, 5)

    env = gym.make('MountainCar-v0')

    for num in numbers:
        file = f"{int(num)}-qtable.npy"
        file_path = f"qtables_22/{file}"
        print(f"path: {file_path}")
        q_table = np.load(file_path, allow_pickle=True)

        step = 0
        discrete_state = get_discrete_state(env.reset())

        done = False
        while not done:

            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            time.sleep(0.03)
            step += 1

