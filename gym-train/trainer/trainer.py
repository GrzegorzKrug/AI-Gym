import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np
import numba
import time
import gym
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Input
from tensorflow.keras.optimizers import Adam

import pygame


class Trainer:
    """"""
    """
    Config
        Lower case keys!
    0: One agent and one env.
    1: One agent and many envs.
    2: Many agents and many envs.
    """
    configs = {
            k.lower(): v for k, v
            in {
                    'OneToOne': 0,
                    'OneToMany': 1,
                    'ManyToMany': 2,
            }.items()
    }

    # configs = {k.lower(): v for k, v in configs.items()}

    def __init__(self,
                 model, environments,
                 # envs_n=10,
                 input_size=2, output_size=2,
                 memory_max_samples=5000,
                 batch_size=200, train_size=2000,
                 train_each_step=False, split_train=False,
                 config='OneToMany',
                 notify_win=2,
                 gamma=0.9,
                 **kw
                 ):
        """

        Args:
            model:
            environments:
            envs_n:
            input_size:
            output_size:
            action_size:
            memory_max_samples:
            batch_size:
            train_size:
            train_each_step:
            split_train:
            config:
                keys are case insensitive
                OneToOne - One agent one env
                OneToMany - One agent, many envs
                ManyToMany - (Not implemented) Many agents with many envs
            notify_win:
                0 - None
                1 - Cumulated data
                2 - Every single one
            **kw:
        """
        if len(kw) > 0:
            print("Uncaught kwargs")
            print(kw)

        "Training settings"
        self.model = model
        self.envs = environments
        self.config = self.configs.get(config.lower(), 0)
        self.config_name = config
        self.input_size = input_size
        self.output_size = output_size
        # self.action_size = action_size

        self.min_train_samples = 10
        self.batch_size = batch_size
        self.train_size = train_size
        # self.max_samples = 200
        self.gamma = gamma

        "Config"
        if self.config == 0:
            self.envs_n = 1
        elif self.config == 1:
            # self.envs_n = envs_n
            self.envs_n = len(self.envs)
        else:
            raise NotImplemented(f"Not implemented config {self.config_name}")
        print(f"Starting Trainer with config: {self.config} ({self.config_name})")

        "Memory settings (fitting)"
        self.memory = np.zeros((memory_max_samples, (input_size * 2 + 3)), dtype=float)
        # print(self.memory.dtype)
        self.memory_write_index = 0
        self.memory_last_index = memory_max_samples - 1
        self.memory_fully_writen = False

        "Last state memory"
        self.states = np.zeros((self.envs_n, self.input_size), dtype=np.float32)
        # self.actions = np.zeros((self.envs_n, self.action_size), dtype=np.float32)
        # self.rewards = np.zeros((self.envs_n, 1), dtype=np.float32)
        self.alive_envs = set(range(self.envs_n))

        self.notify = notify_win

        "Additional trainer setup"
        self.post_init_called = False
        self.__post_init__()
        self.reset()

    def __post_init__(self):
        self._init_set_step_func()
        self._init_set_train_func()

        self.post_init_called = True

    def _init_set_step_func(self):
        cf = self.config
        if cf == 0:
            self._inner_step = self._inner_step_single
        elif cf == 1:
            self._inner_step = self._inner_step_one_agent_many_envs
        else:
            self._inner_step = self._inner_step_many_agents_many_envs

    def _init_set_train_func(self):
        cf = self.config
        if cf == 0:
            self._inner_train = self._inner_train_single
        elif cf == 1:
            self._inner_train = self._inner_train_single
            # self._inner_train = self._inner_train_one_agent_many_envs # Redundant
        else:
            self._inner_train = self._inner_train_many_agents_many_envs

    def reset(self):
        self.states = np.zeros((self.envs_n, self.input_size), dtype=np.float32)
        # self.actions = np.zeros((self.envs_n, self.action_size), dtype=np.int32)
        # self.rewards = np.zeros((self.envs_n, 1), dtype=np.float32)
        self.alive_envs = set(range(self.envs_n))

        for i in range(self.envs_n):
            # print(f"I:{i}")
            state = self.envs[i].reset()
            self.states[i] = state

    def start_training(self, epochs=1, max_iters=10,
                       exploration_type='cos',
                       exploration_ratio=0.35,
                       min_exploration=1e-2,
                       exploration_frequency=3,
                       ):
        for r in range(epochs):
            "Iterate over epochs"
            self.reset()  # Clear environment at start
            current_explore_ratio = np.cos(
                    r / (epochs - 1) * np.pi * exploration_frequency) * exploration_ratio
            current_explore_ratio = np.abs(current_explore_ratio)
            if current_explore_ratio < min_exploration:
                current_explore_ratio = min_exploration

            print(f"Epoch: {r:>3}, exploration: {current_explore_ratio:3.4f}")
            for n in range(max_iters):
                if len(self.alive_envs) <= 0:
                    break
                rnd_act = True if current_explore_ratio >= np.random.random() else False
                self._inner_step(random_action=rnd_act)

            self._inner_train()

        self.model.iters += r + 1

    def _inner_step(self, random_action=False):
        """
        Abstraction
            random_action [boolean] - model choice or random action
        """
        raise NotImplemented("This is prototype")

    def _inner_train(self):
        """ Abstraction """
        raise NotImplemented("This is prototype")

    def _inner_step_single(self, random_action=False):
        old_state = self.states[0]
        if random_action:
            actions = np.random.randint(0, self.action_size, (self.envs_n,))
        else:
            actions = self.predict()[0]
        new_state, reward, end, info = self.envs[0].step(actions)
        sample = np.concatenate([old_state, new_state, actions[0], [reward], [end]],
                                dtype=np.float32)
        self.add_memory(sample)

    def predict(self):
        """DL Q-learning"""
        qvals = self.model.predict(self.states, verbose=False)
        actions = np.argmax(qvals, axis=1)
        return actions

    def get_training_samples(self):
        # print("Random samples:")
        if self.memory_fully_writen:
            k_samples = self.train_size
            inds = random.sample(range(self.memory_last_index), k_samples)
            # samples = np.random.sample(self.memory, axis=0)
        # elif self.memory_write_index <= self.min_samples:
        #     print("return none")
        #     return None
        else:
            k_samples = self.memory_write_index
            inds = random.sample(range(self.memory_write_index), k_samples)
            # samples = np.random.sample(self.memory[self.memory_write_index - 1], axis=0)
        samples = self.memory[inds, :]
        # print(f"inds: {inds}")
        # print(f"Selected samples for fit: {samples.shape}")
        # print(samples)
        # print("aasdasdasdasd")
        return samples

    @property
    def sample_indexes(self):
        return self.input_size, self.input_size * 2,

    def _inner_train_single(self):
        if self.memory_fully_writen or self.memory_write_index > self.min_train_samples:
            train_data = self.get_training_samples()
        else:
            return None
        ind1, ind2 = self.sample_indexes
        old_state = train_data[:, :ind1]
        new_state = train_data[:, ind1:ind2]
        action_inds = train_data[:, ind2].astype(np.int32)
        reward = train_data[:, ind2 + 1].reshape((-1, 1))
        end = train_data[:, ind2 + 2]

        # print("FIT Y")

        # action = np.zeros((action_inds.shape[0], self.output_size * self.action_size))
        # action[:, action_inds] = 1
        # for aci, ind in enumerate(action_inds):
        # action[aci, ind] = 1

        current_qvals = self.model.predict(old_state, verbose=False)
        future_qvals = self.model.predict(new_state, verbose=False)
        future_max = np.max(future_qvals, axis=1).reshape(-1, 1)

        gamma = self.gamma
        Y = current_qvals
        # print(f"iteraing over array: ")
        # print(np.argwhere(end > 0))
        # print()

        for ind in np.argwhere(end > 0):
            Y[ind, action_inds[ind]] = reward[ind]

        for ind in np.argwhere(end <= 0):
            aind = action_inds[ind]
            Y[ind, aind] = reward[ind] + gamma * (future_max[ind] - Y[ind, aind])
            # Y[ind, aind] = reward[ind] + gamma * (future_max[ind])

        # print("Y:")
        # print(Y.shape)
        # print(Y[:5, :])
        # print("State:")
        # print(old_state[:5, :])
        self.model.fit(old_state, Y, batch_size=self.batch_size)

    def _inner_step_one_agent_many_envs(self, random_action=False):
        old_states = self.states[list(self.alive_envs)]

        if random_action:
            actions = np.random.randint(0, self.output_size, (self.envs_n,))
            # print("random:   ", actions)
        else:
            actions = self.model.predict(old_states, verbose=False)
            actions = np.argmax(actions, axis=1)
            # print("predicted:", actions)

        rewards_text = ""

        # print()
        # print(random_action)
        # print(actions)
        for current_i, (act, env_ind) in enumerate(zip(actions, list(self.alive_envs))):
            "Loop over environments"
            new_state, reward, end, info = self.envs[env_ind].step(act)
            # print(reward)
            # reward = self.tweak_reward(None, new_state, reward)

            if self.notify == 2:
                if reward >= 0:
                    rewards_text += f"{reward}, "
            sample = np.concatenate([old_states[current_i, :], new_state, [act], [reward], [end]],
                                    dtype=np.float32)
            if end:
                self.alive_envs.remove(env_ind)

            self.add_memory(sample)
        if len(rewards_text) > 0 and self.notify == 2:
            print(f"Rewards at step: {rewards_text}")

    # def _inner_train_one_agent_many_envs(self):
    #     if self.memory_fully_writen or self.memory_write_index > self.min_samples:
    #         train_data = self.get_training_samples()
    #     else:
    #         return None
    #     ind1, ind2 = self.sample_indexes
    #     old_state = train_data[:, :ind1]
    #     new_state = train_data[:, ind1:ind2]
    #     action_inds = train_data[:, ind2].astype(np.int32)
    #     reward = train_data[:, ind2 + 1].reshape((-1, 1))
    #     end = train_data[:, ind2 + 2]
    #
    #     # print("FIT Y")
    #
    #     action = np.zeros((action_inds.shape[0], self.output_size * self.action_size))
    #     # action[:, action_inds] = 1
    #     for aci, ind in enumerate(action_inds):
    #         # print(aci, ind)
    #         action[aci, ind] = 1
    #
    #     future_qvals = self.model.predict(new_state)
    #     future_max = np.max(future_qvals, axis=1).reshape(-1, 1)
    #
    #     gamma = 0.9
    #     Y = gamma * (future_max + reward)
    #     Y[end] = reward[end]
    #
    #     print("actions", action_inds)
    #     self.model.fit(old_state, Y)

    def _inner_step_many_agents_many_envs(self):
        """"""
        raise NotImplemented

    def _inner_train_many_agents_many_envs(self):
        """"""
        raise NotImplemented

    def add_memory(self, sample):
        self.memory[self.memory_write_index] = sample

        if self.memory_write_index >= self.memory_last_index:
            self.memory_write_index = 0
            self.memory_fully_writen = True
        else:
            self.memory_write_index += 1

    def tweak_reward(self, prev_state, state, reward):
        # print(state)
        reward += np.abs(state[1]) * 20
        return reward

    def render(self, max_iters=100):
        pygame.display.init()

        # rewards = np.zeros(max_iters + 1)
        end = False
        game = self.envs[0]
        state = game.reset()
        game.render()
        time.sleep(2)

        i = 0
        while not end:
            state = state.reshape(1, -1)
            # print(state)
            game.render()
            qvals = self.model.predict(state, verbose=False)
            act = np.argmax(qvals)
            print(state, act, qvals)
            state, reward, end, info = game.step(act)
            time.sleep(0.01)
            # rewards[i] = reward
            i += 1
            if i >= max_iters:
                break

        # print(f"max i: {i}")
        # rewards = rewards[:i]

        time.sleep(5)
        # plt.figure()
        # plt.hist(rewards, bins=50)
        # plt.title("Visual rewards")
        # plt.show()


def simple_model(in_shape, out_shape):
    inp = Input(in_shape)
    lay = Dense(100, activation='relu')(inp)
    lay = Dense(100, activation='relu')(lay)
    out = Dense(out_shape, activation='linear')(lay)

    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['accuracy'])
    return model


def simple_env(n=1):
    envs = [gym.make('LunarLander-v2') for _ in range(n)]
    envs = [gym.make('MountainCar-v0') for _ in range(n)]
    return envs


if __name__ == "__main__":
    envs_n = 15
    input_size, output_size = 8, 4  # Lunar
    input_size, output_size = 2, 3  # Mountain Car
    model = simple_model(input_size, output_size)
    if os.path.isfile("mod1.weights"):
        pass
        # model.load_weights("mod1.weights")

    envs = simple_env(envs_n)
    trening = Trainer(model, envs,
                      config='onetomany', envs_n=envs_n,
                      input_size=input_size, action_size=1,
                      memory_max_samples=envs_n * 400,
                      batch_size=100, train_size=envs_n * 100,
                      notify_win=2,
                      )

    np.set_printoptions(suppress=True, precision=4)

    trening.start_training(epochs=5, max_iters=200, exploration_type='sin', exploration_ratio=0.4)
    rewards = tuple(sample[3] for sample in trening.memory)
    rewards = np.array(rewards)
    print("Rewards")
    print(str(rewards))
    plt.hist(rewards, bins=50)
    # print()
    model.save_weights("mod1.weights")

    "POST Train Render"
    game = simple_env(1)[0]
    end = False
    state = game.reset()
    game.render()
    time.sleep(5)

    rewards = np.zeros_like(rewards)[:201]
    i = 0
    while not end:
        state = state.reshape(1, -1)
        # print(state)
        game.render()
        qvals = model.predict(state, verbose=False)
        act = np.argmax(qvals)
        print(state, act, qvals)
        state, reward, end, info = game.step(act)
        time.sleep(0.01)
        rewards[i] = reward
        i += 1

    print(f"max i: {i}")
    rewards = rewards[:i]

    plt.figure()
    plt.hist(rewards, bins=50)
    plt.title("Visual rewards")
    plt.show()
    time.sleep(10)

    # print(tf.)
    # print(tf.test.is_gpu_available())
