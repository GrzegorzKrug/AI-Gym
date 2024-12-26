import gymnasium as gym
# import gym


import matplotlib.pyplot as plt
# import tensorflow as tf
import random
import numpy as np
# import numba
import time

import keras
import os

from matplotlib.style import use
from yasiu_vis.ykeras import plotLayersWeights

from tensorflow.keras import regularizers
from tensorflow.keras.losses import Huber


# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Concatenate, Dropout, Input
# from tensorflow.keras.optimizers import Adam

import pygame


class TrainerDeepQ:
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

    def __init__(
        self,
        model, environments,
        # envs_n=10,
        input_size=2, action_size=2,
        memory_max_samples=5000,
        #  train_each_step=False, split_train=False,
        config='OneToMany',
        notify_win=2,
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
        "Model"
        self.envs = environments
        "Environments list"
        self.config = self.configs.get(config.lower(), 0)
        """
            0: OneToOne - One agent one env

            1: OneToMany - One agent, many envs

            2: ManyToMany - (Not implemented) Many agents with many envs
        """

        self.config_name = config
        self.input_size = input_size
        # self.output_size = output_size
        self.action_size = action_size

        self.min_train_samples = 10
        # self.max_samples = 200

        "Training parameters"
        self.gamma = None
        "Float: Gamma variable to be overriten by train function"
        self.batch_size = None
        "Int: Variable to be overriten by train function"
        self.train_size = None
        "Int: Variable to be overriten by train function"
        self.rewardTweakScale = 0
        "Float: Variable to be overriten by train function"
        self.rewardTweakEndFlag = False
        "Bool: Variable to be overriten by train function"

        "Config"
        if self.config == 0:
            self.envs_n = 1
        elif self.config == 1:
            # self.envs_n = envs_n
            self.envs_n = len(self.envs)
        else:
            raise NotImplementedError(
                f"Not implemented config {self.config_name}")
        print(f"Starting Trainer with config: {self.config} ({self.config_name})")

        "Memory settings (fitting)"
        self.memory = np.zeros(
            (memory_max_samples, (input_size * 2 + 3)), dtype=float)
        # sample : prevState currentState, endFlag, reward, single action
        "Memory array, Shape: memory_max_samples, stateSize*2 reward, End, action"

        self.memory_write_index = 0
        "Index to write new sample to"
        self.memory_last_index = memory_max_samples - 1
        "Clip write index to this max"
        self.memory_fully_writen = False
        "Flag set when memory is full"

        self.states = np.zeros((self.envs_n, self.input_size), dtype=np.float32)
        "Last state memory"
        # self.actions = np.zeros((self.envs_n, self.action_size), dtype=np.float32)
        # self.rewards = np.zeros((self.envs_n, 1), dtype=np.float32)

        self.alive_envs = set(range(self.envs_n))
        "Set of integers: Indexes of alive environments (not ended)"

        self.notify = notify_win
        """
            notify_win:
                0 - None,
                1 - Cumulated data,
                2 - Every game,
        """

        self.iters = 0

        "Additional trainer setup"
        self.post_init_called = False
        self.__post_init__()
        self.reset()

    def __post_init__(self):
        self._init_set_step_func()
        self._init_set_train_func()

        self.post_init_called = True

    def _init_set_step_func(self):
        """
        Override step function
        """
        config = self.config
        if config == 0:
            self._inner_step = self._inner_step_single
        elif config == 1:
            self._inner_step = self._inner_step_one_agent_many_envs
        else:
            self._inner_step = self._inner_step_many_agents_many_envs

    def _init_set_train_func(self):
        """
        Override training function
        """
        cf = self.config
        if cf == 0:
            self._inner_train = self._inner_train_single
        elif cf == 1:
            self._inner_train = self._inner_train_single
            # self._inner_train = self._inner_train_one_agent_many_envs # Redundant
        else:
            self._inner_train = self._inner_train_many_agents_many_envs

    def reset(self):
        self.states = np.zeros(
            (self.envs_n, self.input_size), dtype=np.float32)
        # self.actions = np.zeros((self.envs_n, self.action_size), dtype=np.int32)
        # self.rewards = np.zeros((self.envs_n, 1), dtype=np.float32)
        self.alive_envs = set(range(self.envs_n))

        for i in range(self.envs_n):
            # print(f"I:{i}")
            state, _ = self.envs[i].reset()
            # print(state)
            self.states[i] = state

    @staticmethod
    def calcExplorationChance(curEp, epMax, cyclesNum, amplitude):
        if cyclesNum > 1:
            cyclesNum = cyclesNum*2-1
        return (np.cos(curEp/epMax*np.pi*cyclesNum)+1)/2

    def start_training(
        self, epochs=1, max_iters=10,
        explorAmplitude=0.35,
        min_exploration=4e-3,
        explorationCyclesNum=3,
        gamma=0.9,

        # TF Parameters
        batch_size=200, train_size=2000,
        rewardTweakScale=0, rewardTweakEndFlag=False,
    ):
        """_summary_

        Parameters
        ----------
        epochs : int, optional
            _description_, by default 1
        max_iters : int, optional
            _description_, by default 10
        explorAmplitude : float, optional
            _description_, by default 0.35
        min_exploration : _type_, optional
            _description_, by default 4e-3
        explorationCyclesNum : int, optional
            _description_, by default 3
        gamma : float, optional
            _description_, by default 0.9
        batch_size : int, optional
            _description_, by default 200
        train_size : int, optional
            _description_, by default 2000
        rewardTweakScale : int, optional
            _description_, by default 0
        rewardTweakEndFlag : bool, optional
            _description_, by default False
        """

        self.gamma = gamma
        self.batch_size = batch_size
        self.train_size = train_size
        self.rewardTweakScale = rewardTweakScale
        self.rewardTweakEndFlag = rewardTweakEndFlag

        for cur_ep in range(epochs):
            "Iterate over epochs"
            self.reset()  # Clear environment at start
            exploreChance = self.calcExplorationChance(
                cur_ep, epochs-1, explorationCyclesNum, explorAmplitude
            )
            exploreChance = np.clip(exploreChance, min_exploration, 1.0)
            # current_explore_ratio = np.cos(
            #     cur_ep * np.pi / exploration_period / 2) * exploration_ratio

            exploreChance = np.abs(exploreChance)
            if exploreChance < min_exploration:
                exploreChance = min_exploration

            print(f"Epoch: {cur_ep:>3}, exploration: {exploreChance:3.4f}")

            for n in range(max_iters):
                if len(self.alive_envs) <= 0:
                    break
                rnd_act = True if exploreChance >= np.random.random() else False
                self._inner_step(random_action=rnd_act)
                "Apply Action to each environment"

            self._inner_train()

        self.iters += cur_ep + 1

    def _inner_step(self, random_action=False):
        """
        Abstraction
            random_action [boolean] - model choice or random action
        """
        raise NotImplemented("This is Abstract")

    def _inner_step_single(self, random_action=False):
        "Step function for single Environment"
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

    def get_training_samples(self, k_samples):
        """
        Return random samples from memory for training.
        """
        if k_samples <= self.memory_write_index:
            "Enough samples"
            inds = random.sample(range(self.memory_last_index), k_samples)

        elif self.memory_fully_writen:
            "Requested to much training samples"
            inds = random.sample(range(self.memory_last_index), k_samples)

        else:
            "Not enough samples, get all"
            k_samples = self.memory_write_index
            inds = random.sample(range(self.memory_write_index), k_samples)

        samples = self.memory[inds, :]

        return samples

    @property
    def sample_indexes(self):
        """Indexes in memmory"""
        return self.input_size, self.input_size * 2,

    def _inner_train(self):
        """ Abstraction """
        raise NotImplemented("This is Abstract")

    def _inner_train_single(self):
        if self.memory_fully_writen or self.memory_write_index > self.min_train_samples:
            train_data = self.get_training_samples(self.train_size)
        else:
            return None

        ind1, ind2 = self.sample_indexes
        old_state = train_data[:, :ind1]
        new_state = train_data[:, ind1:ind2]
        action_inds = train_data[:, ind2].astype(np.int32)
        reward = train_data[:, ind2 + 1]  # .reshape((-1, 1))
        end = train_data[:, ind2 + 2]

        current_qvals = self.model.predict(old_state, verbose=False)
        future_qvals = self.model.predict(new_state, verbose=False)
        # future_max = np.max(future_qvals, axis=1).reshape(-1, 1)
        future_max = np.expand_dims(np.max(future_qvals, axis=1), axis=1)

        gamma = self.gamma
        TargetQ = current_qvals
        
        print(f"Tweak scale: {self.rewardTweakScale:>3.2f}")
        if self.rewardTweakScale > 0 and self.rewardTweakEndFlag:
            reward = self.tweakReward(new_state, old_state, reward, self.rewardTweakScale)

        "Changing Targets for Terminated Envs"
        for ind in np.argwhere(end > 0):
            TargetQ[ind, action_inds[ind]] = reward[ind]

        if self.rewardTweakScale > 0 and not self.rewardTweakEndFlag:
            reward = self.tweakReward(new_state, old_state, reward, self.rewardTweakScale)

        "Changing Targets for Active Envs"
        for ind in np.argwhere(end <= 0):
            aind = action_inds[ind]

            newVal = reward[ind] + gamma * (future_max[ind])  # Deep Learning formula

            TargetQ[ind, aind] = newVal

        self.model.fit(old_state, TargetQ, batch_size=self.batch_size)

    @staticmethod
    def tweakReward(state, prevState, reward, scale=1):
        """
            Custom feedback to increase triaining speed
        """

        return reward + np.clip(np.abs(state[:, 1])*1000, 0, 5) * scale

    def _inner_step_one_agent_many_envs(self, random_action=False):
        """Step function for many environments"""
        old_states = self.states[list(self.alive_envs)]

        if random_action:
            actions = np.random.randint(0, self.action_size, (self.envs_n,))
            # print("random:   ", actions)
        else:
            actions = self.model.predict(old_states, verbose=False)
            actions = np.argmax(actions, axis=1)
            # print("predicted:", actions)

        rewards_text = ""

        for scopeI, (act, env_ind) in enumerate(zip(actions, list(self.alive_envs))):
            "Loop over environments"
            observation, reward, terminated, truncated, info = \
                self.envs[env_ind].step(act)

            if self.notify == 2:
                if reward >= 0:
                    rewards_text += f"{reward}, "
            sample = np.concatenate(
                [old_states[scopeI, :], observation,
                 [act], [reward], [terminated]
                 ], dtype=np.float32
            )

            if terminated:
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
        raise NotImplementedError

    def _inner_train_many_agents_many_envs(self):
        """"""
        raise NotImplementedError

    def add_memory(self, sample):
        self.memory[self.memory_write_index] = sample

        if self.memory_write_index >= self.memory_last_index:
            self.memory_write_index = 0
            self.memory_fully_writen = True
        else:
            self.memory_write_index += 1

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
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(in_shape,)))

    # model.add(keras.layers.Dense(20, activation='leaky_relu', kernel_regularizer=regularizers.l2(1e-7)))
    model.add(keras.layers.Dense(20, activation='leaky_relu'))
    model.add(keras.layers.Dense(20, activation='leaky_relu', kernel_regularizer=regularizers.l2(1e-6)))
    model.add(keras.layers.Dense(20, activation='leaky_relu', kernel_regularizer=regularizers.l2(1e-6)))

    model.add(keras.layers.Dense(out_shape, activation='linear'))
    # model.add(keras.layers.Dense(out_shape, activation='linear'))

    # model = Model(inputs=[inp], outputs=[out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
    )
    return model


def simple_env(n=1, *args, **kwargs):
    # envs = [gym.make('LunarLander-v3') for _ in range(n)]
    envs = [gym.make('MountainCar-v0', *args, **kwargs)
            for _ in range(n)]
    return envs


def plotModel(model, name):
    plotLayersWeights(
        model,
        figsize=(20, 10), dpi=80, scaleWeights=1000
    )
    plt.savefig(os.path.join(os.path.dirname(__file__), "pics", f"{name}.png"))
    plt.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "pics"), exist_ok=True)
    envs_n = 20
    input_size, output_size = 8, 4  # Lunar
    input_size, output_size = 2, 3  # Mountain Car

    model = simple_model(input_size, output_size)
    envs = simple_env(envs_n)

    model_path = os.path.join(os.path.dirname(__file__), "model1.weights.h5")

    if os.path.isfile(model_path):
        model.load_weights(model_path)

    trening = TrainerDeepQ(
        model, envs,
        config='onetomany', envs_n=envs_n,
        input_size=input_size, action_size=3,
        memory_max_samples=envs_n * 400,
        notify_win=2,
    )

    np.set_printoptions(suppress=True, precision=4)

    "Training"
    for ses in range(10):
        print(f"\nSess: {ses}")
        trening.start_training(
            epochs=6, max_iters=150,
            explorAmplitude=0.8, explorationCyclesNum=3,
            batch_size=100, train_size=envs_n * 100,
            rewardTweakScale=0.2/(1+ses),
            gamma=0.9,
        )

        model.save_weights(model_path)
        plotModel(model, f"model_postTraining-ep{ses:>04}")

    # "POST Train Render"
    game = simple_env(1, render_mode="human")[0]
    end = False
    # state, reward, end, info = game.step(act)
    state, info = game.reset()
    # print(state)
    # game.render()
    time.sleep(5)

    rewards = np.zeros(300, dtype=float)
    i = 0
    while not end:
        state = state.reshape(1, -1)
        # print(state)
        game.render()
        qvals = model.predict(state, verbose=False)
        act = np.argmax(qvals)
        print(i, state, act, qvals)
        state, reward, end, truncated, info = game.step(act)
        time.sleep(0.01)
        rewards[i] = reward
        i += 1
        if i > 250:
            break

    # print(f"max i: {i}")
    # rewards = rewards[:i]

    plt.figure()
    plt.hist(rewards, bins=50)
    plt.title("Visual rewards")
    # plt.show()
    time.sleep(5)

    # print(tf.)
    # print(tf.test.is_gpu_available())
