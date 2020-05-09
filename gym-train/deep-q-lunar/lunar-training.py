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

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, concatenate
from keras.models import Model, load_model, Sequential
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from collections import deque
from matplotlib import style
from keras import backend


class Agent:
    def __init__(self,
                 input_shape,
                 action_space,
                 alpha,
                 beta,
                 gamma=0.99,
                 dense1=256,
                 dense2=256):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"

        self.input_shape = input_shape
        self.action_space = action_space
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if settings.LOAD_MODEL:
            try:
                layers = np.load(f"{settings.MODEL_NAME}/model/layers.npy", allow_pickle=True)
                self.dense1, self.dense2 = layers
                print(f"Loaded layers shapes: {settings.MODEL_NAME}")
            except FileNotFoundError:

                self.dense1, self.dense2 = dense1, dense2
            self.actor, self.critic, self.policy = self.create_actor_critic_network()
            loaded = self.load_model()
            if loaded:
                print(f"Loading weights: {settings.MODEL_NAME}")
            else:
                print(f"Not loaded weights: {settings.MODEL_NAME}")

        else:
            self.dense1, self.dense2 = dense1, dense2
            print(f"New model: {settings.MODEL_NAME}")
            self.actor, self.critic, self.policy = self.create_actor_critic_network()

    def create_actor_critic_network(self):
        input1 = Input(shape=self.input_shape)
        delta = Input(shape=(1,))
        initializer = RandomUniform(minval=-1e-1, maxval=1e-1)
        dense1 = Dense(self.dense1, activation='relu',
                       kernel_initializer=initializer)(input1)
        dense2 = Dense(self.dense2, activation='relu',
                       kernel_initializer=initializer)(dense1)

        probs = Dense(self.action_space, activation='softmax')(dense2)
        engines = Dense(self.action_space, activation='tanh')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = backend.clip(y_pred, 1e-8, 1 - 1e-8)
            log_like = y_true * backend.log(out)
            return backend.sum(-log_like * delta)

        actor = Model(inputs=[input1, delta], outputs=[engines])
        actor.compile(optimizer=Adam(self.alpha), loss=custom_loss)

        critic = Model(inputs=[input1], outputs=[values])
        critic.compile(optimizer=Adam(self.beta), loss='mean_squared_error')

        merged = concatenate([engines, probs])
        acts = Dense(self.action_space, activation='tanh')(merged)
        policy = Model(inputs=[input1], outputs=[acts])

        os.makedirs(f"{settings.MODEL_NAME}/model", exist_ok=True)

        plot_model(actor, f"{settings.MODEL_NAME}/model/actor.png")
        plot_model(critic, f"{settings.MODEL_NAME}/model/critic.png")
        plot_model(policy, f"{settings.MODEL_NAME}/model/policy.png")

        with open(f"{settings.MODEL_NAME}/model/actor-summary.txt", 'w') as file:
            actor.summary(print_fn=lambda x: file.write(x + '\n'))
        with open(f"{settings.MODEL_NAME}/model/critic-summary.txt", 'w') as file:
            critic.summary(print_fn=lambda x: file.write(x + '\n'))
        with open(f"{settings.MODEL_NAME}/model/policy-summary.txt", 'w') as file:
            policy.summary(print_fn=lambda x: file.write(x + '\n'))

        return actor, critic, policy

    def save_model(self):
        while True:
            try:
                self.actor.save_weights(f"{settings.MODEL_NAME}/model/actor-weights")
                break
            except OSError:
                time.sleep(0.2)
        while True:
            try:
                self.critic.save_weights(f"{settings.MODEL_NAME}/model/critic-weights")
                break
            except OSError:
                time.sleep(0.2)

        while True:
            try:
                self.policy.save_weights(f"{settings.MODEL_NAME}/model/policy-weights")
                break
            except OSError:
                time.sleep(0.2)
        np.save(f"{settings.MODEL_NAME}/model/layers.npy", (self.dense1, self.dense2))
        return True

    def choose_action_list(self, States):
        Actions = self.policy.predict(States)
        return Actions

    def load_model(self):
        if os.path.isfile(f"{settings.MODEL_NAME}/model/actor-weights") and \
                os.path.isfile(f"{settings.MODEL_NAME}/model/critic-weights") and \
                os.path.isfile(f"{settings.MODEL_NAME}/model/policy-weights"):
            while True:
                try:
                    self.actor.load_weights(f"{settings.MODEL_NAME}/model/actor-weights")
                    break
                except OSError:
                    time.sleep(0.2)

            while True:
                try:
                    self.critic.load_weights(f"{settings.MODEL_NAME}/model/critic-weights")
                    break
                except OSError:
                    time.sleep(0.2)

            while True:
                try:
                    self.policy.load_weights(f"{settings.MODEL_NAME}/model/policy-weights")
                    break
                except OSError:
                    time.sleep(0.2)
            return True

        else:
            return False

    def train(self, train_data):
        self.actor_critic_train(train_data)

    def actor_critic_train(self, train_data):
        Old_states = []
        New_states = []
        Rewards = []
        Dones = []
        Actions = []

        for old_state, action, reward, new_state, done in zip(train_data[0], train_data[1], train_data[2],
                                                              train_data[3], train_data[4]):
            Old_states.append(old_state)
            New_states.append(new_state)
            Actions.append(action)
            Rewards.append(reward)
            Dones.append(done)

        Old_states = np.array(Old_states)
        New_states = np.array(New_states)
        Rewards = np.array(Rewards)
        Actions = np.array(Actions)

        current_critic_value = self.critic.predict(Old_states).ravel()
        future_critic_values = self.critic.predict(New_states).ravel()  # Converting to vector
        # print()
        # print(current_critic_value)
        # print(future_critic_values)
        int_dones = np.array([*map(lambda x: int(not x), Dones)])
        targets = Rewards + self.gamma * int_dones * future_critic_values
        delta = targets - current_critic_value

        self.actor.fit([Old_states, delta], Actions, verbose=0)
        self.critic.fit(Old_states, targets, verbose=0)


def training():
    eps_iter = iter(np.linspace(settings.RAMP_EPS, settings.END_EPS, settings.EPS_INTERVAL))
    time_start = time.time()
    emergency_break = False

    for episode in range(0, settings.EPOCHS):
        try:
            if not (episode + episode_offset) % settings.SHOW_EVERY:
                render = True
            else:
                render = False

            if episode == settings.EPOCHS - 1 or emergency_break:
                eps = 0
                render = True
                if settings.SHOW_LAST:
                    input("Last agent is waiting...")
            elif episode == 0 and settings.SHOW_FIRST or not settings.ALLOW_TRAIN:
                eps = 0
                render = True
            elif episode < settings.EPS_INTERVAL / 4:
                eps = settings.FIRST_EPS
            # elif episode < EPS_INTERVAL:
            #     eps = 0.3
            else:
                try:
                    eps = next(eps_iter)
                except StopIteration:
                    eps_iter = iter(np.linspace(settings.INITIAL_SMALL_EPS, settings.END_EPS, settings.EPS_INTERVAL))
                    eps = next(eps_iter)

            Games = []  # Close screen
            States = []
            for loop_ind in range(settings.SIM_COUNT):
                game = gym.make('LunarLanderContinuous-v2')
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
                if eps > np.random.random():
                    Actions = np.random.random((len(Games), 2)) * 2 - 1
                else:
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
                    # print(Actions[0])
                    if settings.RECORD_GAME:
                        array = Games[0].viewer.get_array()
                        cv2.imwrite(f"{settings.MODEL_NAME}/game-{episode_offset}/{step}.png", array[:, :, [2, 1, 0]])
                    else:
                        time.sleep(settings.RENDER_DELAY)

                if settings.ALLOW_TRAIN:
                    train_data = (Old_states, Actions, Rewards, States, Dones)
                    agent.train(train_data)
                    if not (episode + episode_offset) % 25 and episode > 0:
                        agent.save_model()
                        np.save(f"{settings.MODEL_NAME}/last-episode-num.npy", episode + episode_offset)

                for ind_d in range(len(Games) - 1, -1, -1):
                    if Dones[ind_d]:
                        if ind_d == 0 and render:
                            render = False
                            Games[0].close()

                        All_score.append(Scores[ind_d])
                        All_steps.append(step)

                        stats['episode'].append(episode + episode_offset)
                        stats['eps'].append(eps)
                        stats['score'].append(Scores[ind_d])
                        stats['flighttime'].append(step)

                        Scores.pop(ind_d)
                        Games.pop(ind_d)
                        States.pop(ind_d)

        except KeyboardInterrupt:
            emergency_break = True

        print(f"Step-Ep[{episode + episode_offset:^7} of {settings.EPOCHS + episode_offset}], "
              f"Eps: {eps:>1.3f} "
              f"avg-score: {np.mean(All_score):^8.1f}, "
              f"avg-steps: {np.mean(All_steps):^7.1f}"
              )
        time_end = time.time()
        if emergency_break:
            break
        elif settings.TRAIN_MAX_MIN_DURATION and (time_end - time_start) / 60 > settings.TRAIN_MAX_MIN_DURATION:
            emergency_break = True

    print(f"Run ended: {settings.MODEL_NAME}")
    print(f"Step-Training time elapsed: {(time_end - time_start) / 60:3.1f}m, "
          f"{(time_end - time_start) / (episode + 1):3.1f} s per episode")

    if settings.ALLOW_TRAIN:
        agent.save_model()
        np.save(f"{settings.MODEL_NAME}/last-episode-num.npy", episode + 1 + episode_offset)


def moving_average(array, window_size=None, multi_agents=1):
    size = len(array)

    if not window_size or window_size and size > window_size:
        window_size = size // 20

    window_size *= multi_agents

    while len(array) % window_size or window_size % multi_agents:
        window_size -= 1
        if window_size < 1:
            window_size = 1
            break

    output = []

    for sample_num in range(multi_agents - 1, len(array), multi_agents):
        if sample_num < window_size:
            output.append(np.mean(array[:sample_num + 1]))
        else:
            output.append(np.mean(array[sample_num - window_size: sample_num + 1]))

    if len(array) % window_size:
        output.append(np.mean(array[-window_size:]))

    return output


def plot_results():
    print("Plotting data now...")

    style.use('ggplot')
    plt.figure(figsize=(20, 11))
    X = range(stats['episode'][0], stats['episode'][-1] + 1)
    plt.subplot(411)
    effectiveness = [score / moves for score, moves in zip(stats['score'], stats['flighttime'])]
    plt.scatter(stats['episode'], effectiveness, label='Effectiveness', color='b', marker='o', s=10, alpha=0.5)
    plt.plot(X, moving_average(effectiveness, multi_agents=settings.SIM_COUNT), label='Average', linewidth=3)
    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.3)
    plt.legend(loc=2)

    plt.subplot(412)
    plt.suptitle(f"{settings.MODEL_NAME}\nStats")
    plt.scatter(
            np.array(stats['episode']),
            stats['score'],
            alpha=0.2, marker='s', c='b', s=10, label="Score"
    )

    plt.plot(X, moving_average(stats['score'], multi_agents=settings.SIM_COUNT), label='Average', linewidth=3)
    plt.legend(loc=2)

    plt.subplot(413)
    plt.scatter(stats['episode'], stats['flighttime'], label='Flight-time', color='b', marker='o', s=10, alpha=0.5)
    plt.plot(X, moving_average(stats['flighttime'], multi_agents=settings.SIM_COUNT), label='Average',
             linewidth=3)
    plt.legend(loc=2)

    plt.subplot(414)
    plt.plot(stats['episode'], stats['eps'], label='eps', color='k')
    plt.legend(loc=2)

    if settings.SAVE_PICS and not settings.RECORD_GAME:
        plt.savefig(f"{settings.MODEL_NAME}/scores-{agent.runtime_name}.png")

    if not settings.SAVE_PICS:
        plt.show()

    if settings.SOUND_ALERT:
        os.system("play -nq -t alsa synth 0.3 sine 350")


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.compat.v1.Session(config=config)

    try:
        if settings.LOAD_MODEL:
            episode_offset = np.load(f"{settings.MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
        else:
            episode_offset = 0
    except FileNotFoundError:
        episode_offset = 0

    os.makedirs(f"{settings.MODEL_NAME}/game-{episode_offset}", exist_ok=True)

    "Environment"
    ACTION_SPACE = 2  # Turn left, right or none
    INPUT_SHAPE = (8,)

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "flighttime": []}

    agent = Agent(alpha=1e-5, beta=3e-6, gamma=0.99,
                  input_shape=INPUT_SHAPE,
                  action_space=ACTION_SPACE,
                  dense1=settings.DENSE1,
                  dense2=settings.DENSE2)
    training()
    plot_results()
