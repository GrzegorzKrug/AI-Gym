import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import settings
import datetime
import random
import keras

import time
import os

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, concatenate
from keras.models import Model, load_model, Sequential
from keras.utils import plot_model
from keras.optimizers import Adam
from collections import deque
from matplotlib import style
from keras import backend


class Agent:
    def __init__(self,
                 input_shape,
                 action_space,
                 dual_input=False,
                 min_batch_size=1000,
                 max_batch_size=1000,
                 learining_rate=0.0001,
                 memory_size=10000):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learining_rate
        self.memory = deque(maxlen=memory_size)
        load_success = self.load_model()

        # Bind train command
        self._train = self._dual_train if settings.DUAL_INPUT else self._normal_train

        if load_success:
            print(f"Loading model: {settings.MODEL_NAME}")
        else:
            print(f"New model: {settings.MODEL_NAME}")
            if dual_input:
                self.model = self.create_dual_model()
            else:
                self.model = self.create_normal_model()

        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss='mse',
                           metrics=['accuracy'])
        backend.set_value(self.model.optimizer.lr, self.learning_rate)
        self.model.summary()

    def create_dual_model(self):
        input_area = Input(shape=(self.input_shape[0]))
        layer1a = Dense(64, activation='relu')(input_area)

        input_direction = Input(shape=(self.input_shape[1]))
        layer2a = Dense(32, activation='relu')(input_direction)
        merge_layer = concatenate([layer1a, layer2a], axis=-1)

        layer3 = Dense(64, activation='relu')(merge_layer)
        output = Dense(self.action_space, activation='linear')(layer3)

        model = Model(inputs=[input_area, input_direction], outputs=output)

        plot_model(model, f"{settings.MODEL_NAME}/model.png")
        with open(f"{settings.MODEL_NAME}/model_summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        return model

    def create_normal_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))

        plot_model(model, f"{settings.MODEL_NAME}/model.png")
        with open(f"{settings.MODEL_NAME}/model_summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        return model

    def update_memory(self, state):
        self.memory.append(state)

    def save_model(self):
        while True:
            try:
                self.model.save(f"{settings.MODEL_NAME}/model")
                return True
            except OSError:
                time.sleep(0.2)

    def load_model(self):
        if os.path.isfile(f"{settings.MODEL_NAME}/model"):
            while True:
                try:
                    self.model = load_model(f"{settings.MODEL_NAME}/model")
                    return True
                except OSError:
                    time.sleep(0.2)
        else:
            return False

    def train(self):
        if len(self.memory) < self.min_batch_size:
            return None
        elif settings.TRAIN_ALL_SAMPLES:
            train_data = list(self.memory)
        elif len(self.memory) >= self.max_batch_size:
            train_data = random.sample(self.memory, self.max_batch_size)
            # print(f"Too much data, selecting from: {len(self.memory)} samples")
        else:
            train_data = list(self.memory)

        if settings.STEP_TRAINING or settings.TRAIN_ALL_SAMPLES:
            self.memory.clear()

        self._train(train_data)

    def _normal_train(self, train_data):
        Old_states = []
        New_states = []
        Rewards = []
        Dones = []
        Actions = []

        for old_state, new_state, reward, action, done in train_data:
            Old_states.append(old_state)
            New_states.append(new_state)
            Actions.append(action)
            Rewards.append(reward)
            Dones.append(done)

        Old_states = np.array(Old_states)
        New_states = np.array(New_states)
        old_qs = self.model.predict(Old_states)
        new_qs = self.model.predict(New_states)

        for old_q, new_q, rew, act, done in zip(old_qs, new_qs, Rewards, Actions, Dones):
            if done:
                old_q[act] = rew
            else:
                future_best_val = np.max(new_q)
                old_q[act] = rew + settings.DISCOUNT * future_best_val

        self.model.fit(Old_states, old_qs,
                       verbose=0, shuffle=False, epochs=1)

    def _dual_train(self, train_data):
        Old_states = []
        New_states = []
        Rewards = []
        Dones = []
        Actions = []

        for old_state, new_state, reward, action, done in train_data:
            Old_states.append(old_state)
            New_states.append(new_state)
            Actions.append(action)
            Rewards.append(reward)
            Dones.append(done)

        Old_states = np.array(Old_states)
        New_states = np.array(New_states)

        old_view_area = []
        old_direction = []
        new_view_area = []
        new_direction = []

        for _old_state, _new_state in zip(Old_states, New_states):
            old_view_area.append(_old_state[0])
            old_direction.append(_old_state[1])
            new_view_area.append(_new_state[0])
            new_direction.append(_new_state[1])

        old_qs = self.model.predict([old_view_area, old_direction])
        new_qs = self.model.predict([new_view_area, new_direction])

        for old_q, new_q, rew, act, done in zip(old_qs, new_qs, Rewards, Actions, Dones):
            if done:
                old_q[act] = rew
            else:
                future_best_val = np.max(new_q)
                old_q[act] = rew + settings.DISCOUNT * future_best_val

        self.model.fit([old_view_area, old_direction], old_qs,
                       verbose=0, shuffle=False, epochs=1)


# EPOCHS = settings.EPOCHS
# SIM_COUNT = settings.SIM_COUNT
#
# REPLAY_MEMORY_SIZE = settings.REPLAY_MEMORY_SIZE
# MIN_BATCH_SIZE = settings.MIN_BATCH_SIZE
# MAX_BATCH_SIZE = settings.MAX_BATCH_SIZE
#
# DISCOUNT = settings.DISCOUNT
# AGENT_LR = settings.AGENT_LR
# FREE_MOVE = settings.FREE_MOVE
#
# MODEL_NAME = settings.MODEL_NAME
# LOAD_MODEL = settings.LOAD_MODEL
# ALLOW_TRAIN = settings.ALLOW_TRAIN
# SAVE_PICS = settings.SAVE_PICS
#
# STATE_OFFSET = settings.STATE_OFFSET
# FIRST_EPS = settings.FIRST_EPS
# RAMP_EPS = settings.RAMP_EPS
# INITIAL_SMALL_EPS = settings.INITIAL_SMALL_EPS
# END_EPS = settings.END_EPS
# EPS_INTERVAL = settings.EPS_INTERVAL
#
# SHOW_EVERY = settings.SHOW_EVERY
# RENDER_DELAY = settings.RENDER_DELAY


def training():
    try:
        episode_offset = np.load(f"{settings.MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
    except FileNotFoundError:
        episode_offset = 0

    eps_iter = iter(np.linspace(settings.RAMP_EPS, settings.END_EPS, settings.EPS_INTERVAL))
    time_start = time.time()
    emergency_break = False

    for episode in range(0, settings.EPOCHS):
        try:
            if not settings.STEP_TRAINING and settings.ALLOW_TRAIN:
                agent.train()
                if not (episode + episode_offset) % 100 and episode > 0:
                    agent.save_model()
                    np.save(f"{settings.MODEL_NAME}/last-episode-num.npy", episode + episode_offset)

            Pred_sep.append(len(Predicts[0]))

            if not (episode + episode_offset) % settings.SHOW_EVERY:
                render = True
            else:
                render = False

            if episode == settings.EPOCHS - 1 or emergency_break:
                eps = 0
                render = True
                if settings.SHOW_LAST:
                    input("Last agent is waiting...")
            elif episode == 0 or not settings.ALLOW_TRAIN:
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
                game = None

                state = game.reset()
                Games.append(game)
                States.append(state)

            Dones = [False] * len(Games)
            Scores = [0] * len(Games)
            step = 0
            All_score = []
            All_steps = []
            while len(Games):
                if settings.STEP_TRAINING and settings.ALLOW_TRAIN:
                    agent.train()
                    if not (episode + episode_offset) % 100 and episode > 0:
                        agent.save_model()
                        np.save(f"{settings.MODEL_NAME}/last-episode-num.npy", episode + episode_offset)

                step += 1
                Old_states = np.array(States)

                if eps > np.random.random():
                    Actions = [game.random_action() for game in Games]
                else:

                    if settings.DUAL_INPUT:
                        old_view_area = []
                        old_directions = []
                        for _old_state in Old_states:
                            old_view_area.append(_old_state[0])
                            old_directions.append(_old_state[1])

                        Predictions = agent.model.predict([old_view_area, old_directions])
                    else:
                        Predictions = agent.model.predict(Old_states)

                    Actions = np.argmax(Predictions, axis=1)
                    # if settings.PLOT_FIRST_QS:
                    #     Predicts[0].append(Actions[0])
                    #     Predicts[1].append(Predictions[0][Actions[0]])
                    # elif PLOT_ALL_QS:
                    #     for predict in Predictions:
                    #         Predicts[0].append(predict[0])
                    #         Predicts[1].append(predict[1])
                    #         Predicts[2].append(predict[2])
                    #         Predicts[3].append(predict[3])

                States = []
                assert len(Games) == len(Dones)
                for g_index, game in enumerate(Games):
                    state, reward, done = game.step(action=Actions[g_index])
                    agent.update_memory((Old_states[g_index], state, reward, Actions[g_index], done))
                    Scores[g_index] += reward
                    Dones[g_index] = done
                    States.append(state)

                if render:
                    Games[0].draw(episode+episode_offset)
                    time.sleep(settings.RENDER_DELAY)

                for ind_d in range(len(Games) - 1, -1, -1):
                    if Dones[ind_d]:
                        if ind_d == 0 and render:
                            render = False

                        All_score.append(Scores[ind_d])
                        All_steps.append(step)

                        stats['episode'].append(episode + episode_offset)
                        stats['eps'].append(eps)
                        stats['score'].append(Scores[ind_d])
                        stats['food_eaten'].append(Games[ind_d].food_eaten)
                        stats['moves'].append(step)

                        Scores.pop(ind_d)
                        Games.pop(ind_d)
                        States.pop(ind_d)
                        Dones.pop(ind_d)
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


def moving_average(array, window_size=None):
    size = len(array)
    if not window_size or window_size and size > window_size:
        window_size = size // 10

    if window_size > 1000:
        window_size = 1000

    output = []
    for sample_num, arr_element in enumerate(array):
        arr_slice = array[sample_num-window_size:sample_num]
        if len(arr_slice) < window_size:
            output.append(np.mean(array[0:sample_num+1]))
        else:
            output.append(
                    np.mean(arr_slice)
            )
    return output


def plot_results():
    print("Plotting data now...")
    style.use('ggplot')
    plt.figure(figsize=(20, 11))

    plt.subplot(411)
    effectiveness = [food / moves for food, moves in zip(stats['food_eaten'], stats['moves'])]
    plt.scatter(stats['episode'], effectiveness, label='Effectiveness', color='b', marker='s', s=10, alpha=0.5)
    plt.plot(stats['episode'], moving_average(effectiveness), label='Average', linewidth=3)
    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.3)
    plt.legend(loc=2)

    plt.subplot(412)
    plt.suptitle(f"{settings.MODEL_NAME}\nStats")
    plt.scatter(
            np.array(stats['episode']),
            stats['food_eaten'],
            alpha=0.2, marker='s', c='b', s=10, label="Food_eaten"
    )

    plt.plot(stats['episode'], moving_average(stats['food_eaten']), label='Average', linewidth=3)
    plt.legend(loc=2)

    plt.subplot(413)
    plt.scatter(stats['episode'], stats['moves'], label='Moves', color='b', marker='.', s=10, alpha=0.5)
    plt.plot(stats['episode'], moving_average(stats['moves']), label='Average', linewidth=3)
    plt.legend(loc=2)

    plt.subplot(414)
    plt.scatter(stats['episode'], stats['eps'], label='Epsilon', color='k', marker='.', s=10, alpha=1)
    plt.legend(loc=2)

    if settings.SAVE_PICS:
        plt.savefig(f"{settings.MODEL_NAME}/food-{agent.runtime_name}.png")

    # BIG Q-PLOT
    # plt.figure(figsize=(20, 11))
    # plt.scatter(range(len(Predicts[0])), Predicts[0], c='r', label='up', alpha=0.2, s=3, marker='o')
    # plt.scatter(range(len(Predicts[1])), Predicts[1], c='g', label='right', alpha=0.2, s=3, marker='o')
    # plt.scatter(range(len(Predicts[2])), Predicts[2], c='m', label='down', alpha=0.2, s=3, marker='o')
    # plt.scatter(range(len(Predicts[3])), Predicts[3], c='b', label='left', alpha=0.2, s=3, marker='o')
    # y_min, y_max = np.min(Predicts), np.max(Predicts)
    #
    # for sep in Pred_sep:
    #     last_line, = plt.plot([sep, sep], [y_min, y_max], c='k', linewidth=0.3, alpha=0.2)
    #
    # plt.title(f"{MODEL_NAME}\nMovement 'directions' evolution in time, learning-rate:{AGENT_LR}\n")
    # last_line.set_label("Epoch separator")
    # plt.xlabel("Sample")
    # plt.ylabel("Q-value")
    # plt.legend(loc='best')
    #
    # if SAVE_PICS:
    #     plt.savefig(f"{MODEL_NAME}/Qs-{agent.runtime_name}.png")
    #
    if not settings.SAVE_PICS:
        plt.show()

    if settings.SOUND_ALERT:
        os.system("play -nq -t alsa synth 0.3 sine 350")


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.compat.v1.Session(config=config)

    os.makedirs(MODEL_NAME, exist_ok=True)

    "Environment"
    ACTIONS = 4  # Turn left, right or none
    INPUT_SHAPE = None

    Predicts = [[], [], [], []]
    Pred_sep = []

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "food_eaten": [],
            "moves": []}

    agent = Agent(min_batch_size=MIN_BATCH_SIZE,
                  max_batch_size=MAX_BATCH_SIZE,
                  input_shape=INPUT_SHAPE,
                  action_space=ACTIONS,
                  memory_size=REPLAY_MEMORY_SIZE,
                  learining_rate=AGENT_LR,
                  dual_input=settings.DUAL_INPUT)

    training()
    plot_results()
