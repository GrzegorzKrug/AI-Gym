import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import settings
import datetime
import random
import keras
import time
import gym
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
                 alpha,
                 beta,
                 gamma=0.99):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"

        self.input_shape = input_shape
        self.action_space = action_space
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # load_success = self.load_model()
        load_success = False

        print(f"New model: {settings.MODEL_NAME}")
        self.actor, self.critic, self.policy = self.create_actor_critic_network()

    def create_actor_critic_network(self):
        input = Input(shape=self.input_shape)
        delta = Input(shape=[1])

        dense1 = Dense(256, activation='relu')(input)
        dense2 = Dense(256, activation='relu')(dense1)

        probs = Dense(self.action_space, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = backend.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * backend.log(out)

            return backend.sum(-log_lik * delta)

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(self.alpha), loss=custom_loss)

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(self.beta), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        plot_model(actor, f"{settings.MODEL_NAME}/actor.png")
        plot_model(critic, f"{settings.MODEL_NAME}/critic.png")
        plot_model(policy, f"{settings.MODEL_NAME}/policy.png")

        with open(f"{settings.MODEL_NAME}/actor-summary.txt", 'w') as file:
            actor.summary(print_fn=lambda x: file.write(x + '\n'))
        with open(f"{settings.MODEL_NAME}/critic-summary.txt", 'w') as file:
            critic.summary(print_fn=lambda x: file.write(x + '\n'))
        with open(f"{settings.MODEL_NAME}/policy-summary.txt", 'w') as file:
            policy.summary(print_fn=lambda x: file.write(x + '\n'))

        return actor, critic, policy

    def save_model(self):
        while True:
            try:
                self.actor.save(f"{settings.MODEL_NAME}/actor")
                break
            except OSError:
                time.sleep(0.2)
        while True:
            try:
                self.critic.save(f"{settings.MODEL_NAME}/critic")
                break
            except OSError:
                time.sleep(0.2)

        while True:
            try:
                self.policy.save(f"{settings.MODEL_NAME}/policy")
                break
            except OSError:
                time.sleep(0.2)

        return True

    def choose_action(self, observation):

        state = observation
        probabilities = self.policy.predict(state)
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def load_model(self):
        if os.path.isfile(f"{settings.MODEL_NAME}/actor") and \
                os.path.isfile(f"{settings.MODEL_NAME}/critic") and \
                os.path.isfile(f"{settings.MODEL_NAME}/policy"):
            while True:
                try:
                    self.actor = load_model(f"{settings.MODEL_NAME}/actor")
                    break
                except OSError:
                    time.sleep(0.2)

            while True:
                try:
                    self.critic = load_model(f"{settings.MODEL_NAME}/critic")
                    break
                except OSError:
                    time.sleep(0.2)

            while True:
                try:
                    self.policy = load_model(f"{settings.MODEL_NAME}/policy")
                    break
                except OSError:
                    time.sleep(0.2)
            return True

        else:
            return False

    def train(self, train_data):
        self.actor_critic_train(train_data)

    #     if len(self.memory) < self.min_batch_size:
    #         return None
    #     elif settings.TRAIN_ALL_SAMPLES:
    #         train_data = list(self.memory)
    #     elif len(self.memory) >= self.max_batch_size:
    #         train_data = random.sample(self.memory, self.max_batch_size)
    #         # print(f"Too much data, selecting from: {len(self.memory)} samples")
    #     else:
    #         train_data = list(self.memory)
    #
    #     if settings.STEP_TRAINING or settings.TRAIN_ALL_SAMPLES:
    #         self.memory.clear()
    #
    #     self._normal_train(train_data)

    def actor_critic_train(self, train_data):
        Old_states = []
        New_states = []
        Rewards = []
        Dones = []
        Actions = []

        for old_state, action, reward, new_state, done in train_data:
            Old_states.append(old_state)
            New_states.append(new_state)
            Actions.append(action)
            Rewards.append(reward)
            Dones.append(done)

        current_critic_value = self.critic.predict(Old_states)
        future_critic_values = self.critic.predict((New_states))

        target = Rewards + self.gamma * Dones * future_critic_values
        delta = target - current_critic_value

        Target_Actions = np.zeros(len(train_data), self.action_space)
        for act_ind, action in enumerate(Actions):
            Target_Actions[act_ind, action] = 1.0

        self.actor.fit([Old_states, Target_Actions], Target_Actions, verbose=0)
        self.critic.fit(Old_states, target, verbose=0)


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

                Actions = agent.choose_action(Old_states)
                Dones = []
                Rewards = []
                States = []

                for g_index, game in enumerate(Games):
                    state, reward, done, info = game.step(action=Actions[g_index])
                    # agent.update_memory((Old_states[g_index], state, reward, Actions[g_index], done))
                    Rewards.append(reward)
                    Scores[g_index] += reward
                    Dones.append(done)
                    States.append(state)

                # if render:
                #     Games[0].render()
                #     time.sleep(settings.RENDER_DELAY)

                if settings.STEP_TRAINING and settings.ALLOW_TRAIN:
                    train_data = (Old_states, Actions, Rewards, States)
                    agent.train(train_data)
                    if not (episode + episode_offset) % 100 and episode > 0:
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


def moving_average(array, window_size=None):
    size = len(array)
    if not window_size or window_size and size > window_size:
        window_size = size // 20

    if window_size > 1000:
        window_size = 1000

    elif window_size < 1:
        window_size = 1

    output = []
    for sample_num, _ in enumerate(array):
        arr_slice = array[sample_num - window_size + 1:sample_num + 1]
        if len(arr_slice) < window_size:
            output.append(np.mean(array[0:sample_num + 1]))
            # print(sample_num, array[0:sample_num+1], np.mean(array[0:sample_num+1]))
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
    effectiveness = [score / moves for score, moves in zip(stats['score'], stats['flighttime'])]
    plt.scatter(stats['episode'], effectiveness, label='Effectiveness', color='b', marker='o', s=10, alpha=0.5)
    plt.plot(stats['episode'], moving_average(effectiveness), label='Average', linewidth=3)
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

    plt.plot(stats['episode'], moving_average(stats['score']), label='Average', linewidth=3)
    plt.legend(loc=2)

    plt.subplot(413)
    plt.scatter(stats['episode'], stats['flighttime'], label='Flight-time', color='b', marker='o', s=10, alpha=0.5)
    plt.plot(stats['episode'], moving_average(stats['flighttime']), label='Average', linewidth=3)
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

    os.makedirs(settings.MODEL_NAME, exist_ok=True)

    "Environment"
    ACTION_SPACE = 4  # Turn left, right or none
    INPUT_SHAPE = (8,)

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "flighttime": []}

    agent = Agent(alpha=1e-5, beta=3e-6, gamma=0.99,
                  input_shape=INPUT_SHAPE,
                  action_space=ACTION_SPACE)
    training()
    plot_results()
