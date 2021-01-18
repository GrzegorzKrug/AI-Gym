import tensorflow as tf
import pickle
import numpy as np
import settings
import datetime
import random
import time
import os

from keras.layers import Dense, Flatten, Softmax, Input, concatenate, Conv2D, MaxPool2D
from keras.models import Sequential, load_model, Model
from keras.initializers import RandomUniform
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.optimizers import Adam
from collections import deque
from matplotlib import style
from keras import backend


class CustomTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 0
        self._log_write_dir = self.log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class FlexConvModel:
    def __init__(
            self,
            input_shape,
            output_shape,
            output_method="linear",
            conv_shapes=None,
            node_shapes=None,

            memory_size=10_000,

            alpha=0.99,
            beta=0.99,
            gamma=0.99,
    ):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"
        self.name = settings.MODEL_NAME

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_method = output_method
        self.conv_shapes = conv_shapes
        self.node_shapes = node_shapes
        self.memory_size = memory_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.memory = deque(maxlen=memory_size)
        self.model = None

        if settings.LOAD_MODEL:
            self.load_model()
        else:
            self.make_model()
            print(f"New model: {settings.MODEL_NAME}")

        self.make_model_dir()

    def make_model_dir(self):
        os.makedirs(os.path.dirname(self.path_params), exist_ok=True)

    @property
    def params(self):
        """
        Specify what params to save and load
        Returns:

        """
        return self.input_shape, self.output_shape, self.output_method, self.conv_shapes, self.node_shapes

    @params.setter
    def params(self, new_params):
        """
        Specify what params to save and load
        Returns:

        """
        self.input_shape, self.output_shape, self.output_method, self.conv_shapes, self.node_shapes = new_params

    def load_model(self):
        print(f"Loading model {self.name}")
        self._load_params()
        self.load_model_weights()
        self._load_memory()
        self.make_model()

    def make_model(self):
        pass

    def _load_params(self):
        try:
            pars = np.load(self.path_params, allow_pickle=True)
            self.params = pars
            print(f"Loading params {self.path_params}")
        except FileNotFoundError:
            print(f"Not found params to load {self.path_params}")

    def _save_params(self):
        arr = np.array(self.params, dtype=object)
        np.save(self.path_params, arr)
        print(f"Saving params {self.path_params}")

    def _load_memory(self):
        try:
            mem = np.load(self.path_memory, allow_pickle=True)
            self.memory = mem
            print(f"Loading memory {self.path_memory}")
        except FileNotFoundError:
            print(f"Not found memory file {self.path_memory}")

    def _save_memory(self):
        np.save(self.path_memory, self.memory)
        print(f"Saving memory {self.path_memory}")

        # def actor_critic_train(self, train_data):
        #     Old_states = []
        #     New_states = []
        #     Rewards = []
        #     Dones = []
        #     Actions = []
        #
        #     for old_state, action, reward, new_state, done in train_data:
        #         Old_states.append(old_state)
        #         New_states.append(new_state)
        #         Actions.append(action)
        #         Rewards.append(reward)
        #         Dones.append(int(not done))
        #
        #     Old_states = np.array(Old_states)
        #     Rewards = np.array(Rewards)
        #     Actions = np.array(Actions)
        #     Dones = np.array(Dones)
        #
        #     current_critic_value = self.critic.predict([Old_states]).ravel()
        # future_actions = self.choose_action_list([New_states])
        # future_critic_values = self.critic.predict([New_states]).ravel()
        #
        # target = Rewards + self.gamma * future_critic_values * Dones
        # delta = target - current_critic_value
        # target_actions = np.zeros((len(Actions), self.action_space))
        #
        # for ind, act in enumerate(Actions):
        #     target_actions[ind][act] = 1
        #
        # self.actor.fit([Old_states, delta], target_actions, verbose=0, callbacks=[self.actor_tb])
        # self.critic.fit([Old_states], target, verbose=0, callbacks=[self.critic_tb])

    @property
    def path_model_weights(self):
        return os.path.abspath(f"{settings.MODEL_NAME}{os.sep}model{os.sep}weights")

    @property
    def path_params(self):
        return os.path.abspath(f"{settings.MODEL_NAME}{os.sep}model{os.sep}params-model.npy")

    @property
    def path_memory(self):
        return os.path.abspath(f"{settings.MODEL_NAME}{os.sep}model{os.sep}memory.npy")

    def save_model(self):
        if not settings.SAVE_MODEL:
            print("Saving is not allowed")
            return None

        self._save_weights(self.model, self.path_model_weights)
        self._save_memory()
        self._save_params()

        return True

    @staticmethod
    def _save_weights(mod, path):
        while True:
            try:
                mod.save_weights(path)
                break
            except OSError:
                time.sleep(0.2)
            except AttributeError as err:
                print(f"Can not save weights: {err} to {path}")
                break

    @staticmethod
    def _load_weights(mod, path):
        f1 = os.path.isfile(path)
        if f1:
            while True:
                try:
                    mod.load_weights(path)
                    print(f"Loaded weights: {path}")
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            print(f"No weights: {path}")

    # def choose_action_list(self, States):
    #     probs = self.policy.predict([States])
    #     actions = np.array([np.random.choice(settings.ACTION_SPACE, p=p) for p in probs])
    #     return actions

    def load_model_weights(self):
        self._load_weights(self.model, self.path_model_weights)

    def add_memory(self, data):
        self.memory.append(data)


class DQNAgent(FlexConvModel):
    def make_model(self):
        pass

    def train(self):
        pass


agent = DQNAgent(
        input_shape=settings.INPUT_SHAPE,
        output_shape=settings.OUTPUT_SHAPE,
        conv_shapes=settings.CONV_SHAPES,
        node_shapes=settings.NODE_SHAPES,

)

agent.save_model()
