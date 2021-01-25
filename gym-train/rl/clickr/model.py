import tensorflow as tf
import pickle
import numpy as np
import settings
import datetime
import random
import time
import os

from keras.layers import (
    Dense, Flatten, Dropout,
    Conv2D, MaxPool2D,
    Softmax, Input, concatenate,
)

from keras.models import Sequential, load_model, Model
from keras.initializers import RandomUniform
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.optimizers import Adam
from collections import deque
from matplotlib import style
from keras import backend
from copy import deepcopy


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()

        out = func(*args, **kwargs)

        tend = time.time()
        dur = tend - t0
        name = func.__qualname__
        if dur < 1:
            print(f"Execution: {name:<30} was {dur * 1000:>4.3f} ms")
        elif dur > 60:
            print(f"Execution: {name:<30} was {dur / 60:>4.3f} m")
        else:
            print(f"Execution: {name:<30} was {dur:>4.3f} s")

        return out

    return wrapper


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
            /,
            node_shapes=None,
            compiler=None,
            memory_size=settings.MEMORY_SIZE,

            alpha=0.99,
            beta=0.99,
            gamma=0.99,
    ):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"
        self.name = settings.MODEL_NAME

        self.node_shapes = node_shapes
        self.compiler = compiler
        self.memory_size = memory_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.memory = deque(maxlen=memory_size)
        self.model = None
        self.make_model_dir()

        if settings.LOAD_MODEL:
            self.load_model()
        else:
            self.make_model()
            print(f"New model: {settings.MODEL_NAME}")

    def make_model_dir(self):
        os.makedirs(os.path.dirname(self.path_params), exist_ok=True)

    @property
    def params(self):
        """
        Specify what params to save and load
        Returns:

        """
        return self.node_shapes, self.compiler

    @params.setter
    def params(self, new_params):
        """
        Specify what params to save and load
        Returns:

        """
        self.node_shapes, self.compiler = new_params

    def load_model(self):
        print(f"Loading model {self.name}")
        self._load_params()
        self.make_model()
        self.load_model_weights()
        self._load_memory()

    def make_model(self):
        raise NotImplemented("Define method in your subclass")

    def _load_params(self):
        try:
            pars = np.load(self.path_params, allow_pickle=True)
            self.params = pars
            print(f"Loading params {self.path_params}")
            print(self.params)
        except FileNotFoundError:
            print(f"Not found params to load {self.path_params}")

    def _save_params(self):
        arr = np.array(self.params, dtype=object)
        np.save(self.path_params, arr)
        print(f"Saving params {self.path_params}")

    def _load_memory(self, drop_last=0):
        try:
            mem = np.load(self.path_memory, allow_pickle=True)
            print(f"Loaded memory: {mem.shape} -> {mem.shape[0] - drop_last} samples")
            for ind, sample in enumerate(mem):
                if ind + drop_last >= len(mem):
                    break
                self.add_memory(sample)

            print(f"Loading memory {self.path_memory}")
        except FileNotFoundError:
            print(f"Not found memory file {self.path_memory}")

    @timeit
    def _save_memory(self):
        mem = np.array(self.memory, dtype=object)
        np.save(self.path_memory, mem)
        print(f"Saving memory {self.path_memory}, size: {len(self.memory)}")

    @property
    def path_model_dir(self):
        return os.path.abspath(f"{settings.MODEL_NAME}") + os.sep

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
        # plot_model(self.model, to_file=self.path_model + os.sep + "model.png")

        with open(self.path_model_dir + "summary.txt", 'wt') as fh:
            fh.write(f"Model: {settings.MODEL_NAME}\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        with open(self.path_model_dir + "params_preview.txt", 'wt') as fh:
            fh.write(f"Model: {settings.MODEL_NAME}\n\n")
            pars, optimizer = self.params
            for par in pars:
                fh.write(f"{par}\n")

            fh.write(f"\n")
            fh.write(f"{optimizer}\n")

        return True

    def save(self):
        self.save_model()

    def load(self):
        self.load_model()

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
        check_path = path + ".index"
        f1 = os.path.isfile(check_path)
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
        print(f"Creating dqn model {self.name}")
        layers = []
        CFG = deepcopy(self.node_shapes)
        for cfg in CFG:
            node = cfg.pop('node', None)
            args = cfg.pop('args', ())
            kwargs = cfg

            if not node:
                raise ValueError("Empty node value")
            node = node.lower()
            print(f"Adding {node:>10}, Args: {args}, Kwargs: {kwargs}")

            if node == 'conv2d':
                lay = Conv2D(*args, **kwargs)
            elif node == 'dense':
                lay = Dense(*args, **kwargs)
            elif node == "maxpool":
                lay = MaxPool2D(*args, **kwargs)
            elif node == "dropout":
                lay = Dropout(*args, **kwargs)
            elif node == "input":
                lay = Input(*args, **kwargs)
            elif node == 'flatten':
                lay = Flatten()
            else:
                raise ValueError(f"Unknown layer type: {node}")

            layers.append(lay)

        model = Sequential(layers)
        model.compile(**self.compiler)
        # print(model.summary())
        self.model = model

    def train(self):
        pass


agent = DQNAgent(
        node_shapes=settings.NODE_SHAPES,
        compiler=settings.COMPILER,
)

agent.save_model()
