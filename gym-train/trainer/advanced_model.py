import datetime
import numpy as np
import os
import pickle

from keras.layers import (
    Dense, Flatten, Dropout,
    Conv2D, MaxPool2D,
    Softmax, Input, concatenate,
)

from keras.models import Sequential, load_model, Model
from keras.initializers import RandomUniform
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.optimizers import Adam, SGD

from collections import deque
from copy import deepcopy


class FlatModel:
    def __init__(
            self,
            /,
            node_shapes=None,
            compiler=None,
            memory_size=10000,

            alpha=0.01,
            beta_1=0.9,
            beta_2=0.999,
            # gamma=0.99,
            name=None,
            load_model=False,
            save_model=True,
            save_dir="models",
    ):
        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_str = f"Y{dt.tm_year}M{dt.tm_mon:>02}D{dt.tm_mday:>02}-" \
                           f"{dt.tm_hour:>02}:{dt.tm_min:>02}:{dt.tm_sec:>02}"
        self.name = str(name)

        self.node_shapes = node_shapes
        self.compiler = compiler
        self.memory_size = memory_size

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # self.gamma = gamma

        self.memory = deque(maxlen=memory_size)
        self.model = None
        self.model_name = f"{(type(self).__qualname__)}_{self.name}"  # Captures class name

        self.allow_load = load_model
        self.allow_save = save_model
        self.save_dir = str(save_dir)

        self.iters = 0

        if self.allow_load:
            self.load_all()
        else:
            self.make_model()
            print(f"New model: {self.model_name}")

        self.save_summary()

    def make_model_dir(self):
        os.makedirs(os.path.dirname(self.path_params), exist_ok=True)

    @property
    def params(self):
        """
        Specify what params to save and load
        Returns:

        """
        return self.node_shapes, self.compiler, self.iters

    @params.setter
    def params(self, new_params):
        """
        Specify what params to save and load
        Returns:

        """
        self.node_shapes, self.compiler, self.iters = new_params

    def load_all(self):
        print(f"Loading model {self.name}")
        self.load_params()
        self.make_model()
        self.load_model_weights()
        self.load_memory()

    def load_params(self):
        try:
            pars = np.load(self.path_params, allow_pickle=True)
            self.params = pars
            print(f"Loading params {self.path_params}")
            print(self.params)
        except FileNotFoundError:
            print(f"Not found params to load {self.path_params}")

    def save_params(self):
        arr = np.array(self.params, dtype=object)
        np.save(self.path_params, arr)
        print(f"Saving params {self.path_params}")

    def load_memory(self, drop_last=0, path_override=None):
        try:
            if path_override:
                path = path_override
            else:
                path = self.path_memory

            mem = np.load(path, allow_pickle=True)
            print(f"Loaded memory: {mem.shape} -> {mem.shape[0] - drop_last} samples")
            for ind, sample in enumerate(mem):
                if ind + drop_last >= len(mem):
                    break
                self.add_memory(sample)
            print(f"Loading memory {self.path_memory}")

        except FileNotFoundError:
            print(f"Not found memory file {self.path_memory}")

    def save_memory(self):
        mem = np.array(self.memory, dtype=object)
        np.save(self.path_memory, mem)
        print(f"Saving memory {self.path_memory}, size: {len(self.memory)}")

    @property
    def _main_path(self):
        """Path to main folder for this model"""
        return f"{self.save_dir}{os.path.sep}{self.model_name}"

    @property
    def path_model_dir(self):
        return os.path.abspath(self._main_path) + os.path.sep

    @property
    def path_model_weights(self):
        return os.path.abspath(f"{self._main_path}") + os.path.sep

    @property
    def path_params(self):
        return os.path.abspath(f"{self._main_path}{os.path.sep}model{os.path.sep}params-model.npy")

    @property
    def path_memory(self):
        return os.path.abspath(f"{self._main_path}{os.path.sep}model{os.path.sep}memory.npy")

    def save_all(self):
        """Save model, params and memmory"""
        if not self.allow_save:
            print("Saving is not allowed")
            return None

        # self.make_model_dir()
        self.save_memory()
        self.save_params()
        self._save_weights(self.model, self.path_model_weights)
        self.save_summary()
        # plot_model(self.model, to_file=self.path_model + os.sep + "model.png")

    def save_summary(self):
        self.make_model_dir()

        with open(self.path_model_dir + "params_summary.txt", 'wt') as fh:
            fh.write(f"Model: {self.model_name}\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        with open(self.path_model_dir + "model_preview.txt", 'wt') as fh:
            fh.write(f"Model: {self.model_name}\n\n")

            fh.write(f"Iterations: {self.iters}\n")

            fh.write(f"\nLayers\n")
            pars, optimizer, _ = self.params
            for par in pars:
                fh.write(f"{par}\n")

            fh.write(f"\nOptimizer\n")
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
                print(f"Saving weights {path}")
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

    def make_model(self):
        print(f"Creating dqn model {self.name}")
        layers = []
        CFG = deepcopy(self.node_shapes)
        print(CFG)
        for cfg in CFG:
            node = cfg.pop('node', None)
            args = cfg.pop('args', tuple())
            kwargs = cfg

            if not node:
                raise ValueError("Empty node value")
            node = node.lower()
            print(f"Adding {node:>10}, Args: {args}, Kwargs: {kwargs}")

            if node == 'conv2d':
                lay = Conv2D(*args, **kwargs)
            elif node == 'dense':
                lay = Dense(*args, **kwargs,)
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
        # model.compile(**self.compiler)
        # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        opt_str = self.compiler['optimizer'].lower()
        if opt_str == "adam":
            opt = Adam
        else:
            opt = SGD

        opt_set = self.compiler.copy()
        opt_set.pop('optimizer')
        print(f"opt settings: {opt_set}")

        model.compile(optimizer=opt(learning_rate=self.alpha, beta_1=self.beta_1, beta_2=self.beta_2),
                      **opt_set,
                      )
        self.model = model

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


if __name__ == "__main__":
    NODES_CONFIG = [
            {'node': 'input', 'args': (4,)},
            # {'node': "dropout", 'args': (0.05,)},
            {'node': 'dense', 'args': (30,), 'activation': 'relu'},
            # {'node': "dropout", 'args': (0.05,)},
            {'node': 'dense', 'args': (30,), 'activation': 'relu'},
            {'node': 'dense', 'args': (4,), 'activation': 'softmax'},
    ]

    COMPILER = {
            'optimizer': 'adam',
            'loss': 'mse',
            'metrics': ['accuracy'],
    }

    mod = FlatModel(NODES_CONFIG, compiler=COMPILER, name="test")
    mod.save_all()
    print(mod.runtime_str)
