MODEL_NAME = f"model-102-lin"

NODE_SHAPES = [
        {'node': "conv2d", 'filters': 50,
         # 'kernel_size': (5, 5), 'strides': (3, 3), 'activation': 'relu',
         'kernel_size': (3, 3), 'activation': 'relu',
         'input_shape': (100, 100, 3,)},
        {'node': 'conv2d', 'filters': 50, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
        {'node': "maxpool", 'pool_size': (2, 2)},
        {'node': 'conv2d', 'filters': 100, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
        {'node': 'maxpool', 'pool_size': (2, 2)},
        # {'node': 'conv2d', 'filters': 60, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
        # {'node': 'maxpool', 'pool_size': (2, 2), 'padding': 'same'},
        # {'node': 'conv2d', 'filters': 60, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
        # {'node': 'maxpool', 'pool_size': (2, 2), 'padding': 'same'},
        # {'node': 'conv2d', 'filters': 30, 'kernel_size': (1, 1), 'activation': 'relu', 'padding': 'same'},
        # {'node': 'maxpool', 'pool_size': (2, 2)},
        {'node': 'flatten'},
        {'node': 'dense', 'args': (100,), 'activation': 'relu'},
        {'node': "dropout", 'args': (0.05,)},
        {'node': 'dense', 'args': (100,), 'activation': 'relu'},
        {'node': "dropout", 'args': (0.05,)},
        {'node': 'dense', 'args': (100,), 'activation': 'relu'},
        # {'node': "dropout", 'args': (0.1,)},
        {'node': 'dense', 'args': (64 + 2,), 'activation': 'linear'},
]

COMPILER = {
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['accuracy'],
}

# Training params
ALPHA = 3e-5
BETA = 1e-5
GAMMA = 0.9
MEMORY_SIZE = 10_000

LOAD_MODEL = True
SAVE_MODEL = True
