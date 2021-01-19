MODEL_NAME = f"model_0"

NODE_SHAPES = [
        {'node': "conv2d", 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu',
         'input_shape': (100, 100, 3,)},
        {'node': "maxpool", 'pool_size': (2, 2)},
        {'node': "dropout", 'args': (0.1,)},
        {'node': "conv2d", 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'node': "maxpool", 'pool_size': (2, 2)},
        {'node': 'flatten'},
        {'node': 'dense', 'args': (200,), 'activation': 'relu'},
        {'node': 'dense', 'args': (200,), 'activation': 'relu'},
        {'node': "dropout", 'args': (0.1,)},
        {'node': 'dense', 'args': (65,), 'activation': 'linear'},
]
COMPILER = {
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['accuracy'],
}

# Training params
ALPHA = 3e-5
BETA = 1e-5
GAMMA = 0.95

LOAD_MODEL = True
SAVE_MODEL = True
