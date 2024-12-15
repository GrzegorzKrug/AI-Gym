from trainer import trainer
from froggame import FrogGame
from trainer.advanced_model import FlatModel


def simple_model(in_shape, out_shape):
    inp = Input(in_shape)
    lay = Dense(100, activation='relu')(inp)
    lay = Dense(100, activation='relu')(lay)
    out = Dense(out_shape, activation='linear')(lay)

    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=1e-1),
                  loss='mse',
                  metrics=['accuracy'])
    return model


NODES_CONFIG = [
        {'node': 'input', 'args': (4,)},
        {'node': 'dense', 'args': (10,), 'activation': 'relu'},
        # {'node': "dropout", 'args': (0.05,)},
        {'node': 'dense', 'args': (10,), 'activation': 'relu'},
        {'node': 'dense', 'args': (4,), 'activation': 'linear'},
]

COMPILER = {
        'optimizer': 'adam',
        'loss': 'mae',
        'metrics': ['accuracy'],
}
if __name__ == "__main__":
    N = 10
    games = [FrogGame() for n in range(N)]

    model = FlatModel(
            node_shapes=NODES_CONFIG, compiler=COMPILER, name="3",
            alpha=0.02
    )
    model.load_model_weights()

    tr = trainer.Trainer(
            model=model, environments=games,
            input_size=4, output_size=4,
            gamma=0.1, memory_max_samples=50000,
            batch_size=50,
    )

    tr.start_training(50, max_iters=40)
    model.save_all()

    tr.render(40)
