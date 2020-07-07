MODEL_NAME = "11"
GAME_NUMBER = 1000000
TRAIN_TIMEOUT = 5 * 60
SIM_COUNT = 50

"Train Params"
ALPHA = 1e-2
EPS = 0.35

ALLOW_TRAIN = True
STEP_TRAIN = False
TRAIN_EVERY = 5
MIN_BATCH_SIZE = 50
MAX_BATCH_SIZE = TRAIN_EVERY * SIM_COUNT * 100
MEMOR_MAX_SIZE = SIM_COUNT * 1000
CLEAR_MEMORY = True
DISCOUNT = 0.9

"Model"
LAYERS = (5000, 0.01, 1000)
INPUT_SHAPE = (98 * 8 + 4,)

GAME_TIMEOUT = 150
SKIP_MOVE = -0.1
GOOD_MOVE = -1
LOST_GAME = -15
INVALID_MOVE = -150

"Env"
ACTION_SPACE = 32

"Plot"
GRAPH_CUT_AT = 500  # New plot when x is reached

DEBUG = False