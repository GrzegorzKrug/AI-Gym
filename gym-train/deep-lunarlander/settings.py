SIM_COUNT = 10
EPOCHS = 5000
TRAIN_MAX_MIN_DURATION = 5

SHOW_FIRST = True
SHOW_EVERY_MINUTE = 1  # in minutes
RENDER_DELAY = 0.0003
RENDER_WITH_ZERO_EPS = True

REPLAY_MEMORY_SIZE = 100_000 * SIM_COUNT  # 10 full games 3k each
MIN_BATCH_SIZE = 20 * SIM_COUNT
MAX_BATCH_SIZE = 10_000 * SIM_COUNT
CLEAR_MEMORY_AFTER_TRAIN = True

# Training method
ALLOW_TRAIN = True
LOAD_MODEL = True

MODEL_NAME = f"Model-11"

DENSE1 = 64
DENSE2 = 64
DROPOUT1 = 0.1
DROPOUT2 = 0.2

# Training params
ALPHA = 1e-4
BETA = 1e-4
GAMMA = 0.95

FIRST_EPS = 0.3
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.2
END_EPS = 0
EPS_INTERVAL = 20

# Settings
SAVE_PICS = ALLOW_TRAIN
SHOW_LAST = False
# PLOT_ALL_QS = True
# PLOT_FIRST_QS = False
# COMBINE_QS = True
SOUND_ALERT = True