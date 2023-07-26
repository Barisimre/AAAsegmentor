import monai
import torch

# Constants

RUN_NAME = "new_maxpool_128"


# Paths
GENERAL_PATH = "/home/imreb/AAAsegmentor"
DATA_PATH = f"{GENERAL_PATH}/data"

# DATA_PATH = "/data/AAA"

MODEL_SAVE_PATH = f"{GENERAL_PATH}/models"
RESULTS_SAVE_PATH = f"{GENERAL_PATH}/results"

TEST_PATIENTS = [10, 20, 31, 42, 55, 82]
DATA_LIMIT = -1

# Sizes and transformations
CROP_SIZE = (256, 256, 64)
PATCH_SIZE = (4, 4, 4)
SPACINGS = (1.2, 1.2, 0.9)

CT_WINDOW_MIN = -50
CT_WINDOW_MAX = 150


# Training hyperparameters
DEVICE = "cuda"
LEARNING_RATES = {0: 3e-4, 1000: 1e-4, 1750: 7e-5, 2000: 2e-5}

NUM_EPOCHS = 2500
BATCH_SIZE = 2
LOSS = monai.losses.DiceCELoss(lambda_ce=0.4)

# Transformer stuff
NUM_HEADS = 16
NUM_LAYERS = 12
HIDDEN_FACTOR = 3
CHANNELS = 2
EMBED_DIM = 128
