import monai
import torch

# Constants

RUN_NAME = "run_name"


# Paths
# GENERAL_PATH = "/home/imreb/AAAsegmentor"
GENERAL_PATH = "/home/baris/Documents/AAAsegmentor"
DATA_PATH = f"{GENERAL_PATH}/data"


# DATA_PATH = "/data/AAA"
MODEL_SAVE_PATH = f"{GENERAL_PATH}/models"
RESULTS_SAVE_PATH = f"{GENERAL_PATH}/results"

TEST_PATIENTS = [10, 20, 31, 42, 55, 78]
DATA_LIMIT = 5

# Sizes and transformations
CROP_SIZE = (256, 256, 64)
PATCH_SIZE = (4, 4, 4)
SPACINGS = (1.2, 1.2, 0.9)
EMBED_DIM = 256

CT_WINDOW_MIN = -20
CT_WINDOW_MAX = 120


# Training hyperparameters
DEVICE = "cuda"
LEARNING_RATES = {0: 3e-4, 1000: 1e-4, 1750: 7e-5, 2000: 2e-5}

NUM_EPOCHS = 2500
BATCH_SIZE = 1
LOSS = monai.losses.DiceCELoss(lambda_ce=0.4)

# Transformer stuff
NUM_HEADS = 16
NUM_LAYERS = 12
HIDDEN_FACTOR = 3
