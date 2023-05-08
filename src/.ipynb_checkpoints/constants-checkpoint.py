import monai
import torch


# Constants

RUN_NAME = "patch_8_embed512"


# Paths
GENERAL_PATH = "/home/s1797743/thesis/final/AAAsegmentor"
DATA_PATH = f"/home/s1797743/thesis/AAAdata"
# DATA_PATH = "/data/AAA"
MODEL_SAVE_PATH = f"{GENERAL_PATH}/models"
RESULTS_SAVE_PATH = f"{GENERAL_PATH}/results"

# Sizes and transformations
CROP_SIZE = (128, 128, 128)
PATCH_SIZE = 8
SPACINGS = (1.2, 1.2, 0.9)
# SPACINGS = (0.77, 0.77, 0.62)

CT_WINDOW_MIN = -300
CT_WINDOW_MAX = 300


# Training hyper-parameters
DEVICE = "cuda"
# LEARNING_RATES = {0: 1e-3, 25: 6e-4, 75: 3e-4, 800: 1e-4, 1250: 5e-5, 2000: 5e-6}

# Warm up boy:
LEARNING_RATES = {0: 1e-6, 25: 1e-4, 75: 8e-4, 100: 3e-4, 1000: 1e-4, 1750: 5e-5, 2000: 5e-6}


NUM_EPOCHS = 2500
BATCH_SIZE = 4
LOSS = monai.losses.DiceCELoss(lambda_ce=0.4)
# LOSS = monai.losses.GeneralizedDiceFocalLoss(lambda_gdl=0.5, lambda_focal=1.0)
#

# Transformer stuff
NUM_HEADS = 8
NUM_LAYERS = 12
HIDDEN_FACTOR = 3
