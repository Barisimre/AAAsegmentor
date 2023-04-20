import monai
import torch


# Constants

RUN_NAME = "MyAttempt1_512"


# Paths
GENERAL_PATH = "/home/s1797743/thesis/final/AAAsegmentor"
DATA_PATH = f"/home/s1797743/thesis/AAAdata"
# DATA_PATH = "/data/AAA"
MODEL_SAVE_PATH = f"{GENERAL_PATH}/models"
RESULTS_SAVE_PATH = f"{GENERAL_PATH}/results"

# Sizes and transformations
CROP_SIZE = (128, 128, 128)
PATCH_SIZE = 16
SPACINGS = (1.2, 1.2, 0.9)
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 200


# Training hyper-parameters
DEVICE = "cuda"
LEARNING_RATES = {0: 1e-3, 25: 6e-4, 75: 3e-4, 300: 1e-4, 500: 5e-5}
NUM_EPOCHS = 1000
BATCH_SIZE = 2
LOSS = monai.losses.DiceCELoss(lambda_ce=0.4)


