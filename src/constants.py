import monai
import torch


# Constants

RUN_NAME = "SWINUNETR_moreoverlap_ce_1on1sampling_noscheduler"


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
INITIAL_LEARNING_RATE = 3e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 4
LOSS = monai.losses.DiceCELoss(lambda_ce=0.4)



