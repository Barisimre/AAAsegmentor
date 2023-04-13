import monai
import torch
from src.model.baselines import *

# Constants

RUN_NAME = "UNet"


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
CT_WINDOW_MIN = -40
CT_WINDOW_MAX = 200


# Training hyper-parameters
DEVICE = "cuda"
INITIAL_LEARNING_RATE = 3e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 8
LOSS = monai.losses.DiceLoss()


# Model to be trained. Baseline options are Unet, SWINUNETR
MODEL = UNet.to(DEVICE)

OPTIMIZER = torch.optim.Adam(params=model.parameters(), lr=INITIAL_LEARNING_RATE)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, cooldown=1, patience=2, factor=0.3, verbose=True)
