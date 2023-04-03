from src.constants import *
from src.model.baselines import *
from src.data.data_loaders import train_loader, test_loader
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb

RUN_NAME = "debug"

# Model to be trained. Baseline options are Unet, SWINUNETR
model = UNet

optimizer = torch.optim.Adam(params=model.parameters(), lr=INITIAL_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, cooldown=0, patience=2, factor=0.5)

wandb.init(
    project="AAA",
    entity="barisimre",
    name = RUN_NAME
)

for e in range(NUM_EPOCHS):

    train_single_epoch(model=model, optimizer=optimizer, train_loader=train_loader)

    if e % 50 == 49:
        test_single_epoch(model=model, test_loader=test_loader)

    if e % 10 == 0:
        pass
        # TODO visualise the boi