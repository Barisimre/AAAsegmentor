
from src.constants import *
from src.data.data_loaders import get_loaders
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb
from tqdm import tqdm
from src.model.baselines import *


def main():

	# Model to be trained. Baseline options are Unet, SWINUNETR
    MODEL = SWINUNETR.to(DEVICE)

    OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=INITIAL_LEARNING_RATE)
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=OPTIMIZER, cooldown=4, patience=2, factor=0.3, verbose=True)

    wandb.init(
        project="AAA",
        entity="barisimre",
        name=RUN_NAME
    )

    train_loader, test_loader = get_loaders()

    for e in tqdm(range(NUM_EPOCHS)):

        train_single_epoch(model=MODEL, optimizer=OPTIMIZER, train_loader=train_loader)

        if e % 25 == 0:
            test_loss = test_single_epoch(model=MODEL, test_loader=test_loader)
            # SCHEDULER.step(test_loss)


if __name__ == '__main__':
    main()
