
from src.constants import *
from src.data.data_loaders import get_loaders
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb
from tqdm import tqdm


def main():

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
            SCHEDULER.step(test_loss)


if __name__ == '__main__':
    main()
