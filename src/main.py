from src.constants import *
from src.data.data_loaders import get_loaders
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb
from tqdm import tqdm
from src.model.baselines import *
from src.training.lr_schedule import set_learning_rate
from src.model.my_model import MyModel
import os
import time


def main():
    # Model to be trained. Baseline options are Unet, SWINUNETR
<<<<<<< HEAD

    model = MyModel(in_channels=1,
                    out_channels=3,
                    embed_dim=256,
                    skip_transformer=False,
                    channels=(4, 16, 32, 32, 32, 32),
                    transformer_channels=(2, 8, 16, 16, 16, 16),
                    patch_size=8
                    ).to(DEVICE)
=======
    # model = SWINUNETR.to(DEVICE)
    model = MyModel(in_channels=1, out_channels=3, skip_transformer=False, channels=(32, 32, 32, 32, 32),transformer_channels=8, embed_dim=256).to(DEVICE)
>>>>>>> parent of d8f4d45... One day of cluster work, torch2.0

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATES[0])

    wandb.init(
        project="AAA",
        entity="barisimre",
        name=RUN_NAME
    )

    train_loader, test_loader = get_loaders()
    save_path = f"{RESULTS_SAVE_PATH}/{RUN_NAME}_{time.time()}"
    # os.mkdir(path=save_path)

    for e in tqdm(range(NUM_EPOCHS)):

        best_test_loss = 2.0

        train_single_epoch(model=model, optimizer=optimizer, train_loader=train_loader)

        if e % 25 == 24:
            test_loss = test_single_epoch(model=model, test_loader=test_loader, result_path=save_path)

            #  If test loss is the best, save the model
            if test_loss <= best_test_loss:
                torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/{RUN_NAME}_{test_loss}.pt")
                best_test_loss = test_loss

            # Set new learning rate if it is time
            set_learning_rate(e, optimizer)


if __name__ == '__main__':
    main()
