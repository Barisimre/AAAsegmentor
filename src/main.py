from src.constants import *
from src.data.data_loaders import get_loaders
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb
from tqdm import tqdm
from src.model.baselines import *
from src.training.lr_schedule import set_learning_rate
from src.model.my_model import MyModel


def main():
    # Model to be trained. Baseline options are Unet, SWINUNETR
    # model = SWINUNETR.to(DEVICE)
    model = MyModel(in_channels=1,
                     out_channels=3,
                     lower_channels=16,
                     big_channel=3,
                     patch_size=8,
                     embed_dim=128,
                     skip_transformer=True).to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATES[0])

    wandb.init(
        project="AAA",
        entity="barisimre",
        name=RUN_NAME
    )

    train_loader, test_loader = get_loaders()

    for e in tqdm(range(NUM_EPOCHS)):

        best_test_loss = 2.0

        train_single_epoch(model=model, optimizer=optimizer, train_loader=train_loader)

        if e % 25 == 0:
            test_loss = test_single_epoch(model=model, test_loader=test_loader)

            #  If test loss is the best, save the model
            if test_loss <= best_test_loss:
                torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/{RUN_NAME}_{test_loss}.pt")
                best_test_loss = test_loss

            # Set new learning rate if it is time
            set_learning_rate(e, optimizer)


if __name__ == '__main__':
    main()
