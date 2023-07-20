from src.constants import *
from src.data.data_loaders import get_loaders
from src.training.train import train_single_epoch
from src.training.test import test_single_epoch
import wandb
from tqdm import tqdm
from src.model.baselines import *
from src.training.lr_schedule import set_learning_rate
from src.model.my_model import MyModel
from torch.cuda.amp import GradScaler


def main():
    # Model to be trained. Baseline options are UNet, SWINUNETR
    # model = UNet

    # modes = normal, skip, autoencoder, half_half, no_vit

    model = MyModel(in_channels=1,
                    mid_channels=8,
                    out_channels=3,
                    patch_size=PATCH_SIZE,
                    embed_dim=EMBED_DIM,
                    img_size=CROP_SIZE)

    # model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/focus/all_transformer_seed.pt"))

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    wandb.init(
        project="TreeTransformer",
        entity="barisimre",
        name=RUN_NAME
    )

    train_loader, test_loader = get_loaders()

    for e in tqdm(range(NUM_EPOCHS)):

        best_test_loss = 2.0

        scaler = GradScaler()

        train_single_epoch(model=model, optimizer=optimizer, train_loader=train_loader, scaler=scaler)

        if e % 25 == 0:
            test_loss = test_single_epoch(model=model, test_loader=test_loader, scaler=scaler)

            #  If test loss is the best, save the model
            if test_loss <= best_test_loss:
                torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/{RUN_NAME}_{test_loss}.pt")
                best_test_loss = test_loss

            # Set new learning rate if it is time
            # TODO: set it back
            set_learning_rate(e, optimizer)


if __name__ == '__main__':
    main()
