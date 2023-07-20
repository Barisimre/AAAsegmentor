import wandb
from src.constants import *
from monai.networks import one_hot
from torch.cuda.amp import GradScaler, autocast


def train_single_epoch(model, optimizer, train_loader, scaler=None):
    losses = []

    for d in train_loader:

        img = d['img'].to(DEVICE)
        mask = d['mask'].to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(img)
            outputs = torch.softmax(outputs, dim=1)
            mask = one_hot(mask, num_classes=3)
            loss = LOSS(outputs, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().cpu().item())

    wandb.log({"train_loss": (sum(losses) / len(losses))})
    return sum(losses) / len(losses)
