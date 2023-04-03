import wandb
from src.constants import *
from monai.networks import one_hot


def train_single_epoch(model, optimizer, train_loader):
    model.train()
    losses = []

    for d in train_loader:
        img = d['img'].to(DEVICE)
        mask = d['mask'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(img)

        outputs = torch.softmax(outputs, dim=1)
        mask = one_hot(mask, num_classes=4)

        loss = LOSS(outputs, mask)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    wandb.log({"train_loss": (sum(losses) / len(losses))})
    return sum(losses) / len(losses)
