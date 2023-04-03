import torch
import numpy as  np

def dice(mask, out, index):
    mask = mask == index
    out = out == index
    return 2 * (torch.sum(mask * out) / (torch.sum(mask) + torch.sum(out))).item()


def dice_scores(out, mask):
    return np.array([dice(mask, out, i) for i in range(3)])
