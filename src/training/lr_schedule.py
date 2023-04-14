import torch
import wandb

from constants import *


def set_learning_rate(e, optimizer):
    if e in LEARNING_RATES.keys():

        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATES[e]

            wandb.log({"learning_rate": LEARNING_RATES[e]})
