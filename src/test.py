import monai
import torch
import flash_attn
import wandb
import matplotlib


example = torch.rand(size=(128, 189))
print(example.shape)
print(torch.cuda.is_available())