import torch
import torch.nn as nn
from src.model.blocks import *
from src.model.vit import ViT
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 in_channels=1,
                 mid_channels=1,
                 out_channels=3,
                 patch_size=(4, 4, 4),
                 embed_dim=256,
                 img_size=(128, 128, 128)):
        super(MyModel, self).__init__()

        self.n_classes = out_channels
        self.patch_size = patch_size

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.out_conv1 = DoubleConv(5, out_channels)
        self.out_conv2 = DoubleConv(out_channels, out_channels)


        # Vision Transformer
        self.vit = ViT(embed_dim=embed_dim, channels=mid_channels, patch_size=patch_size, original_img_size=img_size,
                       depth=5)

    def forward(self, x):

        x1 = self.pool(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        x4 = self.pool(x3)

        x, x1, x2, x3, x4 = self.vit([x, x1, x2, x3, x4])

        x1 = torch.nn.functional.interpolate(x1, scale_factor=2, mode='trilinear')
        x2 = torch.nn.functional.interpolate(x2, scale_factor=4, mode='trilinear')
        x3 = torch.nn.functional.interpolate(x3, scale_factor=8, mode='trilinear')
        x4 = torch.nn.functional.interpolate(x4, scale_factor=16, mode='trilinear')

        xs = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.out_conv1(xs)
        x = self.out_conv2(x)

        return x
