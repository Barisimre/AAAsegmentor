import torch
import torch.nn as nn
from src.model.blocks import *
from src.model.vit import ViT
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 in_channels=1,
                 mid_channels=4,
                 out_channels=3,
                 patch_size=(4,4,4),
                 embed_dim=256,
                 img_size=(128, 128, 128)):
        super(MyModel, self).__init__()

        self.n_classes = out_channels
        self.patch_size = patch_size

        self.in_conv = DoubleConv(in_channels=in_channels, out_channels=mid_channels)

        # Unet encoder
        self.down1 = Down(mid_channels, mid_channels)
        self.down2 = Down(mid_channels, mid_channels)
        self.down3 = Down(mid_channels, mid_channels)
        self.down4 = Down(mid_channels, mid_channels)

        # Unet decoder
        self.up1 = Up(mid_channels, mid_channels, mid_channels)
        self.up2 = Up(mid_channels, mid_channels, mid_channels)
        self.up3 = Up(mid_channels, mid_channels, mid_channels)
        self.up4 = Up(mid_channels, mid_channels, mid_channels)

        self.out_conv1 = DoubleConv(mid_channels, out_channels)
        self.out_conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')

        # Vision Transformer
        self.vit = ViT(embed_dim=embed_dim, channels=mid_channels, patch_size=patch_size, original_img_size=img_size, depth=5)


    def forward(self, x):
        residual = self.in_conv(x)
        
        x1 = self.down1(residual)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        residual, x1, x2, x3, x4 = self.vit([residual, x1, x2, x3, x4])

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, residual)

        x = self.out_conv1(x)
        x = self.out_conv2(x)


        return x
