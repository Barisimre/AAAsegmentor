import torch.nn as nn
from src.model.blocks import *
from src.model.vit import ViT
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 channels=(32, 32, 32, 32, 32),
                 patch_size=16,
                 embed_dim=128,
                 levels=4,
                 transformer_channels=16,
                 skip_transformer=False):
        super(MyModel, self).__init__()

        self.n_channels = in_channels
        self.n_classes = out_channels
        self.patch_size = patch_size
        self.skip_transformer = skip_transformer
        self.transformer_channels = transformer_channels

        # Conv block for the input to have some channels
        self.residual_conv = DoubleConv(in_channels=in_channels, out_channels=channels[-1])

        # Unet encoder
        self.down1 = Down(in_channels, channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        # Unet decoder
        self.up1 = Up(channels[4], channels[3])
        self.up2 = Up(channels[3], channels[2])
        self.up3 = Up(channels[2], channels[1])
        self.up4 = Up(channels[1], out_channels)

        # Vision Transformer
        self.vit = ViT(embed_dim=embed_dim, channels=transformer_channels, patch_size=patch_size, levels=levels)

    def forward(self, x):
        residual = self.residual_conv(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        if not self.skip_transformer:
            t = self.transformer_channels
            vit_outs = self.vit([residual[:, :t], x1[:, :t], x2[:, :t], x3[:, :t]])
            residual[:, :t], x1[:, :t], x2[:, :t], x3[:, :t] = vit_outs

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, residual)

        return x
