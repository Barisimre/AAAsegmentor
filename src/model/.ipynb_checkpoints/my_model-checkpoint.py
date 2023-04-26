import torch
import torch.nn as nn
from src.model.blocks import *
from src.model.vit import ViT
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=3,
                 lower_channels=16,
                 big_channel=4,
                 patch_size=8,
                 embed_dim=256,
                 skip_transformer=False):
        super(MyModel, self).__init__()

        self.n_classes = out_channels
        self.patch_size = patch_size
        self.skip_transformer = skip_transformer

        self.in_conv = SingleConvBlock(in_channels=in_channels, out_channels=big_channel, padding='same')

        # Unet encoder
        self.down1 = Down(big_channel, lower_channels)
        self.down2 = Down(lower_channels, lower_channels)
        self.down3 = Down(lower_channels, lower_channels)
        self.down4 = Down(lower_channels, lower_channels)

        # Unet decoder
        self.up1 = Up(lower_channels, lower_channels, lower_channels)
        self.up2 = Up(lower_channels, lower_channels, lower_channels)
        self.up3 = Up(lower_channels, lower_channels, lower_channels)
        self.up4 = Up(lower_channels, big_channel, big_channel)

        self.out_conv = SingleConvBlock(in_channels=big_channel, out_channels=out_channels, padding='same')

        # Vision Transformer
        transformer_channels = [big_channel] + [lower_channels for i in range(4)]
        self.vit = ViT(embed_dim=embed_dim, channels=transformer_channels, patch_size=patch_size)

    def forward(self, x):
        residual = self.in_conv(x)

        x1 = self.down1(residual)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

#         if not self.skip_transformer:
#             t = self.transformer_channels
#             vit_outs = self.vit([residual[:, :t], x1[:, :t], x2[:, :t], x3[:, :t], x4[:, :t]])
#             residual = torch.concat([vit_outs[0], residual[:, t:]], dim=1)
#             x1 = torch.concat([vit_outs[1], x1[:, t:]], dim=1)
#             x2 = torch.concat([vit_outs[2], x2[:, t:]], dim=1)
#             x3 = torch.concat([vit_outs[3], x3[:, t:]], dim=1)
#             x4 = torch.concat([vit_outs[4], x4[:, t:]], dim=1)

        if not self.skip_transformer:
            residual, x1, x2, x3 = self.vit([residual, x1, x2, x3])

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, residual)
        
        x = self.out_conv(x)

        return x
