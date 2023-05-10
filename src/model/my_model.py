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
                 mode="normal",
                 old_embedder = False):
        super(MyModel, self).__init__()

        self.n_classes = out_channels
        self.patch_size = patch_size
        self.mode = mode

        self.in_conv = DoubleConv(in_channels=in_channels, out_channels=big_channel)

        # Unet encoder
        self.down1 = Down(big_channel, lower_channels)
        self.down2 = Down(lower_channels, lower_channels)
        self.down3 = Down(lower_channels, lower_channels)
        self.down4 = Down(lower_channels, lower_channels)

        if self.mode == "autoencoder":
            self.up1 = SingleUp(lower_channels, lower_channels)
            self.up2 = SingleUp(lower_channels, lower_channels)
            self.up3 = SingleUp(lower_channels, lower_channels)
            self.up4 = SingleUp(lower_channels, big_channel)

        else:
            # Unet decoder
            self.up1 = Up(lower_channels, lower_channels, lower_channels)
            self.up2 = Up(lower_channels, lower_channels, lower_channels)
            self.up3 = Up(lower_channels, lower_channels, lower_channels)
            self.up4 = Up(lower_channels, big_channel, big_channel)

        self.out_conv1 = DoubleConv(big_channel, out_channels)
        self.out_conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')

        # Vision Transformer
        transformer_channels = [big_channel] + [lower_channels for i in range(4)]
        if mode == "half_half":
            transformer_channels = np.array(transformer_channels) // 2
        if mode not in ["autoencoder", "skip", "no_down_vit"]:
            self.vit = ViT(embed_dim=embed_dim, channels=transformer_channels, patch_size=patch_size, no_vit=mode=="no_vit", old=old_embedder)

        if mode == "no_down_vit":
            self.vit =  ViT(embed_dim=embed_dim, channels=transformer_channels, patch_size=patch_size, no_vit=mode=="no_vit", old=old_embedder, ablation=True)

    def forward(self, x):
        residual = self.in_conv(x)

        if self.mode == "no_down_vit":
            residual = self.vit()


        x1 = self.down1(residual)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        if self.mode == "half_half":
            t = 16
            vit_outs = self.vit([residual[:, :t], x1[:, :t], x2[:, :t], x3[:, :t], x4[:, :t]])
            residual = torch.concat([vit_outs[0], residual[:, t:]], dim=1)
            x1 = torch.concat([vit_outs[1], x1[:, t:]], dim=1)
            x2 = torch.concat([vit_outs[2], x2[:, t:]], dim=1)
            x3 = torch.concat([vit_outs[3], x3[:, t:]], dim=1)
            x4 = torch.concat([vit_outs[4], x4[:, t:]], dim=1)

        elif self.mode == "normal" or self.mode == "no_vit":
            residual, x1, x2, x3, x4 = self.vit([residual, x1, x2, x3, x4])


        if self.mode != "autoencoder":
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, residual)

        else:
            x = self.up1(x4)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
        x = self.out_conv1(x)
        x = self.out_conv2(x)

        return x
