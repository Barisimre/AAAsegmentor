import torch
import torch.nn as nn
from src.model.blocks import *
from src.model.vit import ViT
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 channels=(4, 8, 16, 16, 16, 16),
                 patch_size=16,
                 embed_dim=128,
                 levels=4,
                 transformer_channels=(2, 4, 8, 8, 8, 8),
                 skip_transformer=False):
        super(MyModel, self).__init__()

        self.n_channels = in_channels
        self.n_classes = out_channels
        self.patch_size = patch_size
        self.skip_transformer = skip_transformer
        self.transformer_channels = transformer_channels

        # First and last blocks, do  not change shape
        self.initial_conv = DoubleConv(in_channels=in_channels, out_channels=channels[0])
        self.end_conv = Up(in_channels=channels[1], out_channels=out_channels)

        # # Unet encoder
        # self.down1 = Down(in_channels, channels[1])
        # self.down2 = Down(channels[1], channels[2])
        # self.down3 = Down(channels[2], channels[3])
        # self.down4 = Down(channels[3], channels[4])

        self.encoder = nn.ModuleList([
            Down(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

        # # Unet decoder
        # self.up1 = Up(channels[4], channels[3])
        # self.up2 = Up(channels[3], channels[2])
        # self.up3 = Up(channels[2], channels[1])
        # self.up4 = Up(channels[1], out_channels)

        self.decoder = nn.ModuleList([
            Up(channels[i], channels[i - 1]) for i in range(len(channels))[::-1]
        ])

        # Vision Transformer
        self.vit = ViT(embed_dim=embed_dim, channels=transformer_channels, patch_size=patch_size, levels=len(channels))

    def forward(self, x):

        # Keep residual, add channels to input
        x = self.initial_conv(x)
        residual = x

        # Encoder
        encoded = []
        for blk in self.encoder:
            x = blk(x)
            print(x.shape)
            encoded.append(x)

        # Vision transformer
        ts = self.transformer_channels
        vit_input = [residual[:, :ts[0]]] + [a[:, :t] for a, t in zip(encoded, ts[1:])]
        vit_outs = self.vit(vit_input)

        residual = torch.concat([vit_outs[0], residual[:, ts[0]:]], dim=1)
        xs = [torch.concat([vit_outs[i + 1], encoded[i][:, ts[i + 1]:]], dim=1) for i in range(len(encoded))]
        # print([a.shape for a in xs])
        print()
        # xs = None

        # TODO: injection

        xs = xs[::-1]
        for i in range(len(xs) - 1):
            x = self.decoder[i](xs[i], xs[i + 1])
            print(x.shape)

        x = self.end_conv(x, residual)

        # residual = self.residual_conv(x)
        #
        # x1 = self.down1(x)
        # x2 = self.down2(x1)
        # x3 = self.down3(x2)
        # x4 = self.down4(x3)
        #
        # if not self.skip_transformer:
        #     t = self.transformer_channels
        #     vit_outs = self.vit([residual[:, :t], x1[:, :t], x2[:, :t], x3[:, :t]])
        #     residual = torch.concat([vit_outs[0], residual[:, t:]], dim=1)
        #     x1 = torch.concat([vit_outs[1], x1[:, t:]], dim=1)
        #     x2 = torch.concat([vit_outs[2], x2[:, t:]], dim=1)
        #     x3 = torch.concat([vit_outs[3], x3[:, t:]], dim=1)
        #
        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # x = self.up4(x, residual)

        return x
