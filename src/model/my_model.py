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
        self.end_conv = Up(in_channels_x1=channels[1], in_channels_x2=channels[0], out_channels=out_channels)

        self.encoder = nn.ModuleList([
            Down(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

        self.decoder = nn.ModuleList([
            Up(channels[i], channels[i - 1], channels[i - 1]) for i in range(len(channels))[::-1][:-1]
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
            encoded.append(x)

        # Vision transformer
        ts = self.transformer_channels
        vit_input = [residual[:, :ts[0]]] + [a[:, :t] for a, t in zip(encoded, ts[1:])]
        vit_outs = self.vit(vit_input)

        # Insert ViT output to residuals
        residual = torch.concat([vit_outs[0], residual[:, ts[0]:]], dim=1)
        xs = [torch.concat([vit_outs[i + 1], encoded[i][:, ts[i + 1]:]], dim=1) for i in range(len(encoded))]

        # Decoder
        xs = xs[::-1]
        for i in range(len(xs) - 1):
            x = self.decoder[i](xs[i], xs[i + 1])

        print(x.shape)
        print(residual.shape)
        x = self.end_conv(x, residual)

        return x
