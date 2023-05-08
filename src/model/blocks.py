from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from collections import OrderedDict


class SingleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dropout=0.1, no_adn=False):
        super().__init__()
        
        if no_adn:
            self.block = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.PReLU(),
                nn.Dropout(p=dropout, inplace=True),
                nn.InstanceNorm3d(out_channels),
            )

    def forward(self, x):
        return self.block(x)


class SingleConvBlockTransposed(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1, no_adn=False):
        super().__init__()
        
        if no_adn:
            self.block = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.PReLU(),
                nn.Dropout(p=dropout, inplace=True),
                nn.InstanceNorm3d(out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2 with residual connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None, no_adn=False, kernel_size=3):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            SingleConvBlock(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding='same', no_adn=no_adn),
            SingleConvBlock(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', no_adn=no_adn),
        )

        # Define a 1x1 convolution to match the number of channels for the residual connection
        if in_channels != out_channels:
            self.match_channels = SingleConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  padding='same', no_adn=no_adn)
        else:
            self.match_channels = None

    def forward(self, x):
        identity = x

        if self.match_channels is not None:
            identity = self.match_channels(identity)

        out = self.double_conv(x)
        out += identity

        return out


class ViTEmbedder(nn.Module):

    def __init__(self, patch_size, embed_dim, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            SingleConvBlock(kernel_size=patch_size, stride=patch_size, in_channels=in_channels,
                            out_channels=embed_dim, no_adn=True),
            # DoubleConv(in_channels=in_channels, out_channels=embed_dim, no_adn=True, kernel_size=3),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ViTDeEmbedder(nn.Module):

    def __init__(self, patch_size, embed_dim, out_channels):
        super().__init__()
        self.conv = SingleConvBlockTransposed(kernel_size=patch_size, stride=patch_size, in_channels=embed_dim,
                                       out_channels=out_channels, no_adn=True)

        self.block = nn.Sequential(
            DoubleConv(in_channels=embed_dim, out_channels=out_channels, kernel_size=3, no_adn=True),
            SingleConvBlockTransposed(kernel_size=patch_size, stride=patch_size, in_channels=out_channels,
                                      out_channels=out_channels, no_adn=True),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            SingleConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class SingleUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.Sequential(

            SingleConvBlockTransposed(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()

        self.up = SingleConvBlockTransposed(in_channels=in_channels1, out_channels=in_channels2, kernel_size=2,
                                            stride=2)
        self.conv = DoubleConv(in_channels2 + in_channels2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.concat([x1, x2], dim=1)

        return self.conv(x)


