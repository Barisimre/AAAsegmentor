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


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection = nn.Linear(in_channels * patch_size**3, embed_dim, bias=False)

    def forward(self, x):
        batch_size, channels, x_dim, y_dim, z_dim = x.shape

        patches_x = x.unfold(2, self.patch_size, self.patch_size)
        patches_y = patches_x.unfold(3, self.patch_size, self.patch_size)
        patches_z = patches_y.unfold(4, self.patch_size, self.patch_size)

        patches = patches_z.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        patches = patches.view(batch_size, -1, self.patch_size**3 * channels)

        embeddings = self.projection(patches)

        return embeddings

class InversePatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection = nn.Linear(embed_dim, in_channels * patch_size**3, bias=False)

    def forward(self, embeddings, x_shape):
        batch_size, _, x_dim, y_dim, z_dim = x_shape

        patches = self.projection(embeddings)

        patches = patches.view(batch_size, -1, self.in_channels, self.patch_size, self.patch_size, self.patch_size)

        num_patches_x = x_dim // self.patch_size
        num_patches_y = y_dim // self.patch_size
        num_patches_z = z_dim // self.patch_size

        patches = patches.view(batch_size, num_patches_x, num_patches_y, num_patches_z, self.in_channels, self.patch_size, self.patch_size, self.patch_size)

        x = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(batch_size, self.in_channels, x_dim, y_dim, z_dim)

        return x


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


