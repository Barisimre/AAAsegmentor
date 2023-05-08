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
        self.projection = nn.Linear(in_channels * patch_size**3, embed_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, x, y, z)
        batch_size, channels, x_dim, y_dim, z_dim = x.shape

        # Create non-overlapping patches
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.unfold(4, self.patch_size, self.patch_size)

        # Flatten patches
        patches_flat = patches.contiguous().view(batch_size, -1, channels * self.patch_size**3)

        # Apply linear embeddings
        embeddings = self.projection(patches_flat)

        print(embeddings.shape)

        return embeddings, patches.shape

class InversePatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.inverse_projection = nn.Linear(embed_dim, in_channels * patch_size**3)
        
    def forward(self, x, patch_shape):
        # x shape: (batch, num_patches, embed_dim)
        batch_size = x.shape[0]

        # Inverse linear embeddings
        patches_flat = self.inverse_projection(x)

        # Reshape patches
        patches = patches_flat.view(*patch_shape).contiguous()

        # Reconstruct the original tensor
        x_dim, y_dim, z_dim = patch_shape[2] * self.patch_size, patch_shape[3] * self.patch_size, patch_shape[4] * self.patch_size
        reconstructed = torch.zeros(batch_size, self.in_channels, x_dim, y_dim, z_dim, device=x.device).contiguous()

        for i in range(patch_shape[2]):
            for j in range(patch_shape[3]):
                for k in range(patch_shape[4]):
                    reconstructed[:, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, k*self.patch_size:(k+1)*self.patch_size] = patches[:, :, i, j, k]

        return reconstructed


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


