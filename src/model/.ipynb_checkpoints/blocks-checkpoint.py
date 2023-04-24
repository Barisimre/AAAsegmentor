from monai.networks.blocks.mlp import MLPBlock
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
from src.constants import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.PReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class ViTEmbedder(nn.Module):

    def __init__(self, patch_size, embed_dim, in_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(kernel_size=patch_size, stride=patch_size, in_channels=in_channels, out_channels=in_channels * 2),
            nn.ReLU(),
            nn.Conv3d(kernel_size=3, stride=1, padding="same", in_channels=in_channels * 2, out_channels=embed_dim)
        )

    def forward(self, x):
        return self.convs(x)


    

<<<<<<< HEAD
=======
# class ViTDeEmbedder(nn.Module):

#     def __init__(self, patch_size, embed_dim, out_channels):
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv3d(kernel_size=3, stride=1, padding="same", in_channels=embed_dim, out_channels=out_channels),
#             nn.ConvTranspose3d(kernel_size=patch_size, stride=patch_size, in_channels=out_channels,
#                                out_channels=out_channels)
#         )

#     def forward(self, x):
#         return self.convs(x)

class ViTDeEmbedder(nn.Module):

    def __init__(self, patch_size, embed_dim, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=False)
        self.convs = nn.Sequential(
            nn.Conv3d(kernel_size=3, stride=1, padding=1, in_channels=embed_dim, out_channels=out_channels * 2),
            nn.ReLU(),
            nn.Conv3d(kernel_size=3, stride=1, padding=1, in_channels=out_channels * 2, out_channels=out_channels),
            nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        # x = self.upsample(x)
        return self.convs(x)

>>>>>>> parent of d8f4d45... One day of cluster work, torch2.0

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# X1 gets upscaled
class Up(nn.Module):

    def __init__(self, in_channels_x1, in_channels_x2, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels_x1, in_channels_x1 // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=(in_channels_x1 // 2) + in_channels_x2, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 qkv_bias: bool = False
                 ):

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SABlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
