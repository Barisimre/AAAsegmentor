import torch.nn as nn
from einops import rearrange
from src.model.blocks import TransformerBlock
import torch


class ViT(nn.Module):

    def __init__(
            self,
            hidden_size: int = 512,
            mlp_dim: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            qkv_bias: bool = False,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads,
                              dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        # attns = []

        for blk in self.blocks:
            x = blk(x)
            # attns.append(a)
        x = self.norm(x)
        return x


# TODO: add positional encoding
class MyVit(nn.Module):

    def __init__(self, embed_dim, patch_size, num_layers=6, mlp_dim=256, levels=3, in_channels=8):
        super().__init__()
        self.levels = levels
        self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.embedders = nn.ModuleList(
            [nn.Conv3d(kernel_size=patch_size, stride=patch_size, in_channels=1, out_channels=embed_dim) for _ in
             range(levels)])

        self.deembedders = nn.ModuleList(
            [nn.ConvTranspose3d(kernel_size=patch_size, stride=patch_size, in_channels=embed_dim, out_channels=1) for _
             in range(levels)])

        self.vit = ViT(hidden_size=embed_dim, num_layers=num_layers, mlp_dim=mlp_dim)

    # input xs --> [(b c x y z), (b c x/2 y/2 z/2) ... ]
    def forward(self, x):
        # TODO
        # split_points = [a.shape[]]
        # all_tokens = torch.concat(x, dim=1)
        # out_tokens = self.vit(all_tokens)
        # xs = []
        #
        # level_sperated = torch.split(out_tokens, split_points, dim=1)
        #
        # for i in range(self.levels):
        #     l = level_sperated[i]
        #     side_len = round(l.shape[1] ** (1. / 3.))
        #     ll = rearrange(l, "c (x y z) em -> c em x y z", x=side_len, y=side_len, z=side_len)
        #
        #     ll = self.deembedders[i](ll)
        #     xs.append(ll)
        #
        # return xs
        return None
