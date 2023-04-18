import einops
import torch.nn as nn
from src.model.blocks import TransformerBlock
import torch


# TODO: add positional encoding
# Untouched ViT from MONAI, used as an internal class
class DefaultViT(nn.Module):

    def __init__(
            self,
            hidden_size: int = 128,
            mlp_dim: int = 256,
            num_layers: int = 12,
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
                              dropout_rate, qkv_bias) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # TODO: positional encoding

        # attns = []

        for blk in self.blocks:
            x = blk(x)
            # TODO
            # attns.append(a)
        x = self.norm(x)
        return x


class ViT(nn.Module):

    def __init__(self, embed_dim, patch_size, channels=8, levels=3):
        super().__init__()

        self.patch_size = patch_size

        self.embedders = nn.ModuleList([
            nn.Conv3d(kernel_size=patch_size, stride=patch_size, in_channels=channels, out_channels=embed_dim)
            for _ in range(levels)])

        self.de_embedders = nn.ModuleList([
            nn.ConvTranspose3d(kernel_size=patch_size, stride=patch_size, in_channels=embed_dim, out_channels=channels)
            for _ in range(levels)])

        self.vit = DefaultViT(hidden_size=embed_dim, mlp_dim=embed_dim * 2)

    # xs: one x per level of the encoder. They should all halve in every size. Channel counts don't matter.
    def forward(self, xs):
        shapes = [int(x.shape[-1] / self.patch_size) for x in xs]

        # Embed
        xs = [einops.rearrange(em(x), "b em x y z -> b (x y z) em") for x, em in zip(xs, self.embedders)]

        token_counts = [x.shape[1] for x in xs]

        # TODO: positional encoding

        # ViT
        xs = self.vit(torch.cat(xs, dim=1))
        xs = torch.split(xs, token_counts, dim=1)

        # De-Embed
        xs = [einops.rearrange(x, "b (x y z) em -> b em x y z", x=s, y=s, z=s) for x, s in zip(xs, shapes)]
        xs = [dem(x) for x, dem in zip(xs, self.de_embedders)]

        return xs
