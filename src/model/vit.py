import einops
import torch.nn as nn
from src.model.blocks import *
import torch
import time
from src.constants import *


class ViT(nn.Module):

    def __init__(self, embed_dim, patch_size, channels, no_vit=False, old=False, ablation=False):
        super().__init__()

        self.patch_size = patch_size
        self.no_vit = no_vit
        self.old = old
        self.ablation = ablation

        seq_lens = {8: 4681,  4: 37448, 2: 299584}

        if ablation:
            seq_lens[8]= 4096

            self.embedder = PatchEmbedding(patch_size=patch_size, in_channels=channels[0], embed_dim=embed_dim)
            self.de_embedder = InversePatchEmbedding(patch_size=patch_size, in_channels=channels[0], embed_dim=embed_dim)

        else:

            if old:

                self.embedders = nn.ModuleList([
                    ViTEmbedder(patch_size=patch_size, in_channels=c, embed_dim=embed_dim) for c in channels
                ])

                self.de_embedders = nn.ModuleList([
                    ViTDeEmbedder(patch_size=patch_size, out_channels=c, embed_dim=embed_dim) for c in channels
                ])
            
            else:

                self.embedders = nn.ModuleList([
                    PatchEmbedding(patch_size=patch_size, in_channels=c, embed_dim=embed_dim) for c in channels
                ])

                self.de_embedders = nn.ModuleList([
                    InversePatchEmbedding(patch_size=patch_size, in_channels=c, embed_dim=embed_dim) for c in channels
                ])



#         encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=NUM_HEADS, dim_feedforward=embed_dim * HIDDEN_FACTOR,
#                                                          dropout=0.1,
#                                                          activation="gelu", layer_norm_eps=1e-5, batch_first=False,
#                                                          norm_first=False)
        
#         encoder_norm = torch.nn.LayerNorm(embed_dim, eps=1e-5)
#         self.vit = torch.nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS, norm=encoder_norm)
        
        if not no_vit:
            self.vit = ViTEncoder(seq_length=seq_lens[patch_size], num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=embed_dim, mlp_dim=embed_dim*HIDDEN_FACTOR, dropout=0.1, attention_dropout=0.1)


    # xs: one x per level of the encoder. They should all halve in every size. Channel counts don't matter.
    def forward(self, xs):
        shapes = [int(x.shape[-1] / self.patch_size) for x in xs]

        if self.ablation:
            shape = xs.shape
            x = self.embedder(xs)
            x = self.vit(x)
            x = self.de_embedder(x, shape)
            return x

        if self.old:
            # Embed
            xs = [einops.rearrange(em(xs[i]), "b em x y z -> b (x y z) em") for i, em in enumerate(self.embedders)]

            token_counts = [x.shape[1] for x in xs]

            # ViT
            if not self.no_vit:
                xs = torch.cat(xs, dim=1)
                xs = self.vit(xs)
                xs = torch.split(xs, token_counts, dim=1)

            # De-Embed
            xs = [einops.rearrange(xs[i], "b (x y z) em -> b em x y z", x=s, y=s, z=s) for i, s in enumerate(shapes)]
            xs = [dem(xs[i]) for i, dem in enumerate(self.de_embedders)]


        else:
            shapes  = [x.shape for x in xs]
            xs = [e(x) for e, x in zip(self.embedders, xs)]
            token_counts = [x.shape[1] for x in xs]
            
            # shapes = [i[1] for i in a]

            if not self.no_vit:
                xs = torch.cat(xs, dim=1)
                xs = self.vit(xs)
                xs = torch.split(xs, token_counts, dim=1)

            xs = [e(x, s) for e, x, s in zip(self.de_embedders, xs, shapes)]
            # xs = [einops.rearrange(xs[i], "b (x y z) em -> b em x y z", x=s, y=s, z=s) for i, s in enumerate(shapes)]
            # xs = [dem(xs[i]) for i, dem in enumerate(self.de_embedders)]


        return xs



class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            inplace: Optional[bool] = None,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class ViTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
