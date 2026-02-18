import torch, torch.nn as nn
from einops.layers.torch import Rearrange
from spinflow.model.blocks.positional_embedding import (
    SinusoidalPositionalEmb2D,
    SinusoidalPositionalEmb1D
)

class SeparableDepthwiseConv2d(nn.Module):
    """
    Depthwise separable convolution with a 3x3 kernel.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthconv_block = nn.Sequential(
            # 1) depthwise conv
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # 2) pointwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.depthconv_block(x)

class ContextAggregator(nn.Module):
    """
    Fuse an arbitrary set of feature tensors into one global vector.

    Each entry in cfg['aggregation']['in_keys'] must contain
      name, type ("2d" | "1d"), feat_dim, token_dim (for 2d).
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.out_key = cfg["out_key"]
        hidden  = cfg["hidden_dim"]

        # ---------------- per-key processing sequence -----------------
        self.attn_enabled = cfg.get("attn_enabled", True)
        if self.attn_enabled:
            heads   = cfg["attn_kwargs"].get("num_heads", 4)
            layers  = cfg["attn_kwargs"].get("num_layers", 4)
            in_keys = cfg["attn_kwargs"]["in_keys"]
            self.groups = nn.ModuleDict()
            for spec in in_keys:
                name, typ, fd = spec["name"], spec["type"], spec["feat_dim"]
                if typ == "2d":
                    h_p, w_p = spec["token_dim"]
                    self.groups[name] = nn.Sequential(
                        nn.AdaptiveAvgPool2d((h_p, w_p)),
                        SeparableDepthwiseConv2d(fd, hidden),
                        SinusoidalPositionalEmb2D(hidden),               # add pos
                        Rearrange("b c h w -> b (h w) c"),           # flatten
                    )
                elif typ == "1d":
                    self.groups[name] = nn.Sequential(
                        nn.Linear(fd, hidden) if fd != hidden else nn.Identity(),
                        SinusoidalPositionalEmb1D(hidden),
                    )
                else:
                    raise ValueError(f"unknown key type {typ}")

            # ---------------- shared Transformer encoder ------------------
            enc_layer = nn.TransformerEncoderLayer(
                d_model        = hidden,
                nhead          = heads,
                dim_feedforward= hidden * 4,
                activation     = "gelu",
                batch_first    = True,
                norm_first     = True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    # -----------------------------------------------------------------
    def forward(self, inputs: dict) -> torch.Tensor:
        if not self.attn_enabled:
            assert self.out_key in inputs, \
                f"ContextAggregator requires {self.out_key} in inputs"
            return inputs[self.out_key]
        
        token_chunks = []
        for name, mod in self.groups.items():
            if name not in inputs:
                raise KeyError(f"input dict missing key '{name}'")
            token_chunks.append(mod(inputs[name]))    # (B,N_i,H)

        tokens = torch.cat(token_chunks, dim=1)       # (B, ΣN_i, H)
        tokens = self.encoder(tokens)                 # self-attention
        return tokens.mean(dim=1)                     # (B, H) # Aggreagate to a single feature
