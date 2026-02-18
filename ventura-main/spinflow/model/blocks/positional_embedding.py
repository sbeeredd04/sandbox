"""
Borrowed from:
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
"""

import math
import torch
import torch.nn as nn

from einops import rearrange

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    k = dim // 4
    if k == 1:                               # d_model == 4
        omega = torch.zeros(1, dtype=dtype)  # single frequency = 0
    else:
        omega = torch.arange(k, dtype=dtype) / (k - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class SinusoidalPositionalEmb2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding.")
        self.d_model = d_model
        self._cached_shape = None  # (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Tensor of shape [B, C, H, W] with positional encoding added
        """
        B, C, H, W = x.shape
        if self._cached_shape != (H, W):
            # Create and register new pos encoding
            pe = posemb_sincos_2d(H, W, self.d_model, dtype=x.dtype).T.reshape(self.d_model, H, W).to(x.device)
            self.register_buffer("pos_enc", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]
            self._cached_shape = (H, W)

        return x + self.pos_enc
    
class SinusoidalPositionalEmb1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be divisible by 2 for 1D positional encoding.")

        # Precompute up to max_len and register once
        position = torch.arange(max_len).unsqueeze(1).float()              # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )                                                                  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)                                 # (max_len,d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                       # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                       # odd dims
        pe = pe.unsqueeze(0)                                               # (1,max_len,d_model)

        # persistent=True so pe shows up in state_dict & moves with the module
        self.register_buffer("pos_enc", pe)                                
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]  
        """
        B, L, C = x.shape
        if C != self.d_model:
            raise RuntimeError(f"Model dim {C} != initialized {self.d_model}")
        # slice to the sequence length L
        return x + self.pos_enc[:, :L, :].to(x.dtype)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len=6):
#         super().__init__()

#         # Compute the positional encoding once
#         pos_enc = torch.zeros(max_seq_len, d_model)
#         pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pos_enc[:, 0::2] = torch.sin(pos * div_term)
#         pos_enc[:, 1::2] = torch.cos(pos * div_term)
#         pos_enc = pos_enc.unsqueeze(0)

#         # Register the positional encoding as a buffer to avoid it being
#         # considered a parameter when saving the model
#         self.register_buffer('pos_enc', pos_enc)

#     def forward(self, x):
#         # Add the positional encoding to the input
#         x = x + self.pos_enc[:, :x.size(1), :]
#         return x