import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer for PyTorch.

    Parameters
    ----------
    cond_dim : int
        Dimensionality of the conditioning vector  (second axis of `conditioning`).
    num_channels : int
        Number of channels in the feature map to modulate (F).
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()

        # Linear layers that predict bias (β) and scale (γ) per channel
        self.proj_add = nn.Linear(cond_dim, num_channels, bias=True)
        self.proj_mult = nn.Linear(cond_dim, num_channels, bias=True)

        # Original TF code initialises with zeros → identity transform
        nn.init.zeros_(self.proj_add.weight)
        nn.init.zeros_(self.proj_add.bias)
        nn.init.zeros_(self.proj_mult.weight)
        nn.init.zeros_(self.proj_mult.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        conv_features: torch.Tensor,   # [B, F, H, W]  or  [B, F]
        conditioning: torch.Tensor     # [B, cond_dim]
    ) -> torch.Tensor:
        """
        Apply FiLM modulation:  (1 + γ) ⊙ x  +  β

        Returns
        -------
        torch.Tensor
            Same shape as `conv_features`.
        """
        if conditioning.dim() != 2:
            raise ValueError(
                f"`conditioning` must be rank-2 [B,cond_dim]; got {conditioning.shape}"
            )

        beta  = self.proj_add(conditioning)   # [B, F]
        gamma = self.proj_mult(conditioning)  # [B, F]

        if conv_features.dim() == 4:          # [B,F,H,W]  → broadcast β,γ
            beta  = beta.unsqueeze(-1).unsqueeze(-1)   # [B,F,1,1]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B,F,1,1]
        elif conv_features.dim() != 2:        # anything else is unsupported
            raise ValueError(
                f"`conv_features` must be rank-2 or rank-4; got {conv_features.shape}"
            )

        # Based on original paper, where gamma is identity at initialization
        return (1.0 + gamma) * conv_features + beta
    

if __name__ == "__main__":
    B, F, H, W = 8, 64, 32, 32
    cond_dim   = 128

    x   = torch.randn(B, F, H, W)
    cond = torch.randn(B, cond_dim)

    film = FiLMConditioning(cond_dim=cond_dim, num_channels=F)
    y = film(x, cond)                       # y has the same shape as x