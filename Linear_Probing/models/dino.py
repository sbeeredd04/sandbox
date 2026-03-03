import torch
import torch.nn as nn


# Maps short names → torch.hub model names
_HUB_NAMES = {
    "vits14":     "dinov2_vits14",
    "vitb14":     "dinov2_vitb14",
    "vitl14":     "dinov2_vitl14",
    "vitg14":     "dinov2_vitg14",
    "vits14_reg": "dinov2_vits14_reg",
    "vitb14_reg": "dinov2_vitb14_reg",
    "vitl14_reg": "dinov2_vitl14_reg",
    "vitg14_reg": "dinov2_vitg14_reg",
}

# Embed dims per variant
_EMBED_DIMS = {
    "vits14": 384,  "vits14_reg": 384,
    "vitb14": 768,  "vitb14_reg": 768,
    "vitl14": 1024, "vitl14_reg": 1024,
    "vitg14": 1536, "vitg14_reg": 1536,
}


class DINOv2(nn.Module):

    def __init__(self, variant: str = "vitb14"):
        super().__init__()
        assert variant in _HUB_NAMES, f"Unknown variant '{variant}'. Choose from {list(_HUB_NAMES)}"
        self.variant = variant
        self.embed_dim = _EMBED_DIMS[variant]
        self.model = torch.hub.load("facebookresearch/dinov2", _HUB_NAMES[variant])
        # Remove classification head if present
        if hasattr(self.model, "head"):
            self.model.head = nn.Identity()
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CLS token features (B, embed_dim)."""
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out[:, 0]

    def __repr__(self):
        return f"DINOv2(variant={self.variant}, embed_dim={self.embed_dim})"
