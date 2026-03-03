import torch
import torch.nn as nn
import timm


# Maps short names → timm pretrained tags
_TIMM_TAGS = {
    "vitb16": "vit_base_patch16_224.mae",
    "vitl16": "vit_large_patch16_224.mae",
    "vith14": "vit_huge_patch14_224.mae",
}

# Embed dims per variant
_EMBED_DIMS = {
    "vitb16": 768,
    "vitl16": 1024,
    "vith14": 1280,
}


class MAE(nn.Module):

    def __init__(self, variant: str = "vitl16"):
        super().__init__()
        assert variant in _TIMM_TAGS, f"Unknown variant '{variant}'. Choose from {list(_TIMM_TAGS)}"
        self.variant = variant
        self.embed_dim = _EMBED_DIMS[variant]
        self.model = timm.create_model(_TIMM_TAGS[variant], pretrained=True, num_classes=0)
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)[:, 0]

    def __repr__(self):
        return f"MAE(variant={self.variant}, embed_dim={self.embed_dim})"
