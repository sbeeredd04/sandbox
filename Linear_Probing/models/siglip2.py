import torch
import torch.nn as nn
from transformers import Siglip2VisionModel


# Maps short names → HuggingFace model IDs (FixRes variants)
_HF_MODELS = {
    "vitb16":    "google/siglip2-base-patch16-224",
    "vitl16":    "google/siglip2-large-patch16-256",
    "so400m14":  "google/siglip2-so400m-patch14-384",
}

# Embed dims per variant
_EMBED_DIMS = {
    "vitb16":   768,
    "vitl16":   1024,
    "so400m14": 1152,
}

# Native image sizes per variant
_IMAGE_SIZES = {
    "vitb16":   224,
    "vitl16":   256,
    "so400m14": 384,
}


class SigLIP2(nn.Module):

    def __init__(self, variant: str = "vitb16"):
        super().__init__()
        assert variant in _HF_MODELS, f"Unknown variant '{variant}'. Choose from {list(_HF_MODELS)}"
        self.variant = variant
        self.embed_dim = _EMBED_DIMS[variant]
        self.image_size = _IMAGE_SIZES[variant]
        self.model = Siglip2VisionModel.from_pretrained(_HF_MODELS[variant])
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mean-pooled patch features (B, embed_dim)."""
        B, C, H, W = x.shape

        # SigLIP2 VisionModel requires an attention mask and spatial shapes.
        # For FixRes (fixed resolution), every pixel is valid → all ones.
        mask = torch.ones(B, H, W, dtype=torch.bool, device=x.device)
        shapes = torch.tensor([[H, W]], dtype=torch.long, device=x.device).expand(B, -1)

        out = self.model(
            pixel_values=x,
            pixel_attention_mask=mask,
            spatial_shapes=shapes,
        )

        # last_hidden_state: (B, num_patches, embed_dim) — no CLS token
        return out.last_hidden_state.mean(dim=1)

    def __repr__(self):
        return f"SigLIP2(variant={self.variant}, embed_dim={self.embed_dim})"
