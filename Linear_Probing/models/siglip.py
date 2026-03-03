import torch
import torch.nn as nn
from transformers import SiglipVisionModel


# Maps short names → HuggingFace model IDs
_HF_MODELS = {
    "vitb16": "google/siglip-base-patch16-224",
    "vitl16": "google/siglip-large-patch16-256",
}

# Embed dims per variant
_EMBED_DIMS = {
    "vitb16": 768,
    "vitl16": 1024,
}


class SigLIP(nn.Module):

    def __init__(self, variant: str = "vitb16"):
        super().__init__()
        assert variant in _HF_MODELS, f"Unknown variant '{variant}'. Choose from {list(_HF_MODELS)}"
        self.variant = variant
        self.embed_dim = _EMBED_DIMS[variant]
        self.model = SiglipVisionModel.from_pretrained(_HF_MODELS[variant])
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.model(pixel_values=x)
        
        # last_hidden_state: (B, num_patches, embed_dim) — no CLS token for SigLIP instead use last hidden state 
        return out.last_hidden_state.mean(dim=1)

    def __repr__(self):
        return f"SigLIP(variant={self.variant}, embed_dim={self.embed_dim})"
