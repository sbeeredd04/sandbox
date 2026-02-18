"""
Deep Geometric Moments (DGM) encoder for RAE.

This module wraps the DGM ResNet34 model as an encoder compatible with the RAE
(Representation Autoencoder) framework. Instead of using a Vision Transformer
(like DINOv2, MAE, or SigLIP2), DGM uses a CNN backbone with learned geometric
basis functions to produce spatially-aware feature representations.

=== How DGM Works (High-Level) ===

DGM (Deep Geometric Moments) computes features by:
1. Processing input images through a ResNet-like CNN backbone to get spatial
   feature maps at 32×32 resolution.
2. Learning geometric basis functions on a coordinate grid (Legendre-polynomial
   inspired), which are transformed via learned affine parameters.
3. Computing element-wise products of CNN features and basis functions to produce
   "geometric moment" features — capturing both appearance and spatial structure.
4. Globally pooling these products to produce a compact moment descriptor (xy1).

The key outputs are:
  - grid: [B, 256, H, W]  — spatial feature maps (basis × CNN features)
  - xy1:  [B, 256]         — global moment descriptor (avg-pooled grid)

=== Adapting DGM to RAE ===

RAE expects encoders to follow a simple protocol:
  - patch_size: int  — conceptual patch size (used to compute number of tokens)
  - hidden_size: int — feature dimension of the output tokens
  - forward(x) → [B, N, C] — returns N patch tokens of dimension C

DGM's grid features are naturally spatial, so we:
  1. Extract grid features at 32×32 resolution
  2. Adaptively pool to 16×16 (= 256 tokens, matching DINOv2's token count)
  3. Reshape from [B, C, H, W] to [B, N, C] where N = H×W = 256, C = 256

This gives us:
  - patch_size = 16  (256 input / 16×16 spatial = 16 pixels per token)
  - hidden_size = 256 (DGM's internal feature dimension)
  - 256 tokens per image (same as DINOv2-B with 224 input / patch_size=14)

=== Key Differences from ViT-based Encoders ===

| Property         | DINOv2-B        | DGM ResNet34    |
|------------------|-----------------|-----------------|
| Backbone         | ViT-B/14        | ResNet34 + GM   |
| hidden_size      | 768             | 256             |
| patch_size       | 14              | 16 (conceptual) |
| Num tokens       | 256 (16×16)     | 256 (16×16)     |
| Global feature   | CLS token       | xy1 (moments)   |
| Pre-training     | Self-supervised | Supervised      |
| Normalization    | ImageNet μ/σ    | [0,1] range     |

Reference:
  Deep Geometric Moments — /scratch/sbeeredd/sandbox/Deep-Geometric-Moment/
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_encoder


# ---------------------------------------------------------------------------
# DGM model import helper
# ---------------------------------------------------------------------------
# The DGM model (resnet_gm.py) lives in a sibling project directory outside
# RAE. We add it to sys.path so we can import the ResNet34 factory function.
# This avoids duplicating the model code and keeps a single source of truth.
_DGM_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Deep-Geometric-Moment')
)
if _DGM_MODEL_DIR not in sys.path:
    sys.path.insert(0, _DGM_MODEL_DIR)

# Import the ResNet34 factory from the DGM project.
# This creates a MyResNet1 instance with [3,4,6,3] blocks and 256-dim features.
from resnet_gm import ResNet34 as _build_resnet34


@register_encoder()
class DGMEncoder(nn.Module):
    """
    DGM (Deep Geometric Moments) encoder wrapper for RAE.

    This encoder uses a ResNet34 backbone augmented with geometric moment
    computation. It is frozen during RAE training — only the decoder learns.

    The encoder produces 256 spatial tokens (16×16) with 256-dim features,
    obtained by adaptive-pooling the DGM grid features from 32×32 to 16×16.

    Args:
        checkpoint_path (str):
            Path to the pretrained DGM ResNet34 checkpoint (.pth.tar).
            The checkpoint may contain a 'state_dict' key with 'module.'
            prefix from DataParallel training — this is handled automatically.
        num_classes (int):
            Number of classification classes the DGM model was trained with.
            Default: 1000 (ImageNet). This only affects the classifier head
            which is not used for encoding.
        target_spatial_size (int):
            Target spatial resolution for adaptive pooling. The grid features
            (originally 32×32) are pooled to (target × target) before being
            reshaped to patch tokens. Default: 16 → 256 tokens total.
        freeze (bool):
            Whether to freeze encoder weights. Default: True.
            In RAE, encoders are always frozen; only the decoder is trained.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 1000,
        target_spatial_size: int = 16,
        freeze: bool = True,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Build DGM ResNet34 model
        # ----------------------------------------------------------------
        # The DGM model operates at 32×32 internal spatial resolution
        # (input images are downsampled by the initial conv with stride=8).
        # It produces 256-dimensional features at every spatial position.
        self.model = _build_resnet34(num_classes=num_classes)

        # ----------------------------------------------------------------
        # Load pretrained checkpoint
        # ----------------------------------------------------------------
        self._load_checkpoint(checkpoint_path)

        # ----------------------------------------------------------------
        # RAE-required attributes (Stage1Protocol)
        # ----------------------------------------------------------------
        # patch_size: conceptual patch size used by RAE to compute the number
        # of spatial tokens. With a 256×256 input and 16×16 output grid:
        #   base_patches = (encoder_input_size / patch_size)² = (256/16)² = 256
        self.patch_size = 16

        # hidden_size: the feature dimension of each output token.
        # DGM uses df=256 throughout its architecture.
        self.hidden_size = 256

        # Target spatial size for adaptive pooling (default 16 → 16×16 = 256 tokens)
        self.target_spatial_size = target_spatial_size

        # ----------------------------------------------------------------
        # Freeze encoder weights (standard RAE practice)
        # ----------------------------------------------------------------
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load DGM pretrained weights, handling DataParallel 'module.' prefix.

        The DGM model was originally trained with nn.DataParallel, so saved
        checkpoint keys have a 'module.' prefix that must be stripped for
        loading into the unwrapped model.

        Args:
            checkpoint_path: Path to .pth.tar checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"DGM checkpoint not found at: {checkpoint_path}\n"
                f"Please download or symlink the pretrained DGM ResNet34 weights."
            )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # The checkpoint may store weights under different keys depending
        # on how it was saved. Handle the common formats:
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix from DataParallel training
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k[len('module.'):] if k.startswith('module.') else k
            cleaned[new_key] = v

        # Load weights. strict=False allows missing/extra keys (e.g., classifier
        # head trained with different num_classes won't cause errors).
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[DGMEncoder] Missing keys: {missing}")
        if unexpected:
            print(f"[DGMEncoder] Unexpected keys: {unexpected}")
        print(f"[DGMEncoder] Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract DGM features and reshape to RAE token format.

        Args:
            x: Input images [B, 3, H, W].
               In RAE, images arrive pre-normalized:
                 x = (raw_image - encoder_mean) / encoder_std
               For DGM we set mean=0, std=1 so images arrive in ~[0, 1] range
               (from transforms.ToTensor()), which is what DGM expects.

        Returns:
            tokens: [B, N, C] where N = target_spatial_size² = 256, C = 256.

        Internal pipeline:
            1. DGM forward → (cl, (grid, xy1), imgr)
               - grid: [B, 256, 32, 32] spatial features
               - xy1:  [B, 256] global moment descriptor
            2. Adaptive pool grid: [B, 256, 32, 32] → [B, 256, 16, 16]
            3. Reshape: [B, 256, 16, 16] → [B, 256, 256] (N=256 tokens, C=256 dim)
        """
        # ----------------------------------------------------------------
        # Step 1: Forward through DGM model
        # ----------------------------------------------------------------
        # return_moments=True gives us the grid and xy1 features
        # cl = classification logits (unused for encoding)
        # grid = spatial features (element-wise product of bases × CNN features)
        # xy1 = global average-pooled moment descriptor
        # imgr = saliency visualization (unused)
        _, (grid, xy1), _ = self.model(x, return_moments=True)
        # grid: [B, 256, H, W] where H=W=32 for 256×256 input (stride-8 conv)
        # xy1:  [B, 256] global descriptor

        # ----------------------------------------------------------------
        # Step 2: Adaptive pool to fixed spatial resolution
        # ----------------------------------------------------------------
        # Pool from native 32×32 to 16×16, producing 256 tokens.
        # This makes the token count consistent with ViT-based encoders
        # (DINOv2-B with 224 input / patch_size=14 → 16×16 = 256 tokens).
        grid = F.adaptive_avg_pool2d(
            grid, (self.target_spatial_size, self.target_spatial_size)
        )
        # grid: [B, 256, 16, 16]

        # ----------------------------------------------------------------
        # Step 3: Reshape to RAE patch token format [B, N, C]
        # ----------------------------------------------------------------
        B, C, H, W = grid.shape
        tokens = grid.reshape(B, C, H * W).permute(0, 2, 1)
        # tokens: [B, N=256, C=256]

        return tokens
