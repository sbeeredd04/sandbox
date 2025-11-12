# Import custom modules for EMA (Exponential Moving Average) during optimization
from tto.ema import EMAModel
# Import SigLIP model for alternative CLIP-like objective
from tto.siglip import SigLIP
# Import pretrained VQGAN wrapper for maskgit-vqgan tokenizer option
from tto.vqgan_wrapper import PretrainedVQGAN
from tto.continuous_tokenizer_wrapper import ContinuousTokenizerWrapper
from tto.imagefolder_wrapper import ImageFolderWrapper

# Type hints for better code clarity
from typing import cast, Callable, Literal
from dataclasses import dataclass

# Standard numerical and deep learning libraries
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torchvision

# PIL for image saving
from PIL import Image

# Torchvision v2 transforms for data augmentation
import torchvision.transforms.v2 as v2
# Einops for tensor reshaping and einstein summation
from einops import rearrange, einsum
# Jaxtyping for tensor shape annotations
from jaxtyping import Float
# OpenCLIP for loading pretrained CLIP models
import open_clip

# TiTok specific modules for VAE quantization and main tokenizer
from titok.modeling.quantizer import DiagonalGaussianDistribution
from titok.modeling.titok import TiTok

# Progress bar for optimization iterations
import tqdm

@dataclass
class TestTimeOptConfig:
    # Path to pretrained TiTok checkpoint or "maskgit-vqgan" for alternative tokenizer
    titok_checkpoint: str = "yucornetto/tokenizer_titok_l32_imagenet"
    
    # Whether to optimize tokens after quantization (discrete) or before (continuous)
    # False = optimize continuous tokens, then quantize during decode (more flexible)
    # True = quantize first, then optimize discrete tokens (more structured)
    optimize_post_quantization_tokens: bool = False
    
    # For VAE quantization, whether to use deterministic mean or sample from distribution
    vae_deterministic_sampling: bool = True
    
    # Learning rate for Adam optimizer during token optimization
    lr: float = 1e-1
    
    # Exponential moving average decay rate for smoothing token updates
    # 0 = no EMA, higher values = more smoothing
    ema_decay: float = 0.
    
    # Amount of noise to add to tokens during optimization for exploration
    # None = no noise, higher values = more exploration
    token_noise: float | None = None
    
    # Regularization weight to prevent tokens from drifting too far
    # None = no regularization, higher values = stronger constraint
    reg_weight: float | None = None
    
    # Type of regularization to apply
    # "seed" = penalize deviation from original tokens
    # "zero" = penalize large token magnitudes
    # None = no regularization
    reg_type: None | Literal["seed", "zero"] = None
    
    # Number of optimization iterations to run
    num_iter: int = 600
    
    # Whether to use automatic mixed precision (AMP) for faster training
    enable_amp: bool = False


@dataclass
class TestTimeOptInfo:
    # Current iteration number
    i: int
    # Current token values (potentially with EMA applied)
    tokens: Float[Tensor, "b d 1 n"]
    # Decoded image from current tokens
    img: Float[Tensor, "b c h w"]
    # Loss value for current iteration
    loss: Float[Tensor, "b"]


# Type alias for objective functions that take images and return per-batch losses
ObjectiveT = Callable[[Float[Tensor, "b c h w"]], Float[Tensor, "b"]]


class TestTimeOpt(nn.Module):
    # Main class for test-time optimization of image tokens
    # Optimizes latent tokens to better satisfy a given objective (e.g., CLIP similarity)
    
    def __init__(self, config: TestTimeOptConfig, objective: ObjectiveT):
        super().__init__()
        self.config = config
        self.objective = objective  # Objective function to minimize (e.g., -CLIP_similarity)
        
                # Load the appropriate tokenizer model
        if config.titok_checkpoint == "maskgit-vqgan":
            self.titok = PretrainedVQGAN().eval()
        elif config.titok_checkpoint.startswith("continuous_tokenizer:"):
            # Format: "continuous_tokenizer:MODEL_TYPE:CHECKPOINT_PATH"
            parts = config.titok_checkpoint.split(":")
            model_type = parts[1] if len(parts) > 1 else 'SoftVQ'
            checkpoint_path = parts[2] if len(parts) > 2 else None
            
            self.titok = ContinuousTokenizerWrapper(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
            ).eval()
        elif config.titok_checkpoint.startswith("imagefolder:"):
            # Format: "imagefolder:MODEL_NAME" or "imagefolder:MODEL_NAME:CHECKPOINT_PATH"
            # e.g., "imagefolder:MSVR10P2-4096" or "imagefolder:MSVR10P2-4096:/path/to/checkpoint.pt"
            parts = config.titok_checkpoint.split(":")
            model_name = parts[1] if len(parts) > 1 else 'MSVR10P2-4096'
            checkpoint_path = parts[2] if len(parts) > 2 else None
            
            self.titok = ImageFolderWrapper(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            ).eval()
        else:
            self.titok = TiTok.from_pretrained(config.titok_checkpoint).eval()
            self.titok.requires_grad_(False)
        
        # Set to evaluation mode (disable dropout, batch norm training, etc.)
        self.eval()

    def decode(self, tokens: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        # Decode tokens back to pixel space (image)
        # tokens: (batch, token_dim, 1, num_tokens) -> img: (batch, 3, height, width)
        
        if isinstance(self.titok, ContinuousTokenizerWrapper):
            if tokens.dim() == 4:
                tokens = tokens.squeeze(2)
                tokens = tokens.transpose(1, 2)
            
            if not self.config.optimize_post_quantization_tokens:
                if self.titok.quantize_mode in ['softvq', 'vq']:
                    tokens, _, _ = self.titok.quantize(tokens)
            
            dec = self.titok.decode(tokens)
            return dec
        elif isinstance(self.titok, ImageFolderWrapper):
            # ImageFolder uses shape (b, d, h, w)
            # tokens shape: (b, d, 1, n) -> need to reshape to (b, d, h, w)
            b, d, _, n = tokens.shape
            h = w = int(n ** 0.5)  # Assume square spatial layout
            tokens_reshape = tokens.squeeze(2).view(b, d, h, w)
            
            if not self.config.optimize_post_quantization_tokens:
                # Quantize before decoding if optimizing continuous tokens
                tokens_reshape, _ = self.titok.quantize(tokens_reshape)
            
            dec = self.titok.decode(tokens_reshape)
            return dec
        else:
            def _maybe_quantize(tokens):
                # Apply quantization if we're optimizing pre-quantization tokens
                if self.config.optimize_post_quantization_tokens:
                    # Tokens are already quantized, return as-is
                    return tokens
                else:
                    # Need to quantize continuous tokens before decoding
                    if self.titok.quantize_mode == "vae":
                        # VAE-style quantization: sample from diagonal gaussian
                        assert isinstance(self.titok, TiTok)
                        tokens = self.titok.quantize(tokens)
                        # Either use mean (deterministic) or sample (stochastic)
                        return tokens.mean if self.config.vae_deterministic_sampling else tokens.sample()
                    else:
                        # VQ-VAE style quantization: lookup nearest codebook vectors
                        return self.titok.quantize(tokens)[0]  # Returns (quantized_tokens, indices, loss)
            
            # Quantize if necessary, then decode to RGB image
            tokens = _maybe_quantize(tokens)
            dec = self.titok.decode(tokens)
            return dec

    def encode(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        # Encode image to latent tokens
        # img: (batch, 3, height, width) -> tokens: (batch, token_dim, 1, num_tokens)
        
        if isinstance(self.titok, ContinuousTokenizerWrapper):
            tok = self.titok.encode(pixel_values=img)
            
            if self.config.optimize_post_quantization_tokens:
                if self.titok.quantize_mode in ['softvq', 'vq']:
                    tok, _, info = self.titok.quantize(tok)            
            tok = tok.transpose(1, 2).unsqueeze(2)
            return tok
        elif isinstance(self.titok, ImageFolderWrapper):
            # ImageFolder encoder outputs (b, d, h, w)
            tok = self.titok.encoder(img)
            
            if self.config.optimize_post_quantization_tokens:
                # Quantize first if optimizing post-quantization tokens
                tok, _ = self.titok.quantize(tok)
            
            # Reshape to (b, d, 1, n) format
            b, d, h, w = tok.shape
            tok = tok.view(b, d, 1, h * w)
            return tok
        else:
            # Use TiTok encoder with learnable latent tokens as query vectors
            tok = self.titok.encoder(pixel_values=img, latent_tokens=self.titok.latent_tokens)
            
            if self.config.optimize_post_quantization_tokens:
                # If optimizing post-quantization, apply quantization now
                if self.titok.quantize_mode == "vae":
                    # VAE quantization: get mean or sample from gaussian
                    tok = DiagonalGaussianDistribution(tok)
                    return tok.mean if self.config.vae_deterministic_sampling else tok.sample()
                else:
                    # VQ quantization: get discrete tokens
                    assert isinstance(tok, Tensor)
                    return self.titok.quantize(tok)[0]
            
            # Return continuous tokens for optimization
            return tok

    def _token_noise_schedule(self, i):
        # Compute noise scaling factor that decays over time
        # Uses cosine schedule that ramps to 0 at 2/3 of total iterations
        t = i / (self.config.num_iter - 1)  # Normalize to [0, 1]
        t = max(0, min(1, 1.5 * t))  # Scale to reach 1.0 at iteration 2/3
        return 0.5 * (1 + np.cos(np.pi * t))  # Cosine decay from 1.0 to 0.0

    def forward(self, seed: Float[Tensor, "b c h w"] | None, seed_tokens: Float[Tensor, "b d 1 n"] | None = None, callback: Callable[[TestTimeOptInfo], bool | None] | None = None, token_reset_callback: Callable[[TestTimeOptInfo], Float[Tensor, "b d 1 n"] | None] | None = None):
        # Main optimization loop - optimizes tokens to minimize objective function
        # seed: Optional input image to encode as starting point
        # seed_tokens: Optional pre-computed tokens as starting point
        # callback: Optional function called each iteration for monitoring/early stopping
        # token_reset_callback: Optional function to reset tokens if optimization gets stuck
        
        assert not self.training  # Ensure we're in eval mode
        
        # Initialize optimization tokens from either image or provided tokens
        if seed is not None:
            if seed_tokens is not None:
                raise ValueError("must provide seed_tokens or seed but not both")
            with torch.no_grad():
                # Encode input image to get initial tokens
                print(f"Encoding seed image to tokens...")
                opt_tokens = self.encode(seed)
                print(f"Initial tokens shape: {opt_tokens.shape}")
        else:
            if seed_tokens is None:
                raise ValueError("must provide either seed_tokens or seed")
            # Use provided tokens directly
            opt_tokens = seed_tokens.detach().clone()

        # Setup for gradient-based optimization
        opt_tokens.requires_grad_(True)  # Enable gradient computation for tokens
        opt = torch.optim.Adam(params=[opt_tokens], lr=self.config.lr)  # Adam optimizer
        scaler = GradScaler(enabled=self.config.enable_amp)  # For mixed precision training
        
        # Setup exponential moving average to smooth token updates
        ema = EMAModel([opt_tokens], decay=self.config.ema_decay, min_decay=self.config.ema_decay)
        
        # Save original tokens for regularization
        orig_tokens = opt_tokens.detach().clone()

        # Main optimization loop
        for i in tqdm.tqdm(range(self.config.num_iter), desc="Optimization Iterations"):
            
            # Add noise to tokens for exploration (if configured)
            if self.config.token_noise is not None:
                with torch.no_grad():
                    # Noise scale decreases over time according to schedule
                    noise_scale = self._token_noise_schedule(i)
                    opt_tokens.add_(self.config.token_noise * noise_scale * torch.randn_like(opt_tokens))
                    
            # Forward pass with optional mixed precision
            with torch.autocast(device_type=orig_tokens.device.type, dtype=torch.float16, enabled=self.config.enable_amp):

                # Decode tokens to image
                dec = self.decode(opt_tokens)
                
                # Save the image for debugging
                if i % 50 == 0:
                    img_array = (dec[0].detach().cpu().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    Image.fromarray(img_array).save(f"debug_iter_{i:04d}.png")
                    
                # Compute objective loss (e.g., negative CLIP similarity)
                loss = self.objective(dec)
                
                # Add regularization term if configured
                if self.config.reg_weight is not None:
                    assert self.config.reg_type is not None
                    if self.config.reg_type == "seed":
                        # L2 penalty for deviation from original tokens
                        reg = self.config.reg_weight * torch.mean((opt_tokens - orig_tokens)**2, dim=(1, 2, 3))
                    elif self.config.reg_type == "zero":
                        # L2 penalty for large token magnitudes
                        reg = self.config.reg_weight * torch.mean(opt_tokens**2, dim=(1, 2, 3))
                    else:
                        assert False
                else:
                    reg = 0
                
                # Combine loss and regularization
                sum_loss = torch.sum(loss + reg, dim=0)
                
            #
            
            # Backward pass and optimizer step
            scaler.scale(sum_loss).backward()  # Compute gradients (scaled for AMP)
            scaler.step(opt)  # Update tokens
            scaler.update()  # Update gradient scaler
            opt.zero_grad()  # Clear gradients for next iteration

            # Optional token reset mechanism (if optimization gets stuck)
            with torch.no_grad():
                if token_reset_callback is not None:
                    tokens_reset = token_reset_callback(TestTimeOptInfo(i=i, tokens=opt_tokens, img=dec, loss=loss))
                    if tokens_reset is not None:
                        opt_tokens.copy_(tokens_reset.detach())

            # Update exponential moving average
            ema.step()
            
            # Optional callback for monitoring and early stopping
            with ema.average_parameters(), torch.no_grad():
                if callback is not None:
                    # Note: callback gets EMA tokens but non-EMA image for monitoring
                    if callback(TestTimeOptInfo(i=i, tokens=opt_tokens, img=dec, loss=loss)):
                        break  # Early stopping if callback returns True
        
        # Final decode with EMA-averaged tokens
        with ema.average_parameters(), torch.no_grad():
            # Clamp output to valid image range [0, 1]
            return torch.clamp(self.decode(opt_tokens), 0.0, 1.0)


class AugmentationHelper:
    # Helper class to apply data augmentations to images during optimization
    # Uses multiple augmented views for more robust CLIP/SigLIP similarity computation
    
    def __init__(self, num_augmentations: int, img_size):
        # num_augmentations: Number of different augmented views to create per image
        # img_size: Target size for cropping
        self.num_augmentations = num_augmentations
        
        if num_augmentations >= 1:
            # Create augmentation pipeline: random crop + horizontal flip
            self.augmentations = v2.Compose([
                v2.RandomCrop(size=img_size),  # Random spatial crop
                v2.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
            ])
        else:
            # No augmentations if num_augmentations < 1
            self.augmentations = None

    def __call__(self, x: Float[Tensor, "b c h_in w_in"]) -> Float[Tensor, "num_aug b c h w"]:
        # Apply augmentations and return stacked views
        # x: Input images (batch, channels, height, width)
        # Returns: (num_augmentations, batch, channels, height, width)
        
        if self.augmentations is None:
            # No augmentation, just add dimension for consistency
            return x.unsqueeze(0)
        else:
            # Create multiple augmented views by applying transforms multiple times
            return torch.stack([self.augmentations(x) for _ in range(self.num_augmentations)])


class CLIPObjective(nn.Module):
    # Objective function based on CLIP text-image similarity
    # Optimizes tokens to make decoded images more similar to a text prompt
    
    device_indicator: Tensor  # Dummy tensor to track which device the model is on

    def __init__(self, prompt: str | list[str] | None = None, neg_prompt: str | list[str] | None = None, cfg_scale: float = 1., num_augmentations: int = 0, pretrained: tuple[str, str] = ("ViT-B-32", "laion2b_s34b_b79k")):
        # prompt: Text description to optimize towards
        # neg_prompt: Optional negative prompt to optimize away from (classifier-free guidance)
        # cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger guidance)
        # num_augmentations: Number of augmented views for robustness
        # pretrained: Tuple of (model_name, checkpoint_name) for CLIP
        
        super().__init__()

        # Register buffer to track device (parameters automatically move with model)
        self.register_buffer("device_indicator", torch.tensor(0))
        
        # Load pretrained CLIP model, preprocessing transforms, and tokenizer
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(pretrained[0], pretrained=pretrained[1])
        self.clip_tokenizer = cast(open_clip.SimpleTokenizer, open_clip.get_tokenizer("ViT-B-32"))
        
        # Setup augmentation helper for multiple views
        self.augment = AugmentationHelper(num_augmentations=num_augmentations, img_size=self.clip_model.visual.image_size)
        
        # Set to evaluation mode
        self.eval()

        # Initialize prompt and cached text features
        self._prompt = prompt
        self._prompt_feat = None  # Lazy computation (computed on first access)
        self._neg_prompt = neg_prompt
        self._neg_prompt_feat = None
        
        # Convert cfg_scale to negative prompt weight
        # cfg_scale=1.0 → weight=0 (no negative guidance)
        # cfg_scale=2.0 → weight=-1 (equal positive/negative weight)
        self.neg_prompt_weight = 1 - cfg_scale

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        # Reset cached features when prompt changes
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        # Lazily compute and cache text features for prompt
        # Returns normalized CLIP text embeddings
        assert not self.training
        if self._prompt_feat is None:
            # Encode text with CLIP text encoder
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.prompt))
            # Normalize to unit length for cosine similarity
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._prompt_feat = prompt_feat
        return self._prompt_feat

    @property
    def neg_prompt(self):
        return self._neg_prompt

    @prompt.setter
    def neg_prompt(self, prompt):
        # Reset cached features when negative prompt changes
        self._neg_prompt_feat = None
        self._neg_prompt = prompt

    @property
    @torch.no_grad
    def neg_prompt_feat(self) -> Float[Tensor, "#b d"]:
        # Lazily compute and cache text features for negative prompt
        assert not self.training
        if self._neg_prompt_feat is None:
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.neg_prompt))
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._neg_prompt_feat = prompt_feat
        return self._neg_prompt_feat

    def preprocess(self, img):
        # Apply differentiable preprocessing: resize to CLIP input size + normalize
        # This is differentiable so gradients can flow through to tokens
        resize = self.clip_preprocess.transforms[0]  # Resize transform
        normalize = self.clip_preprocess.transforms[4]  # Normalization transform
        
        # Resize if needed (bilinear interpolation is differentiable)
        if not (img.shape[-1] == img.shape[-2] == resize.size):
            img = F.interpolate(img, size=resize.size, mode="bilinear")
        
        # Apply CLIP normalization (mean/std)
        img = normalize(img)
        return img

    def tokenize(self, text):
        # Convert text to token IDs for CLIP
        if isinstance(text, str):
            text = [text]  # Wrap single string in list
        return self.clip_tokenizer(text).to(self.device_indicator.device)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        # Compute CLIP-based loss for given images
        # Returns negative similarity (minimize to maximize CLIP score)
        
        assert not self.training
        
        # Apply augmentations to get multiple views
        augs = self.augment(img)  # Shape: (num_aug, batch, channels, height, width)
        num_augs = augs.shape[0]
        
        # Flatten augmentation and batch dimensions for batch processing
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        
        # Encode images with CLIP vision encoder
        image_feats = self.clip_model.encode_image(self.preprocess(augs))
        
        # Normalize image features to unit length
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        
        # Reshape back to separate augmentations and batches
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        
        # Compute cosine similarity between image and text features
        # einsum: (num_aug, batch, dim) @ (dim, batch) -> (num_aug, batch)
        # Mean over augmentations to get robust similarity estimate
        similarity = torch.mean(einsum(image_feats, self.prompt_feat.mT, "n b d, d b -> n b"), dim=0)
        
        # Apply classifier-free guidance if negative prompt provided
        if self.neg_prompt is not None:
            # Compute similarity with negative prompt
            neg_similarity = torch.mean(einsum(image_feats, self.neg_prompt_feat.mT, "n b d, d b -> n b"), dim=0)
            # Combine: maximize (prompt_similarity - weight * neg_prompt_similarity)
            # Return negative because we minimize loss
            return -similarity - self.neg_prompt_weight * neg_similarity
        else:
            # Simple case: just maximize similarity with prompt
            return -similarity  # Negative because optimizer minimizes


class SigLIPObjective(nn.Module):
    # Alternative objective using SigLIP instead of CLIP
    # SigLIP uses a different contrastive loss (sigmoid instead of softmax)
    
    def __init__(self, prompt: str | list[str] | None = None, num_augmentations: int = 0):
        # prompt: Text description to optimize towards
        # num_augmentations: Number of augmented views for robustness
        
        super().__init__()
        # Load SigLIP model
        self.siglip = SigLIP()
        
        # Setup augmentation helper (SigLIP uses 224x224 images)
        self.augment = AugmentationHelper(num_augmentations=num_augmentations, img_size=224)
        
        # Set to evaluation mode
        self.eval()

        # Initialize prompt and cached text features
        self._prompt = prompt
        self._prompt_feat = None  # Lazy computation

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        # Reset cached features when prompt changes
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        # Lazily compute and cache text features for prompt
        assert not self.training
        if self._prompt_feat is None:
            # Encode text with SigLIP text encoder
            self._prompt_feat = self.siglip.encode_text(self._prompt)
        return self._prompt_feat

    def preprocess(self, img):
        # Resize image to SigLIP input size (224x224)
        return F.interpolate(img, size=224, mode="bilinear")

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        # Compute SigLIP-based loss for given images
        # Returns negative similarity (minimize to maximize SigLIP score)
        
        assert not self.training
        
        # Apply augmentations to get multiple views
        augs = self.augment(img)  # Shape: (num_aug, batch, channels, height, width)
        num_augs = augs.shape[0]
        
        # Flatten augmentation and batch dimensions for batch processing
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        
        # Encode images with SigLIP image encoder (differentiable for gradients)
        image_feats = self.siglip.encode_img(self.preprocess(augs), differentiable=True)
        
        # Reshape back to separate augmentations and batches
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        
        # Compute SigLIP similarity and average over augmentations
        # Return negative because we minimize loss
        return -torch.mean(self.siglip.similarity(image_embeds=image_feats, text_embeds=self.prompt_feat.unsqueeze(0)), dim=0)


class MultiObjective(nn.Module):
    # Combine multiple objective functions with weighted sum
    # Enables optimizing for multiple goals simultaneously (e.g., aesthetics + content)
    
    def __init__(self, objectives: list[nn.Module], weights: list[float]):
        # objectives: List of objective functions (e.g., [CLIPObjective, SigLIPObjective])
        # weights: Corresponding weights for each objective (e.g., [0.7, 0.3])
        
        super().__init__()
        self.weights = weights  # Weight for each objective
        self.objectives = nn.ModuleList(objectives)  # List of objective modules

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        # Compute weighted combination of all objectives
        # img: Input images
        # Returns: Weighted sum of individual objective losses
        
        # Initialize loss to zero (same shape as batch)
        loss = torch.zeros_like(img[:, 0, 0, 0])
        
        # Add weighted contribution from each objective
        for w, o in zip(self.weights, self.objectives):
            loss = loss + w * o(img)  # w: weight, o: objective function
        
        return loss
