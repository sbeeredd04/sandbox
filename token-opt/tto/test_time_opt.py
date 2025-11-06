
from tto.ema import EMAModel
from tto.siglip import SigLIP
from tto.vqgan_wrapper import PretrainedVQGAN
from tto.continuous_tokenizer_wrapper import ContinuousTokenizerWrapper

from typing import cast, Callable, Literal
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torchvision
from PIL import Image
import torchvision.transforms.v2 as v2

from einops import rearrange, einsum

from jaxtyping import Float

import open_clip
from titok.modeling.quantizer import DiagonalGaussianDistribution
from titok.modeling.titok import TiTok
import tqdm

@dataclass
class TestTimeOptConfig:
    titok_checkpoint: str = "yucornetto/tokenizer_titok_l32_imagenet"
    optimize_post_quantization_tokens: bool = False
    vae_deterministic_sampling: bool = True
    lr: float = 1e-1
    ema_decay: float = 0.
    token_noise: float | None = None
    reg_weight: float | None = None
    reg_type: None | Literal["seed", "zero"] = None
    num_iter: int = 600
    enable_amp: bool = False
@dataclass
class TestTimeOptInfo:
    i: int
    tokens: Float[Tensor, "b d 1 n"]
    img: Float[Tensor, "b c h w"]
    loss: Float[Tensor, "b"]
ObjectiveT = Callable[[Float[Tensor, "b c h w"]], Float[Tensor, "b"]]
class TestTimeOpt(nn.Module):
    
    def __init__(self, config: TestTimeOptConfig, objective: ObjectiveT):
        super().__init__()
        self.config = config
        self.objective = objective
        
        if config.titok_checkpoint == "maskgit-vqgan":
            print("Using pretrained MaskGIT-VQGAN!")
            self.titok = PretrainedVQGAN()
        elif config.titok_checkpoint.startswith("continuous_tokenizer:"):
            parts = config.titok_checkpoint.split(":")
            model_type = parts[1].upper() if len(parts) > 1 else 'SOFTVQ'
            checkpoint_path = parts[2] if len(parts) > 2 else None
            
            kwargs = {}
            if len(parts) > 3:
                for kv in parts[3:]:
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                        kwargs[k] = v
            
            print(f"Using Continuous Tokenizer ({model_type})!")
            self.titok = ContinuousTokenizerWrapper(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                num_latent_tokens=kwargs.get('num_latent_tokens', 64),
                codebook_embed_dim=kwargs.get('codebook_embed_dim', 32),
                num_codebooks=kwargs.get('num_codebooks', 4),
                codebook_size=kwargs.get('codebook_size', 8192),
            )
        else:
            self.titok = TiTok.from_pretrained(config.titok_checkpoint)
        
        self.eval()

    def decode(self, tokens: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        
        if isinstance(self.titok, ContinuousTokenizerWrapper):
            if tokens.dim() == 4:
                tokens = tokens.squeeze(2)
                tokens = tokens.transpose(1, 2)
            
            if not self.config.optimize_post_quantization_tokens:
                if self.titok.quantize_mode in ['softvq', 'vq']:
                    tokens, _, _ = self.titok.quantize(tokens)
            
            dec = self.titok.decode(tokens)
            return dec
        else:
            def _maybe_quantize(tokens):
                if self.config.optimize_post_quantization_tokens:
                    return tokens
                else:
                    if self.titok.quantize_mode == "vae":
                        assert isinstance(self.titok, TiTok)
                        tokens = self.titok.quantize(tokens)
                        return tokens.mean if self.config.vae_deterministic_sampling else tokens.sample()
                    else:
                        return self.titok.quantize(tokens)[0]
            
            tokens = _maybe_quantize(tokens)
            dec = self.titok.decode(tokens)
            return dec

    def encode(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        
        if isinstance(self.titok, ContinuousTokenizerWrapper):
            tok = self.titok.encode(pixel_values=img)
            
            if self.config.optimize_post_quantization_tokens:
                if self.titok.quantize_mode in ['softvq', 'vq']:
                    tok, _, _ = self.titok.quantize(tok)
            
            tok = tok.transpose(1, 2).unsqueeze(2)
            return tok
        else:
            tok = self.titok.encoder(pixel_values=img, latent_tokens=self.titok.latent_tokens)
            
            if self.config.optimize_post_quantization_tokens:
                if self.titok.quantize_mode == "vae":
                    tok = DiagonalGaussianDistribution(tok)
                    return tok.mean if self.config.vae_deterministic_sampling else tok.sample()
                else:
                    assert isinstance(tok, Tensor)
                    return self.titok.quantize(tok)[0]
            
            return tok

    def _token_noise_schedule(self, i):
        t = i / (self.config.num_iter - 1)
        t = max(0, min(1, 1.5 * t))
        return 0.5 * (1 + np.cos(np.pi * t))

    def forward(self, seed: Float[Tensor, "b c h w"] | None, seed_tokens: Float[Tensor, "b d 1 n"] | None = None, callback: Callable[[TestTimeOptInfo], bool | None] | None = None, token_reset_callback: Callable[[TestTimeOptInfo], Float[Tensor, "b d 1 n"] | None] | None = None):
        
        assert not self.training
        
        if seed is not None:
            if seed_tokens is not None:
                raise ValueError("must provide seed_tokens or seed but not both")
            with torch.no_grad():
                opt_tokens = self.encode(seed)
        else:
            if seed_tokens is None:
                raise ValueError("must provide either seed_tokens or seed")
            opt_tokens = seed_tokens.detach().clone()

        opt_tokens.requires_grad_(True)
        opt = torch.optim.Adam(params=[opt_tokens], lr=self.config.lr)
        scaler = GradScaler(enabled=self.config.enable_amp)
        
        ema = EMAModel([opt_tokens], decay=self.config.ema_decay, min_decay=self.config.ema_decay)
        
        orig_tokens = opt_tokens.detach().clone()

        for i in tqdm.tqdm(range(self.config.num_iter), desc="Optimization Iterations"):
            
            if self.config.token_noise is not None:
                with torch.no_grad():
                    noise_scale = self._token_noise_schedule(i)
                    opt_tokens.add_(self.config.token_noise * noise_scale * torch.randn_like(opt_tokens))
                    
            with torch.autocast(device_type=orig_tokens.device.type, dtype=torch.float16, enabled=self.config.enable_amp):
                
                dec = self.decode(opt_tokens)
                
                if i % 50 == 0:
                    img_array = (dec[0].detach().cpu().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    Image.fromarray(img_array).save(f"debug_iter_{i:04d}.png")
                    
                loss = self.objective(dec)
                
                if self.config.reg_weight is not None:
                    assert self.config.reg_type is not None
                    if self.config.reg_type == "seed":
                        reg = self.config.reg_weight * torch.mean((opt_tokens - orig_tokens)**2, dim=(1, 2, 3))
                    elif self.config.reg_type == "zero":
                        reg = self.config.reg_weight * torch.mean(opt_tokens**2, dim=(1, 2, 3))
                    else:
                        assert False
                else:
                    reg = 0
                
                sum_loss = torch.sum(loss + reg, dim=0)
                
            scaler.scale(sum_loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            with torch.no_grad():
                if token_reset_callback is not None:
                    tokens_reset = token_reset_callback(TestTimeOptInfo(i=i, tokens=opt_tokens, img=dec, loss=loss))
                    if tokens_reset is not None:
                        opt_tokens.copy_(tokens_reset.detach())

            ema.step()
            
            with ema.average_parameters(), torch.no_grad():
                if callback is not None:
                    if callback(TestTimeOptInfo(i=i, tokens=opt_tokens, img=dec, loss=loss)):
                        break
        
        with ema.average_parameters(), torch.no_grad():
            return torch.clamp(self.decode(opt_tokens), 0.0, 1.0)
class AugmentationHelper:
    
    def __init__(self, num_augmentations: int, img_size):
        self.num_augmentations = num_augmentations
        
        if num_augmentations >= 1:
            self.augmentations = v2.Compose([
                v2.RandomCrop(size=img_size),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentations = None

    def __call__(self, x: Float[Tensor, "b c h_in w_in"]) -> Float[Tensor, "num_aug b c h w"]:
        
        if self.augmentations is None:
            return x.unsqueeze(0)
        else:
            return torch.stack([self.augmentations(x) for _ in range(self.num_augmentations)])
class CLIPObjective(nn.Module):
    
    device_indicator: Tensor

    def __init__(self, prompt: str | list[str] | None = None, neg_prompt: str | list[str] | None = None, cfg_scale: float = 1., num_augmentations: int = 0, pretrained: tuple[str, str] = ("ViT-B-32", "laion2b_s34b_b79k")):
        
        super().__init__()

        self.register_buffer("device_indicator", torch.tensor(0))
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(pretrained[0], pretrained=pretrained[1])
        self.clip_tokenizer = cast(open_clip.SimpleTokenizer, open_clip.get_tokenizer("ViT-B-32"))
        
        self.augment = AugmentationHelper(num_augmentations=num_augmentations, img_size=self.clip_model.visual.image_size)
        
        self.eval()

        self._prompt = prompt
        self._prompt_feat = None
        self._neg_prompt = neg_prompt
        self._neg_prompt_feat = None
        
        self.neg_prompt_weight = 1 - cfg_scale

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
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
