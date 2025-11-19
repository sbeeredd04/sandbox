import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
import sys
import yaml
import os

# Simple: find GigaTok relative to this file's location
# This file is at: sandbox/token-opt/tto/gigatok_wrapper.py
# GigaTok is at:   sandbox/GigaTok/
_GIGATOK_DIR = None

def get_gigatok_dir():
    global _GIGATOK_DIR
    if _GIGATOK_DIR is None:
        # Get directory of this file
        this_file = Path(__file__).resolve()
        # Go up to sandbox: token-opt/tto/gigatok_wrapper.py -> token-opt/tto -> token-opt -> sandbox
        sandbox_dir = this_file.parent.parent.parent
        _GIGATOK_DIR = sandbox_dir / 'GigaTok'
        
        if not _GIGATOK_DIR.exists():
            raise FileNotFoundError(f"GigaTok not found at {_GIGATOK_DIR}")
        
        if str(_GIGATOK_DIR) not in sys.path:
            sys.path.append(str(_GIGATOK_DIR))
    
    return _GIGATOK_DIR


class GigaTokWrapper(nn.Module):
    
    # Predefined model configurations (will be populated dynamically)
    @staticmethod
    def _get_configs():
        gigatok_dir = get_gigatok_dir()
        return {
            'BL256': {
                'config_path': str(gigatok_dir / 'configs' / 'vq' / 'VQ_BL256.yaml'),
                'description': 'Base encoder, Large decoder, 256 tokens',
            },
            'XLXXL256': {
                'config_path': str(gigatok_dir / 'configs' / 'vq' / 'VQ_XLXXL256.yaml'),
                'description': 'XL encoder, XXL decoder, 256 tokens (3B params)',
            },
        }
    
    @property
    def CONFIGS(self):
        return self._get_configs()
    
    def __init__(self, config_name: str = 'BL256', checkpoint_path: str = None):

        super().__init__()
        
        # Import GigaTok models (lazy import after path is set)
        from tokenizer.tokenizer_image.vq.vq_vit_model import VQVitModelPlus, VQVitModelPlusArgs
        
        # Load configuration
        if config_name in self.CONFIGS:
            config_path = self.CONFIGS[config_name]['config_path']
            print(f"Using predefined config: {config_name}")
            print(f"  {self.CONFIGS[config_name]['description']}")
            print(f"  Config path: {config_path}")
        elif Path(config_name).exists() and config_name.endswith('.yaml'):
            config_path = config_name
            print(f"Using custom config: {config_path}")
        else:
            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available: {list(self.CONFIGS.keys())} or provide YAML path"
            )
        
        # Verify config file exists before trying to open it
        config_path_obj = Path(config_path)
        print(f"\nVerifying config file...")
        print(f"  Path: {config_path_obj}")
        print(f"  Absolute path: {config_path_obj.absolute()}")
        print(f"  Exists: {config_path_obj.exists()}")
        
        if not config_path_obj.exists():
            # Print directory contents to help debug
            parent_dir = config_path_obj.parent
            print(f"\n  Config file not found!")
            print(f"  Parent directory: {parent_dir}")
            print(f"  Parent exists: {parent_dir.exists()}")
            if parent_dir.exists():
                print(f"  Contents of parent directory:")
                try:
                    for item in parent_dir.iterdir():
                        print(f"    - {item.name}")
                except Exception as e:
                    print(f"    Error listing directory: {e}")
            
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"GigaTok directory: {GIGATOK_DIR}\n"
                f"Please ensure the GigaTok repository is properly cloned with all config files."
            )
        
        # Parse YAML config
        print(f"  ✓ Config file exists, loading...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Config loaded successfully")
        
        model_init_args = config['model']['init_args']
        
        # Create model args
        model_args = VQVitModelPlusArgs(**model_init_args)
        
        # Initialize model
        self.model = VQVitModelPlus(model_args)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract state dict from checkpoint
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
                print("  Using EMA weights")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"\tMissing keys: {len(missing)}")
            if unexpected:
                print(f"\tUnexpected keys: {len(unexpected)}")
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint provided - using random initialization")
        
        self.model.eval()
        
        # Store config for reference
        self.num_latent_tokens = model_args.num_latent_tokens
        self.codebook_size = model_args.codebook_size
        self.codebook_embed_dim = model_args.codebook_embed_dim
        self.z_channels = model_args.z_channels
        
        print(f"\nModel configuration:")
        print(f"  Latent tokens: {self.num_latent_tokens}")
        print(f"  Codebook size: {self.codebook_size}")
        print(f"  Codebook dim: {self.codebook_embed_dim}")
        print(f"  Z channels: {self.z_channels}")
    
    def encode(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        # Encode image to 1D latent tokens
        # Input: (b, 3, 256, 256) -> Output: (b, codebook_embed_dim, 1, num_latent_tokens)
        
        with torch.no_grad():
            # CNN spatial encoder
            h = self.model.encoder(x)  # (b, z_channels, h, w)
            print(f"[DEBUG encode] After encoder: h.shape = {h.shape}")
            
            # ViT 2D->1D encoder: compresses spatial features to 1D sequence
            z = self.model.s2to1encoder(h)  # Expected: (b, num_latent_tokens, z_channels)
            print(f"[DEBUG encode] After s2to1encoder: z.shape = {z.shape}, z.dim() = {z.dim()}")
            
            # Handle different output shapes from s2to1encoder
            if z.dim() == 3:
                # Expected case: (b, num_latent_tokens, z_channels) -> (b, z_channels, 1, num_latent_tokens)
                print(f"[DEBUG encode] z is 3D, reshaping for Conv2d")
                z = z.permute(0, 2, 1).unsqueeze(2)  # (b, z_channels, 1, num_latent_tokens)
            elif z.dim() == 4:
                # Already 4D from s2to1encoder
                print(f"[DEBUG encode] z is already 4D: {z.shape}")
                # Check if it's (b, n, d, 1) and needs to be (b, d, 1, n)
                b, dim1, dim2, dim3 = z.shape
                if dim3 == 1 and dim1 == self.num_latent_tokens:
                    # Shape is (b, num_latent_tokens, z_channels, 1) -> (b, z_channels, 1, num_latent_tokens)
                    z = z.squeeze(3).permute(0, 2, 1).unsqueeze(2)
                elif dim2 == 1:
                    # Shape is (b, z_channels, 1, num_latent_tokens) - already correct
                    pass
                else:
                    raise ValueError(f"Unexpected 4D shape from s2to1encoder: {z.shape}")
            else:
                raise ValueError(f"Unexpected dimension from s2to1encoder: z.dim() = {z.dim()}, shape = {z.shape}")
            
            print(f"[DEBUG encode] Before quant_conv: z.shape = {z.shape}")
            
            # Project to codebook dimension using 1x1 Conv2d
            z = self.model.quant_conv(z)  # (b, codebook_embed_dim, 1, num_latent_tokens)
            print(f"[DEBUG encode] After quant_conv: z.shape = {z.shape}")
        
        return z
    
    def quantize(self, z: Float[Tensor, "b d 1 n"]) -> tuple[Float[Tensor, "b d 1 n"], dict]:
        # Vector quantization: map continuous tokens to discrete codebook entries
        # Input/Output: (b, codebook_embed_dim, 1, num_latent_tokens)
        
        with torch.no_grad():
            # VectorQuantizer expects 4D input (b, c, h, w)
            # Our shape (b, d, 1, n) is interpreted as (b, channels, height=1, width=num_tokens)
            # This is already in the correct format, pass directly
            print(f"[DEBUG quantize] Input z.shape = {z.shape}")
            
            # VQ: find nearest codebook entry for each token
            z_q, vq_loss, _ = self.model.quantize(z)  # (b, d, 1, n) -> (b, d, 1, n)
            
            print(f"[DEBUG quantize] Output z_q.shape = {z_q.shape}")
        
        return z_q, {"loss": vq_loss}
    
    def decode(self, z: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        # Decode 1D latent tokens back to image
        # Input: (b, codebook_embed_dim, 1, num_latent_tokens) -> Output: (b, 3, 256, 256)
        
        with torch.no_grad():
            print(f"[DEBUG decode] Input z.shape = {z.shape}")
            
            # post_quant_conv is Conv2d, input: (b, codebook_embed_dim, 1, num_latent_tokens)
            h = self.model.post_quant_conv(z)  # (b, z_channels, 1, num_latent_tokens)
            print(f"[DEBUG decode] After post_quant_conv: h.shape = {h.shape}")
            
            # Reshape for 1D->2D decoder: (b, z_channels, 1, n) -> (b, n, z_channels)
            h = h.squeeze(2).permute(0, 2, 1)  # (b, num_latent_tokens, z_channels)
            print(f"[DEBUG decode] After reshape: h.shape = {h.shape}")
            
            # ViT 1D->2D decoder: expand 1D sequence back to 2D spatial features
            h = self.model.s1to2decoder(h)  # (b, num_latent_tokens, z_channels) -> (b, z_channels, h, w)
            print(f"[DEBUG decode] After s1to2decoder: h.shape = {h.shape}")
            
            # CNN spatial decoder: upsample to full resolution image
            x_recon = self.model.decoder(h)  # (b, z_channels, h, w) -> (b, 3, 256, 256)
            print(f"[DEBUG decode] After decoder: x_recon.shape = {x_recon.shape}")
        
        return x_recon
    
    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        z = self.encode(x)
        z_q, _ = self.quantize(z)
        x_recon = self.decode(z_q)
        return x_recon


def load_gigatok_model(config_name: str = 'BL256', checkpoint_path: str = None):
    return GigaTokWrapper(config_name=config_name, checkpoint_path=checkpoint_path)
