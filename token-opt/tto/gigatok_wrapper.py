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

        with torch.no_grad():
            # CNN encoder: (b, 3, 256, 256) -> (b, z_channels, h, w)
            h = self.model.encoder(x)
            
            # ViT encoder 2D->1D: (b, z_channels, h, w) -> (b, num_latent_tokens, z_channels)
            z = self.model.s2to1encoder(h)
            
            # Project to codebook dimension: (b, num_latent_tokens, z_channels) -> (b, num_latent_tokens, codebook_embed_dim)
            z = self.model.quant_conv(z)
            
            # Reshape to (b, codebook_embed_dim, 1, num_latent_tokens) for consistency
            z = z.permute(0, 2, 1).unsqueeze(2)  # (b, d, 1, n)
        
        return z
    
    def quantize(self, z: Float[Tensor, "b d 1 n"]) -> tuple[Float[Tensor, "b d 1 n"], dict]:

        with torch.no_grad():
            # Reshape for quantizer: (b, d, 1, n) -> (b, n, d)
            z_in = z.squeeze(2).permute(0, 2, 1)
            
            # Quantize: (b, n, d) -> (b, n, d)
            z_q, vq_loss, _ = self.model.quantize(z_in)
            
            # Reshape back: (b, n, d) -> (b, d, 1, n)
            z_q = z_q.permute(0, 2, 1).unsqueeze(2)
        
        return z_q, {"loss": vq_loss}
    
    def decode(self, z: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:

        with torch.no_grad():
            # Reshape: (b, d, 1, n) -> (b, n, d)
            z_in = z.squeeze(2).permute(0, 2, 1)
            
            # Post-quantization conv: (b, n, d) -> (b, n, z_channels)
            h = self.model.post_quant_conv(z_in)
            
            # ViT decoder 1D->2D: (b, n, z_channels) -> (b, z_channels, h, w)
            h = self.model.s1to2decoder(h)
            
            # CNN decoder: (b, z_channels, h, w) -> (b, 3, 256, 256)
            x_recon = self.model.decoder(h)
        
        return x_recon
    
    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        z = self.encode(x)
        z_q, _ = self.quantize(z)
        x_recon = self.decode(z_q)
        return x_recon


def load_gigatok_model(config_name: str = 'BL256', checkpoint_path: str = None):
    return GigaTokWrapper(config_name=config_name, checkpoint_path=checkpoint_path)
