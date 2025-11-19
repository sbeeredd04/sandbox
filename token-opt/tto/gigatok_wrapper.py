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
                'description': 'Base encoder, Large decoder, 256 tokens (622M params)',
                'gdrive_id': '1VnBu4aj0N7lFa1_wKBdTgLNZmaNgFvwi',
                'checkpoint_name': 'VQ_BL256_e200.pt',
            },
            'XLXXL256': {
                'config_path': str(gigatok_dir / 'configs' / 'vq' / 'VQ_XLXXL256.yaml'),
                'description': 'XL encoder, XXL decoder, 256 tokens (3B params)',
                'gdrive_id': '1HK_bV_zklLfGmIHGE4gMwjfhLLKi6Z3G',
                'checkpoint_name': 'VQ_XLXXL256_e300.pt',
            },
        }
    
    @property
    def CONFIGS(self):
        return self._get_configs()
    
    def __init__(self, config_name: str = 'BL256', checkpoint_path: str = None, auto_download: bool = True):

        super().__init__()
        
        # Import GigaTok models (lazy import after path is set)
        from tokenizer.tokenizer_image.vq.vq_vit_model import VQVitModelPlus, VQVitModelPlusArgs
        
        # Load configuration
        use_predefined = False
        if config_name in self.CONFIGS:
            config_info = self.CONFIGS[config_name]
            config_path = config_info['config_path']
            use_predefined = True
            print(f"Using predefined config: {config_name}")
            print(f"  {config_info['description']}")
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
        
        # Auto-download pretrained checkpoint if requested and using predefined config
        if checkpoint_path is None and use_predefined and auto_download:
            print(f"\nNo checkpoint provided, auto-downloading pretrained weights...")
            checkpoint_path = self._download_checkpoint(config_name, config_info)
        
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
    
    def _download_checkpoint(self, model_name: str, config_info: dict) -> str:
        """Download pretrained checkpoint from Google Drive"""
        cache_dir = Path("pretrained") / "gigatok"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / config_info['checkpoint_name']
        
        # If file exists, validate it
        if checkpoint_path.exists():
            try:
                # Quick validation - try to load checkpoint keys
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'ema' in checkpoint or 'model' in checkpoint or 'state_dict' in checkpoint:
                    print(f"Using cached checkpoint: {checkpoint_path}")
                    return str(checkpoint_path)
                else:
                    print(f"Cached checkpoint appears invalid, re-downloading...")
                    checkpoint_path.unlink()
            except Exception as e:
                print(f"Cached checkpoint appears corrupted ({e}), re-downloading...")
                checkpoint_path.unlink()
        
        # Download from Google Drive
        gdrive_id = config_info['gdrive_id']
        
        print(f"Downloading {model_name} checkpoint from Google Drive...")
        print(f"  File ID: {gdrive_id}")
        print(f"  Destination: {checkpoint_path}")
        print(f"  This may take a few minutes (file is large ~500MB-3GB)...")
        
        try:
            # Try using gdown first (handles Google Drive properly)
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={gdrive_id}"
                gdown.download(url, str(checkpoint_path), quiet=False)
                print(f"  Download completed successfully using gdown")
            except ImportError:
                # Fallback to manual instructions if gdown not available
                print(f"\n  ERROR: 'gdown' library not found.")
                print(f"  Install it with: pip install gdown")
                print(f"  Or manually download from:")
                print(f"    https://drive.google.com/file/d/{gdrive_id}/view")
                print(f"  And save to: {checkpoint_path}")
                raise RuntimeError("gdown library required for automatic download")
            
            # Validate the download
            print(f"  Validating downloaded checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'ema' not in checkpoint and 'model' not in checkpoint and 'state_dict' not in checkpoint:
                raise RuntimeError("Downloaded file does not appear to be a valid checkpoint")
            
            print(f"  Checkpoint validated successfully")
            return str(checkpoint_path)
            
        except Exception as e:
            # Clean up partial download
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise RuntimeError(
                f"Failed to download checkpoint: {e}\n\n"
                f"MANUAL DOWNLOAD INSTRUCTIONS:\n"
                f"1. Visit: https://drive.google.com/file/d/{gdrive_id}/view\n"
                f"2. Download the file manually\n"
                f"3. Save it to: {checkpoint_path}\n"
                f"4. Re-run your code\n\n"
                f"Or install gdown: pip install gdown"
            )
    
    def encode(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        # Encode image to 1D latent tokens
        # Input: (b, 3, 256, 256) -> Output: (b, codebook_embed_dim, 1, num_latent_tokens)
        
        # Normalize input from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        
        with torch.no_grad():
            # CNN spatial encoder
            h = self.model.encoder(x)
            
            # ViT 2D->1D encoder
            z = self.model.s2to1encoder(h)
            
            # Handle different output shapes from s2to1encoder
            if z.dim() == 3:
                # (b, num_latent_tokens, z_channels) -> (b, z_channels, 1, num_latent_tokens)
                z = z.permute(0, 2, 1).unsqueeze(2)
            elif z.dim() == 4:
                # Check if already in correct format
                b, dim1, dim2, dim3 = z.shape
                if dim3 == 1 and dim1 == self.num_latent_tokens:
                    # (b, num_latent_tokens, z_channels, 1) -> (b, z_channels, 1, num_latent_tokens)
                    z = z.squeeze(3).permute(0, 2, 1).unsqueeze(2)
                elif dim2 != 1:
                    raise ValueError(f"Unexpected 4D shape from s2to1encoder: {z.shape}")
            else:
                raise ValueError(f"Unexpected dimension from s2to1encoder: z.dim() = {z.dim()}, shape = {z.shape}")
            
            # Project to codebook dimension
            z = self.model.quant_conv(z)
        
        return z
    
    def quantize(self, z: Float[Tensor, "b d 1 n"]) -> tuple[Float[Tensor, "b d 1 n"], dict]:
        # Vector quantization: map continuous tokens to discrete codebook entries
        # Input/Output: (b, codebook_embed_dim, 1, num_latent_tokens)
        
        with torch.no_grad():
            # VectorQuantizer expects 4D input (b, c, h, w)
            # Shape (b, d, 1, n) is interpreted as (b, channels, height=1, width=num_tokens)
            z_q, vq_loss, _ = self.model.quantize(z)
        
        return z_q, {"loss": vq_loss}
    
    def decode(self, z: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        # Decode 1D latent tokens back to image
        # Input: (b, codebook_embed_dim, 1, num_latent_tokens) -> Output: (b, 3, 256, 256)
        
        with torch.no_grad():
            # Project back to z_channels dimension
            h = self.model.post_quant_conv(z)
            
            # ViT 1D->2D decoder (expects 4D input)
            h = self.model.s1to2decoder(h)
            
            # CNN spatial decoder
            x_recon = self.model.decoder(h)
        
        # Denormalize output from [-1, 1] to [0, 1]
        x_recon = (x_recon + 1.0) / 2.0
        
        return x_recon
    
    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        z = self.encode(x)
        z_q, _ = self.quantize(z)
        x_recon = self.decode(z_q)
        return x_recon


def load_gigatok_model(config_name: str = 'BL256', checkpoint_path: str = None):
    return GigaTokWrapper(config_name=config_name, checkpoint_path=checkpoint_path)
