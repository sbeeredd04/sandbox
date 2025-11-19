import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
import sys
import yaml

# Add GigaTok to path
sys.path.append('/home/sbeeredd/sandbox/GigaTok')

from tokenizer.tokenizer_image.vq.vq_vit_model import VQVitModelPlus, VQVitModelPlusArgs


class GigaTokWrapper(nn.Module):
    
    # Predefined model configurations
    CONFIGS = {
        'BL256': {
            'config_path': '/home/sbeeredd/sandbox/GigaTok/configs/vq/VQ_BL256.yaml',
            'description': 'Base encoder, Large decoder, 256 tokens',
        },
        'XLXXL256': {
            'config_path': '/home/sbeeredd/sandbox/GigaTok/configs/vq/VQ_XLXXL256.yaml',
            'description': 'XL encoder, XXL decoder, 256 tokens (3B params)',
        },
    }
    
    def __init__(self, config_name: str = 'BL256', checkpoint_path: str = None):

        super().__init__()
        
        # Load configuration
        if config_name in self.CONFIGS:
            config_path = self.CONFIGS[config_name]['config_path']
            print(f"Using predefined config: {config_name}")
            print(f"  {self.CONFIGS[config_name]['description']}")
        elif Path(config_name).exists() and config_name.endswith('.yaml'):
            config_path = config_name
            print(f"Using custom config: {config_path}")
        else:
            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available: {list(self.CONFIGS.keys())} or provide YAML path"
            )
        
        # Parse YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
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
