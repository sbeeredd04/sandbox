import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import sys
from pathlib import Path

# Dynamically find continuous_tokenizer directory
def find_continuous_tokenizer_dir():
    # Check if continuous_tokenizer is already in sys.path
    for path in sys.path:
        ct_path = Path(path)
        if ct_path.name == 'continuous_tokenizer' and ct_path.exists():
            return ct_path
        # Check if it's a subdirectory
        potential_path = ct_path / 'continuous_tokenizer'
        if potential_path.exists():
            return potential_path
    
    # Try common locations
    common_locations = [
        Path('../../continuous_tokenizer'),
        Path('../continuous_tokenizer'),
        Path('sandbox/continuous_tokenizer'),
        Path('/home/sbeeredd/sandbox/continuous_tokenizer'),
    ]
    
    for loc in common_locations:
        if loc.exists():
            return loc.resolve()
    
    raise FileNotFoundError("Could not find continuous_tokenizer directory. Please ensure it's cloned and in sys.path.")

# Find and add to path
CONTINUOUS_TOKENIZER_DIR = find_continuous_tokenizer_dir()
if str(CONTINUOUS_TOKENIZER_DIR) not in sys.path:
    sys.path.append(str(CONTINUOUS_TOKENIZER_DIR))

from modelling.tokenizer import SoftVQModel, VQModel, ModelArgs
from huggingface_hub import hf_hub_download


class ContinuousTokenizerWrapper(nn.Module):
    
    def __init__(self, checkpoint_path: str = None, model_type: str = 'SoftVQ', 
                 num_latent_tokens: int = 64, codebook_embed_dim: int = 32, **model_kwargs):
        super().__init__()
        
        default_config = {
            'image_size': 256,
            'codebook_size': 8192,
            'num_codebooks': 4,
            'tau': 0.07,
            'enc_type': 'vit',
            'dec_type': 'vit',
            'encoder_model': 'vit_large_patch14_dinov2.lvd142m',
            'decoder_model': 'vit_large_patch14_dinov2.lvd142m',
            'enc_pretrained': True,
            'dec_pretrained': False,
            'enc_patch_size': 16,
            'dec_patch_size': 16,
            'enc_tuning_method': 'full',
            'dec_tuning_method': 'full',
        }
        default_config.update(model_kwargs)
        
        config = ModelArgs(
            num_latent_tokens=num_latent_tokens,
            codebook_embed_dim=codebook_embed_dim,
            **default_config
        )
        
        if model_type == 'SoftVQ':
            self.model = SoftVQModel(config)
            self.quantize_mode = 'softvq'
        elif model_type == 'VQ':
            self.model = VQModel(config)
            self.quantize_mode = 'vq'
        else:
            raise ValueError(f"model_type must be 'SoftVQ' or 'VQ', got {model_type}")
        
        if checkpoint_path:
            if checkpoint_path.startswith('SoftVQVAE/') or checkpoint_path.startswith('MAETok/'):
                print(f"Downloading checkpoint from HuggingFace: {checkpoint_path}")
                ckpt_path = hf_hub_download(repo_id=checkpoint_path, filename="model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
        
        self.num_latent_tokens = num_latent_tokens
        self.codebook_embed_dim = codebook_embed_dim
        
        self.eval()
    
    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def decoder(self):
        return self.model.decoder
    
    def encode(self, pixel_values: Float[Tensor, "b c h w"], **kwargs) -> Float[Tensor, "b n d"]:
        h = self.model.encoder(pixel_values)
        h = self.model.quant_conv(h)
        return h
    
    def quantize(self, z: Float[Tensor, "b n d"]) -> tuple | Float[Tensor, "b n d"]:
        if self.model.quantize is None:
            return z
        
        quant, losses, info = self.model.quantize(z)
        return quant, losses, info
    
    def decode(self, z: Float[Tensor, "b n d"], **kwargs) -> Float[Tensor, "b c h w"]:
        h = self.model.post_quant_conv(z)
        img = self.model.decoder(h, None, None, None)
        return img
    
    def forward(self, pixel_values: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        return self.model(pixel_values)[0]


def load_softvq_model(checkpoint_path: str = None, **kwargs):
    return ContinuousTokenizerWrapper(
        checkpoint_path=checkpoint_path,
        model_type='SoftVQ',
        **kwargs
    )


def load_vq_model(checkpoint_path: str = None, **kwargs):
    return ContinuousTokenizerWrapper(
        checkpoint_path=checkpoint_path,
        model_type='VQ',
        **kwargs
    )
