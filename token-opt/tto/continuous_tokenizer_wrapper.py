import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../continuous_tokenizer'))

from modelling.tokenizer import SoftVQModel, VQModel, ModelArgs


class ContinuousTokenizerWrapper(nn.Module):
    
    def __init__(self, checkpoint_path: str = None, model_type: str = 'SoftVQ', 
                 num_latent_tokens: int = 64, codebook_embed_dim: int = 32, 
                 num_codebooks: int = 4, codebook_size: int = 8192,
                 enc_type: str = 'vit', dec_type: str = 'vit',
                 encoder_model: str = 'vit_large_patch14_dinov2.lvd142m',
                 decoder_model: str = 'vit_large_patch14_dinov2.lvd142m',
                 encoder_pretrained: bool = True, decoder_pretrained: bool = False,
                 image_size: int = 256, **model_kwargs):
        super().__init__()
        
        config_dict = {
            'num_latent_tokens': num_latent_tokens,
            'codebook_embed_dim': codebook_embed_dim,
            'num_codebooks': num_codebooks,
            'codebook_size': codebook_size,
            'enc_type': enc_type,
            'dec_type': dec_type,
            'encoder_model': encoder_model,
            'decoder_model': decoder_model,
            'encoder_pretrained': encoder_pretrained,
            'decoder_pretrained': decoder_pretrained,
            'image_size': image_size,
            **model_kwargs
        }
        
        config = ModelArgs(**config_dict)
        
        if model_type.lower() in ['softvq', 'soft']:
            self.model = SoftVQModel(config)
            self.quantize_mode = 'softvq'
        elif model_type.lower() == 'vq':
            self.model = VQModel(config)
            self.quantize_mode = 'vq'
        else:
            raise ValueError(f"model_type must be 'SoftVQ' or 'VQ', got {model_type}")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'ema' in state_dict:
                state_dict = state_dict['ema']
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
        
        self.num_latent_tokens = num_latent_tokens
        self.codebook_embed_dim = codebook_embed_dim
        self._encoder = self.model.encoder
        self._decoder = self.model.decoder
        
        self.eval()
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder
    
    def encode(self, pixel_values: Float[Tensor, "b c h w"], **kwargs) -> Float[Tensor, "b n d"]:
        h = self.model.encoder(pixel_values)
        h = self.model.quant_conv(h)
        return h
    
    def quantize(self, z: Float[Tensor, "b n d"]) -> tuple:
        if self.model.quantize is None:
            return z, {}, {}
        
        quant, losses, info = self.model.quantize(z)
        return quant, losses, info
    
    def decode(self, z: Float[Tensor, "b n d"], **kwargs) -> Float[Tensor, "b c h w"]:
        h = self.model.post_quant_conv(z)
        img = self.model.decoder(h, None, None, None)
        return img
    
    def forward(self, pixel_values: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        recon, losses, info = self.model(pixel_values)
        return recon


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
