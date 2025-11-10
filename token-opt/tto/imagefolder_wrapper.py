import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
import sys
from math import sqrt

# Add ImageFolder to path
sys.path.append('/home/sbeeredd/sandbox/ImageFolder')

from tokenizer.tokenizer_image.xqgan_model import VQModel, ModelArgs
from huggingface_hub import hf_hub_download



class ImageFolderWrapper(nn.Module):
    
    # Model configs based on YAML files
    CONFIGS = {
        'MSVR10P2-4096': {
            'codebook_size': 4096,
            'codebook_embed_dim': 32,
            'codebook_l2_norm': True,
            'enc_type': 'dinov2',
            'dec_type': 'dinov2',
            'encoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'decoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'num_latent_tokens': 121,
            'product_quant': 2,
            'v_patch_nums': [1, 1, 2, 3, 3, 4, 5, 6, 8, 11],
            'abs_pos_embed': True,
            'share_quant_resi': 4,
            'codebook_drop': 0.1,
            'half_sem': True,
            'start_drop': 3,
            'hf_url': 'https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-4096/best_ckpt.pt',
        },
        'MSVR10P2-8192': {
            'codebook_size': 8192,
            'codebook_embed_dim': 32,
            'codebook_l2_norm': True,
            'enc_type': 'dinov2',
            'dec_type': 'dinov2',
            'encoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'decoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'num_latent_tokens': 121,
            'product_quant': 2,
            'v_patch_nums': [1, 1, 2, 3, 3, 4, 5, 6, 8, 11],
            'abs_pos_embed': True,
            'share_quant_resi': 4,
            'codebook_drop': 0.1,
            'half_sem': True,
            'start_drop': 3,
            'hf_url': 'https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-8192/best_ckpt.pt',
        },
        'MSVR10P2-16384': {
            'codebook_size': 16384,
            'codebook_embed_dim': 32,
            'codebook_l2_norm': True,
            'enc_type': 'dinov2',
            'dec_type': 'dinov2',
            'encoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'decoder_model': 'vit_base_patch14_dinov2.lvd142m',
            'num_latent_tokens': 121,
            'product_quant': 2,
            'v_patch_nums': [1, 1, 2, 3, 3, 4, 5, 6, 8, 11],
            'abs_pos_embed': True,
            'share_quant_resi': 4,
            'codebook_drop': 0.1,
            'half_sem': True,
            'start_drop': 3,
            'hf_url': 'https://huggingface.co/qiuk6/XQ-GAN/resolve/main/MSVR10P2-16384/best_ckpt.pt',
        },
    }
    
    def __init__(self, model_name: str = 'MSVR10P2-4096', checkpoint_path: str = None) : 
        
        super().__init__()
        
        if model_name not in self.CONFIGS:
            raise ValueError(f"Model name {model_name} not recognized. Available models: {list(self.CONFIGS.keys())}")
        
        config_dict = self.CONFIGS[model_name]
        hf_url = config_dict.pop('hf_url')
        
        #create ModelArgs with inference
        config = ModelArgs(
            **config_dict, 
            semantic_guide='none', 
            detail_guide='none', 
            test_model=True, 
            commit_loss_beta=0.25, 
            entropy_loss_ratio=0.0, 
        )
        
        #create model 
        self.model = VQModel(config)
        
        #load checkpoint
        if checkpoint_path is None:
            print(f"Downloading checkpoint for {model_name} from {hf_url}...")
            checkpoint_path = self._download_checkpoint(model_name, hf_url)
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        #compatiibility 
        self.quantize_mode = 'vq'
        self.latent_tokens = None
        self.num_latent_tokens = config.num_latent_tokens // config.product_quant
        self.product_quant = config.product_quant
        self.v_patch_nums = config.v_patch_nums
        self.codebook_embed_dim = config.codebook_embed_dim
        
        print(f"Model {model_name} loaded successfully.")
        print(f"\t - Codebook size: {config.codebook_size} x {config.product_quant} quants")
        print(f"\t - Number of latent tokens: {self.num_latent_tokens} ")
        print(f"\t - Multi-scale levels: {len(config.v_patch_nums)} ")
        
    #download the checkpoint
    def _download_checkpoint(self, model_name: str, hf_url: str) -> str:
        cache_dir = Path("pretrained") / "imagefolder"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / f"{model_name}.pt"
        
        if checkpoint_path.exists():
            print(f"✓ Using cached checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
        
        # Download using wget (simpler than huggingface_hub for direct URLs)
        import urllib.request
        print(f"Downloading to {checkpoint_path}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Downloaded successfully")
        return str(checkpoint_path)
    

    def encoder(self, x: Float[Tensor, "b c h w"], latent_tokens=None) -> Float[Tensor, "b t d"]:
        assert latent_tokens is None
    
        with torch.no_grad():
            h = self.model.encode(x)
            
        return h
    
    def quantize(self, z: Float[Tensor, "b d h w"]) -> tuple[Float[Tensor, "b d h w"], dict]:
        with torch.no_grad():
            b, c, l, _ = z.shape
            
            if self.product_quant > 1:
                # Split for product quantization
                h_list = z.chunk(chunks=self.product_quant, dim=2)
                quant_list = []
                vq_loss_list = []
                
                for i, h in enumerate(h_list):
                    h = h.view(b, -1, int(sqrt(l // self.product_quant)), 
                              int(sqrt(l // self.product_quant)))
                    quant, _, vq_loss, _, _ = self.model.quantizes[i].forward(
                        h, ret_usages=False, dropout=None
                    )
                    quant_list.append(quant)
                    vq_loss_list.append(vq_loss)
                
                z_quant = torch.cat(quant_list, dim=1)
                vq_loss = sum(vq_loss_list) / len(vq_loss_list)
            else:
                z_quant, _, vq_loss, _, _ = self.model.quantize.forward(
                    z, ret_usages=False, dropout=None
                )
        
        return z_quant, {"loss": vq_loss}
    
    def decode(self, z_quant: Float[Tensor, "b d h w"]) -> Float[Tensor, "b c h w"]:
        with torch.no_grad():
            x_recon = self.model.decode(z_quant)
        
        return x_recon
    
    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
       h = self.encoder(x)
       z_quant, _ = self.quantize(h)
       x_recon = self.decode(z_quant)
       return x_recon
   
def load_imagefolder_model(model_name: str = 'MSVR10P2-4096', checkpoint_path: str = None):
    return ImageFolderWrapper(model_name=model_name, checkpoint_path=checkpoint_path)