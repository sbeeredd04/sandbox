from email.mime import image
import time
from pyparsing import alphas
import torch
import deepinv
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
image_size = 32

checkpoint_path = "./checkpoints/ddpm_mnist.pth"
model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=checkpoint_path).to(device)

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

model.eval()

n_samples = 32

with torch.no_grad():
    x = torch.randn(n_samples, 1, image_size, image_size).to(device)
    
    for t in reversed(range(timesteps)):
        t_tensor = torch.ones(n_samples, device=device).long() * t
        
        predicted_noise = model(x, t_tensor, type_t="timestep")
        
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]
        
        if t > 0: 
            noise = torch.randn_like(x)
        else:
            noise = 0
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise