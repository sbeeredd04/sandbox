import torch
import torch.nn as nn
import sys
import os

# Add Deep-Geometric-Moment to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../Deep-Geometric-Moment'))

try:
    from model import ResNet18
except ImportError:
    print("Warning: Could not import DGM model. DGM loss will not be available.")
    ResNet18 = None


def load_dgm_model(model_path, num_classes=1000, hw=32, device='cuda'):
    if ResNet18 is None:
        raise ImportError("DGM model not available. Check Deep-Geometric-Moment path.")
    
    dgm_model = ResNet18(num_classes=num_classes, hw=hw).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    dgm_model.load_state_dict(state_dict, strict=False)
    dgm_model.eval()
    
    # Freeze all parameters
    for param in dgm_model.parameters():
        param.requires_grad = False
    
    return dgm_model


def compute_dgm_loss(x, x_hat, dgm_model, loss_type='mse'):
    # Extract moments from original image (no gradients needed)
    with torch.no_grad():
        _, moments_x = dgm_model(x, return_moments=True)
    
    # Extract moments from reconstructed image (gradients flow through this)
    _, moments_x_hat = dgm_model(x_hat, return_moments=True)
    
    # Compute loss between moment sets
    dgm_loss = torch.tensor(0.0, device=x.device)
    
    if loss_type == 'mse':
        for mx, mx_hat in zip(moments_x, moments_x_hat):
            dgm_loss += torch.mean((mx - mx_hat) ** 2)
    elif loss_type == 'l1':
        for mx, mx_hat in zip(moments_x, moments_x_hat):
            dgm_loss += torch.mean(torch.abs(mx - mx_hat))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse' or 'l1'.")
    
    # Average across all levels
    return dgm_loss / len(moments_x)
