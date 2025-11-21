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


def compute_dgm_loss(x, x_hat, dgm_model, loss_type='mse', input_size=None, dgm_size=32):
    """
    Compute DGM loss by resizing inputs to match pretrained DGM model resolution.
    
    Args:
        x: Original images at native resolution (B, 3, H, W)
        x_hat: Reconstructed images at native resolution (B, 3, H, W)
        dgm_model: Pretrained DGM model expecting (B, 3, dgm_size, dgm_size)
        loss_type: 'mse' or 'l1'
        input_size: Native image size (if None, auto-detected from x)
        dgm_size: DGM model expected size (default: 32)
    """
    # Auto-detect input size if not provided
    if input_size is None:
        input_size = x.shape[-1]  # Assumes square images
    
    # Resize both to DGM resolution if needed
    if input_size != dgm_size:
        x_resized = torch.nn.functional.interpolate(
            x, size=(dgm_size, dgm_size), mode='bilinear', align_corners=False
        )
        x_hat_resized = torch.nn.functional.interpolate(
            x_hat, size=(dgm_size, dgm_size), mode='bilinear', align_corners=False
        )
    else:
        x_resized = x
        x_hat_resized = x_hat
    
    # Extract both xb (weighted features) and moment from original image (no gradients needed)
    with torch.no_grad():
        _, (xb_x, moment_x) = dgm_model(x_resized, return_moments=True)
    
    # Extract both xb and moment from reconstructed image (gradients flow through this)
    _, (xb_x_hat, moment_x_hat) = dgm_model(x_hat_resized, return_moments=True)
    
    # Compute loss on weighted features (xb) - spatial loss
    if loss_type == 'mse':
        xb_loss = torch.mean((xb_x - xb_x_hat) ** 2)
        moment_loss = torch.mean((moment_x - moment_x_hat) ** 2)
    elif loss_type == 'l1':
        xb_loss = torch.mean(torch.abs(xb_x - xb_x_hat))
        moment_loss = torch.mean(torch.abs(moment_x - moment_x_hat))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse' or 'l1'.")
    
    # Combine both losses: xb captures spatial structure, moment captures global statistics
    # Scale moment loss up since it's much smaller
    dgm_loss = xb_loss + 10.0 * moment_loss
    
    return dgm_loss
