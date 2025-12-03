import torch
import torch.nn as nn
import sys
import os

# Add Deep-Geometric-Moment to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../Deep-Geometric-Moment'))

# Import both CIFAR and ImageNet DGM models
try:
    from model import ResNet18 as ResNet18_CIFAR
    CIFAR_DGM_AVAILABLE = True
except ImportError:
    CIFAR_DGM_AVAILABLE = False

try:
    from resnet_gm import ResNet18 as ResNet18_ImageNet, ResNet34 as ResNet34_ImageNet
    IMAGENET_DGM_AVAILABLE = True
except ImportError:
    IMAGENET_DGM_AVAILABLE = False


def load_dgm_model(model_path, num_classes=1000, hw=256, model_type='imagenet', arch='resnet34', device='cuda'):
    # Select appropriate model architecture
    if model_type == 'cifar':
        if not CIFAR_DGM_AVAILABLE:
            raise ImportError("CIFAR DGM model not available. Check model.py")
        dgm_model = ResNet18_CIFAR(num_classes=num_classes, hw=hw).to(device)
    elif model_type == 'imagenet':
        if not IMAGENET_DGM_AVAILABLE:
            raise ImportError("ImageNet DGM model not available. Check resnet_gm.py")
        if arch == 'resnet18':
            dgm_model = ResNet18_ImageNet(device=device, num_classes=num_classes).to(device)
        elif arch == 'resnet34':
            dgm_model = ResNet34_ImageNet(device=device, num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Unknown architecture: {arch}. Choose 'resnet18' or 'resnet34'")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cifar' or 'imagenet'")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
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


def compute_dgm_loss(x, x_hat, dgm_model, loss_type='mse', input_size=256, dgm_size=256, model_type='imagenet', return_reconstruction=False):
    """
    Compute DGM loss between input and reconstruction images.
    
    Args:
        x: Input images in VQGAN range [-1, 1]
        x_hat: Reconstructed images in VQGAN range [-1, 1]
        dgm_model: Pre-trained DGM ResNet model
        loss_type: Type of loss ('mse', 'l1', 'l2_norm')
        input_size: Size of input images (default 256)
        dgm_size: Size expected by DGM model (256 for ImageNet DGM - gets downsampled to 32 internally)
        model_type: 'cifar' or 'imagenet'
        return_reconstruction: Whether to return DGM reconstruction visualization
        
    Returns:
        dgm_loss: Scalar loss value
        dgm_recon: (optional) DGM reconstruction for visualization
    """
    # Auto-detect input size if not provided
    if input_size is None:
        input_size = x.shape[-1]  # Assumes square images
    
    # DGM ResNet expects images in [0, 1] range (RGB pixel values)
    # Convert from VQGAN range [-1, 1] to [0, 1]
    x_01 = (x + 1.0) / 2.0
    x_hat_01 = (x_hat + 1.0) / 2.0
    
    # DGM ResNet34 expects 256x256 input (conv02 has stride=8, 256->32)
    # Change from 32 to 256!
    dgm_size = 256  # ImageNet DGM expects 256x256, downsamples to 32x32 internally
    
    if input_size != dgm_size:
        x_resized = torch.nn.functional.interpolate(
            x_01, size=(dgm_size, dgm_size), mode='bilinear', align_corners=False
        )
        x_hat_resized = torch.nn.functional.interpolate(
            x_hat_01, size=(dgm_size, dgm_size), mode='bilinear', align_corners=False
        )
    else:
        x_resized = x_01
        x_hat_resized = x_hat_01
    
    # Clamp to ensure [0, 1] range after interpolation
    x_resized = torch.clamp(x_resized, 0.0, 1.0)
    x_hat_resized = torch.clamp(x_hat_resized, 0.0, 1.0)
    
    # Different models have different outputs
    dgm_recon = None
        
    if model_type == 'imagenet':
        # ImageNet model: now supports return_moments=True -> (imgr, (xb, m))
        with torch.no_grad():
            _, (xb_x, m_x), imgr_input = dgm_model(x_resized, return_moments=True)

        _, (xb_x_hat, m_x_hat), imgr_hat = dgm_model(x_hat_resized, return_moments=True)
        if return_reconstruction:
            dgm_recon = imgr_input.repeat(1, 3, 1, 1)  # Convert to 3-channel for visualization
            dgm_recon = dgm_recon * 2.0 - 1.0  # Back to VQGAN range [-1, 1]
        
        # Compute loss on weighted features xb (spatial structure)
        if loss_type == 'mse':
            dgm_loss = torch.mean((xb_x - xb_x_hat) ** 2)
        elif loss_type == 'l1':
            dgm_loss = torch.mean(torch.abs(xb_x - xb_x_hat))
        elif loss_type == 'l2_norm':
            dgm_loss = torch.mean(torch.sqrt(torch.sum((xb_x - xb_x_hat) ** 2, dim=1)))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse', 'l1', or 'l2_norm'.")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cifar' or 'imagenet'")
    
    if return_reconstruction:
        return dgm_loss, dgm_recon
    return dgm_loss