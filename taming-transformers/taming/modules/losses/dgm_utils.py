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
    print("Warning: CIFAR DGM model (model.py) not available")

try:
    from resnet_gm import ResNet18 as ResNet18_ImageNet, ResNet34 as ResNet34_ImageNet
    IMAGENET_DGM_AVAILABLE = True
except ImportError:
    IMAGENET_DGM_AVAILABLE = False
    print("Warning: ImageNet DGM model (resnet_gm.py) not available")


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
    
    print(f"Loaded {model_type} DGM model ({arch}) with num_classes={num_classes}, hw={hw}")
    return dgm_model


def compute_dgm_loss(x, x_hat, dgm_model, loss_type='mse', input_size=None, dgm_size=256, model_type='imagenet'):
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
    
    # Different models have different outputs
    if model_type == 'cifar':
        # CIFAR model: expects return_moments=True
        with torch.no_grad():
            _, (xb_x, moment_x) = dgm_model(x_resized, return_moments=True)
        
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
        dgm_loss = xb_loss + 1.0 * moment_loss
        
    elif model_type == 'imagenet':
        # ImageNet model: returns (cl, imgr) where imgr is the geometric moment map
        with torch.no_grad():
            _, imgr_x = dgm_model(x_resized)
        
        _, imgr_x_hat = dgm_model(x_hat_resized)
        
        # Compute loss on geometric moment maps
        if loss_type == 'mse':
            dgm_loss = torch.mean((imgr_x - imgr_x_hat) ** 2)
        elif loss_type == 'l1':
            dgm_loss = torch.mean(torch.abs(imgr_x - imgr_x_hat))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse' or 'l1'.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cifar' or 'imagenet'")
    
    return dgm_loss
