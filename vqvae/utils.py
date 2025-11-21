import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import sys
sys.path.append('../Deep-Geometric-Moment/')
from model import ResNet18

# Optional imports for BLOCK dataset
try:
    from datasets.block import BlockDataset, LatentBlockDataset
    BLOCK_AVAILABLE = True
except ImportError:
    BLOCK_AVAILABLE = False

def load_cifar(cifar100=False):
    dataset_class = datasets.CIFAR100 if cifar100 else datasets.CIFAR10
    train = dataset_class(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = dataset_class(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val

def load_imagenet():
    train = datasets.ImageNet(root="data/imagenet", split='train',
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.ImageNet(root="data/imagenet", split='val',
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(256),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val

def load_imagenet_val(data_root='/scratch/sbeeredd/sandbox/imagenet_val', image_size=256):
    """Load ImageNet validation dataset from custom path
    
    Args:
        data_root: Path to imagenet validation data
        image_size: Size to resize images to (default: 256 for full res, use 32 for CIFAR-size)
    """
    val = datasets.ImageFolder(root=os.path.join(data_root, 'val'),
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    # Use val for both train and val (we're just training on val set)
    return val, val


def load_block():
    if not BLOCK_AVAILABLE:
        raise ImportError("BlockDataset not available. Install datasets.block module.")
    
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    if not BLOCK_AVAILABLE:
        raise ImportError("LatentBlockDataset not available. Install datasets.block module.")
    
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, data_root=None, image_size=None):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar(cifar100=False)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'CIFAR100':
        training_data, validation_data = load_cifar(cifar100=True)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'IMAGENET':
        training_data, validation_data = load_imagenet()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = 1.0  # ImageNet pre-normalized

    elif dataset == 'IMAGENET_VAL':
        if data_root is None:
            data_root = '/scratch/sbeeredd/sandbox/imagenet_val'
        if image_size is None:
            image_size = 32  # Default to 32 for DGM compatibility
        training_data, validation_data = load_imagenet_val(data_root, image_size)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = 1.0  # Pre-normalized

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: CIFAR10, CIFAR100, IMAGENET, IMAGENET_VAL, BLOCK, and LATENT_BLOCK are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')


#load the Deep-Geometric-Moment frozen model for computing auxiliary losses
def load_dgm_model(model_path, num_classes=100, hw=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dgm_model = ResNet18(num_classes=num_classes, hw=hw).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    dgm_model.load_state_dict(state_dict, strict=False)
    dgm_model.eval()
    
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