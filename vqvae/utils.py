import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import sys
sys.path.append('../Deep-Geometric-Moment/')

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

def load_imagenet_val(data_root='/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC', image_size=256):
    """Load ImageNet validation set organized by class folders.
    
    Args:
        data_root: Path to CLS-LOC directory containing organized val/ folder
        image_size: Target image size (default 256)
    
    Returns:
        train_dataset, val_dataset (both point to val set for val-only training)
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

def load_imagenet_full(train_root='/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC/train',
                       val_root='/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC/val',
                       image_size=256):
    """Load full ImageNet training and validation datasets.
    
    Args:
        train_root: Path to training data directory with class folders (1000 classes)
        val_root: Path to validation data directory with class folders (organized)
        image_size: Target image size (default 256)
    
    Returns:
        train_dataset, val_dataset
    """
    train = datasets.ImageFolder(root=train_root,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    
    val = datasets.ImageFolder(root=val_root,
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    return train, val


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
            data_root = '/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC'
        if image_size is None:
            image_size = 256  # Default to 256 for ImageNet
        training_data, validation_data = load_imagenet_val(data_root, image_size)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = 1.0  # Pre-normalized

    elif dataset == 'IMAGENET_FULL':
        # Full ImageNet with separate train and val directories
        if image_size is None:
            image_size = 256
        train_root = '/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC/train'
        val_root = '/scratch/sbeeredd/imagenet/ILSVRC/Data/CLS-LOC/val'
        training_data, validation_data = load_imagenet_full(train_root, val_root, image_size)
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
            'Invalid dataset: CIFAR10, CIFAR100, IMAGENET, IMAGENET_VAL, IMAGENET_FULL, BLOCK, and LATENT_BLOCK are supported.')

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
def load_dgm_model(model_path, num_classes=1000, hw=256, model_type='imagenet', arch='resnet18'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        elif loss_type == 'l2_norm':
            xb_loss = torch.mean(torch.sqrt(torch.sum((xb_x - xb_x_hat) ** 2, dim=1)))
            moment_loss = torch.mean(torch.sqrt(torch.sum((moment_x - moment_x_hat) ** 2, dim=1)))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse', 'l1', or 'l2_norm'.")
        
        # Combine both losses: xb captures spatial structure, moment captures global statistics
        dgm_loss = xb_loss + 100.0 * moment_loss
        
    elif model_type == 'imagenet':
        # ImageNet model: returns (cl, imgr) where imgr is the geometric moment map
        with torch.no_grad():
            _, imgr_x = dgm_model(x_resized)
        
        _, imgr_x_hat = dgm_model(x_hat_resized)
        
        # Compute loss on geometric moment maps
        if loss_type == 'mse':
            xb_loss = torch.mean((imgr_x - imgr_x_hat) ** 2)
        elif loss_type == 'l1':
            xb_loss = torch.mean(torch.abs(imgr_x - imgr_x_hat))
        elif loss_type == 'l2_norm':
            xb_loss = torch.mean(torch.sqrt(torch.sum((imgr_x - imgr_x_hat) ** 2, dim=1)))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse', 'l1', or 'l2_norm'.")
        
        dgm_loss = xb_loss
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cifar' or 'imagenet'")
    
    return dgm_loss