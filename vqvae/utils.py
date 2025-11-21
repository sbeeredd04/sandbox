import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
import sys
sys.path.append('../Deep-Geometric-Moment/')
from model import ResNet18

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
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))

    val = datasets.ImageNet(root="data/imagenet", split='val',
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           ]))
    return train, val


def load_block():
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


def load_data_and_data_loaders(dataset, batch_size):
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
            'Invalid dataset: CIFAR10, CIFAR100, IMAGENET, BLOCK, and LATENT_BLOCK are supported.')

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
    dgm_model.load_state_dict(torch.load(model_path, map_location=device))
    dgm_model.eval()
    for param in dgm_model.parameters():
        param.requires_grad = False
    return dgm_model

def compute_dgm_loss(x, x_hat, dgm_model, loss_type='mse'):
    # Extract moments from original image (no gradients needed)
    with torch.no_grad():
        _, moments_x = dgm_model(x, return_moments=True)
    
    # Extract moments from reconstructed image (gradients flow through this)
    _, moments_x_hat = dgm_model(x_hat, return_moments=True)
    
    dgm_loss = 0
    if loss_type == 'mse':
        for mx, mx_hat in zip(moments_x, moments_x_hat):
            dgm_loss += torch.mean((mx - mx_hat) ** 2)
    elif loss_type == 'l1':
        for mx, mx_hat in zip(moments_x, moments_x_hat):
            dgm_loss += torch.mean(torch.abs(mx - mx_hat))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse' or 'l1'.")
    
    return dgm_loss / len(moments_x)