import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import sys
import torchvision

#add Deep-Geometric-Moment to path for utils
sys.path.append('../Deep-Geometric-Moment/')

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, metrics will not be logged to W&B")

parser = argparse.ArgumentParser()

# Hyperparameters
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')
parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root (for IMAGENET_VAL)")
parser.add_argument("--image_size", type=int, default=None, help="Image size for IMAGENET_VAL (default: 32 for DGM, or 256 for full res)")

# DGM loss parameters
parser.add_argument("--use_dgm_loss", action="store_true", help="Use DGM auxiliary loss")
parser.add_argument("--dgm_model_path", type=str, default="/scratch/sbeeredd/sandbox/Deep-Geometric-Moment/checkpoints/res34_model_best.pth.tar", help="Path to pretrained DGM model")
parser.add_argument("--dgm_loss_weight", type=float, default=0.01, help="Weight for DGM loss")
parser.add_argument("--dgm_loss_type", type=str, default='mse', choices=['mse', 'l1'], help="Type of DGM loss")
parser.add_argument("--dgm_model_type", type=str, default='imagenet', choices=['cifar', 'imagenet'], help="DGM model type (cifar or imagenet)")
parser.add_argument("--dgm_arch", type=str, default='resnet34', choices=['resnet18', 'resnet34'], help="DGM architecture (for imagenet)")
parser.add_argument("--dgm_image_size", type=int, default=256, help="DGM model image size (32 for CIFAR, 256 for ImageNet, auto if None)")

# Wandb logging
parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
parser.add_argument("--wandb_project", type=str, default="vqvae-dgm", help="W&B project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb if requested
if args.use_wandb and WANDB_AVAILABLE:
    wandb_run_name = args.wandb_run_name or f"{args.dataset}_dgm{args.dgm_loss_weight}" if args.use_dgm_loss else args.dataset
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config=vars(args)
    )
    print(f"W&B logging enabled: {args.wandb_project}/{wandb_run_name}")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

# Load data and define batch data loaders
training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(args.dataset, args.batch_size, args.data_root, args.image_size)


model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

# Load DGM model if using DGM loss
dgm_model = None
dgm_model_type = 'cifar'
dgm_size = 32

if args.use_dgm_loss:
    if not args.dgm_model_path:
        raise ValueError("--dgm_model_path must be provided when using --use_dgm_loss")
    
    # Determine model type and configuration based on dataset
    if args.dataset in ['IMAGENET', 'IMAGENET_VAL', 'IMAGENET_FULL']:
        dgm_model_type = args.dgm_model_type if args.dgm_model_type == 'imagenet' else 'imagenet'
        num_classes = 1000
        dgm_size = args.dgm_image_size if args.dgm_image_size else 256
        dgm_arch = args.dgm_arch
    elif args.dataset == 'CIFAR100':
        dgm_model_type = 'cifar'
        num_classes = 100
        dgm_size = args.dgm_image_size if args.dgm_image_size else 32
        dgm_arch = 'resnet18'
    else:  # CIFAR10 and others
        dgm_model_type = 'cifar'
        num_classes = 10
        dgm_size = args.dgm_image_size if args.dgm_image_size else 32
        dgm_arch = 'resnet18'
    
    dgm_model = utils.load_dgm_model(
        args.dgm_model_path, 
        num_classes=num_classes, 
        hw=dgm_size,
        model_type=dgm_model_type,
        arch=dgm_arch
    )
    print(f"DGM Configuration: model_type={dgm_model_type}, arch={dgm_arch}, "
          f"num_classes={num_classes}, image_size={dgm_size}x{dgm_size}")

# Set up optimizer and training loop
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'dgm_losses': [],
    'embedding_losses': [],
}

import time
start_time = time.time()


def train():
    
    print(f"=="*10 + " Training Started " + "=="*10)
    print(f"Using DGM Loss: {args.use_dgm_loss}, DGM Loss Weight: {args.dgm_loss_weight}, DGM Loss Type: {args.dgm_loss_type}")
    
    # Calculate steps per epoch for logging reconstructions
    steps_per_epoch = len(training_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Logging metrics every step, reconstructions every epoch")
    
    global_step = 0  # Initialize global step counter

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()
        
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss
        
        # Add DGM loss if enabled
        dgm_loss = torch.tensor(0.0).to(device)
        if args.use_dgm_loss and dgm_model is not None:
            dgm_loss = utils.compute_dgm_loss(x, x_hat, dgm_model, loss_type=args.dgm_loss_type, 
                                             dgm_size=dgm_size, model_type=dgm_model_type)
            loss = loss + args.dgm_loss_weight * dgm_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["dgm_losses"].append(dgm_loss.cpu().detach().numpy())
        results["embedding_losses"].append(embedding_loss.cpu().detach().numpy())
        results["n_updates"] = i
        
        # Calculate current epoch
        current_epoch = i / steps_per_epoch
        
        # Log to wandb EVERY step with simplified metrics
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "train/recon_error": recon_loss.cpu().detach().item(),
                "train/total_loss": loss.cpu().detach().item(),
                "train/embedding_loss": embedding_loss.cpu().detach().item(),
                "train/perplexity": perplexity.cpu().detach().item(),
                "train/epoch": current_epoch,
            }
            if args.use_dgm_loss:
                log_dict["train/dgm_loss"] = dgm_loss.cpu().detach().item()
            
            wandb.log(log_dict, step=global_step)
            global_step += 1

        # Print to console every log_interval steps
        if i % args.log_interval == 0:
            # save model and print values
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            dgm_loss_str = ''
            if args.use_dgm_loss:
                dgm_loss_str = f'DGM Loss: {np.mean(results["dgm_losses"][-args.log_interval:]):.6f}'
            
            recon_error_mean = np.mean(results["recon_errors"][-args.log_interval:])
            loss_mean = np.mean(results["loss_vals"][-args.log_interval:])
            perplexity_mean = np.mean(results["perplexities"][-args.log_interval:])
            
            print('Update #', i, 'Recon Error:',
                  recon_error_mean,
                  'Loss', loss_mean,
                  'Perplexity:', perplexity_mean,
                  dgm_loss_str)
        
        # Log reconstructions and validation metrics every epoch
        if i > 0 and i % steps_per_epoch == 0:
            current_epoch_int = int(i / steps_per_epoch)
            if args.use_wandb and WANDB_AVAILABLE:
                with torch.no_grad():
                    model.eval()
                    
                    # Compute validation metrics
                    val_recon_errors = []
                    val_losses = []
                    val_perplexities = []
                    val_embedding_losses = []
                    val_dgm_losses = []
                    
                    for val_batch_idx, (x_val, _) in enumerate(validation_loader):
                        if val_batch_idx >= 10:  # Limit to 10 batches for speed
                            break
                        x_val = x_val.to(device)
                        val_embedding_loss, x_val_recon, val_perplexity = model(x_val)
                        val_recon_loss = torch.mean((x_val_recon - x_val)**2) / x_train_var
                        val_loss = val_recon_loss + val_embedding_loss
                        
                        if args.use_dgm_loss and dgm_model is not None:
                            val_dgm_loss = utils.compute_dgm_loss(x_val, x_val_recon, dgm_model, 
                                                                 loss_type=args.dgm_loss_type,
                                                                 dgm_size=dgm_size, 
                                                                 model_type=dgm_model_type)
                            val_loss = val_loss + args.dgm_loss_weight * val_dgm_loss
                            val_dgm_losses.append(val_dgm_loss.cpu().detach().numpy())
                        
                        val_recon_errors.append(val_recon_loss.cpu().detach().numpy())
                        val_losses.append(val_loss.cpu().detach().numpy())
                        val_perplexities.append(val_perplexity.cpu().detach().numpy())
                        val_embedding_losses.append(val_embedding_loss.cpu().detach().numpy())
                    
                    # Log validation metrics
                    val_log_dict = {
                        "val/recon_error": np.mean(val_recon_errors),
                        "val/total_loss": np.mean(val_losses),
                        "val/embedding_loss": np.mean(val_embedding_losses),
                        "val/perplexity": np.mean(val_perplexities),
                        "val/epoch": current_epoch_int,
                    }
                    if args.use_dgm_loss and val_dgm_losses:
                        val_log_dict["val/dgm_loss"] = np.mean(val_dgm_losses)
                    
                    # Get a batch for reconstruction visualization
                    (x_vis, _) = next(iter(validation_loader))
                    x_vis = x_vis.to(device)
                    _, x_vis_recon, _ = model(x_vis)
                    
                    # Create separate grids for inputs and reconstructions (like VQGAN)
                    n_images = min(8, x_vis.shape[0])
                    input_grid = torchvision.utils.make_grid(x_vis[:n_images], nrow=4, normalize=True, value_range=(-1, 1))
                    recon_grid = torchvision.utils.make_grid(x_vis_recon[:n_images], nrow=4, normalize=True, value_range=(-1, 1))
                    
                    val_log_dict["val/inputs"] = wandb.Image(input_grid, caption=f"Epoch {current_epoch_int} - Inputs")
                    val_log_dict["val/reconstructions"] = wandb.Image(recon_grid, caption=f"Epoch {current_epoch_int} - Reconstructions")
                    
                    wandb.log(val_log_dict, step=global_step)
                    global_step += 1
                    
                    print(f"\n[Epoch {current_epoch_int}] Validation - Recon Error: {val_log_dict['val/recon_error']:.6f}, "
                          f"Loss: {val_log_dict['val/total_loss']:.6f}, Perplexity: {val_log_dict['val/perplexity']:.2f}")
                    model.train()


if __name__ == "__main__":
    train()
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("W&B run finished")
