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
parser.add_argument("--dgm_model_path", type=str, default="", help="Path to pretrained DGM model")
parser.add_argument("--dgm_loss_weight", type=float, default=0.1, help="Weight for DGM loss")
parser.add_argument("--dgm_loss_type", type=str, default='mse', choices=['mse', 'l1'], help="Type of DGM loss")

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
if args.use_dgm_loss:
    if not args.dgm_model_path:
        raise ValueError("--dgm_model_path must be provided when using --use_dgm_loss")
    # DGM model expects 32x32 images (CIFAR resolution)
    # For ImageNet, we resize to 32x32 and use CIFAR10 model (10 classes)
    if args.dataset in ['IMAGENET', 'IMAGENET_VAL']:
        num_classes = 10  # Use CIFAR10-trained model for ImageNet (resized to 32x32)
        hw = 32
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        hw = 32
    else:  # CIFAR10 and others
        num_classes = 10
        hw = 32
    dgm_model = utils.load_dgm_model(args.dgm_model_path, num_classes=num_classes, hw=hw)
    print(f"Loaded DGM model from {args.dgm_model_path} with num_classes={num_classes}, image size {hw}x{hw}")

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
step_times = []


def train():
    
    print(f"=="*10 + " Training Started " + "=="*10)
    print(f"Using DGM Loss: {args.use_dgm_loss}, DGM Loss Weight: {args.dgm_loss_weight}, DGM Loss Type: {args.dgm_loss_type}")
    
    # Calculate steps per epoch for logging reconstructions
    steps_per_epoch = len(training_loader)
    print(f"Steps per epoch: {steps_per_epoch}")

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        step_start_time = time.time()
        
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss
        
        # Add DGM loss if enabled
        dgm_loss = torch.tensor(0.0).to(device)
        if args.use_dgm_loss and dgm_model is not None:
            dgm_loss = utils.compute_dgm_loss(x, x_hat, dgm_model, loss_type=args.dgm_loss_type)
            loss = loss + args.dgm_loss_weight * dgm_loss

        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["dgm_losses"].append(dgm_loss.cpu().detach().numpy())
        results["embedding_losses"].append(embedding_loss.cpu().detach().numpy())
        results["n_updates"] = i

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
            embedding_loss_mean = np.mean(results["embedding_losses"][-args.log_interval:])
            dgm_loss_mean = np.mean(results["dgm_losses"][-args.log_interval:]) if args.use_dgm_loss else 0
            step_time_mean = np.mean(step_times[-args.log_interval:]) if step_times else 0
            current_epoch = i / steps_per_epoch
            elapsed_time = time.time() - start_time
            
            print('Update #', i, 'Recon Error:',
                  recon_error_mean,
                  'Loss', loss_mean,
                  'Perplexity:', perplexity_mean,
                  dgm_loss_str)
            
            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "train/recon_error": recon_error_mean,
                    "train/total_loss": loss_mean,
                    "train/embedding_loss": embedding_loss_mean,
                    "train/perplexity": perplexity_mean,
                    "train/learning_rate": args.learning_rate,
                    "train/epoch": current_epoch,
                    "timing/step_time": step_time_mean,
                    "timing/elapsed_time": elapsed_time,
                    "timing/steps_per_second": 1.0 / step_time_mean if step_time_mean > 0 else 0,
                }
                if args.use_dgm_loss:
                    log_dict["train/dgm_loss"] = dgm_loss_mean
                wandb.log(log_dict, step=i)
        
        # Log reconstructions and validation metrics every epoch
        if i % steps_per_epoch == 0 and i > 0:
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
                            val_dgm_loss = utils.compute_dgm_loss(x_val, x_val_recon, dgm_model, loss_type=args.dgm_loss_type)
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
                    }
                    if args.use_dgm_loss and val_dgm_losses:
                        val_log_dict["val/dgm_loss"] = np.mean(val_dgm_losses)
                    
                    # Get a batch for reconstruction visualization
                    (x_vis, _) = next(iter(validation_loader))
                    x_vis = x_vis.to(device)
                    _, x_vis_recon, _ = model(x_vis)
                    
                    # Create grid of original and reconstructed images
                    n_images = min(8, x_vis.shape[0])
                    comparison = torch.cat([x_vis[:n_images], x_vis_recon[:n_images]])
                    grid = torchvision.utils.make_grid(comparison, nrow=n_images, normalize=True, value_range=(-1, 1))
                    
                    val_log_dict["reconstructions"] = wandb.Image(grid, caption=f"Top: Original, Bottom: Reconstructed (Step {i})")
                    
                    wandb.log(val_log_dict, step=i)
                    model.train()


if __name__ == "__main__":
    train()
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("W&B run finished")
