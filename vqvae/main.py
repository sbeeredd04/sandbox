import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import sys

#add Deep-Geometric-Moment to path for utils
sys.path.append('../Deep-Geometric-Moment/')

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

# DGM loss parameters
parser.add_argument("--use_dgm_loss", action="store_true", help="Use DGM auxiliary loss")
parser.add_argument("--dgm_model_path", type=str, default="", help="Path to pretrained DGM model")
parser.add_argument("--dgm_loss_weight", type=float, default=0.1, help="Weight for DGM loss")
parser.add_argument("--dgm_loss_type", type=str, default='mse', choices=['mse', 'l1'], help="Type of DGM loss")

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

# Load data and define batch data loaders
training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(args.dataset, args.batch_size)


model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

# Load DGM model if using DGM loss
dgm_model = None
if args.use_dgm_loss:
    if not args.dgm_model_path:
        raise ValueError("--dgm_model_path must be provided when using --use_dgm_loss")
    num_classes = 100 if args.dataset == 'CIFAR100' else 1000 if args.dataset == 'IMAGENET' else 10
    hw = 256 if args.dataset == 'IMAGENET' else 32
    dgm_model = utils.load_dgm_model(args.dgm_model_path, num_classes=num_classes, hw=hw)
    print(f"Loaded DGM model from {args.dgm_model_path} with image size {hw}x{hw}")

# Set up optimizer and training loop
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'dgm_losses': [],
}


def train():

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
            dgm_loss = utils.compute_dgm_loss(x, x_hat, dgm_model, loss_type=args.dgm_loss_type)
            loss = loss + args.dgm_loss_weight * dgm_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["dgm_losses"].append(dgm_loss.cpu().detach().numpy())
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
            
            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]),
                  dgm_loss_str)


if __name__ == "__main__":
    train()
