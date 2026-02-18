"""
Linear Probing on ImageNet using a pretrained DGM ResNet34 backbone.

What is linear probing?
-----------------------
Linear probing is a simple way to evaluate how good the features learned by a
neural network are.  The idea is:
  1. Load a pretrained model (our DGM ResNet34).
  2. FREEZE all the backbone weights — we do NOT update them.
  3. Throw away the old classification head and add a fresh linear layer.
  4. Train ONLY that new linear layer on ImageNet.
  5. The accuracy tells us how "linearly separable" the learned features are.

If it's high, the backbone learned really useful features!

Usage:
------
    python linear_probe.py --config config.yaml

Requirements:
-------------
    torch, torchvision, wandb, pyyaml
    (All available in the 'myenv' conda environment)
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
import time
import random
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import wandb

# We need to import the DGM model definition from the project.
# Add the Deep-Geometric-Moment directory to Python's path so we can import it.
DGM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "Deep-Geometric-Moment")
sys.path.insert(0, DGM_DIR)
from resnet_gm import ResNet34  # The DGM ResNet34 model factory


# =============================================================================
# Helper: Load config from YAML file
# =============================================================================
def load_config(config_path):
    """Read the YAML config file and return a dictionary."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# Helper: Build ImageNet data loaders
# =============================================================================
def build_dataloaders(cfg):
    """
    Create PyTorch DataLoaders for ImageNet train and val sets.

    The DGM model uses an 8×8 stride-8 conv as its first layer, so it
    expects 256×256 input images to produce 32×32 internal feature maps.

    Train transforms: RandomResizedCrop(256) + RandomHorizontalFlip + Normalize
    Val transforms:   Resize(292) + CenterCrop(256) + Normalize
    """
    image_size = cfg["data"]["image_size"]  # 256

    # Standard ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # --- Training transforms ---
    # RandomResizedCrop: randomly crop and resize to 256×256
    # RandomHorizontalFlip: flip horizontally 50% of the time
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # --- Validation transforms ---
    # Resize to 292, then center-crop to 256 (standard practice)
    val_transform = transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Use torchvision's ImageFolder — it reads from folders named by class
    print(f"Loading training data from: {cfg['data']['train_dir']}")
    train_dataset = datasets.ImageFolder(
        root=cfg["data"]["train_dir"],
        transform=train_transform,
    )

    print(f"Loading validation data from: {cfg['data']['val_dir']}")
    val_dataset = datasets.ImageFolder(
        root=cfg["data"]["val_dir"],
        transform=val_transform,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Num classes:   {len(train_dataset.classes)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# Helper: Load pretrained DGM backbone (frozen) + new linear head
# =============================================================================
def build_model(cfg, device):
    """
    1. Create a DGM ResNet34 model.
    2. Load the pretrained checkpoint weights.
    3. Freeze all backbone parameters.
    4. Replace the classification head with a fresh linear layer.

    Returns:
        model:       The full model (backbone frozen, head trainable)
        linear_head: Reference to the new linear layer (for the optimizer)
    """
    num_classes = cfg["model"]["num_classes"]

    # --- Step 1: Create the model architecture ---
    print("Creating DGM ResNet34 model...")
    model = ResNet34(device=device, num_classes=num_classes)

    # --- Step 2: Load pretrained weights ---
    ckpt_path = cfg["model"]["checkpoint"]
    print(f"Loading pretrained weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # The checkpoint was saved with DataParallel, so keys have 'module.' prefix.
    # We need to remove that prefix to load into a non-DataParallel model.
    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove "module." prefix if present
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    # Load the weights (strict=True ensures all keys match)
    model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint accuracy: {checkpoint.get('best_acc', '?')}%")

    # --- Step 3: Freeze ALL parameters ---
    # This means no gradients will be computed for the backbone during training.
    for param in model.parameters():
        param.requires_grad = False
    print("Froze all backbone parameters.")

    # --- Step 4: Replace the linear head with a fresh one ---
    feature_dim = cfg["model"]["feature_dim"]  # 256
    model.linear = nn.Linear(feature_dim, num_classes)
    # The new linear layer's parameters are trainable by default (requires_grad=True)
    print(f"Replaced classification head: Linear({feature_dim} -> {num_classes})")

    # Move model to GPU
    model = model.to(device)

    # Return the model and a reference to the trainable linear head
    return model, model.linear


# =============================================================================
# Helper: Compute accuracy (top-1 and top-5)
# =============================================================================
def compute_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the indices of the top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape: (maxk, batch_size)

        # Check which predictions match the target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            # For each k, count how many of the top-k predictions are correct
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            results.append(acc.item())

        return results


# =============================================================================
# Training: One epoch
# =============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg):
    model.train()  # Set to train mode (affects dropout, batchnorm in backbone)

    # Tracking variables
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    num_samples = 0
    log_interval = cfg["wandb"]["log_interval"]

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data to GPU
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass through the frozen backbone + trainable linear head
        # The model's forward returns: (logits, (grid, moments), imgr)
        # when return_moments=True (default). We only need the logits.
        with torch.no_grad():
            # Run backbone with no gradient tracking (saves memory)
            output, (grid, moments), imgr = model(images, return_moments=True)

        # Re-run ONLY the linear head with gradients enabled
        # moments is the 256-dim feature vector we want to classify
        logits = model.linear(moments)

        # Compute cross-entropy loss
        loss = criterion(logits, targets)

        # Backward pass — only updates the linear layer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = targets.size(0)
        top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
        running_loss += loss.item() * batch_size
        running_top1 += top1 * batch_size
        running_top5 += top5 * batch_size
        num_samples += batch_size

        # Print and log progress every N batches
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / num_samples
            avg_top1 = running_top1 / num_samples
            avg_top5 = running_top5 / num_samples
            elapsed = time.time() - start_time
            imgs_per_sec = num_samples / elapsed

            print(
                f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f}  Top-1: {avg_top1:.2f}%  Top-5: {avg_top5:.2f}%  "
                f"Speed: {imgs_per_sec:.0f} img/s"
            )

            # Log to wandb
            wandb.log({
                "train/loss": avg_loss,
                "train/top1_acc": avg_top1,
                "train/top5_acc": avg_top5,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/imgs_per_sec": imgs_per_sec,
                "epoch": epoch + 1,
                "global_step": epoch * len(train_loader) + batch_idx,
            })

    # End-of-epoch averages
    epoch_loss = running_loss / num_samples
    epoch_top1 = running_top1 / num_samples
    epoch_top5 = running_top5 / num_samples
    return epoch_loss, epoch_top1, epoch_top5


# =============================================================================
# Validation: One epoch
# =============================================================================
@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """
    Evaluate the model on the validation set.
    Everything runs with torch.no_grad() since we're just evaluating.
    """
    model.eval()  # Set to eval mode

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    num_samples = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        output, (grid, moments), imgr = model(images, return_moments=True)
        logits = model.linear(moments)

        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
        running_loss += loss.item() * batch_size
        running_top1 += top1 * batch_size
        running_top5 += top5 * batch_size
        num_samples += batch_size

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Val Batch [{batch_idx+1}/{len(val_loader)}] "
                f"Top-1: {running_top1/num_samples:.2f}%  "
                f"Top-5: {running_top5/num_samples:.2f}%"
            )

    elapsed = time.time() - start_time
    val_loss = running_loss / num_samples
    val_top1 = running_top1 / num_samples
    val_top5 = running_top5 / num_samples

    print(
        f"  === Validation Results ===  "
        f"Loss: {val_loss:.4f}  Top-1: {val_top1:.2f}%  Top-5: {val_top5:.2f}%  "
        f"Time: {elapsed:.1f}s"
    )

    # Log to wandb
    wandb.log({
        "val/loss": val_loss,
        "val/top1_acc": val_top1,
        "val/top5_acc": val_top5,
        "epoch": epoch + 1,
    })

    return val_loss, val_top1, val_top5


# =============================================================================
# Main training loop
# =============================================================================
def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="DGM Linear Probing on ImageNet")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    print("=" * 60)
    print("DGM ResNet34 — Linear Probing on ImageNet")
    print("=" * 60)
    print(f"Config loaded from: {args.config}")

    # --- Set random seed for reproducibility ---
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True  # Faster convolutions when input sizes don't change
    print(f"Random seed: {seed}")

    # --- Set GPU ---
    gpu_id = cfg["gpu_id"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")

    # --- Initialize wandb ---
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,  # Log all our hyperparameters
    )
    print(f"wandb project: {cfg['wandb']['project']}")

    # --- Build data loaders ---
    print("\n--- Data ---")
    train_loader, val_loader = build_dataloaders(cfg)

    # --- Build model ---
    print("\n--- Model ---")
    model, linear_head = build_model(cfg, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (linear head only)")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # --- Optimizer: only optimize the linear head ---
    optimizer = optim.SGD(
        linear_head.parameters(),  # ONLY the linear layer
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    print(f"\n  Optimizer: SGD(lr={cfg['training']['lr']}, "
          f"momentum={cfg['training']['momentum']}, "
          f"wd={cfg['training']['weight_decay']})")

    # --- Learning rate scheduler: MultiStepLR ---
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["training"]["lr_milestones"],
        gamma=cfg["training"]["lr_gamma"],
    )
    print(f"  LR schedule: milestones={cfg['training']['lr_milestones']}, "
          f"gamma={cfg['training']['lr_gamma']}")

    # --- Loss function ---
    criterion = nn.CrossEntropyLoss().to(device)

    # --- Training loop ---
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_top1 = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{cfg['training']['epochs']} "
              f"(LR: {optimizer.param_groups[0]['lr']:.6f}) ---")

        # Train
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )

        # Validate
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device, epoch
        )

        # Step the learning rate scheduler
        scheduler.step()

        # Log epoch summary to wandb
        wandb.log({
            "epoch_summary/train_loss": train_loss,
            "epoch_summary/train_top1": train_top1,
            "epoch_summary/train_top5": train_top5,
            "epoch_summary/val_loss": val_loss,
            "epoch_summary/val_top1": val_top1,
            "epoch_summary/val_top5": val_top5,
            "epoch_summary/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
        })

        # Save checkpoint if this is the best validation accuracy so far
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            save_path = os.path.join(save_dir, "linear_probe_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "linear_head_state_dict": linear_head.state_dict(),
                "best_top1": best_top1,
                "optimizer": optimizer.state_dict(),
            }, save_path)
            print(f"  ** New best! Top-1: {best_top1:.2f}% — saved to {save_path}")

        # Also save latest checkpoint every epoch
        save_path = os.path.join(save_dir, "linear_probe_latest.pth")
        torch.save({
            "epoch": epoch + 1,
            "linear_head_state_dict": linear_head.state_dict(),
            "best_top1": best_top1,
            "optimizer": optimizer.state_dict(),
        }, save_path)

    # --- Done! ---
    print("\n" + "=" * 60)
    print(f"Training complete!  Best validation Top-1 accuracy: {best_top1:.2f}%")
    print("=" * 60)

    wandb.log({"best_val_top1": best_top1})
    wandb.finish()


if __name__ == "__main__":
    main()
