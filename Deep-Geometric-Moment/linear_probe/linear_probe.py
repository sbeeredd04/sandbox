"""
Linear Probing on ImageNet using a pretrained DGM ResNet34 backbone.
Caches features once, then trains linear head on cached tensors (very fast).
Logs imgr reconstructions to wandb as 2×5 grids (originals + reconstructions).
"""

import os, sys, time, random, argparse, yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from resnet_gm import ResNet34

# ImageNet class names (index → human-readable) — load directly from known path
import importlib.util as _ilu
_classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ImageFolder", "imagenet_classes.py")
_spec = _ilu.spec_from_file_location("imagenet_classes", _classes_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
IDX2CLASS = _mod.imagenet_idx2classname


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_image_loader(cfg, split):
    """Deterministic image loader for one-time feature extraction."""
    image_size = cfg["data"]["image_size"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    root = cfg["data"]["train_dir"] if split == "train" else cfg["data"]["val_dir"]
    ds = datasets.ImageFolder(root, transform=tf)
    print(f"{split.capitalize()} samples: {len(ds)}, Classes: {len(ds.classes)}")
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                      shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)


@torch.no_grad()
def extract_features(model, loader, device, probe_layer, desc="Extracting",
                     save_first_batch=False):
    """Run frozen DGM backbone once. Returns (moments, labels) on CPU.
    If save_first_batch=True, also returns (images_cpu, imgr_cpu) for viz."""
    all_moments, all_labels = [], []
    first_batch = None
    with torch.amp.autocast("cuda"):
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=desc)):
            images = images.to(device, non_blocking=True)
            _, moments, imgr = model(images, return_moments=False, layer=probe_layer)
            all_moments.append(moments.cpu())
            all_labels.append(targets)
            if save_first_batch and batch_idx == 0:
                first_batch = (images.cpu(), imgr.cpu())
    return torch.cat(all_moments).float(), torch.cat(all_labels), first_batch


def build_model(cfg, device):
    model = ResNet34(device=device, num_classes=cfg["model"]["num_classes"])
    ckpt = torch.load(cfg["model"]["checkpoint"], map_location="cpu")
    cleaned = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(cleaned, strict=True)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, acc {ckpt.get('best_acc','?')}%)")

    for p in model.parameters():
        p.requires_grad = False

    model.linear = nn.Linear(cfg["model"]["feature_dim"], cfg["model"]["num_classes"])
    model = model.to(device)
    model.eval()  # Keep backbone in eval mode (correct BN stats)
    return model


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            c = correct[:k].reshape(-1).float().sum(0)
            res.append(c.mul_(100.0 / bs).item())
        return res


def get_class_name(idx):
    """Get human-readable ImageNet class name for a given index."""
    if IDX2CLASS is not None:
        name = IDX2CLASS.get(idx, str(idx))
        # Shorten long class names (take first part before comma)
        return name.split(",")[0] if len(name) > 25 else name
    return str(idx)


# ImageNet normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log_reconstruction_grid(images, imgr, logits, targets):
    """
    Build a matplotlib 2-row × 5-col grid:
      Row 1: original images
      Row 2: imgr reconstructions
    Each column has GT and Pred class name as title.
    Returns a wandb.Image ready to log.
    """
    n = min(5, images.size(0))
    device = images.device

    mean = MEAN.to(device)
    std = STD.to(device)
    orig = (images[:n] * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    recon = imgr[:n].repeat(1, 3, 1, 1).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    _, preds = logits[:n].topk(1, dim=1)
    pred_names = [get_class_name(p.item()) for p in preds.squeeze(-1)]
    gt_names = [get_class_name(t.item()) for t in targets[:n]]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    for i in range(n):
        axes[0, i].imshow(orig[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"GT: {gt_names[i]}", fontsize=8, color="green", wrap=True)

        axes[1, i].imshow(recon[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Pred: {pred_names[i]}", fontsize=8, color="red", wrap=True)

    fig.suptitle("Top: Original  |  Bottom: imgr Reconstruction", fontsize=10)
    plt.tight_layout()

    wandb_img = wandb.Image(fig)
    plt.close(fig)
    return wandb_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = cfg["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
    probe_layer = cfg["model"]["probe_layer"]

    print("=" * 60)
    print(f"DGM ResNet34 — Linear Probe (layer {probe_layer}) on ImageNet (train→val)")
    print("=" * 60)

    run_name = cfg["wandb"]["run_name"].format(probe_layer=probe_layer)
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)

    # ── extract & cache features (one-time backbone pass) ───────────
    model = build_model(cfg, device)
    linear_head = model.linear

    print("\nCaching features (runs backbone once over each split)...")
    train_feats, train_labels, _ = extract_features(
        model, build_image_loader(cfg, "train"), device, probe_layer, "Cache train")
    val_feats, val_labels, first_val_batch = extract_features(
        model, build_image_loader(cfg, "val"), device, probe_layer, "Cache val",
        save_first_batch=True)
    print(f"  Train: {train_feats.shape}  Val: {val_feats.shape}")

    # keep first val batch images+imgr for reconstruction grid, free the rest
    viz_images, viz_imgr = first_val_batch
    del model          # free backbone VRAM (linear_head stays)
    torch.cuda.empty_cache()
    print("Backbone freed from GPU — training on cached features\n")

    # tensor dataloaders (pure tensor copies, very fast)
    train_loader = DataLoader(TensorDataset(train_feats, train_labels),
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels),
                            batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    optimizer = optim.SGD(linear_head.parameters(), lr=cfg["training"]["lr"],
                          momentum=cfg["training"]["momentum"],
                          weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=cfg["training"]["lr_milestones"],
                                                gamma=cfg["training"]["lr_gamma"])
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        # ── train on cached features ────────────────────────────────
        linear_head.train()
        running_loss = running_t1 = running_t5 = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1:02d}/{cfg['training']['epochs']}",
                     bar_format="{l_bar}{bar:30}{r_bar}")

        for batch_idx, (feats, targets) in enumerate(pbar):
            feats   = feats.to(device,   non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = linear_head(feats)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            t1, t5 = accuracy(logits, targets)
            running_loss += loss.item() * bs
            running_t1 += t1 * bs
            running_t5 += t5 * bs
            n += bs

            global_step = epoch * len(train_loader) + batch_idx

            pbar.set_postfix(loss=f"{loss.item():.3f}", t1=f"{t1:.1f}",
                             avg=f"{running_t1/n:.1f}")

            if batch_idx % cfg["wandb"]["log_interval"] == 0:
                wandb.log({"batch/loss": loss.item(), "batch/top1": t1, "batch/top5": t5,
                            "batch/avg_top1": running_t1/n, "epoch": epoch+1},
                           step=global_step)

        pbar.close()
        scheduler.step()

        train_loss = running_loss / n
        train_t1 = running_t1 / n
        train_t5 = running_t5 / n
        print(f"  Epoch {epoch+1:02d} Train | Loss: {train_loss:.4f} | Top-1: {train_t1:.2f}% | Top-5: {train_t5:.2f}%")

        # ── evaluate on cached val features ─────────────────────────
        linear_head.eval()
        val_t1_sum = val_t5_sum = val_n = 0

        with torch.no_grad():
            for feats, targets in tqdm(val_loader, desc=f"Val   E{epoch+1:02d}"):
                feats   = feats.to(device,   non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = linear_head(feats)
                t1, t5 = accuracy(logits, targets)
                bs = targets.size(0)
                val_t1_sum += t1 * bs
                val_t5_sum += t5 * bs
                val_n += bs

        val_t1 = val_t1_sum / val_n
        val_t5 = val_t5_sum / val_n
        print(f"  Epoch {epoch+1:02d} Val   | Top-1: {val_t1:.2f}% | Top-5: {val_t5:.2f}%")

        # reconstruction grid using cached first val batch
        with torch.no_grad():
            viz_feats = val_feats[:viz_images.size(0)].to(device)
            viz_logits = linear_head(viz_feats)
        grid_img = log_reconstruction_grid(viz_images, viz_imgr, viz_logits.cpu(),
                                           val_labels[:viz_images.size(0)])

        wandb.log({"epoch/train_loss": train_loss,
                    "epoch/train_top1": train_t1, "epoch/train_top5": train_t5,
                    "epoch/val_top1": val_t1, "epoch/val_top5": val_t5,
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                    "reconstruction_grid": grid_img},
                   step=(epoch + 1) * len(train_loader) - 1)

        if val_t1 > best_top1:
            best_top1 = val_t1
            torch.save({"epoch": epoch+1, "state_dict": linear_head.state_dict(),
                         "best_top1": best_top1}, os.path.join(save_dir, "linear_probe_best.pth"))
            print(f"  ** New best val Top-1: {best_top1:.2f}%")

    print(f"\nDone! Best Val Top-1: {best_top1:.2f}%")
    wandb.log({"best_top1": best_top1})
    wandb.finish()


if __name__ == "__main__":
    main()
