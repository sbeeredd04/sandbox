"""
Linear Probing on ImageNet with a frozen pretrained encoder.
Caches features once, then trains linear head on cached tensors (very fast).

Usage:
    python linear_probe.py --config configs/dinov2_vitb14.yaml
"""

import os, sys, random, argparse, yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_encoder(cfg, device):

    name = cfg["model"]["name"]

    if name == "dinov2":
        from models.dino import DINOv2
        encoder = DINOv2(variant=cfg["model"]["variant"]).to(device)
        return encoder, encoder.embed_dim

    if name == "mae":
        from models.mae import MAE
        encoder = MAE(variant=cfg["model"]["variant"]).to(device)
        return encoder, encoder.embed_dim

    if name == "siglip":
        from models.siglip import SigLIP
        encoder = SigLIP(variant=cfg["model"]["variant"]).to(device)
        return encoder, encoder.embed_dim

    if name == "siglip2":
        from models.siglip2 import SigLIP2
        encoder = SigLIP2(variant=cfg["model"]["variant"]).to(device)
        return encoder, encoder.embed_dim

    raise ValueError(f"Unknown model: {name}")
                     

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bs).item()
                for k in topk]


def build_image_loader(cfg, split):

    size = cfg["data"]["image_size"]
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    root = cfg["data"]["train_dir"] if split == "train" else cfg["data"]["val_dir"]
    ds = datasets.ImageFolder(root, transform=tf)
    
    print(f"{split.capitalize()} samples: {len(ds)},  Classes: {len(ds.classes)}")
    
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)


@torch.no_grad()
def extract_features(encoder, loader, device, desc="Extracting"):

    all_feats, all_labels = [], []
    with torch.amp.autocast("cuda"):
        for images, targets in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            feats = encoder(images)
            all_feats.append(feats.cpu())
            all_labels.append(targets)
    return torch.cat(all_feats).float(), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # seed
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    cudnn.benchmark = True

    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")

    model_tag = f"{cfg['model']['name']}_{cfg['model'].get('variant', '')}"
    print("=" * 60)
    print(f"Linear Probe — {model_tag} — ImageNet (train→val)")
    print("=" * 60)

    wandb.init(project=cfg["wandb"]["project"],name=cfg["wandb"]["run_name"], config=cfg)

    # ── extract & cache features (one-time encoder pass) ────────────
    encoder, embed_dim = load_encoder(cfg, device)
    print(f"Encoder: {model_tag}  |  embed_dim={embed_dim}")

    print("\nCaching features (runs encoder once over each split)...")
    train_feats, train_labels = extract_features(encoder, build_image_loader(cfg, "train"), device, "Cache train")
    val_feats, val_labels = extract_features(encoder, build_image_loader(cfg, "val"), device, "Cache val")
    print(f"  Train: {train_feats.shape}  Val: {val_feats.shape}")

    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from GPU — training on cached features\n")

    # tensor dataloaders (pure tensor copies, very fast)
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # linear head
    head = nn.Linear(embed_dim, cfg["model"]["num_classes"]).to(device)
    nn.init.trunc_normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    optimizer = optim.SGD(head.parameters(),lr=cfg["training"]["lr"],momentum=cfg["training"]["momentum"],weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg["training"]["lr_milestones"],gamma=cfg["training"]["lr_gamma"])
    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", model_tag)
    os.makedirs(save_dir, exist_ok=True)
    best_top1 = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        
        # ── train on cached features ────────────────────────────────
        head.train()
        loss_sum = t1_sum = t5_sum = n = 0

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1:02d}/{cfg['training']['epochs']}")
        for batch_idx, (feats, targets) in enumerate(pbar):
            feats   = feats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = head(feats)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            t1, t5 = accuracy(logits, targets)
            loss_sum += loss.item() * bs
            t1_sum   += t1 * bs
            t5_sum   += t5 * bs
            n        += bs

            pbar.set_postfix(loss=f"{loss.item():.3f}", t1=f"{t1:.1f}", avg=f"{t1_sum/n:.1f}")

            step = epoch * len(train_loader) + batch_idx
            
            if batch_idx % cfg["wandb"]["log_interval"] == 0:
                wandb.log({"batch/loss": loss.item(),"batch/top1": t1, "batch/top5": t5,"batch/avg_top1": t1_sum / n,"epoch": epoch + 1}, step=step)

        pbar.close()
        scheduler.step()

        train_loss = loss_sum / n
        train_t1   = t1_sum / n
        train_t5   = t5_sum / n
        print(f"  Epoch {epoch+1:02d} Train | Loss {train_loss:.4f} | "f"Top-1 {train_t1:.2f}% | Top-5 {train_t5:.2f}%")

        # ── evaluate on cached val features ─────────────────────────
        head.eval()
        val_t1_sum = val_t5_sum = val_n = 0

        with torch.no_grad():
            for feats, targets in tqdm(val_loader, desc=f"Val   E{epoch+1:02d}"):
                feats   = feats.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = head(feats)
                t1, t5 = accuracy(logits, targets)
                bs = targets.size(0)
                val_t1_sum += t1 * bs
                val_t5_sum += t5 * bs
                val_n      += bs

        val_t1 = val_t1_sum / val_n
        val_t5 = val_t5_sum / val_n
        print(f"  Epoch {epoch+1:02d} Val   | "f"Top-1 {val_t1:.2f}% | Top-5 {val_t5:.2f}%")

        wandb.log({"epoch/train_loss": train_loss, "epoch/train_top1": train_t1, "epoch/train_top5": train_t5, "epoch/val_top1": val_t1, "epoch/val_top5": val_t5, "epoch/lr": optimizer.param_groups[0]["lr"]}, step=(epoch + 1) * len(train_loader) - 1)

        if val_t1 > best_top1:
            best_top1 = val_t1
            torch.save({"epoch": epoch + 1, "state_dict": head.state_dict(), "best_top1": best_top1},os.path.join(save_dir, "best.pth"))
            print(f"  ** New best val Top-1: {best_top1:.2f}%")

    print(f"\nDone! Best Val Top-1: {best_top1:.2f}%")
    wandb.log({"best_top1": best_top1})
    wandb.finish()


if __name__ == "__main__":
    main()
