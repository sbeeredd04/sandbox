#!/bin/bash
# ===========================================================================
# run_dgm.sh — Launch RAE training with DGM (Deep Geometric Moments) encoder
# ===========================================================================
#
# This script trains a Representation Autoencoder (RAE) using DGM ResNet34
# as the frozen encoder instead of the default DINOv2. The training follows
# the RAE paper (Zheng et al., 2025) but replaces the ViT-based encoder
# with a CNN + geometric moments approach.
#
# === Architecture Overview ===
#
#   [Image 256×256] → [DGM ResNet34 (frozen)] → [256-dim, 16×16 tokens]
#                                                        ↓
#                                              [ViT-B Decoder (trained)]
#                                                        ↓
#                                              [Reconstructed Image 256×256]
#
# === Training Stages ===
#
# Stage 1 (this script): Train the decoder to reconstruct images from DGM
#   latent representations. Uses L1 + LPIPS + GAN losses.
#   - Encoder: DGM ResNet34 (frozen, 256-dim features, 256 tokens)
#   - Decoder: ViT-B (512-dim hidden, 12 layers, trainable)
#   - Discriminator: DINOv1 ViT-S/8 based (with spectral normalization)
#
# Stage 2 (separate script): Train a diffusion transformer (DiT) to generate
#   DGM latent codes conditioned on class labels.
#
# === Usage ===
#
#   # Stage 1: Train decoder
#   bash run_dgm.sh stage1
#
#   # Stage 2: Train diffusion model (after stage 1 completes)
#   bash run_dgm.sh stage2
#
#   # Default (no argument): runs stage 1
#   bash run_dgm.sh
#
# === Requirements ===
#
#   - DGM pretrained checkpoint at:
#     ../Deep-Geometric-Moment/checkpoints/res34_model_best.pth.tar
#   - ImageNet training data at: data/imagenet/train/
#   - DINOv1 discriminator at: models/discs/dino_vit_small_patch8_224.pth
#   - At least 2 GPUs recommended (adjust NPROC below for your setup)
#
# ===========================================================================

set -e  # Exit on any error

# ===========================================================================
# Configuration — Adjust these for your environment
# ===========================================================================

# Number of GPUs to use (auto-detect available GPUs)
NPROC=${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NPROC" -eq 0 ]; then
    echo "ERROR: No GPUs detected. RAE training requires at least 1 GPU."
    exit 1
fi
echo "=== Using $NPROC GPU(s) for training ==="

# ImageNet data path
DATA_PATH="${DATA_PATH:-data/imagenet/train}"

# Image resolution
IMAGE_SIZE=256

# Precision (bf16 recommended for A100/L40S, fp16 for older GPUs)
PRECISION="${PRECISION:-bf16}"

# Results directory (checkpoints, logs, samples)
RESULTS_DIR="${RESULTS_DIR:-ckpts}"

# Stage selection (default: stage1)
STAGE="${1:-stage1}"

# Working directory: ensure we run from RAE/src/ so relative config paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# ===========================================================================
# NCCL Configuration (for multi-GPU stability)
# ===========================================================================
# These settings help prevent NCCL timeout errors on multi-GPU setups,
# especially on systems with slower interconnects.
export NCCL_TIMEOUT=1800           # 30 minute timeout (default is 300s)
export NCCL_IB_TIMEOUT=23         # InfiniBand timeout
export NCCL_DEBUG=WARN            # Show warnings only (set to INFO for debug)
export OMP_NUM_THREADS=8          # Limit CPU threads per process

# ===========================================================================
# Verify prerequisites
# ===========================================================================

# Check DGM checkpoint exists
DGM_CKPT="../Deep-Geometric-Moment/checkpoints/res34_model_best.pth.tar"
# Resolve relative to src/ directory
DGM_CKPT_RESOLVED="$(cd "$SCRIPT_DIR" && realpath "$DGM_CKPT" 2>/dev/null || echo "$SCRIPT_DIR/$DGM_CKPT")"
if [ ! -f "$DGM_CKPT_RESOLVED" ]; then
    echo "ERROR: DGM checkpoint not found at: $DGM_CKPT_RESOLVED"
    echo ""
    echo "Please ensure the pretrained DGM ResNet34 weights are available."
    echo "Expected location: $DGM_CKPT_RESOLVED"
    echo ""
    echo "You can create a symlink if the checkpoint is elsewhere:"
    echo "  ln -s /path/to/res34_model_best.pth.tar $DGM_CKPT_RESOLVED"
    exit 1
fi
echo "=== DGM checkpoint found: $DGM_CKPT_RESOLVED ==="

# Check ImageNet data
if [ ! -d "$DATA_PATH" ]; then
    echo "WARNING: ImageNet data not found at: $DATA_PATH"
    echo "Set DATA_PATH environment variable to point to your ImageNet train directory."
    echo "Example: DATA_PATH=/path/to/imagenet/train bash run_dgm.sh"
fi

# ===========================================================================
# Stage 1: Train RAE Decoder with DGM Encoder
# ===========================================================================
#
# This stage trains only the decoder while keeping the DGM encoder frozen.
# The decoder learns to reconstruct images from DGM's 256-dim latent tokens.
#
# Loss breakdown:
#   Total = L1_recon + perceptual_weight * LPIPS + disc_weight * adaptive_w * GAN
#
# Training schedule:
#   Epochs 0-5:  L1 only (decoder learns basic reconstruction)
#   Epochs 6-7:  L1 + LPIPS + discriminator updates begin
#   Epochs 8-15: L1 + LPIPS + GAN (full adversarial training)
#
# ===========================================================================

if [ "$STAGE" = "stage1" ]; then
    echo ""
    echo "==========================================================="
    echo "  RAE Stage 1: Training Decoder with DGM Encoder"
    echo "==========================================================="
    echo "  Encoder:    DGM ResNet34 (frozen, 256-dim, 256 tokens)"
    echo "  Decoder:    ViT-B (512-dim hidden, 12 layers, trainable)"
    echo "  Input:      ${IMAGE_SIZE}×${IMAGE_SIZE} ImageNet images"
    echo "  GPUs:       $NPROC"
    echo "  Precision:  $PRECISION"
    echo "  Config:     configs/stage1/training/DGM_decB.yaml"
    echo "==========================================================="
    echo ""

    torchrun \
        --nproc_per_node=$NPROC \
        --master_port=29500 \
        train_stage1.py \
        --config ../configs/stage1/training/DGM_decB.yaml \
        --data-path "$DATA_PATH" \
        --results-dir "$RESULTS_DIR" \
        --image-size $IMAGE_SIZE \
        --precision $PRECISION \
        --compile \
        --wandb \
        2>&1 | tee -a "$RESULTS_DIR/dgm_stage1_train.log"


# ===========================================================================
# Stage 2: Train Diffusion Transformer on DGM Latents
# ===========================================================================
#
# This stage trains a DiT (Diffusion Transformer) to generate DGM latent
# representations conditioned on ImageNet class labels.
#
# The DiT learns to denoise from random Gaussian noise to valid DGM latents.
# At inference, generated latents are decoded by the Stage-1 decoder to images.
#
# Prerequisites:
#   - Completed Stage 1 training (decoder checkpoint)
#   - Computed latent normalization statistics (stat.pt)
#   - Update configs/stage2/training/ImageNet256/DiTDH-XL_DGM.yaml with:
#     - pretrained_decoder_path: path to trained decoder
#     - normalization_stat_path: path to computed stats
#
# ===========================================================================

elif [ "$STAGE" = "stage2" ]; then
    echo ""
    echo "==========================================================="
    echo "  RAE Stage 2: Training Diffusion on DGM Latents"
    echo "==========================================================="
    echo "  Latent shape: [256, 16, 16] (C, H, W)"
    echo "  DiT:          XL with DDT head"
    echo "  GPUs:         $NPROC"
    echo "  Precision:    $PRECISION"
    echo "  Config:       configs/stage2/training/ImageNet256/DiTDH-XL_DGM.yaml"
    echo "==========================================================="
    echo ""

    torchrun \
        --nproc_per_node=$NPROC \
        --master_port=29501 \
        train.py \
        --config ../configs/stage2/training/ImageNet256/DiTDH-XL_DGM.yaml \
        --data-path "$DATA_PATH" \
        --results-dir "$RESULTS_DIR" \
        --image-size $IMAGE_SIZE \
        --precision $PRECISION \
        --compile \
        --wandb \
        2>&1 | tee -a "$RESULTS_DIR/dgm_stage2_train.log"

else
    echo "ERROR: Unknown stage '$STAGE'. Use 'stage1' or 'stage2'."
    echo "Usage: bash run_dgm.sh [stage1|stage2]"
    exit 1
fi

echo ""
echo "=== Training complete! ==="
echo "Checkpoints saved to: $RESULTS_DIR/"
