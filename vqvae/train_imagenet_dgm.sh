#!/bin/bash
# VQVAE + DGM on ImageNet (Single GPU)
# GPU 1 - With DGM auxiliary loss (weight 1.0)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CUDA_VISIBLE_DEVICES=1 python main.py \
    --dataset IMAGENET_FULL \
    --image_size 256 \
    --batch_size 256 \
    --n_updates 100000 \
    --n_hiddens 128 \
    --n_residual_hiddens 32 \
    --n_residual_layers 2 \
    --embedding_dim 64 \
    --n_embeddings 512 \
    --beta 0.25 \
    --learning_rate 3e-4 \
    --log_interval 100 \
    --use_dgm_loss \
    --dgm_model_path /scratch/sbeeredd/sandbox/Deep-Geometric-Moment/checkpoints/res34_model_best.pth.tar \
    --dgm_loss_weight 1.0 \
    --dgm_loss_type l2_norm \
    --dgm_model_type imagenet \
    --dgm_arch resnet34 \
    --dgm_image_size 256 \
    --use_wandb \
    --wandb_project vqvae-imagenet \
    --wandb_run_name vqvae_dgm_w1.0_imagenet_256 \
    -save \
    --filename vqvae_dgm_imagenet
