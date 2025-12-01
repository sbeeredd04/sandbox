#!/bin/bash
# VQVAE Baseline on ImageNet (Single GPU)
# GPU 0 - No DGM loss, standard reconstruction training

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset IMAGENET_FULL \
    --image_size 256 \
    --batch_size 512 \
    --n_updates 100000 \
    --n_hiddens 128 \
    --n_residual_hiddens 32 \
    --n_residual_layers 2 \
    --embedding_dim 64 \
    --n_embeddings 512 \
    --beta 0.25 \
    --learning_rate 3e-4 \
    --log_interval 100 \
    --use_wandb \
    --wandb_project vqvae-imagenet \
    --wandb_run_name vqvae_baseline_imagenet_256 \
    -save \
    --filename vqvae_baseline_imagenet
