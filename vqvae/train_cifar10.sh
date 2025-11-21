#!/bin/bash
# Train VQVAE on CIFAR-10 with DGM loss

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset CIFAR10 \
    --batch_size 256 \
    --n_updates 50000 \
    --n_hiddens 128 \
    --n_residual_hiddens 32 \
    --n_residual_layers 2 \
    --embedding_dim 64 \
    --n_embeddings 512 \
    --beta 0.25 \
    --learning_rate 3e-4 \
    --log_interval 100 \
    --use_dgm_loss \
    --dgm_model_path ../Deep-Geometric-Moment/cifar10/chkpt/model_best.pth.tar \
    --dgm_loss_weight 0.01 \
    --dgm_loss_type mse \
    --use_wandb \
    --wandb_project vqvae-dgm \
    --wandb_run_name cifar10_dgm_w0.01 \
    -save \
    --filename cifar10_dgm
