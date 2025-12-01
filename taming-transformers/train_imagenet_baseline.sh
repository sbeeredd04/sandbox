#!/bin/bash
# VQGAN Baseline on ImageNet (Single GPU)
# GPU 2 - Standard LPIPS perceptual loss

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CUDA_VISIBLE_DEVICES=2 python main.py \
    --base configs/vqgan_imagenet_full_baseline.yaml \
    -t \
    --gpus 0, \
    --name vqgan_baseline_imagenet \
    --seed 42
