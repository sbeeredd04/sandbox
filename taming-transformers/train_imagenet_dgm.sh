#!/bin/bash
# VQGAN + DGM on ImageNet (Single GPU)
# GPU 3 - DGM replaces LPIPS (weight 1.0)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CUDA_VISIBLE_DEVICES=3 python main.py \
    --base configs/vqgan_dgm_imagenet_full.yaml \
    -t \
    --gpus 0, \
    --name vqgan_dgm_imagenet \
    --seed 42
