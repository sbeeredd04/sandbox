#!/bin/bash

# Training script for VQGAN baseline (no DGM) on CIFAR10

CUDA_VISIBLE_DEVICES=1 python main.py \
    --base configs/vqgan_dgm_cifar10.yaml \
    -t \
    --gpus 0, \
    --name vqgan_cifar10_baseline \
    --seed 42 \
    --dgm_weight 0.0
