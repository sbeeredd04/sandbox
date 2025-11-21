#!/bin/bash

# Training script for VQGAN with frozen DGM model on CIFAR10
# Single GPU training

CUDA_VISIBLE_DEVICES=0 python main.py \
    --base configs/vqgan_dgm_cifar10.yaml \
    -t \
    --gpus 0, \
    --name vqgan_dgm_cifar10 \
    --seed 42
