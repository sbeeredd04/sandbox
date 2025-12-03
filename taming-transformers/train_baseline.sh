#!/bin/bash
# Baseline experiment: LPIPS only (no DGM) on GPU 2

#clean up the runs and zombie processes

CUDA_VISIBLE_DEVICES=2 python main.py \
  --base configs/vqgan_baseline_imagenet_val.yaml \
  -t \
  --name vqgan_baseline_iamgenet_val_32_dgm_visual \
  --seed 42 \
  --no-test