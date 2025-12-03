#!/bin/bash
# DGM low weight experiment: LPIPS + DGM (weight=0.5) on GPU 3

#clean up the runs and zombie processes

CUDA_VISIBLE_DEVICES=3 python main.py \
  --base configs/vqgan_dgm_low_weight_imagenet_val.yaml \
  -t \
  --name vqgan_dgm_low_weight_mse_0.01_32_dgm_visual \
  --seed 42 \
  --no-test 
