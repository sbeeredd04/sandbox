CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/vqgan_dgm_imagenet_val.yaml \
  -t \
  --name vqgan_dgm_imagenet_val_mse_0.1_32_dgm_visual \
  --seed 42 \
  --no-test 
