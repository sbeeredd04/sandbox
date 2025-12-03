

#clean up the runs and zombie processes

CUDA_VISIBLE_DEVICES=1 python main.py \
  --base configs/vqgan_dgm_imagenet_val_replace.yaml \
  -t \
  --name vqgan_dgm_imagenet_val_split_replace_dgm_visual \
  --seed 42 \
  --no-test