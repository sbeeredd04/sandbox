# Training Spinflow


## Generate training splits

Generates the large scale 50k dataset of spatial language (left, right, straight) conditioned trajectory mask generator
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/frodobots8k_processed --output_dir ./data/frodobots8k_processed/divergence_splits_entity_50k --use_cache --val_ratio 0.02 --test_ratio 0.05 --max_samples 50000 --split_file full_trackfiltered.txt
```

Generates a subset of the 50k dataset, but for post-training the model to follow segmentation masks and language-centric object targets.

```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/frodobots8k_processed --output_dir ./data/frodobots8k_processed/divergence_splits_entity_50k --use_cache --cache_dir ./data/frodobots8k_processed/divergence_splits_depth_50k --max_samples 50000
```

## Spatial Language Pre-training

Trains a frodobots SD2 pretrained model for generating spatial language conditioned trajectories on four GPUs.
```bash
python scripts/trainers/train_path.py dataset=planning/frodo8k_turn_goal model=planning/cfg/controlnet_base_spatial trainer=standard_four
```

Trains a frodobots + fai cotrained SD2 model on spatial language conditioned trajectories on four GPUs
```bash
python scripts/trainers/train_path.py dataset=planning/frodo8k_turn_goal dataset@dataset_small=planning/fai_turn_goal model=planning/cfg/fai_controlnet_base_spatial trainer=standard_four model.weights_ckpt=model_ckpts/ControlNetPlanning/controlnet_base_spatial/20250701/024010/best-0029-0.7649.ckpt
```

## Semantic Language + Mask Conditioned Steering

