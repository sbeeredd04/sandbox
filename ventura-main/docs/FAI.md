# Generating dataset for FAI data

1. Download bags
```bash
python scripts/fai/process_bags.py
```

Replace this with the following to process just the test set

```bash
python scripts/fai/process_bags.py --cfg ./scripts/fai/config/urban_spinflow.yaml --csv ./data/fai_spinflow_raw/lhy_meta
```

2. Apply non-track based velocity filters. Set the following params in `./scripts/preprocessing/config/fai_mining.yaml`.
```yaml
pipeline:
  apply_prefilters: True
  visualize_samples: True
```
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_raw ./data/fai_processed full_raw.txt --gpus 0,1,2,3 --jobs 2 --cfg_file scripts/preprocessing/config/fai_mining.yaml
```

Run the following to apply these changes on the fai spinflow dataset
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_raw ./data/fai_spinflow_processed full_raw.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

3. Run tracking. Assumes 4 GPU configuration with 14 jobs per GPU
```bash
./scripts/preprocessing/track_points_parallel.sh ./data/fai_raw ./data/fai_processed ./scripts/preprocessing/config/fai_mining.yaml --gpus 0,1,2,3,4,5,6,7 --jobs 4
```

Runs tracking for spinflow set
```bash
./scripts/preprocessing/track_points_parallel.sh ./data/fai_spinflow_raw ./data/fai_spinflow_processed ./scripts/preprocessing/config/fai_spinflow_mining.yaml --gpus 0,1,2,3 --jobs 12
```

3a. Compute ride info and odometry files. Set the following params in `scripts/preprocessing/config/fai_mining.yaml`. Note this command just computes it for all path_tracker.h5 files.

```yaml
pipeline:
  compute_ride_infos: True
```

```bash 
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_mining.yaml
```

Run below to run on test set
```bash 
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_processed ./data/fai_spinflow_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 1 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

4. Postfilter tracked points. Set the following params in `./scripts/preprocessing/config/fai_mining.yaml`.

```yaml
pipeline:
  apply_postfilters: True
```

```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_mining_postfiltering.yaml
```

5. Compute entity masks. Set the following params in `scripts/preprocessing/config/fai_mining.yaml`.

```yaml
pipeline:
  compute_entity_masks: True
```
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 1 --cfg_file scripts/preprocessing/config/fai_mining_entitymasks.yaml
```

6. Generate training splits for training.

Unconditional pretraining
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_depth --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.05
```

Mask based training
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_depth --split_file full_entitymasks.txt
```

Run the following to compute the odometry statistics for all of the current tracked files
```bash
python scripts/preprocessing/compute_odom_stats.py --cfg_file scripts/preprocessing/config/fai_mining.yaml
```

## Image Planning Training

Sample mini split (20k) for entity targets with curvature balancing + odometry filtering
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_entitylang_20k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve --max_samples 20000
```

Sample full split (100k) for entity targets with curvature balancing + odometry filtering
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_entitylang_100k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve
```

## Running Language Label Postprocessing Pipeline

0. Download outstanding bags
```bash
python scripts/fai/process_bags.py --cfg ./scripts/fai/config/urban_spinflow.yaml --csv ./data/fai_spinflow_raw/lhy_meta
```

Configure the pipeline args as follows. This will
1. Convert the annotated json to full_raw.txt files
```yaml
process_language_labels: True
visualize_samples: True
```

```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_raw ./data/fai_spinflow_processed full_raw.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

2. Run path tracker generation
```bash
./scripts/preprocessing/track_points_parallel.sh ./data/fai_spinflow_raw ./data/fai_spinflow_processed ./scripts/preprocessing/config/fai_spinflow_mining.yaml --gpus 0,1,2,3,4,5,6,7 --jobs 3
```

Manually inspect the masks, update the `data/fai_spinflow_processed/bad_tracks.txt` file with hard bad labels, remove the following mask files

```
python scripts/migrations/remove_bad_tracks.py ./data/fai_spinflow_processed ./data/fai_spinflow_processed/bad_tracks.txt


3. Compute ride infos and run VLM language diversification
```yaml
compute_ride_infos: True
```
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_processed ./data/fai_spinflow_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

4. Apply postfilters on masks
```yaml
apply_postfilters: True
```
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_processed ./data/fai_spinflow_processed full_rideinfos.txt --gpus 0,1,2,3,4,5,6,7 --jobs 2 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

5. Run VLM language diversification
```yaml
compute_entity_masks: True
```
```bash
./scripts/preprocessing/run_engine_parallel.sh ./data/fai_spinflow_processed ./data/fai_spinflow_processed full_trackfiltered.txt --gpus 0,1,2,3,4,5,6,7 --jobs 8 --cfg_file scripts/preprocessing/config/fai_spinflow_mining.yaml
```

6. Sample splits for mask based training

Fai unconditioned dataset
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/mix_language_splits_80k_13k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter none
```

Fai spinflow conditioned dataset
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_spinflow_processed --output_dir ./data/fai_spinflow_processed/language_splits_13k --split_file full_entitymasks.txt --val_ratio 0.05 --test_ratio 0.03 --filter none
```

5. Copy over fai_spinflow files to main dataset. Make sure to manually copy over the train split samples too!!

```bash
python scripts/migrations/merge_datasets.py ./data/fai_spinflow_processed ./data/fai_processed --mode copy
```

6. Sample splits for action prediction training. Here we sample fai and spinflow datasets simultaneously since we're just learning to ground!
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/aligned_action_splits_80k_13k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve --frame_horizon 40 --min_alignment_pct 0.5
```

Fai spinflow conditioned dataset [Optional]
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_spinflow_processed --output_dir ./data/fai_spinflow_processed/action_splits_12k --split_file full_entitymasks.txt --val_ratio 0.05 --test_ratio 0.03 --filter curve
```


<!-- 
## Policy Training

Sample mini split with curvature balancing
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_curve_20k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve --max_samples 20000
```

Sample full split with curvature balancing
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_curve_100k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve
```

Sample full split with curvature balancing + odometry filtering
```bash
python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_curvefiltered_100k --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.02 --filter curve
``` -->