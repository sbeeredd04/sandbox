#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  run_when_gpus_free.sh
#
#  Wait until *all* GPUs are idle, then execute a fixed list of commands.
# ---------------------------------------------------------------------------

set -euo pipefail        # safer bash defaults

: "${START_DELAY_MIN:=0}"

# -------- helper -----------------------------------------------------------
all_gpus_idle() {
  # Returns 0 if no processes are using the GPUs, 1 otherwise
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# -------- wait loop --------------------------------------------------------
echo "[ $(timestamp) ] Checking GPU status …"
while ! all_gpus_idle; do
  echo "[ $(timestamp) ] GPUs busy – waiting 60 s …"
  sleep 60
done
echo "[ $(timestamp) ] GPUs are free."

# -------- optional start delay --------------------------------------------
if (( START_DELAY_MIN > 0 )); then
  echo "[ $(timestamp) ] Delaying start for ${START_DELAY_MIN} min …"
  # minute-by-minute countdown so you can see progress in logs
  for ((m=START_DELAY_MIN; m>0; m--)); do
    echo "[ $(timestamp) ] … ${m} min remaining"
    sleep 60
  done
fi

echo "[ $(timestamp) ] Starting jobs!"

# --------------------------------------------------------------------------- #
# Commands to run, one per line, will be executed sequentially.
# --------------------------------------------------------------------------- #
# cmds=(
#   # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_raw ./data/fai_processed full_raw.txt         --gpus 0,1,2,3 --jobs 2 --cfg_file scripts/preprocessing/config/fai_mining.yaml'
#   # './scripts/preprocessing/track_points_parallel.sh ./data/fai_raw ./data/fai_processed ./scripts/preprocessing/config/fai_mining.yaml --gpus 0,1,2,3 --jobs 14'
# # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_raw.txt         --gpus 0,1,2,3 --jobs 4 --cfg_file scripts/preprocessing/config/fai_mining_postfiltering.yaml'
# # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_trackfiltered.txt --gpus 0,1,2,3 --jobs 1 --cfg_file scripts/preprocessing/config/fai_mining_entitymasks.yaml'
# # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_trackfiltered.txt --gpus 0,1,2,3 --jobs 4 --cfg_file scripts/preprocessing/config/fai_mining.yaml'
# # 'python scripts/preprocessing/gen_split_divergence.py --root_dir ./data/fai_processed --output_dir ./data/fai_processed/divergence_splits_depth --split_file full_trackfiltered.txt --val_ratio 0.03 --test_ratio 0.05'
# # 'python scripts/trainers/train_path.py dataset=planning/frodo8k_turn_goal dataset@dataset_small=planning/fai_turn_goal model=planning/cfg/fai_controlnet_base_spatial trainer=standard_four model.weights_ckpt=model_ckpts/ControlNetPlanning/controlnet_base_spatial/20250701/024010/best-0029-0.7649.ckpt'
# )

# Commands for launching entity mask captioning
# cmds=(
  # 'python scripts/migrations/remove_end_frame.py ./data/fai_processed'
  # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_trackfiltered.txt --gpus 0,1,2,3 --jobs 4 --cfg_file scripts/preprocessing/config/fai_mining.yaml'
  # './scripts/preprocessing/run_engine_parallel.sh ./data/fai_processed ./data/fai_processed full_rideinfos.txt --gpus 0,1,2,3 --jobs 1 --cfg_file scripts/preprocessing/config/fai_mining_entitymasks.yaml'
# )

# Commands for autosync weights and shutdown host machine
# cmds=(
#   'rsync -azP /data/model_ckpts_awslarge2x/ fieldai-aws:/data/model_ckpts_awslarge2x/'
#   'sudo shutdown -h now'
# )

# 

# 1. Copy over FAI spinflow to FAI directory
# 2. Launch model for training
cmds=(
  # 'python scripts/migrations/merge_datasets.py ./data/fai_spinflow_processed ./data/fai_processed --mode copy'
  # 'python scripts/trainers/train_path.py dataset=planning/fai_stlang_goal model=planning/cfg/fai_controlnet_base_stlang trainer=standard_eight dataset.subdirs.0.path=mix_language_splits_80k_12k'
  # 'python scripts/trainers/train_policy.py dataset=policy/fai_lhy model=planning/simplepolicy/spinflow_linear_marigold model.normalize_actions=False dataset.normalize_actions=False trainer=standard_eight dataset.subdirs.0.path=action_splits_80k_12k model.pretrained.vision_encoder.ckpt=model_ckpts/ControlNetPlanning/fai_controlnet_base_stlang/20250814/105548/best-0049-0.7435.ckpt model.use_gt_feats=True'
  'python scripts/migrations/merge_datasets.py ./data/fai_spinflow_processed ./data/fai_processed --mode copy'
  'python scripts/trainers/train_path.py dataset=planning/fai_stlang_goal model=planning/cfg/fai_controlnet_base_stlang trainer=standard_eight dataset.subdirs.0.path=mix_language_splits_80k_13k model.optimizer.kwargs.lr=0.0003 model.pipeline.unet.input_keys.1.dropout_prob=0.05'
)

# -------- run sequentially --------------------------------------------------
trap 'echo; echo "[ $(timestamp) ] Script interrupted – killing current job."; exit 130' INT

for ((i=0; i<${#cmds[@]}; i++)); do
  echo
  echo "[ $(timestamp) ] >>> (${i}/${#cmds[@]}) ${cmds[$i]}"
  eval "${cmds[$i]}"
done

echo
echo "[ $(timestamp) ] ✅  All jobs finished successfully."
