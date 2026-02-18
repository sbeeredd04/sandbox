#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_jobs.sh ROOT_DIR OUT_DIR CFG_FILE [--gpus 0,1] [--jobs N]
[[ $# -lt 3 ]] && { sed -n '2,14p' "$0"; exit 1; }
ROOT_DIR=$1; OUT_DIR=$2; CFG_FILE=$3; shift 3

GPUS="0,1,2,3"; JOBS_PER_GPU=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) GPUS=$2;  shift 2;;
    --jobs) JOBS_PER_GPU=$2; shift 2;;
    *) echo "unknown flag $1"; exit 1;;
  esac
done
IFS=',' read -ra GPU_ARR <<<"$GPUS"

# Discover rides
RIDES=()
for dir in "${OUT_DIR}"/output_rides_*; do
  [[ -d $dir ]] && RIDES+=("$(basename "$dir")")
done
(( ${#RIDES[@]} )) || { echo "No output_rides_* dirs in $OUT_DIR"; exit 1; }

# Build GPU slots
SLOTS=(); for ((i=0;i<JOBS_PER_GPU;i++)); do for g in "${GPU_ARR[@]}"; do SLOTS+=("$g"); done; done
TOTAL_SLOTS=${#SLOTS[@]}
declare -A PID2SLOT
cleanup(){ echo -e "\nStopping…"; for p in "${!PID2SLOT[@]}"; do kill "$p" 2>/dev/null || true; done; wait; }
trap cleanup INT TERM

# Build TASKS = each <ride_tag>:<row_idx>
TASKS=()
for ride_tag in "${RIDES[@]}"; do
  split_path="${OUT_DIR}/${ride_tag}/full_raw.txt"
  [[ -f "$split_path" ]] || { echo "Missing $split_path"; continue; }
  # Missing or empty file? skip
  if [[ ! -s "$split_path" ]]; then
    echo "[${ride_tag}] No split file or empty: $split_path — skipping"
    continue
  fi

  # Count data rows only: skip header (NR==1), ignore blank lines (NF)
  nrows=$(awk 'NR>1 && NF {c++} END {print c+0}' "$split_path") || nrows=0

  if (( nrows == 0 )); then
    echo "[${ride_tag}] No data rows in $split_path — skipping"
    continue
  fi

  # Build per-row tasks
  for ((i=0; i<nrows; i++)); do
    TASKS+=("${ride_tag}:${i}")
  done
  echo "Found $nrows rows in $split_path for ride $ride_tag"
done
echo "▶ ${#TASKS[@]} segments across ${#RIDES[@]} rides on GPU(s) {${GPUS}}  (${JOBS_PER_GPU} job/GPU)"


run_segment(){             # run_segment <ride_tag> <row_idx> <gpu>
  local ride_tag=$1 row_idx=$2 gpu=$3
  local split_path="${OUT_DIR}/${ride_tag}/full_raw.txt"
  local tag="[${ride_tag}#${row_idx} gpu${gpu}]"
  echo "$tag START $(date)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    python -u scripts/preprocessing/track_points_online_oneshot.py \
      --root_dir "$ROOT_DIR" \
      --split_path "$split_path" \
      --out_dir   "$OUT_DIR" \
      --cfg_file  "$CFG_FILE" \
      --dataset   split \
      --rows      "$row_idx" \
    2>&1 | sed -u "s/^/$tag /"
  echo "$tag END   $(date)"
}

# Launch scheduler
for task in "${TASKS[@]}"; do
  IFS=':' read -r ride_tag row_idx <<<"$task"
  while :; do
    for idx in "${!SLOTS[@]}"; do
      free=true
      for p in "${!PID2SLOT[@]}"; do
        [[ ${PID2SLOT[$p]} == "$idx" ]] && { free=false; break; }
      done
      $free || continue
      (run_segment "$ride_tag" "$row_idx" "${SLOTS[$idx]}") & PID2SLOT[$!]=$idx
      break 2
    done
    wait -n 2>/dev/null || true
    for p in "${!PID2SLOT[@]}"; do kill -0 "$p" 2>/dev/null || unset 'PID2SLOT[$p]'; done
  done
done

wait
echo "✓ all segments finished $(date)"
