#!/usr/bin/env bash
# run_rides_parallel.sh – launch run_engine.py for every output_rides_* folder
#
# Minimal usage:
#   ./run_rides_parallel.sh <split_dir> <split_file>
#
# Optional flags:
#   --gpus 0,1,2          which GPU IDs to cycle through (default 0)
#   --jobs N              concurrent jobs **per GPU**   (default 1)
#   --cfg_file cfg.yaml   extra config file passed to run_engine.py
#
# Example (4 jobs per GPU 0-3):
#   ./run_rides_parallel.sh ./data/frodobots8k_processed full_raw.txt \
#       --gpus 0,1,2,3 --jobs 4 --cfg_file configs/my_exp.yaml

set -euo pipefail

##############################################################################
# ── Parse positional args ───────────────────────────────────────────────────
##############################################################################
[[ $# -lt 2 ]] && {
  echo "Usage: $0 <input_dir> <split_dir> <split_file> [--gpus 0,1] [--jobs N] [--cfg_file cfg.yaml]"
  exit 1
}

INPUT_DIR=$1
SPLIT_DIR=$2
SPLIT_FILE=$3
shift 3

##############################################################################
# ── Optional flags ----------------------------------------------------------
##############################################################################
GPUS="0"; JOBS_PER_GPU=1; CFG_FILE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)      GPUS=$2;        shift 2;;
    --jobs)      JOBS_PER_GPU=$2; shift 2;;
    --cfg_file)  CFG_FILE=$2;    shift 2;;
    *) echo "unknown flag $1"; exit 1;;
  esac
done
IFS=',' read -ra GPU_ARR <<<"$GPUS"

##############################################################################
# ── Discover rides ----------------------------------------------------------
##############################################################################
RIDES=()
for dir in "$INPUT_DIR"/output_rides_*; do
  [[ -d $dir ]] && RIDES+=("$(basename "$dir")")
done
(( ${#RIDES[@]} == 0 )) && { echo "No output_rides_* dirs found"; exit 1; }

##############################################################################
# ── Build the GPU slot table -----------------------------------------------
##############################################################################
SLOTS=()
for g in "${GPU_ARR[@]}"; do
  for ((i=0;i<JOBS_PER_GPU;i++)); do SLOTS+=("$g"); done
done

declare -A PID2SLOT
cleanup(){ echo; for p in "${!PID2SLOT[@]}"; do kill "$p" 2>/dev/null || true; done; wait; }
trap cleanup INT TERM

##############################################################################
# ── Worker ------------------------------------------------------------------
##############################################################################
run_ride() {                    # run_ride <ride_tag> <gpu>
  local ride_tag=$1             # e.g. output_rides_17
  local gpu=${2:-}
  [[ -z $gpu ]] && { echo "[run_ride] missing GPU arg"; exit 1; }

  local split_path="${SPLIT_DIR}/${ride_tag}/${SPLIT_FILE}"
  local tag="[${ride_tag} gpu${gpu}]"

  echo "$tag START $(date)  split=$split_path"

  # Build cfg argument only if provided
  local cfg_args=()
  [[ -n $CFG_FILE ]] && cfg_args=( --cfg_file "$CFG_FILE" )

  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONUNBUFFERED=1 \
    python -u scripts/preprocessing/run_engine.py \
      --split "$split_path" \
      --ride  "$ride_tag" \
      "${cfg_args[@]}" \
    2>&1 | sed -u "s/^/$tag /"

  echo "$tag END   $(date)"
}

echo "▶ running ${#RIDES[@]} rides on GPU(s) {${GPUS}}  (${JOBS_PER_GPU} job/GPU)"

##############################################################################
# ── Main loop ---------------------------------------------------------------
##############################################################################
for ride_tag in "${RIDES[@]}"; do
  while :; do
    for idx in "${!SLOTS[@]}"; do
      # slot free?
      free=true
      for p in "${!PID2SLOT[@]}"; do
        [[ ${PID2SLOT[$p]} == "$idx" ]] && { free=false; break; }
      done
      $free || continue

      (run_ride "$ride_tag" "${SLOTS[$idx]}") & PID2SLOT[$!]=$idx
      break 2                              # launched → next ride
    done

    # no slot free → wait for any job to finish, then clean pids
    wait -n 2>/dev/null || true
    for p in "${!PID2SLOT[@]}"; do
      kill -0 "$p" 2>/dev/null || unset 'PID2SLOT[$p]'
    done
  done
done

wait
echo "✓ all rides finished  $(date)"
