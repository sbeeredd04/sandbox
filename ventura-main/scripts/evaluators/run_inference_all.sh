#!/usr/bin/env bash
set -Eeuo pipefail

# -------------------------------
# Config — edit these lists
# -------------------------------
# Placeholder token to be replaced with each bag path:
BAG_PLACEHOLDER="__BAG_PATH__"
VIDEO_PREFIX_PLACEHOLDER="__VIDEO_PREFIX__"


# List of bag files to process:
BAGS=(
  # Sidewalk
  # "/data/aih/madero/2025-09-17/ferrite7_2025-09-17-17-03-00_test/merged.bag"
  "/data/aih/madero/2025-09-17/ferrite7_2025-09-17-17-37-00_test/merged.bag"
  # "/data/aih/madero/2025-09-18/ferrite7_2025-09-18-15-22-00_test/merged.bag"

  # Dirt path
  # "/data/aih/madero/2025-09-18/ferrite7_2025-09-18-11-13-00_test/merged.bag"
  # "/data/aih/madero/2025-09-18/ferrite7_2025-09-18-13-02-00_test/merged.bag"
)

# List of Python CLI command templates (use the PLACEHOLDER in the right spot)
CMDS=(
    "python scripts/evaluators/run_bag_inference.py model=planning/simplepolicy/spinflow_linear_marigold_inference model.weights_ckpt=/data/model_ckpts_awslarge2x/SimplePolicy/spinflow_linear_marigold/20250905/052855/best-0049-0.0818.ckpt +robot_config=deployment/config/robot_offline.yaml +bag_path=${BAG_PLACEHOLDER} +rgb_pred_path=${VIDEO_PREFIX_PLACEHOLDER}_pred.mp4 +rgb_path=${VIDEO_PREFIX_PLACEHOLDER}_rgb.mp4" 
    # "python scripts/evaluators/run_bag_inference.py model=planning/simplepolicy/spinflow_linear_marigold_inference model.weights_ckpt=/data/model_ckpts_awslarge2x/SimplePolicy/spinflow_linear_marigold/20250901/053426/best-0049-0.0684.ckpt +robot_config=deployment/config/robot_offline.yaml +bag_path=${BAG_PLACEHOLDER} +rgb_pred_path=${VIDEO_PREFIX_PLACEHOLDER}_pred.mp4 +rgb_path=${VIDEO_PREFIX_PLACEHOLDER}_rgb.mp4"
    # "python scripts/evaluators/run_bag_inference.py model=planning/lelan/lelan_clip model.weights_ckpt=/data/model_ckpts_awslarge2x/LeLaNPolicy/lelan_clip/20250821/052037/best-0099-0.1688.ckpt +robot_config=deployment/config/robot_offline.yaml +bag_path=${BAG_PLACEHOLDER} +rgb_pred_path=${VIDEO_PREFIX_PLACEHOLDER}_pred.mp4 +rgb_path=${VIDEO_PREFIX_PLACEHOLDER}_rgb.mp4"
    # "python scripts/evaluators/run_bag_inference.py model=planning/convoi +robot_config=deployment/config/robot_convoi_offline.yaml +bag_path=${BAG_PLACEHOLDER} +rgb_pred_path=${VIDEO_PREFIX_PLACEHOLDER}_pred.mp4 +rgb_path=${VIDEO_PREFIX_PLACEHOLDER}_rgb.mp4"
)

VIDEO_PREFIXES=(
  "spinflow_linear_pretrained"
  # "spinflow_linear_nopretrained"
  # "lelan_clip"
  # "convoi"
)

# Behavior flags (optional)
CONTINUE_ON_ERROR=0   # set to 1 to continue even if a command fails
DRY_RUN=0             # set to 1 to only print commands without running them
LOG_DIR="./bag_runs_logs"  # logs for each run (created automatically)

# -------------------------------
# Helpers
# -------------------------------
ts() { date +"%Y-%m-%d_%H-%M-%S"; }
mkdir -p "$LOG_DIR"

# Validate arrays
if [[ "${#CMDS[@]}" -ne "${#VIDEO_PREFIXES[@]}" ]]; then
  echo "ERROR: CMDS (${#CMDS[@]}) and VIDEO_PREFIXES (${#VIDEO_PREFIXES[@]}) must have the same length." >&2
  exit 2
fi

# -------------------------------
# Double for-loop
# -------------------------------
for i in "${!CMDS[@]}"; do
  tmpl="${CMDS[$i]}"
  vprefix="${VIDEO_PREFIXES[$i]}"

  for bag in "${BAGS[@]}"; do
    if [[ ! -f "$bag" ]]; then
      echo "WARNING: Bag not found: $bag" >&2
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && continue || exit 1
    fi

    # Make a safe version of the bag basename you can optionally append to prefix
    bag_base="$(basename "$bag")"
    safe_bag="${bag_base//[^a-zA-Z0-9._-]/_}"

    # If you want unique outputs per bag, uncomment the next line:
    # vprefix_effective="${vprefix}_${safe_bag}"
    # Otherwise, keep the provided prefix as-is:
    vprefix_effective="${vprefix}"

    # Replace placeholders
    cmd="${tmpl//${BAG_PLACEHOLDER}/${bag}}"
    cmd="${cmd//${VIDEO_PREFIX_PLACEHOLDER}/${vprefix_effective}}"

    echo "----------------------------------------------------------------"
    echo "[$(ts)] Running [cmd #$((i+1))/${#CMDS[@]}] on bag:"
    echo "Bag: $bag"
    echo "Cmd: $cmd"
    echo "----------------------------------------------------------------"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      continue
    fi

    log_file="${LOG_DIR}/run_${safe_bag}_cmd${i}_$(ts).log"

    set +e
    bash -lc "$cmd" 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" -ne 0 ]]; then
      echo "ERROR: Command failed (exit $status). See log: $log_file" >&2
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || exit "$status"
    else
      echo "SUCCESS: See log: $log_file"
    fi
  done
done

echo "All done."