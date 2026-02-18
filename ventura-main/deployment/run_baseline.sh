#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") <model_name> <weights_ckpt_path> [-- extra_hydra_overrides]
...
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

# Raw inputs
RAW_MODEL="$1"
WEIGHTS_CKPT="$2"
shift 2

# Default robot config (can be overridden for convoi)
ROBOT_CFG="deployment/config/robot.yaml"

# Apply the name mappings
case "$RAW_MODEL" in
  lelan)
    MODEL_NAME="planning/lelan/lelan_clip"
    ;;
  spinflow)
    MODEL_NAME="planning/simplepolicy/spinflow_linear_marigold_inference"
    ;;
  convoi)
    MODEL_NAME="planning/convoi"
    ROBOT_CFG="deployment/config/robot_convoi.yaml"   # <-- use convoi config
    ;;
  *)
    MODEL_NAME="$RAW_MODEL"
    ;;
esac

# Build Hydra override args (add robot config)
HYDRA_ARGS=( "model=${MODEL_NAME}" "model.weights_ckpt=${WEIGHTS_CKPT}" "+robot_config=${ROBOT_CFG}" )

# Parse extra args or passthrough
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
    PASSTHRU=( "$@" )
  else
    PASSTHRU=()
    for arg in "$@"; do
      HYDRA_ARGS+=( "$arg" )
    done
  fi
else
  PASSTHRU=()
fi

# --- conda activation detection ---------------------------------------------
CONDA_ENV="${CONDA_ENV:-spinflow_deployment}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
  else
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
      if [[ -f "$d/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$d/etc/profile.d/conda.sh"
        break
      fi
    done
  fi
fi
if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] Could not locate conda.sh. Set CONDA_SH=/path/to/conda.sh" >&2
  exit 2
fi

ACTIVATION="source \"${CONDA_SH}\" && conda activate \"${CONDA_ENV}\""

mk_cmd_str() {
  local out=""
  for a in "$@"; do
    out+=$(printf '%q ' "$a")
  done
  echo "$out"
}

CMD0=$(mk_cmd_str python deployment/src/commander.py)
CMD1=$(mk_cmd_str python deployment/src/navigate.py "${HYDRA_ARGS[@]}" "${PASSTHRU[@]}")

# --- tmux setup --------------------------------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found on PATH. Please install tmux." >&2
  exit 3
fi

SESSION="${TMUX_SESSION:-spinflow}"
ATTACH="${TMUX_ATTACH:-1}"
FORCE="${TMUX_FORCE:-0}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$FORCE" == "1" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "[ERROR] tmux session '$SESSION' already exists. Set TMUX_FORCE=1 to replace." >&2
    exit 4
  fi
fi

tmux new-session -d -s "$SESSION" -n run
tmux send-keys -t "${SESSION}:0.0" "${ACTIVATION}" C-m
tmux send-keys -t "${SESSION}:0.0" "${CMD0}" C-m

tmux split-window -h -t "${SESSION}:0"
sleep 2
tmux send-keys -t "${SESSION}:0.1" "${ACTIVATION}" C-m
tmux send-keys -t "${SESSION}:0.1" "${CMD1}" C-m

echo "[tmux] Launched:"
echo "  • pane 0: ${CMD0} (env: ${CONDA_ENV})"
echo "  • pane 1: ${CMD1} (env: ${CONDA_ENV})"
echo "  session: ${SESSION}"

tmux select-pane -t "${SESSION}:0.1"
if [[ "$ATTACH" == "1" ]]; then
  exec tmux attach-session -t "$SESSION"
fi
