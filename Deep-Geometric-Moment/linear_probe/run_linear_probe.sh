#!/bin/bash
# =============================================================================
# run_linear_probe.sh
# =============================================================================
# Launch script for DGM ResNet34 linear probing on ImageNet.
#
# Usage:
#   bash run_linear_probe.sh
#
# This script:
#   1. Activates the correct conda environment (myenv)
#   2. Sets the GPU to use
#   3. Runs the linear probing script with the YAML config
# =============================================================================

# --- Configuration ---
# Which GPU to use (0-indexed). Change if you want a different GPU.
export CUDA_VISIBLE_DEVICES=0

# Conda environment name
CONDA_ENV="myenv"

# Path to the config file (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"

# --- Activate conda ---
# Try common conda init paths
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/packages/apps/mamba/2.0.8/etc/profile.d/conda.sh" ]; then
    source "/packages/apps/mamba/2.0.8/etc/profile.d/conda.sh"
fi

echo "============================================"
echo " DGM ResNet34 — Linear Probing on ImageNet"
echo "============================================"
echo ""

# Activate the environment
echo "Activating conda environment: ${CONDA_ENV}"
conda activate ${CONDA_ENV}
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# --- Run the linear probing script ---
python "${SCRIPT_DIR}/linear_probe.py" \
    --config "${CONFIG_PATH}" \
    2>&1 | tee "${SCRIPT_DIR}/linear_probe.log"

echo ""
echo "Done! Log saved to ${SCRIPT_DIR}/linear_probe.log"
