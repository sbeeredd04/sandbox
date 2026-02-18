#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    cat <<EOF
Usage: $0 DATA_PATH OUT_DIR MIN_RIDE MAX_RIDE
EOF
    exit 1
fi

DATA_PATH="$1"
OUT_DIR="$2"
MIN_RIDE="$3"
MAX_RIDE="$4"

echo "Processing rides $MIN_RIDE..$MAX_RIDE sequentially"

for ride_id in $(seq $MIN_RIDE $MAX_RIDE); do
    PREFIX="[ride $ride_id]"
    echo "$PREFIX START $(date)"
    
    # Run the Python script and print output to terminal
    set +e
    python scripts/preprocessing/track_points_online_reinit.py \
        --data_path "$DATA_PATH" \
        --split_path "${OUT_DIR}/output_rides_${ride_id}/full_raw.txt" \
        --out_dir "$OUT_DIR" \
        --dataset split
    RET=$?
    set -e
    
    if [ "$RET" -eq 0 ]; then
        echo "$PREFIX FINISH $(date)"
    else
        echo "$PREFIX ERROR($RET) $(date)"
    fi
done

echo "All rides completed at $(date)."