#!/usr/bin/env bash
#
# Usage:
#   ./run_rides.sh <split_dir> [ride_id]
#
# If <ride_id> is provided, runs the engine for only that ride:
#   ./run_rides.sh /path/to/splits output_rides_5
#
# If <ride_id> is omitted, loops from output_rides_0 through output_rides_40.

set -euo pipefail

# Check for at least one argument (split directory)
if [ $# -lt 4 ]; then
  echo "Usage: $0 <split_dir> <min_ride> <max_ride> <split_file> [ride_id]"
  echo "Example: $0 ./data/frodobots8k_processed 0 40 full_raw.txt"
  exit 1
fi

split_dir="$1"
min_ride="$2"  # minimum ride index
max_ride="$3"  # maximum ride index
split_file="$4"  # split file name (e.g., full_raw.txt, )
ride_arg="${5:-}"  # optional ride ID argument

if [ -n "$ride_arg" ]; then
  ride_id="$ride_arg"
  split_path="${split_dir}"
  echo "🏃 Running for ride: $ride_id"
  echo "Using split path: $split_path"
  python scripts/preprocessing/run_engine.py --split "$split_path" --ride "$ride_id"
else
  for i in $(seq "$min_ride" "$max_ride"); do
    ride_id="output_rides_$i"
    split_path="${split_dir}/${ride_id}/${split_file}"
    echo "🏃 Running for ride: $ride_id"
    echo "Using split path: $split_path"
    python scripts/preprocessing/run_engine.py --split "$split_path" --ride "$ride_id"
  done
fi
