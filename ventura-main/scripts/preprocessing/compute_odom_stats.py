"""
This function should load a config file and compute the odometry statistics
for a dataset.
"""
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import hickle as hkl
from joblib import Parallel, delayed

from scripts.utils.log_utils import logging
from spinflow.dataset.frodo_helpers import (
    get_available_sequences,
    get_frodo_id,
    set_frodo_dir
)
from scripts.mapping.compute_odom import (
    odom_window_metrics
)


def load_ride_infos(data_dir, split_file):
    name = split_file.stem
    ext = split_file.suffix.strip('.')
    split_files = get_available_sequences(data_dir, name=name, ext=ext)
    if not split_files:
        logging.error(f"No split files found for {split_file} in {data_dir}.")
        return None
    
    # Load each split file and extract ride paths
    ride_infos = []
    for split_file in split_files:
        ride_df = pd.read_csv(split_file, header=0)
        ride_infos.append(ride_df)
    full_df = pd.concat(ride_infos, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['ride_name', 'start_frame', 'end_frame'])

    return full_df

def _compute_disp_by_axis(ride_name, start_frame, end_frame, data_dir, window_len):
    # unpack ride_name
    ride_id, did0, did1, timestamp = ride_name.split(' ')
    ride_dir = set_frodo_dir(data_dir, ride_id, did0, did1, timestamp)
    seq_dir = ride_dir / f"seq_{start_frame}"
    odometry_path = seq_dir / "odometry_info.h5"
    if not odometry_path.exists():
        logging.warning(f"Odometry file {odometry_path} missing for ride {ride_name}.")
        return None

    data_dict = hkl.load(odometry_path)
    odom_np = np.array(data_dict['poses'])  # (F,8)
    if odom_np.shape[0] < window_len + 1:
        logging.warning(f"Not enough frames ({odom_np.shape[0]}) for window size {window_len+1}.")
        return None
    
    # take exactly horizon+1 frames from the start
    odom_window = odom_np[: window_len + 1, :]
    metrics = odom_window_metrics(odom_window, window_len)
    return metrics['disp_by_axis']    # shape (N-H,3,2)


def compute_and_log_stats(total_disp_by_axis: np.ndarray, log_path: str):
    """
    Compute mean, std, and 1σ/2σ/3σ min/max thresholds for each axis
    from total_disp_by_axis (shape: N×3×2) and log to console & file.

    Parameters
    ----------
    total_disp_by_axis : np.ndarray
        Array of shape (N,3,2), where [:,i,0] are per-window mins
        and [:,i,1] are per-window maxs for axis i in (x,y,z).
    log_path : str
        Path to the output log file.
    """
    # set up a logger that writes to both stdout and a file
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        mins = total_disp_by_axis[:, i, 0]
        maxs = total_disp_by_axis[:, i, 1]

        mean_min = np.mean(mins)
        std_min  = np.std(mins)
        mean_max = np.mean(maxs)
        std_max  = np.std(maxs)

        logging.info(f"Axis {ax!r}: mean_min={mean_min:.4f}, std_min={std_min:.4f}, "
                    f"mean_max={mean_max:.4f}, std_max={std_max:.4f}")

        for k in (1, 2, 3):
            thr_min = mean_min - k * std_min
            thr_max = mean_max + k * std_max
            logging.info(f"  {k}σ thresholds: min ≥ {thr_min:.4f}, max ≤ {thr_max:.4f}")

    logging.info("Done computing displacement-by-axis statistics.\n")

def main(cfg, split_file, output_file):
    """Computes odometry statistics from the dataset config."""
    data_dir = Path(cfg['out_dir'])
    ride_info_df = load_ride_infos(data_dir, split_file)
    
    if ride_info_df is None or ride_info_df.empty:
        logging.error("No ride information found. Exiting.")
        return

    logging.info(f"Loaded {len(ride_info_df)} rides from {split_file}.")
    window_len = cfg['window_length']
    rows = ride_info_df[['ride_name', 'start_frame', 'end_frame']].values

    # run in parallel
    disp_list = Parallel(
        n_jobs=cfg.get('n_jobs', 32),
        verbose=10
    )(
        delayed(_compute_disp_by_axis)(
            ride_name, start_frame, end_frame, data_dir, window_len
        )
        for (ride_name, start_frame, end_frame) in rows
    )

    # stack only the successful results
    total_disp_by_axis = np.vstack([d for d in disp_list if d is not None])

    # clip displacements
    total_disp_by_axis = np.clip(total_disp_by_axis, -12.8, 12.8)  # Assuming limits from config
    logging.info(f"Aggregated disp_by_axis shape: {total_disp_by_axis.shape}")
    
    # Compute the mean and std, and 1 std, 2std, 3 std thresholds for each axis
    compute_and_log_stats(total_disp_by_axis, output_file)

    logging.info(f"Odometry statistics saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute odometry statistics from dataset config.")
    parser.add_argument("--cfg_file", type=Path, default="./scripts/preprocessing/config/fai_mining.yaml", help="Path to the dataset config file.")
    parser.add_argument("--split_file", type=Path, default="full_rideinfos.txt", help="Path to the split file (optional).")
    parser.add_argument("--output_file", type=Path, default="odom_stats.txt", help="Output file to save the statistics.")
    args = parser.parse_args()

    assert args.cfg_file.exists(), f"Config file {args.cfg_file} does not exist."
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.split_file, args.output_file)