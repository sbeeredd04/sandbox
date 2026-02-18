#!/usr/bin/env python3
"""
sync_odom.py
------------------
Resample odometry CSVs to camera timestamps using pymlg (Lie-group) interpolation.

Assumptions
-----------
Odometry CSV columns:
    timestamp,x,y,z,qx,qy,qz,qw      # quaternion (x y z w) order

Camera timestamp file:
    plain text, one float/int per line.

Dependencies
------------
pip install numpy pandas scipy pymlg
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np
# import pandas as pd
from pymlg import SE2

import spinflow.dataset.frodo_helpers as fh

# -----------------------------------------------------------------------------#
# I/O helpers
# -----------------------------------------------------------------------------#
def load_odometry_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load odometry CSV via numpy.
    Returns:
        t_odom: (N,) float64
        p_odom: (N,3) float64
        yaw_odom: (N,4) float64
    """
    # load all 8 columns, skip the header line
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    t_odom = data[:, 0]
    p_odom = data[:, 1:3]
    yaw_odom = data[:, 3]
    return t_odom, p_odom, yaw_odom


def load_camera_timestamps(ts_path: Path, abs_frames: np.ndarray) -> np.ndarray:
    """Read camera timestamps from HDF5 into a NumPy array."""
    ts = np.loadtxt(ts_path, delimiter=',', skiprows=1, dtype=np.float64)
    try:
        camera_ts = ts[abs_frames, 1]
    except IndexError:
        raise IndexError(f"Camera timestamps in {ts_path} do not match frames in HDF5 file.")

    return camera_ts


# ---------------------------------------------------------------------------- #
# SE(3) interpolation
# ---------------------------------------------------------------------------- #

def interpolate_geodesic(T0: SE2, T1: SE2, alpha: float) -> SE2:
    """Compute T0 · Exp( α · Log(T0⁻¹ T1) )."""
    xi = SE2.Log(SE2.inverse(T0) @ T1)   # ℝ⁶ tangent vector
    return T0 @ SE2.Exp(alpha * xi)


def resample_SE2(
    t_odom:    np.ndarray,
    p_odom:    np.ndarray,
    yaw_odom:  np.ndarray,
    cam_ts:    np.ndarray
) -> Tuple[np.ndarray, List[SE2]]:
    """
    For each camera timestamp t in cam_ts, perform SE(2) geodesic interpolation
    between the two odometry poses that bracket t.

    Assumes:
      - t_odom is strictly increasing
      - cam_ts[0]  >= t_odom[0]
      - cam_ts[-1] <= t_odom[-1]

    Inputs
    ------
    t_odom   : (N,)   monotonic odometry timestamps
    p_odom   : (N,2)  odometry positions [x, y]
    yaw_odom : (N,)   odometry headings (radians)
    cam_ts   : (M,)   camera timestamps

    Returns
    -------
    ts_out : (M,)       identical to cam_ts
    poses  : List[SE2]  one SE2 pose per cam_ts
    """
    # sanity check
    if cam_ts[0] < t_odom[0] or cam_ts[-1] > t_odom[-1]:
        raise ValueError("Camera timestamps must lie within odometry time span")

    # pre-build SE2 from odometry
    T_odom: List[SE2] = [
        SE2.Exp(np.array([px, py, yaw]))
        for (px, py), yaw in zip(p_odom, yaw_odom)
    ]

    # find right‐bracket index for each camera ts
    # idx = k means t_odom[k-1] <= cam_ts < t_odom[k]
    idx_right = np.searchsorted(t_odom, cam_ts, side="right")
    idx_left  = idx_right - 1

    poses: List[SE2] = []
    for j, t in enumerate(cam_ts):
        i0, i1 = idx_left[j], idx_right[j]
        t0, t1 = t_odom[i0], t_odom[i1]
        alpha       = (t - t0) / (t1 - t0)

        T0, T1 = T_odom[i0], T_odom[i1]
        # compute relative tangent and geodesic step
        xi      = SE2.Log(SE2.inverse(T0) @ T1)
        T_interp = T0 @ SE2.Exp(alpha * xi)

        poses.append(T_interp)

    # ts_out is exactly cam_ts, one‐to‐one
    return cam_ts.copy(), poses


def load_camera_frames(h5_path: Path) -> np.ndarray:
    """Load camera frames from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if "frames" in f:
            frames = f['frames']['front_camera'][()]
        else:
            raise KeyError(f"No 'frames' dataset in {h5_path}")
    return frames.astype(int)

# ---------------------------------------------------------------------------- #
# Per‐sequence processing
# ---------------------------------------------------------------------------- #

def process_one(pair: dict):
    odom_path, ts_path, h5_path, out_path = \
        pair["odom_path"], pair["ts_path"], pair["h5_path"], pair["out_path"]

    # 1) load odom + cam timestamps
    t_odom, p_odom, yaw_odom  = load_odometry_csv(odom_path)
    abs_frames_np           = load_camera_frames(h5_path)
    cam_ts                  = load_camera_timestamps(ts_path, abs_frames_np)

    # 2) interpolate in SE(3)
    ts_out, poses = resample_SE2(t_odom, p_odom, yaw_odom, cam_ts)

    # 3) assemble numpy array for saving
    M = len(poses)
    out_arr = np.empty((M, 8), dtype=np.float64)
    out_arr[:, 0] = ts_out
    for i, T in enumerate(poses):
        out_arr[i, 1:4] = T.translation()
        out_arr[i, 4:8] = T.quaternion()  # x, y, z, w

    # 4) save via numpy.savetxt
    header = "timestamp,x,y,z,qx,qy,qz,qw"
    np.savetxt(
        out_path,
        out_arr,
        fmt="%.9f",            # adjust precision if needed
        delimiter=",",
        header=header,
        comments=""            # no '#' before header
    )
    print(f"✓ {odom_path.name} → {out_path.name} ({M} frames)")

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def get_odom_frodo_id(path: Path) -> str:
    """Extract Frodo ID from the odometry CSV file path."""
    dt_array = path.parent.name.split('_')[1:]
    drive_timestamp = [f'{dt_array[0]}_{dt_array[1]}', str(dt_array[2])]

    ride_id = str(path.parent.parent.name.split('_')[-1])
    return (ride_id, *drive_timestamp)  # (ride_id, drive_id, timestamp)

def main():
    ap = argparse.ArgumentParser(
        description="Interpolate odometry CSVs with pymlg’s SE(3) tools."
    )
    ap.add_argument("--odom_dir", required=True, type=Path,
                    help="Directory containing odometry *.csv")
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="TXT file of camera timestamps")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Directory to write interpolated CSVs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load all camera timestamp and odometry CSV files. Only process pairs
    h5_files = fh.get_available_sequences(args.data_dir, ext='h5')
    h5_frodo_ids = [fh.get_frodo_id(f) for f in h5_files]
    id_to_h5 = {fid: f for fid, f in zip(h5_frodo_ids, h5_files)}
    split_set = set(map(tuple, h5_frodo_ids))
    

    odom_files = fh.get_available_sequences(args.odom_dir, ext='csv')
    odom_frodo_ids = [get_odom_frodo_id(f) for f in odom_files]

    raw_data_dir = "/robodata/public_datasets/frodobots8k"
    # Filter common sequences between h5 and odom files
    valid_pairs = [
        {
            "odom_path": odom_file, 
            "h5_path": id_to_h5[odom_id], 
            "out_path": args.out_dir / f"ride_{odom_id[0]}_{odom_id[1]}_{odom_id[2]}.csv",
            "ts_path": Path(raw_data_dir) / f"output_rides_{odom_id[0]}" / f"ride_{odom_id[1]}_{odom_id[2]}" / f"front_camera_timestamps_{odom_id[1].split('_')[0]}.csv"
        }
        for odom_file, odom_id in zip(odom_files, odom_frodo_ids) if odom_id in split_set
    ]

    for pair in valid_pairs:
        process_one(pair)

if __name__ == "__main__":
    main()
