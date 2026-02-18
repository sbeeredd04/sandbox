import numpy as np
import pandas as pd
from scripts.utils.log_utils import logging

import re
import json
import torch
import decord

import h5py
from typing import Any, Dict, Union
from pathlib import Path
import yaml

from spinflow.dataset.frodo_helpers import get_frodo_raw_id
from spinflow.dataset.frodo_constants import (DEFAULT_GPS_VALUE)

def combine_ride_splits(root_dir: str, split_file: str) -> pd.DataFrame:
    """"Combines all ride splits into a single DataFrame."""
    root_dir = Path(root_dir)
    assert root_dir.is_dir(), f"Root directory {root_dir} does not exist or is not a directory."
    
    split_files = list(root_dir.glob(f"output_rides_*/{split_file}"))
    assert len(split_files) > 0, f"No split files found in {root_dir} with name {split_file}."

    all_data = []
    for split_file in split_files:
        try:
            df = pd.read_csv(split_file, header=0)
            if not df.empty:
                all_data.append(df)
                logging.info(f"Loaded {len(df)} samples from {split_file}.")
            else:
                logging.warning(f"Split file {split_file} is empty.")
        except pd.errors.EmptyDataError:
            logging.error(f"Split file {split_file} is empty or could not be read.")
    assert len(all_data) > 0, "No valid data found in any split files."
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined DataFrame contains {len(combined_df)} samples.")
    return combined_df

def combine_ride_graphs(root_dir: str, graph_lut_file: str) -> pd.DataFrame:
    """Merge every ride→graph LUT under *root_dir/output_rides_*/…*.

    Each row’s `graph_path` is rewritten so everything **up to (and including)**
    the first directory that starts with `output_rides_` is replaced by
    `root_dir`.
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    lut_paths = root.glob(f"output_rides_*/maps/metadata/{graph_lut_file}")
    frames = []
    for lut in lut_paths:
        try:
            df = pd.read_csv(lut)
        except pd.errors.EmptyDataError:
            logging.warning(f"skip empty {lut}")
            continue
        if df.empty:
            continue

        def rebase(p: str) -> str:
            parts = Path(p).parts
            # keep from the first ‘output_rides_…’ on
            try:
                i = next(j for j, s in enumerate(parts) if s.startswith("output_rides_"))
            except StopIteration:          # no match → leave unchanged
                return p
            return str(root / Path(*parts[i:]))

        df["graph_path"] = df["graph_path"].map(rebase)
        frames.append(df)

    if not frames:
        raise RuntimeError("no valid LUTs found")

    return pd.concat(frames, ignore_index=True)

def load_video(video_path, start_frame=None, end_frame=None, format="numpy"):
    """
    Loads a video file and returns its frames.
    
    Args:
        video_path (str or Path): Path to the video file.
        format (str): Format to return the frames in, either 'numpy' or 'pandas'.
    
    Returns:
        np.ndarray or pd.DataFrame: Video frames in the specified format.
    """
    try:
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path))
        start_frame = int(start_frame) if start_frame is not None else 0
        end_frame = int(end_frame) if end_frame is not None else len(vr) - 1
        frames = vr.get_batch(range(start_frame, end_frame + 1))

        if format == "numpy":
            return np.clip(frames.numpy(), 0, 255).astype(np.uint8)
        elif format == "torch":
            return torch.clip(frames, 0, 255).to(torch.uint8)
        else:
            raise ValueError("Unsupported video format. Use 'numpy' or 'torch'.")
    except Exception as e:
        logging.error(f"Failed to load video from {video_path}: {e}")
        return None

def load_graph_lut(graph_lut_path):
    """Loads a pandas df mapping ride IDs to their corresponding graph nodes."""
    try:
        df = pd.read_csv(graph_lut_path)
        if not {"ride", "ride_dir", "graph_path"}.issubset(df.columns):
            logging.error(f"[ERROR] graph_lut.csv in {graph_lut_path} is missing required columns.")
            return None
        return df
    except Exception as e:
        logging.error(f"Failed to load graph LUT from {graph_lut_path}: {e}")
        return None

def load_timestamps(timestamps_path):
    """
    Loads timestamps from a CSV file.
    Returns a numpy array of timestamps.
    """
    try:
        timestamps = np.loadtxt(timestamps_path, delimiter=',', skiprows=1, dtype=np.float64)
        if timestamps.ndim == 1:
            timestamps = timestamps.reshape(-1, 1)
        return timestamps[:, -1] # [N,]
    except Exception as e:
        logging.error(f"Failed to load timestamps from {timestamps_path}: {e}")
        return None
    
def load_controls(cmd_vel_path):
    """
    Loads control data from a CSV file.
    Returns a numpy array of control commands.
    """
    try:
        controls = np.loadtxt(cmd_vel_path, delimiter=',', skiprows=1, dtype=float)
        return controls[:, [0, 1, -1]] # [linear, angular, timestamp]
    except Exception as e:
        logging.error(f"Failed to load control data from {cmd_vel_path}: {e}")
        return None
    
def load_odom(odom_path, format='numpy'):
    assert format in ['numpy', 'pandas'], "Unsupported odometry loader format. Use 'numpy' or 'pandas'."
    try:
        odometry = pd.read_csv(odom_path, header=0)

        # Rearrange colummns to [timestamp, x, y, z, qw, qx, qy, qz]
        if not {"timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"}.issubset(odometry.columns):
            logging.error(f"[ERROR] odometry_data.csv in {odom_path} is missing required columns.")
            return None
        
        odometry = odometry[["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]].dropna()
        if format == 'pandas':
            return odometry.astype(np.float64)
        elif format == 'numpy':
            odometry_np = odometry.to_numpy(dtype=np.float64)
            if odometry_np.shape[1] != 8:
                logging.error(f"[ERROR] odometry_data.csv in {odom_path} has incorrect number of columns.")
                return None
            return odometry_np

    except Exception as e:
        logging.error(f"Failed to load odometry data from {odom_path}: {e}")
        return None

def load_gps(gps_path, format='numpy'):
    """
    Loads GPS data from a CSV file or default file strecture
    """
    if gps_path.is_dir():
        # Attempt to use default filename
        ride_infos = get_frodo_raw_id(gps_path)
        gps_path = gps_path / f"gps_data_{ride_infos[1]}.csv"

    try:
        if format not in ['numpy', 'pandas']:
            raise ValueError("Unsupported gps loader format. Use 'numpy' or 'pandas'.")

        if format == 'pandas':
            df = pd.read_csv(gps_path)
            if not {"latitude", "longitude", "timestamp"}.issubset(df.columns):
                logging.error(f"[ERROR] gps_data.csv in {gps_path} is missing required columns.")
                return None
            gps_data = df[["latitude", "longitude", "timestamp"]].dropna()
            gps_data = gps_data[(gps_data['latitude'] != DEFAULT_GPS_VALUE) &
                            (gps_data['longitude'] != DEFAULT_GPS_VALUE)]

            # Convert ms to s
            gps_data['timestamp'] = gps_data['timestamp'].astype(np.float64) / 1000.0
        elif format == 'numpy':
            gps_data = np.loadtxt(gps_path, delimiter=',', skiprows=1, dtype=np.float64)
            invalid_mask = gps_data == DEFAULT_GPS_VALUE
            gps_data = gps_data[~np.any(invalid_mask, axis=1)]  # Remove rows with default GPS values

            # Convert ms to s
            gps_data[:, -1] = gps_data[:, -1].astype(np.float64) / 1000.0
        
        return gps_data # [latitude, longitude, timestamp]
    except Exception as e:
        logging.error(f"Failed to load GPS data from {gps_path}: {e}")
        return None
    
def load_inertial(imu_path, format='numpy'):
    """
    Loads inertial data from a CSV file.
    Returns a numpy array of inertial data.
    """
    try:
        if format not in ['numpy', 'pandas']:
            raise ValueError("Unsupported inertial loader format. Use 'numpy' or 'pandas'.")
       
        df = pd.read_csv(imu_path)
        heading = imudf_to_heading(df, compass_col='compass')
        if heading is None or len(heading) == 0:
            logging.error(f"[ERROR] imu_data.csv in {imu_path} is empty or invalid.")
            return None
        
        if format == 'pandas':
            inertial_data = pd.DataFrame(heading, columns=['timestamp', 'heading'])
            inertial_data['timestamp'] = inertial_data['timestamp'].astype(np.float64)
            inertial_data['heading'] = inertial_data['heading'].astype(np.float64)
            inertial_data = inertial_data.dropna()
        elif format == 'numpy':
            # filter out mask for invalid values
            inertial_data = heading[~np.isnan(heading[:, 1])].astype(np.float64)

        return inertial_data # [timestamp, heading]
    except Exception as e:
        logging.error(f"Failed to load inertial data from {imu_path}: {e}")
        return None

def load_intrinsics(intrinsics_path):
    try:
        # Load yaml as dictionary
        with open(intrinsics_path, 'r') as f:
            intrinsics = yaml.safe_load(f)
        # Convert to numpy array
        fields = {
            'D': [1, 5],
            'K': [3, 3],
            'R': [3, 3],
            'P': [3, 4],
        }
        assert all(k in intrinsics for k in fields), f"Missing fields in {intrinsics_path}"
        intrinsics.update({
            k: np.array(v).reshape(fields[k]) for k, v in intrinsics.items() if k in fields
        })
        return intrinsics
    except Exception as e:
        logging.error(f"Failed to load intrinsics from {intrinsics_path}: {e}")
        return None

def load_extrinsics(extrinsics_path):
    try:
        with open(extrinsics_path, 'r') as f:
            extrinsics = yaml.safe_load(f)
        
        # Loop through all transforms and convert to numpy arrays
        for key, value in extrinsics.items():
            extrinsics[key] = np.array(value, dtype=np.float64).reshape(4, 4) if isinstance(value, list) else value

        return extrinsics
    except Exception as e:
        logging.error(f"Failed to load extrinsics from {extrinsics_path}: {e}")
        return None

def imudf_to_heading(imu_df, compass_col='compass'):
    """
    Vectorized magnetometer → compass heading.

    Inputs:
      imu_df : pandas DataFrame with columns ['compass, 'imu', 'timestamp']
      y_offset : float, offset to subtract from magy (in same units as mag)
      z_offset : float, offset to subtract from magz

    Returns:
      np.ndarray of shape (N,2): [
        [ts_0, heading_0_deg],
        [ts_1, heading_1_deg],
        ...
      ]
    where 0° = North, increasing clockwise.
    """
    extracted = imu_df[compass_col].str.extract(
        r'\[\["(?P<magx>-?\d+)",\s*"(?P<magy>-?\d+)",\s*"(?P<magz>-?\d+)",\s*"(?P<ts>[\d\.]+)"\]\]'
    ).astype(float)

    mag_y = extracted['magy'].to_numpy()
    mag_z = extracted['magz'].to_numpy()
    ts    = extracted['ts'].to_numpy()

    # Compute y and z offset by averaging min max
    y_offset = (np.nanmin(mag_y) + np.nanmax(mag_y)) / 2.0
    z_offset = (np.nanmin(mag_z) + np.nanmax(mag_z)) / 2.0

    # apply bias offsets
    y_cal = mag_y - y_offset
    z_cal = mag_z - z_offset

    # compute North & West components
    north = -z_cal      # "forward" vector
    west  =  y_cal      # left vector

    # atan2(west, north) yields angle from North, positive clockwise
    heading_rad = np.arctan2(west, north)
    heading_rad = np.mod(heading_rad, 2 * np.pi)
    heading_deg = np.degrees(heading_rad)

    return np.column_stack((ts, heading_deg))

def load_langlabel_info(langlabel_path, root_dir):
    """
    Convert language label json files to a pandas DataFrame.
    with the following columns:
    - ride_name (str): Name of the ride.
    - start_frame (int): Start frame of the ride.
    - end_frame (int): End frame of the ride.
    """
    label_dict = json.loads(langlabel_path.read_text())
    video_meta = label_dict['video']

    m = re.match(
        r'^(output_rides_\d{4}(?:-\d{2}){5})_(ride_.+?_\d{4}(?:-\d{2}){5}_\d+)_(.+)$', 
        video_meta
    )
    if not m:
        raise ValueError(f"Unrecognized video meta format: {video_meta}")
    out_dir, ride_dir, video_name = m.groups()

    video_path = root_dir / out_dir / ride_dir / video_name
    assert video_path.exists(), f"Video path {video_path} does not exist."

    out_suffix = out_dir.replace("output_rides_", "")
    ride_parts = ride_dir.replace("ride_", "").split("_")  # e.g. [name, date, idx]
    ride_name = " ".join([out_suffix] + ride_parts)

    # Open with decord and prepare seconds→frame mapping
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    num_frames = len(vr)

    # Try to build per-frame timestamps (seconds). Fallback to fps.
    ts_sec = None
    try:
        # get_frame_timestamp(i) returns a scalar or a tuple; take the first value
        ts_list = []
        for i in range(num_frames):
            t = vr.get_frame_timestamp(i)
            # handle tuple/list returns (e.g., (pts_sec, ?))
            if isinstance(t, (list, tuple, np.ndarray)):
                ts_list.append(float(t[0]))
            else:
                ts_list.append(float(t))
        ts_sec = np.asarray(ts_list, dtype=np.float64)
        # Ensure strictly non-decreasing (some containers can have tiny jitter)
        ts_sec = np.maximum.accumulate(ts_sec)
    except Exception:
        ts_sec = None

    if ts_sec is None:
        # CFR fallback: seconds × fps
        fps = float(vr.get_avg_fps())
        def sec_to_frame(t: float) -> int:
            return int(np.clip(round(t * fps), 0, num_frames - 1))
    else:
        # VFR (or exact) mapping: choose nearest timestamp
        def sec_to_frame(t: float) -> int:
            t = float(t)
            j = int(np.searchsorted(ts_sec, t, side="left"))
            if j <= 0:
                idx = 0
            elif j >= ts_sec.size:
                idx = ts_sec.size - 1
            else:
                idx = j if (ts_sec[j] - t) < (t - ts_sec[j - 1]) else (j - 1)
            return int(idx)

    rows = []
    for ann in label_dict.get("annotations", []):
        t0, t1 = float(ann["start"]), float(ann["end"])
        lab = ann.get("label", "").strip()

        # Map to nearest indices and clamp to [0, num_frames-1]
        i0 = sec_to_frame(t0)
        i1 = sec_to_frame(t1)
        # Ensure start <= end
        if i1 < i0:
            i0, i1 = i1, i0

        rows.append({
            "ride_name": ride_name,
            "start_frame": i0,
            "end_frame": i1,
            "label": lab
        })

    return pd.DataFrame(rows, columns=["ride_name", "start_frame", "end_frame", "label"])

