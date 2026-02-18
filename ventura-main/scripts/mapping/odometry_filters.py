
from pathlib import Path
import hickle as hkl

from PIL import Image
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

# Custom imports
from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)
from spinflow.util.math_utils import (
    odom_to_local_pose
)
from spinflow.util.vis_utils import (
    plot_odometry_topdown
)
from scripts.utils.log_utils import logging

DEBUG_MODE = False

def filter_by_odometry(
    root_dir: str | Path,
    ride_name: str,
    start_frame: int,
    end_frame: int,
    *,
    frame_horizon: int = 80,
    filter_backwards: bool = True,
    filter_stops: bool = True,
    smoothing_window: int = 2,      # Smooths every 2 frames
    back_thresh: float = -0.05,     # any dx < -5 cm counts as backward
    stop_thresh: float = 0.05,      # < 1 cm between frames counts as stop
    num_stops: int = 2,             # Number of stops allowed in the sequence
) -> bool:
    """
    Return **True**  → keep this sequence  
           **False** → reject it because at least one successive pose pair
                       (within `frame_horizon` frames from `start_frame`)
                       • moves *backwards* in x, or
                       • moves less than `stop_thresh` metres (i.e. “stops”)

    Notes
    -----
    * `smoothed_poses` are already expressed **relative to the first frame** so
      +x is forward, +y left, +z up.
    * Only the first `frame_horizon` rows are inspected – perfect for short
      subsequences.
    * Set `smoothing_window>1` to median-filter the raw position trace before
      taking finite-differences (helps noisy wheel odom).  A value of 1 leaves
      the data untouched.
    """
    # ---------- locate & load odom ----------------------------------------
    ride_parts = ride_name.split(" ")
    if len(ride_parts) < 4:
        logging.error(f"Bad ride_name='{ride_name}'")
        return False

    ride_dir = set_frodo_dir(root_dir, *ride_parts)
    odom_path = ride_dir / f"seq_{start_frame}" / "odometry_info.h5"
    if not odom_path.exists():
        logging.error(f"{odom_path} missing.")
        return False
    try:
        info = hkl.load(odom_path)
    except Exception as e:
        logging.error(f"Failed to load {odom_path}: {e}")
        return False
    
    if "smoothed_poses" not in info:
        logging.error(f"{odom_path} has no 'smoothed_poses'.")
        return False
    
    if info['smoothed_poses'].shape[0] < frame_horizon:
        logging.error(f"Not enough poses in {odom_path} (need at least {frame_horizon}).")
        return False

    poses = info["smoothed_poses"][:frame_horizon]          # (N,8)
    if poses.shape[0] < 2:                                  # need ≥2 poses
        return False

    poses = odom_to_local_pose(poses, mode="se3")            # still (N,8)
    xyz   = poses[:, 1:4]                                    # (N,3)

    # ---------- optional median filter to reduce jitter -------------------
    if smoothing_window > 1:
        # xyz = median_filter(xyz, size=(smoothing_window, 1))
        xyz = uniform_filter1d(xyz, size=smoothing_window, axis=0, mode='nearest')

    # ---------- finite-difference ----------------------------------------
    d_xyz = np.diff(xyz, axis=0)                             # (N-1,3)
    d_x   = d_xyz[:, 0]                                      # forward component
    d_norm= np.linalg.norm(d_xyz, axis=1)                    # total displacement

    if DEBUG_MODE:
        data_dict = {
            "odom": xyz[None,...]
        }
        plt_img = plot_odometry_topdown(
            data_dict,
            keys=["odom"],
            labels=["Odom Trajectory"],
            batch_index=0,
            figsize=(8, 8),
            dpi=150,
            axis_limits=[0, -9.6, 9.6, 9.6]
        )
        Image.fromarray(plt_img).save("test_odom_filter.png")
        logging.info(f"Saved odometry plot to 'test_odom_filter.png'.")

    # ---------- criteria --------------------------------------------------
    if filter_backwards and np.any(d_x < back_thresh):
        logging.debug("Rejected – backwards step detected.")
        return False

    # num_stops_detected = np.sum(d_norm < stop_thresh)
    # if filter_stops and num_stops_detected >= num_stops:
    #     logging.debug("Rejected – stop detected.")
    #     return False

    return True