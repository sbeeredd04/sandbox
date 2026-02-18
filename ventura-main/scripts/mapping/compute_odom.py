from pathlib import Path
import numpy as np
import pandas as pd
import hickle as hkl
import math
from numpy.lib.stride_tricks import sliding_window_view

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline, interp1d

from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)
from scripts.utils.log_utils import logging
from scripts.utils.loader_utils import (
    load_odom,
    load_timestamps
)

try:
    from pymlg import SO3, SE3
except ImportError:
    logging.error("Please install 'pymlg' package to use this script.")
    raise ImportError("Missing 'pymlg' package. Install it with: pip install pymlg")
# ── tiny helpers to “batchify” scalar SE3 ops ───────────────────────────
def se3_inverse_batch(Ts: np.ndarray) -> np.ndarray:         # (B,4,4)
    return np.stack([SE3.inverse(T) for T in Ts], axis=0)

def se3_log_batch(Ts: np.ndarray) -> np.ndarray:             # (B,6)
    return np.stack([SE3.Log(T).reshape(-1) for T in Ts], 0)

def se3_exp_batch(xis: np.ndarray) -> np.ndarray:            # (B,4,4)
    return np.stack([SE3.Exp(xi) for xi in xis], axis=0)

# ── quat+xyz → SE3 ----------------------------------------------------------
def quat_to_se3(q_wxyz: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]           # SciPy order x y z w
    mats   = np.zeros((len(q_wxyz), 4, 4))
    mats[:, 3, 3]  = 1.0
    mats[:, :3, :3] = R.from_quat(q_xyzw).as_matrix()
    mats[:, :3,  3] = xyz
    return mats

def quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def angular_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def odom_window_metrics(odom_np: np.ndarray, horizon_frames: int):
    """
    Vectorised version: returns three 1-D arrays (length N-H) so that
    entry *k* corresponds to the centred window whose **middle frame index**
    is k + H//2.

    Returns
    -------
    dist      : (N-H,)  Euclidean start→end distance  [metres]
    max_dyaw  : (N-H,)  max |yaw - yaw_start|         [radians]
    max_omega : (N-H,)  max |d(yaw)/dt|               [deg/s]
    """
    H       = horizon_frames
    N       = odom_np.shape[0]
    window  = H + 1                                    # samples in each window

    # ---------------- positions --------------------
    pos  = odom_np[:, 1:4]                             # (N,3)
    dist = np.linalg.norm(pos[H:] - pos[:-H], axis=1) # (N-H,)

    # ---------------- yaw series ------------------
    q      = odom_np[:, 4:8]                          # (N,4)
    yaw    = np.array([quat_to_yaw(*row) for row in q])
    yaw_win = sliding_window_view(yaw, window_shape=window)  # (N-H,window)

    yaw0       = yaw_win[:, 0:1]                      # (N-H,1)
    disp_all   = angular_diff(yaw0, yaw_win)          # broadcast → (N-H,window)
    max_dyaw   = np.max(np.abs(disp_all), axis=1)     # (N-H,)

    # ---------------- instantaneous ω -------------
    dy   = angular_diff(yaw_win[:, :-1], yaw_win[:, 1:])     # (N-H,H)
    ts   = odom_np[:, 0]
    dt   = sliding_window_view(ts, window_shape=window)
    dt   = dt[:, 1:] - dt[:, :-1]                     # (N-H,H)
    omega = np.abs(np.degrees(dy) / (dt+1e-5))               # deg/s
    max_omega = np.max(omega, axis=1)                 # (N-H,)

    # # ---- displacement by axis --------------------
    pos_win = sliding_window_view(pos, H, axis=0).transpose(0, 2, 1)
    # 2) displacements relative to first frame: still (N-H,3,W)
    disp    = pos_win - pos_win[:, :1, :]
    # 3) min/max over the window axis → (N-H,3)
    disp_min = disp.min(axis=1)
    disp_max = disp.max(axis=1)
    # 4) stack min/max into last dim → (N-H,3,2)
    disp_by_axis = np.stack((disp_min, disp_max), axis=2)

    return {
        "total_disp": dist, 
        "max_dyaw": max_dyaw, 
        "max_domega": max_omega,                 # each length N-H
        "disp_by_axis": disp_by_axis              # (N-H, 3, 2)
    }

# ── batch Lie–group interpolation without library broadcasting -------------
def interpolate_se3(
    cam_ts:     np.ndarray,        # (F,)
    odo_ts:     np.ndarray,        # (N,)
    odo_xyz:    np.ndarray,        # (N,3)
    odo_q_wxyz: np.ndarray         # (N,4)  qw qx qy qz
) -> np.ndarray:                   # → (F,8) ts x y z  qw qx qy qz
    # 1) Convert odometry poses to SE3 matrices
    T_odo = quat_to_se3(odo_q_wxyz, odo_xyz)             # (N,4,4)

    # 2) Bracket each camera timestamp
    idx0 = np.searchsorted(odo_ts, cam_ts, side="right") - 1
    idx0 = np.clip(idx0, 0, len(odo_ts) - 2)
    idx1 = idx0 + 1
    alpha = ((cam_ts - odo_ts[idx0]) /
             (odo_ts[idx1] - odo_ts[idx0] + 1e-6))[:, None]     # (F,1)

    T0, T1 = T_odo[idx0], T_odo[idx1]                    # (F,4,4)
    # 3) Relative motion & SE3 interpolation
    Delta = se3_inverse_batch(T0) @ T1                   # (F,4,4)
    xi    = se3_log_batch(Delta)                         # (F,6)
    T_d   = se3_exp_batch(alpha * xi)                    # (F,4,4)
    T_cam = T0 @ T_d                                     # (F,4,4)

    # 4) Unpack xyz + quaternion (qw qx qy qz)
    xyz_cam = T_cam[:, :3, 3]
    try:
        q_xyzw  = R.from_matrix(T_cam[:, :3, :3]).as_quat()
    except Exception as e:
        import pdb; pdb.set_trace()
        logging.error(f"Failed to convert rotation matrix to quaternion: {e}")
    q_wxyz  = q_xyzw[:, [3, 0, 1, 2]]
    return np.hstack([cam_ts.reshape(-1, 1), xyz_cam, q_wxyz])                  # (F,7)

def _equidistant_on_polyline(pts: np.ndarray, delta: float) -> np.ndarray:
    """
    Exact equidistant sampling on a polyline via segment interpolation.

    pts   : (M,3) dense polyline vertices
    delta : target spacing in metres

    returns X_eq : (K,3) with ||X_eq[i+1]-X_eq[i]|| == delta (up to fp error),
                   last step may be < delta if total length isn't multiple.
    """
    if len(pts) == 1:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])

    if total < 1e-12:
        return pts[[0]].copy()

    # target distances 0, delta, 2*delta, ..., ≤ total
    k_max = int(np.floor(total / delta))
    targets = delta * np.arange(k_max + 1, dtype=np.float64)
    # ensure we start exactly at 0
    targets[0] = 0.0

    # piecewise-linear interpolation along segments
    Xeq = np.empty((len(targets), 3), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        # find segment index so that cum[j] ≤ s ≤ cum[j+1]
        while j + 1 < len(cum) and s > cum[j + 1]:
            j += 1
        if j + 1 >= len(cum):
            Xeq[i] = pts[-1]
            continue
        denom = (cum[j + 1] - cum[j])
        alpha = 0.0 if denom <= 1e-12 else (s - cum[j]) / denom
        Xeq[i] = pts[j] + alpha * (pts[j + 1] - pts[j])
    return Xeq


def smooth_and_resample_poses(
    interp_poses: np.ndarray,  # (F,8) [ts, x,y,z, qw,qx,qy,qz]
    delta_s: float
) -> np.ndarray:
    """
    Smooth XYZ with a spline (vs. cumulative distance), then take **exact**
    equidistant samples at spacing `delta_s`. Orientations are taken from
    odometry by interpolating in time, but XYZ comes from the equidistant
    samples (so spacing stays exact).
    """
    ts   = interp_poses[:, 0]
    xyz  = interp_poses[:, 1:4]
    quat = interp_poses[:, 4:8]

    # ----- cumulative distance on the *input* path --------------------------
    if len(xyz) < 2:
        return interp_poses.copy()
    seg     = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cumdist = np.concatenate(([0.0], np.cumsum(seg)))
    total   = float(cumdist[-1])
    if total < 1e-12:
        return interp_poses[[0]].copy()

    # ----- smooth XYZ as functions of s (cumulative distance) ---------------
    # small smoothing helps kill jitters; tune s_smooth if you like
    s_smooth = 0.1 # max(0.0, 0.0)
    k = min(3, len(cumdist) - 1)  # spline degree
    sx = UnivariateSpline(cumdist, xyz[:, 0], s=s_smooth, k=k)
    sy = UnivariateSpline(cumdist, xyz[:, 1], s=s_smooth, k=k)
    sz = UnivariateSpline(cumdist, xyz[:, 2], s=s_smooth, k=k)

    # densify the smooth curve to a polyline (for exact arclength on polyline)
    samples_per_meter = 500  # higher → smaller approximation error
    M = max(int(total * samples_per_meter), 1000)
    s_dense = np.linspace(0.0, total, M)
    poly = np.stack([sx(s_dense), sy(s_dense), sz(s_dense)], axis=1)  # (M,3)

    # ----- exact equidistant XYZ on the smooth polyline ---------------------
    X_eq = _equidistant_on_polyline(poly, delta_s)  # (K,3)
    K = len(X_eq)

    # ----- map distances → timestamps for orientation interpolation ---------
    # Use original (cumdist, ts) relation (monotonic). Clamp if smooth path is longer.
    s_targets = delta_s * np.arange(K, dtype=np.float64)
    s_targets = np.minimum(s_targets, cumdist[-1])
    f_ts = interp1d(cumdist, ts, kind="linear", bounds_error=False, fill_value=ts[-1])
    ts_new = f_ts(s_targets)

    # ----- get orientations from odom @ ts_new, but keep equidistant XYZ ----
    poses_oriented = interpolate_se3(
        cam_ts     = ts_new,
        odo_ts     = ts,
        odo_xyz    = xyz,
        odo_q_wxyz = quat
    )  # (K,8): [ts, x,y,z, qw,qx,qy,qz]

    poses_oriented[:, 1:4] = X_eq  # preserve exact spacing positions
    return poses_oriented

# def smooth_and_resample_poses(
#     interp_poses: np.ndarray,  # (F,8) [ts, x,y,z, qw,qx,qy,qz]
#     delta_s: float
# ) -> np.ndarray:
#     """
#     Resample interpolated poses so that each is ~delta_s meters apart,
#     using the original odometry and SE3 interpolation.
#     """
#     # Extract timestamps & positions
#     ts = interp_poses[:, 0]
#     xyz = interp_poses[:, 1:4]
#     qwxyz = interp_poses[:, 4:8]  # (qw, qx, qy, qz) format

#     # Compute cumulative distance along path
#     deltas = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
#     cumdist = np.concatenate(([0.0], np.cumsum(deltas)))
#     total = cumdist[-1]

#     # New sample distances
#     num = max(int(np.floor(total / delta_s)), 1)
#     new_dist = np.linspace(0.0, total, num + 1)

#     # Map distances back to timestamps
#     f_ts = interp1d(cumdist, ts, kind='linear')
#     ts_new = f_ts(new_dist)

#     # Use SE3 interpolation helper for both position & orientation
#     smoothed = interpolate_se3(
#         cam_ts=ts_new,
#         odo_ts=ts,
#         odo_xyz=xyz,
#         odo_q_wxyz=qwxyz
#     )
#     return smoothed

def compute_odometry_goals(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str,
    delta_s: float = 0.1
) -> bool:
    """
    Interpolates odometry data to match camera timestamps, extracts
    pose for each frame in start->end, saves the results to odometry file
    """
    parts = ride_name.split(' ')
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    ride_id = parts[0]
    driveid0 = parts[1]
    driveid1 = parts[2]
    timestamp = "_".join(parts[3:])

    # ---- 1) Load all the data
    root_ride_dir = set_frodo_dir(root_dir, *parts)
    timestamps_path = root_ride_dir / f"front_camera_timestamps_{driveid0}.csv"
    odometry_path = root_ride_dir / f"odometry_data_{driveid0}.csv"
    if not timestamps_path.exists() or not odometry_path.exists():
        logging.error(f"Missing timestamps or odometry file for ride {ride_name}.")
        return False

    cam_ts = load_timestamps(timestamps_path)          # (N_cam,)
    odo    = load_odom(odometry_path)              # (N_odo, 8)
    assert odo.shape[1] == 8, f"Expected odometry shape (N,8), got {odo.shape}"
    
    if cam_ts is None or odo is None:
        logging.error(f"Failed to load timestamps or odometry for ride {ride_name}.")
        return False

    # Extract frame range from start 
    odom_end_frame = min(len(cam_ts) - 1, start_frame + 1000) # Up to 100 seconds in the future
    tgt_ts = cam_ts[start_frame : odom_end_frame]
    n_frames = len(tgt_ts)
    if n_frames == 0:
        logging.warning("No frames in requested range.")
        return False

    # ---- 2) Pre-compute SE(3) objects ---------------------------------
    odo_ts   = odo[:, 0]
    xyz      = odo[:, 1:4]
    qwxyz   = odo[:, 4:] # (qw, qx, qy, qz) quaternion format

    interp_odo = interpolate_se3(
        cam_ts=tgt_ts, 
        odo_ts=odo_ts, 
        odo_xyz=xyz, 
        odo_q_wxyz=qwxyz
    )  # (F,7)  x y z  qw qx qy qz

    # Remove odometry rows that are nans or infs
    invalid_odom = np.any(np.isnan(interp_odo), axis=1) | np.any(np.isinf(interp_odo), axis=1)
    assert not np.any(invalid_odom), "compute_odometry_goals(): Interpolated odometry contains NaNs or Infs."
    if interp_odo.shape[0] != n_frames:
        logging.error(f"Interpolated odometry shape mismatch: expected {n_frames}, got {interp_odo.shape[0]}")
        return False

    smoothed = smooth_and_resample_poses(interp_odo, delta_s)

    # ---- 4) Save -------------------------------------------------------    
    ride_dir = set_frodo_dir(out_dir.parent, *parts) / f"seq_{start_frame}"
    ride_dir.mkdir(parents=True, exist_ok=True)

    out_path = ride_dir / "odometry_info.h5"
    output = {
        "columns": ["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"],
        "timestamp": tgt_ts,
        "poses": interp_odo,
        "smoothed_poses": smoothed,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    hkl.dump(output, out_path, mode="w")

    logging.info(f"Saved interpolated odometry → {out_path}")
    return True