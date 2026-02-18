import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp_batch(q0, q1, t):
    """
    Vectorized quaternion slerp.
    q0,q1: (M,4) quaternions in (qw,qx,qy,qz)
    t    : (M,) interpolation factor in [0,1]
    returns (M,4)
    """
    # normalise
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True).clip(1e-12, None)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True).clip(1e-12, None)

    dot = np.sum(q0 * q1, axis=1)
    # take shortest path
    flip = dot < 0.0
    q1[flip] = -q1[flip]
    dot[flip] = -dot[flip]

    EPS = 1e-7
    # close to 1 → lerp
    lin = dot > 1.0 - EPS
    out = np.empty_like(q0)

    # linear (safe)
    if np.any(lin):
        tl = t[lin][:, None]
        out[lin] = q0[lin] + tl * (q1[lin] - q0[lin])
        out[lin] /= np.linalg.norm(out[lin], axis=1, keepdims=True).clip(1e-12, None)

    # slerp
    if np.any(~lin):
        d   = dot[~lin]
        th  = np.arccos(d)
        sth = np.sqrt(1.0 - d * d)
        tt  = t[~lin]
        w0  = np.sin((1.0 - tt) * th) / sth
        w1  = np.sin(tt * th) / sth
        out[~lin] = (q0[~lin] * w0[:, None]) + (q1[~lin] * w1[:, None])
        out[~lin] /= np.linalg.norm(out[~lin], axis=1, keepdims=True).clip(1e-12, None)

    return out

def se2_matrix(x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Given arrays x, y, yaw of shape (N,), return an array of SE(2) mats shape (N,3,3).
    """
    N = x.shape[0]
    c = np.cos(yaw)
    s = np.sin(yaw)

    T = np.zeros((N, 3, 3), dtype=float)
    # rotation
    T[:, 0, 0] = c
    T[:, 0, 1] = -s
    T[:, 1, 0] = s
    T[:, 1, 1] = c
    # translation
    T[:, 0, 2] = x
    T[:, 1, 2] = y
    # homogeneous row
    T[:, 2, 2] = 1.0

    return T

def se3_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """
    Given arrays x, y, z of shape (N,) and quaternion q_wxyz of shape (N, 4),
    return an array of SE(3) mats shape (N,4,4).
    """
    N = x.shape[0]
    Rt = R.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # Convert to SciPy's (x y z w) order

    T = np.zeros((N, 4, 4), dtype=float)
    T[:, :3, :3] = Rt
    T[:, :3, 3] = np.column_stack([x, y, z])
    T[:, 3, 3] = 1.0

    return T

def se3_to_odom(odom: np.ndarray) -> np.ndarray:
    """
    Convert an SE(3) pose to odometry format [x, y, z, qw, qx, qy, qz].
    Parameters
    ----------
    odom : (N, 4, 4) ndarray
        SE(3) poses.

    Returns
    -------
    out : (N, 8) ndarray
        Odometry format [x, y, z, qw, qx, qy, qz].
    """
    if odom.shape[1:] != (4, 4):
        raise ValueError("odom must be of shape (N, 4, 4)")

    xyz = odom[:, :3, 3]  # Extract translation
    Rt = odom[:, :3, :3]  # Extract rotation matrix
    q_xyzw = R.from_matrix(Rt).as_quat()
    
    return np.column_stack([xyz, q_xyzw[:, [3, 0, 1, 2]]]) 

def odom_to_local_pose(
    odom: np.ndarray,
    mode: str = "se3",
) -> np.ndarray:
    """
    Convert a global odometry trace to **local coordinates w.r.t. the first pose**.

    Parameters
    ----------
    odom : (N,8) ndarray
        Global poses ordered `[x  y  z  qw  qx  qy  qz]`
        (scalar–first quaternion, right-handed).
    mode : {"se2","se3"}, default="se2"
        • "se2" → return local **(x,y,yaw)** – identical to the old function.  
        • "se3" → return local **(x,y,z,qw,qx,qy,qz)**.

    Returns
    -------
    out :
        - If *mode="se2"*  → shape **(N,3)**  = `[x_local, y_local, yaw_local]`
        - If *mode="se3"*  → shape **(N,7)**  = `[x_local, y_local, z_local, qw, qx, qy, qz]`
    """
    if odom.shape[1] != 8:
        raise ValueError("odom must be (N,8) with [ts, x y z qw qx qy qz]")

    if mode not in {"se3"}:
        raise ValueError("mode must be 'se3'")

    odom_ts   = odom[:, 0:1]          # (N,1)
    xyz_world = odom[:, 1:4]          # (N,3)
    q_wxyz    = odom[:, 4:]          # (N,4) scalar-first

    # ------------------------------------------------------------------ #
    #                      3-D  (x,y,z,q)   branch                       #
    # ------------------------------------------------------------------ #
    # Build SE(3) matrices
    Tt_world = se3_matrix(
        xyz_world[:, 0], xyz_world[:, 1], xyz_world[:, 2], q_wxyz
    )                                                # (N,4,4)

    T0_inv   = np.linalg.inv(Tt_world[0])
    Tt_local = T0_inv @ Tt_world                     # (N,4,4)

    # local translation
    xyz_local = Tt_local[:, :3, 3]                   # (N,3)

    # local rotation → quaternion (scalar-first)
    Rt_local  = Tt_local[:, :3, :3]
    q_xyzw_local = R.from_matrix(Rt_local).as_quat() # (N,4) xyzw
    q_wxyz_local = q_xyzw_local[:, [3, 0, 1, 2]]     # back to wxyz

    return np.column_stack([odom_ts, xyz_local, q_wxyz_local])