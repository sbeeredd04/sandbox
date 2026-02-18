import logging
import numpy as np
from collections import deque
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage

import rospy
import rosbag
import tf2_ros
from sensor_msgs.msg import CompressedImage, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import Path
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion

import torch
import torch.nn as nn
from torchvision import transforms

from spinflow.util.vis_utils import (
    densify_points,
    draw_xyz_on_image
)
from spinflow.util.path_utils import (
    blend_mask
)

def compressed_imgmsg_to_pil(image_msg: CompressedImage) -> PILImage:
    """
    Convert a ROS1 sensor_msgs/CompressedImage to a PIL.Image (RGB).
    Handles JPEG/PNG, grayscale, BGR, and BGRA inputs.
    """
    # 1) Decode compressed bytes with OpenCV
    buf = np.frombuffer(image_msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) # In BGR format for some reason
    if img is None:
        raise ValueError("cv2.imdecode failed (data may be corrupt or empty)")

    # 2) Normalize to RGB for PIL
    if img.ndim == 2:
        # Grayscale → L → RGB
        return PILImage.fromarray(img, mode="L").convert("RGB")
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 3:
            # BGR → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(rgb)
        elif c == 4:
            # BGRA → RGBA → drop alpha to RGB (or keep RGBA if you prefer)
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return PILImage.fromarray(rgba).convert("RGB")
        else:
            raise ValueError(f"Unsupported channel count: {c}")
    else:
        raise ValueError(f"Unexpected image shape from imdecode: {img.shape}")

def pil_to_compressed_imgmsg(
    pil_img,
    frame_id: str,
    stamp: rospy.Time = None,
    fmt: str = "jpeg",
    quality: int = 90
) -> CompressedImage:
    """
    Convert a PIL Image to a ROS CompressedImage via OpenCV encoding.

    Args:
        pil_img (PIL.Image.Image): Input PIL image (RGB).
        frame_id (str): TF frame ID to set in the header.
        stamp (rospy.Time, optional): Timestamp for the header. Defaults to now.
        fmt (str, optional): "jpeg" or "png". Defaults to "jpeg".
        quality (int, optional): JPEG quality (0-100) or PNG compression (0-9). Defaults to 90.

    Returns:
        CompressedImage: The ROS message ready to publish.
    """
    # 1) Convert PIL→numpy (RGB) then to BGR for OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 2) Choose extension and encode params
    ext = ".jpg" if fmt.lower().startswith("j") else ".png"
    if ext == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        png_level = max(0, min(9, quality // 10))
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_level]

    # 3) Encode to memory buffer
    success, buf = cv2.imencode(ext, cv_img, params)
    if not success:
        raise RuntimeError(f"cv2.imencode failed for format {fmt}")

    # 4) Build and return CompressedImage
    msg = CompressedImage()
    msg.header = Header(frame_id=frame_id, stamp=stamp or rospy.Time.now())
    msg.format = fmt
    msg.data = buf.tobytes()
    return msg

def intrinsics_to_camera_info_msg(
    intr: dict,
    frame_id: str,
    stamp: rospy.Time = None,
    *,
    width: int = None,
    height: int = None,
    base_msg: CameraInfo = None,
) -> CameraInfo:
    """
    Build a sensor_msgs/CameraInfo from an intrinsics dict, mirroring the style of
    pil_to_compressed_imgmsg (explicit header, minimal magic, safe defaults).

    Args:
        intr (dict): Intrinsics dictionary. Expected keys (if available):
            - 'K' : (3,3) or flat(9)
            - 'P' : (3,4) or flat(12)
            - 'R' : (3,3) or flat(9)  (optional; defaults to identity if not provided)
            - 'D' : (N,) distortion coefficients (optional; defaults to [])
            - 'dist_model' or 'distortion_model' : string (optional; defaults to "plumb_bob")
            - 'image_width', 'image_height' : ints (optional; used as fallback size)
        frame_id (str): TF frame for the header.
        stamp (rospy.Time, optional): Header stamp (defaults to now()).
        width (int, optional): Output image width (overrides intr['image_width'] if given).
        height (int, optional): Output image height (overrides intr['image_height'] if given).
        base_msg (CameraInfo, optional): If provided, fields will be initialized from this
            message and then overwritten by calibrated values.

    Returns:
        CameraInfo: ROS message ready for publishing.
    """
    # --- 1) Start with a fresh (or base) message
    msg = CameraInfo()
    if base_msg is not None:
        # Copy over fields we may keep if intr doesn't provide them
        msg.height = base_msg.height
        msg.width = base_msg.width
        msg.distortion_model = base_msg.distortion_model
        msg.D = list(base_msg.D)
        msg.K = list(base_msg.K)
        msg.R = list(base_msg.R)
        msg.P = list(base_msg.P)

    # --- 2) Header
    msg.header = Header(frame_id=frame_id, stamp=stamp or rospy.Time.now())

    # --- 3) Image size (prefer explicit args, then intrinsics dict, then current)
    w = int(width if width is not None else intr.get("image_width", msg.width or 0) or 0)
    h = int(height if height is not None else intr.get("image_height", msg.height or 0) or 0)
    if w <= 0 or h <= 0:
        # Keep whatever was in base_msg, otherwise leave as zeroes
        pass
    else:
        msg.width = w
        msg.height = h

    # --- 4) Distortion model
    dist_model = intr.get("dist_model", intr.get("distortion_model", None))
    msg.distortion_model = str(dist_model) if dist_model else (msg.distortion_model or "plumb_bob")

    # --- 5) Numeric helpers (accept (3,3)/(3,4) or flat)
    def _to_list(arr, shape=None, length=None):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64)
        if shape is not None:
            try:
                a = a.reshape(shape)
            except Exception:
                # If reshape fails, fall back to flat
                a = a.ravel()
        if length is not None:
            a = a.ravel()
            if a.size != length:
                # Pad/truncate to required length for safety
                out = np.zeros((length,), dtype=np.float64)
                n = min(length, a.size)
                out[:n] = a[:n]
                return out.tolist()
        return a.ravel().tolist()

    # K (3x3 → 9)
    K_list = _to_list(intr.get("K", None), shape=(3, 3), length=9)
    if K_list is not None:
        msg.K = K_list

    # R (3x3 → 9) — default to identity if not present and not already set
    R_list = _to_list(intr.get("R", None), shape=(3, 3), length=9)
    if R_list is not None:
        msg.R = R_list
    elif not msg.R or sum(msg.R) == 0.0:
        msg.R = np.eye(3, dtype=np.float64).ravel().tolist()

    # P (3x4 → 12)
    P_list = _to_list(intr.get("P", None), shape=(3, 4), length=12)
    if P_list is not None:
        msg.P = P_list

    # D (any length)
    D_list = intr.get("D", None)
    if D_list is not None:
        msg.D = np.asarray(D_list, dtype=np.float64).ravel().tolist()

    # (Optional) binning / ROI could be copied from base_msg if you need them.
    return msg

def rescale_intrinsics(camera_info: dict, image_size: tuple) -> dict:
    """
    Rescale camera intrinsics to a new image size.

    Args:
        camera_info (dict): Original camera info dictionary with 'K', 'D', 'R', 'image_width', 'image_height'.
        image_size (tuple): Target image size as (width, height).

    Returns:
        dict: New camera info dictionary with updated intrinsics.
    """
    W_new, H_new = image_size
    W0 = int(camera_info.get('image_width', W_new))
    H0 = int(camera_info.get('image_height', H_new))
    K0 = np.asarray(camera_info['K'], np.float64)
    P0 = np.asarray(camera_info['P'], np.float64)

    # Scale factors
    sx, sy = W_new / float(W0), H_new / float(H0)
    K = K0.copy()
    K[:2, :] *= np.array([sx, sy]).reshape(2, 1)

    P = P0.copy()
    P[:2, :3] *= np.array([sx, sy]).reshape(2, 1)

    new_camera_info = camera_info.copy()
    new_camera_info['K'] = K
    new_camera_info['P'] = P
    new_camera_info['image_width'] = W_new
    new_camera_info['image_height'] = H_new
    return new_camera_info

def apply_intrinsics_to_image(
    pil_img: PILImage,
    camera_info: dict,
    interpolation: int = PILImage.BILINEAR,
    alpha: float = 0.0                  # 0=tighter crop, 1=keep FOV (may add borders)
) -> PILImage:
    img = np.array(pil_img)
    H, W = img.shape[:2]

    W0 = int(camera_info.get('image_width', W))
    H0 = int(camera_info.get('image_height', H))
    K0 = np.asarray(camera_info['K'], np.float64)
    D  = np.asarray(camera_info['D'], np.float64).ravel()
    R  = np.asarray(camera_info.get('R', np.eye(3)), np.float64)

    # 1) Scale K from calibration size -> current image size
    sx, sy = W / float(W0), H / float(H0)
    K = K0.copy()
    K[:2, :3] *= [sx, sy, 1.0]

    # 2) Center-crop to target aspect (default: keep width, crop height to 16:9)
    target_size = (camera_info.get('image_width'), camera_info.get('image_height'))
    if target_size:
        Wt, Ht = target_size
        tgt_ar = Wt / float(Ht)
    else:
        Wt, Ht = W, int(round(W * 9.0 / 16.0))
        tgt_ar = 16.0 / 9.0

    cur_ar = W / float(H)
    if abs(cur_ar - tgt_ar) > 1e-3:
        if cur_ar > tgt_ar:  # crop width
            newW = int(round(H * tgt_ar)); x0 = (W - newW) // 2
            img = img[:, x0:x0 + newW]; K[0,2] -= x0; W = newW
        else:                # crop height
            newH = int(round(W / tgt_ar)); y0 = (H - newH) // 2
            img = img[y0:y0 + newH, :]; K[1,2] -= y0; H = newH

    # Optional resize to target_size
    interp_map = {PILImage.NEAREST: cv2.INTER_NEAREST, PILImage.BILINEAR: cv2.INTER_LINEAR,
                  PILImage.BICUBIC: cv2.INTER_CUBIC, PILImage.LANCZOS: cv2.INTER_LANCZOS4}
    if target_size and (W, H) != (Wt, Ht):
        img = cv2.resize(img, (Wt, Ht), interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR))
        sx2, sy2 = Wt / float(W), Ht / float(H)
        K[0,0] *= sx2; K[1,1] *= sy2
        K[0,2] *= sx2; K[1,2] *= sy2
        W, H = Wt, Ht

    # 3) Undistort/rectify in one go using OpenCV
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha, (W, H))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, newK, (W, H), cv2.CV_32FC1)
    out = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # If R is identity, you could replace the two lines above with:
    # out = cv2.undistort(img, K, D, None, newK)
    return PILImage.fromarray(out)

def waypoints_to_path_msg(
    outputs: dict[str, torch.Tensor],
    frame_id: str,
    msg_ts: float,
    pred_key: str = "action_pred",
    num_samples: int = 20
):
    """Creates a nav_msgs/Path message from predicted waypoints."""
    actions = outputs[pred_key]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()[0]
    if actions.ndim == 3:
        actions = actions[0]
    assert actions.ndim == 2 and actions.shape[1] == 3, \
        f"Expected actions shape (N, 3), got {actions.shape}"
    
    # Densify waypoints for smoother tracking
    actions = densify_points(actions, num_samples)

    T = actions.shape[0]
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = rospy.Time.from_sec(msg_ts)
    if T == 0:
        return path  # empty

    # Compute planar yaw from forward differences; final uses last segment
    yaws = np.zeros(T, dtype=float)
    if T >= 2:
        diffs = np.diff(actions[:, :2], axis=0)  # (T-1, 2)
        for i in range(T - 1):
            dx, dy = diffs[i]
            if dx == 0.0 and dy == 0.0:
                # if degenerate, copy previous yaw (or 0 if none)
                yaws[i] = yaws[i - 1] if i > 0 else 0.0
            else:
                yaws[i] = np.arctan2(dy, dx)
        # final waypoint tangent = last non-degenerate segment
        last_dx, last_dy = diffs[-1]
        yaws[-1] = np.arctan2(last_dy, last_dx) if (last_dx != 0.0 or last_dy != 0.0) else yaws[-2]
    else:
        yaws[0] = 0.0  # single point: default yaw

    # Fill Path. Orientation uses yaw; position uses (x,y,z)
    poses = []
    for i in range(T):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.seq = i
        ps.header.stamp = path.header.stamp
        ps.pose.position = Point(
            x=float(actions[i, 0]), y=float(actions[i, 1]), z=float(actions[i, 2])
        )
        quat = R.from_euler('z', yaws[i]).as_quat()  # Convert yaw to quaternion
        ps.pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3])
        )
        poses.append(ps)
    path.poses = poses
    return path

def pointcloud2_to_numpy(pointcloud_msg):
    """
    Convert a PointCloud2 message to a numpy array.
    
    Args:
        pointcloud_msg (PointCloud2): The PointCloud2 message.
        
    Returns:
        np.ndarray: The converted numpy array of points.
    """
    return np.array(list(pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True)))


def camera_info_to_dict(cam: CameraInfo) -> dict:
    """
    Convert a CameraInfo message to a dictionary with properly shaped matrices.
    """
    # helper to grab a flat list and reshape
    to_mat = lambda attr, shape: np.array(getattr(cam, attr), float).reshape(shape)

    return {
        'image_height':           cam.height,
        'image_width':            cam.width,
        'distortion_model': cam.distortion_model,
        'K':                 to_mat('K', (3, 3)),
        'D':                 np.asarray(cam.D, float),
        'R':                 to_mat('R', (3, 3)),
        'P':                 to_mat('P', (3, 4)),
        'binning_x':        getattr(cam, 'binning_x', 0),
        'binning_y':        getattr(cam, 'binning_y', 0),
        # Neglect region_of_interest for now
    }

def compute_projection_matrices(
    camera_info_dict: dict, 
    T_cam_base: np.ndarray
) -> np.ndarray:
    """
    Computes a 4x3 and 3x4 projection matrix from optical pixel coordinates
    to base coordinates and vice versa.
    
    Args:
        camera_info_dict (dict): The camera info dictionary.
        T_cam_world (np.ndarray): The transformation matrix from camera to world.
        
    Returns:
        np.ndarray: The 4x3 projection matrix.
    """
    K = camera_info_dict['K']
    R = camera_info_dict['R']
    
    # Compute the 4x3 projection matrix
    T_base_cam = np.linalg.inv(T_cam_base)
    T_canon = np.eye(4)
    T_canon[:3, :3] = R

    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = K[:3, :3]

    T_world_to_optical = P_pix_cam @ T_canon @ T_base_cam

    # Computes 3x4 projection matrix
    T_canon[:3, :3] = np.linalg.inv(R)
    P_pix_cam[:3, :3] = np.linalg.inv(K[:3, :3])
    T_optical_to_world = T_cam_base @ T_canon @ P_pix_cam

    return {
        "intrinsics": camera_info_dict,
        "T_base_to_optical": T_world_to_optical,
        "T_optical_to_base": T_optical_to_world
    }

def tf_to_se3(tf_matrix: tf2_ros.Buffer) -> np.ndarray:
    """
    Convert a tf2_ros transform to a 4x4 SE(3) matrix.
    
    Args:
        tf_matrix (tf2_ros.Buffer): The transform buffer.
        
    Returns:
        np.ndarray: The SE(3) matrix.
    """
    translation = tf_matrix.transform.translation
    rotation = tf_matrix.transform.rotation

    se3_matrix = np.eye(4)
    se3_matrix[:3, 3] = [translation.x, translation.y, translation.z]
    
    # Convert quaternion to rotation matrix
    q = [rotation.x, rotation.y, rotation.z, rotation.w]
    r = R.from_quat(q).as_matrix()
    
    se3_matrix[:3, :3] = r

    return se3_matrix

def pose_to_se3(pose: Pose):
    """
    Convert a Pose message to a 4x4 SE(3) matrix.
    
    Args:
        pose (Pose): The pose message.
        
    Returns:
        np.ndarray: The SE(3) matrix.
    """
    se3_matrix = np.eye(4)
    se3_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    
    # Convert quaternion to rotation matrix
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    r = R.from_quat(q).as_matrix()
    
    se3_matrix[:3, :3] = r

    return se3_matrix

def get_compensated_cloud(
    cloud_ts: np.ndarray,
    clouds: deque,
    target_ts: float,
    tolerance: float = np.inf,
    assume_sorted: bool = True
):
    """Computes synchronized, ego-motion compensated point cloud."""
    ts = np.asarray(cloud_ts, dtype=np.float64)
    if ts.ndim != 1 or ts.size == 0 or len(clouds) != ts.size:
        raise ValueError("cloud_ts must be 1D non-empty and match clouds length")

    def _nearest_index_sorted(ts: np.ndarray, t: float) -> int:
        """Assumes ts is 1D, ascending. Returns index of closest entry to t."""
        j = np.searchsorted(ts, t, side="left")
        if j == 0: return 0
        if j == ts.size: return ts.size - 1
        # pick nearer of neighbors; ties go to the left (j-1)
        return j if (ts[j] - t) < (t - ts[j-1]) else (j - 1)

    idx = (
        _nearest_index_sorted(ts, float(target_ts))
        if assume_sorted
        else int(np.nanargmin(np.abs(ts - float(target_ts))))
    )

    dt = abs(ts[idx] - target_ts)
    logging.info(f"Compensating point cloud at {target_ts} (dt={dt:.3f}s)")
    if dt > tolerance:
        logging.warning(f"Target timestamp {target_ts} is outside tolerance {tolerance}. ")
        return None, None
    
    cloud_np = pointcloud2_to_numpy(clouds[idx])
    return ts[idx], cloud_np

def viz_predictions(
    # model_inputs: dict,
    obs: PILImage,
    model_outputs: dict,
    obs_range: tuple,
    infos: dict,
    obs_key: str = "rgb_image",
    pred_key: str = "action_pred",
    mask_key: str = "path_mask_pred",
    # mask_key: str = "target_pred_cont",
    vis_path: bool = True
):
    """
    predictions: dict[str, torch.Tensor]
    img: PIL.Image
    infos: dict[str, np.ndarray]
    """
    img_np = np.array(obs)
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    if vis_path:
        ann_img = draw_xyz_on_image(
            img_np,
            model_outputs[pred_key],
            infos,
            num_points=100
        )
    else:
        ann_img = img_np.copy()

    if mask_key in model_outputs:
        # Blend the path mask with the annotation image
        # path_mask_th = model_outputs[mask_key] > 0.5
        path_mask_th = model_outputs[mask_key] * 0.5 # Halve probabilities for blending
        H, W = ann_img.shape[:2]
        path_mask_th = transforms.Resize((H, W))(path_mask_th)
        path_mask = path_mask_th.cpu().numpy().squeeze()

        ann_img = blend_mask(ann_img, path_mask)
    
    return PILImage.fromarray(ann_img).convert("RGB")

class ImagePreprocessor(nn.Module):
    """
    Converts a PIL image → Tensor, resizes to (H,W), then remaps [0,1]→[out_min,out_max].
    
    The Normalize layer is configured so that
        x_raw ∈ [0,1] → (x_raw - mean) / std ∈ [out_min, out_max]
    """
    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        out_min: float = -1.0,
        out_max: float = +1.0,
    ):
        super().__init__()
        rng = out_max - out_min
        self.out_min = out_min
        self.out_max = out_max
        self.rng = rng
        # mean = -out_min / rng
        # std  =  1.0   / rng
        self.pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            # transforms.Normalize(
            #     mean=[mean] * channels,
            #     std =[std]  * channels
            # ),
        ])

    def forward(self, img):
        """
        Args:
          img: PIL.Image
        Returns:
          Tensor[1,C,H,W] with values in [out_min, out_max]
        """
        img = self.pipeline(img).unsqueeze(0)  # [1, C, H, W] 
        img = torch.clamp(img, 0, 1)
        img = img * self.rng + self.out_min # Convert to [out_min, out_max]
        return img # [1, C, H, W]

def smooth_robot_actions(
    odom_ts: deque,          # deque[float] or np.ndarray-like
    odom_poses: deque,       # deque[np.ndarray (4x4 SE3)]
    img_ts: deque,           # deque[float] or np.ndarray-like
    action_pred: deque       # deque[np.ndarray (K,3)], latest is current
) -> np.ndarray:
    """
    Odometry-aligned EMA of per-frame predicted trajectories.

    - Take latest image timestamp t_now and predicted (K,3) in *current base* frame.
    - Transform prediction to world with T_w_b(t_now).
    - Roll previous fused world-trajectory forward to the new horizon.
    - EMA-blend with current prediction; lock first τ seconds to avoid twitch.
    - Transform fused world-trajectory back to current base frame; return (K,3).

    Keeps state across calls in smooth_robot_actions._state.
    """
    # ---- tunables ----
    DT = 0.0                 # s between waypoints (model assumption)
    ALPHA = 0.35             # EMA weight for the new plan
    LOCK_S = 0.4             # lock first τ seconds of horizon
    # -------------------

    # ---- helpers ----
    def _as_np(a):
        return np.asarray(list(a), dtype=np.float64) if isinstance(a, deque) else np.asarray(a, dtype=np.float64)

    def _nearest_T_w_b(t_query: float, ts: np.ndarray, Ts: list[np.ndarray]) -> np.ndarray:
        """Nearest-neighbor odom pose; identity if unavailable."""
        if ts.size == 0 or len(Ts) == 0:
            return np.eye(4, dtype=np.float64)
        j = np.searchsorted(ts, t_query, side="left")
        if j == 0: idx = 0
        elif j >= ts.size: idx = ts.size - 1
        else: idx = j if (ts[j] - t_query) < (t_query - ts[j-1]) else (j-1)
        T = np.asarray(Ts[idx], dtype=np.float64)
        return T if T.shape == (4,4) else np.eye(4, dtype=np.float64)

    def _transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 4x4 to (N,3)."""
        pts_h = np.c_[pts, np.ones((len(pts), 1), dtype=np.float64)]
        out = (T @ pts_h.T).T[:, :3]
        return out

    def _roll_forward(prev_traj_w: np.ndarray, prev_t_abs: np.ndarray,
                      t_now: float, new_t_abs: np.ndarray):
        """
        Drop past points (t < t_now), re-interp onto new absolute time grid.
        Returns None,None if insufficient support.
        """
        keep = prev_t_abs >= t_now
        if keep.sum() < 2:
            return None, None
        t_ref = prev_t_abs[keep]
        traj_ref = prev_traj_w[keep]  # (M,3)
        # Interp each coord to new grid (clip to bounds)
        traj_new = np.column_stack([
            np.interp(new_t_abs, t_ref, traj_ref[:, i], left=traj_ref[0, i], right=traj_ref[-1, i])
            for i in range(3)
        ])
        return traj_new, new_t_abs

    # ---- inputs → latest sample ----
    img_ts_np   = _as_np(img_ts)
    odom_ts_np  = _as_np(odom_ts)
    if img_ts_np.size == 0 or len(action_pred) == 0:
        # nothing to smooth; return zeros 8x3 to be safe
        return np.zeros((8,3), dtype=np.float64)

    t_now = float(img_ts_np[-1])
    pred_b = np.asarray(action_pred[-1], dtype=np.float64)  # (K,3) in base frame now
    if pred_b.ndim != 2 or pred_b.shape[1] != 3:
        raise ValueError("Each action_pred entry must be (K,3).")
    K = pred_b.shape[0]
    horizon_abs = t_now + np.arange(K, dtype=np.float64) * DT

    # Get T_w_b(t_now) and its inverse
    T_w_b_now = _nearest_T_w_b(t_now, odom_ts_np, list(odom_poses))
    T_b_w_now = np.linalg.inv(T_w_b_now)

    # Current prediction → world
    pred_w = _transform(T_w_b_now, pred_b)  # (K,3)

    # ---- state (persist across calls) ----
    st = getattr(smooth_robot_actions, "_state", None)
    if st is None or ("t_abs" not in st) or ("traj_w" not in st):
        # bootstrap
        fused_w = pred_w.copy()
        t_abs   = horizon_abs.copy()
    else:
        prev_w  = st["traj_w"]
        prev_t  = st["t_abs"]
        # Time moved backwards? reset.
        if t_now < prev_t.min():
            fused_w = pred_w.copy()
            t_abs   = horizon_abs.copy()
        else:
            prev_aligned, _ = _roll_forward(prev_w, prev_t, t_now, horizon_abs)
            if prev_aligned is None:
                fused_w = pred_w.copy()
                t_abs   = horizon_abs.copy()
            else:
                # EMA blend; lock near-term
                fused_w = (1.0 - ALPHA) * prev_aligned + ALPHA * pred_w
                lock_mask = (horizon_abs - t_now) < LOCK_S
                fused_w[lock_mask] = prev_aligned[lock_mask]
                t_abs = horizon_abs

    # Update persistent state (in world coords, absolute time)
    smooth_robot_actions._state = {"traj_w": fused_w.copy(), "t_abs": t_abs.copy()}

    # Return fused plan in current base frame
    fused_b_now = _transform(T_b_w_now, fused_w)
    return fused_b_now

def save_path_to_file(path_msg: Path, filename: str, path_topic: str):
    """
    Save a nav_msgs/Path message to a rosbag file.
    """
    with rosbag.Bag(filename, 'w') as bag:
        bag.write(path_topic, path_msg)

def load_path_from_file(filename: str, path_topic: str) -> Path:
    """
    Load a nav_msgs/Path message from a rosbag file.
    """
    with rosbag.Bag(filename, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=[path_topic]):
            return msg
    raise RuntimeError("No Path message found in file.")
