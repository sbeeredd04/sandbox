import cv2
import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path
from profilehooks import timecall

import av

from skimage import measure          # find_contours
from scipy.interpolate import splprep, splev
import shapely.geometry as geom

import torch
import kornia as K
import kornia.filters as KF
import kornia.enhance as KE

def save_video_np(
        video: np.ndarray, path: str, fps: int = 15,
        codec: str = 'h264', quality: int = 20
    ) -> None:
    """
    video: (N, H, W, 3) uint8 ndarray
    """
    path = Path(path)
    assert video.ndim == 4 and video.shape[-1] == 3, "video must be (N, H, W, 3)"
    assert video.dtype == np.uint8, "video must be uint8"
    assert path.suffix in ['.mp4', '.avi', '.mov'], \
        "path must end with .mp4, .avi, or .mov"

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(path, mode='w', format='mp4')
    stream = container.add_stream(codec, rate=fps, options={
        'preset': 'medium',  # 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
        'crf': str(quality),  # Constant Rate Factor (0-51, lower is better quality)
    })
    stream.width = video.shape[2]
    stream.height = video.shape[1]
    stream.pix_fmt = 'yuv420p'

    for frame in video:
        frame_av = av.VideoFrame.from_ndarray(frame, format='rgb24')
        packet = stream.encode(frame_av)
        container.mux(packet)

    # flush the encoder
    packet = stream.encode()
    if packet:
        container.mux(packet)

    container.close()
    return True

def gray_world_white_balance_no_clip(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gray‐world white balance that also guarantees no channel will clip above 1.0.

    Args:
        x  (B, 3, H, W)  float in [0, 1].

    Returns:
        x_bal (B, 3, H, W)  float in [0, 1], white‐balanced but never clipped.
    """
    B, C, H, W = x.shape
    assert C == 3, "Expected 3 channels (RGB)"

    # 1) compute per‐channel mean over H×W:
    mean_rgb = x.mean(dim=[2, 3], keepdim=True)  # (B, 3, 1, 1)
    mean_gray = mean_rgb.mean(dim=1, keepdim=True)  # (B, 1, 1, 1)

    # 2) “ideal” gray‐world scale:
    scale_gray = mean_gray / (mean_rgb + eps)       # (B, 3, 1, 1)

    # 3) compute per‐channel max over H×W:
    max_rgb = x.amax(dim=[2, 3], keepdim=True)    # (B, 3, 1, 1)

    # 4) channel‐wise scale so that max*scale ≤ 1.0  ⟹  scale_max = 1.0 / max_rgb 
    scale_max = 1.0 / (max_rgb + eps)             # (B, 3, 1, 1)

    # 5) final scale = min(scale_gray, scale_max), broadcasting over (B,3,1,1):
    scale = torch.minimum(scale_gray, scale_max)  # (B, 3, 1, 1)

    # 6) apply and clamp in [0,1]:
    x_bal = x * scale
    return x_bal.clamp(0.0, 1.0)

def enhance_ground_batch(
        img_bchw: torch.Tensor,
        do_whitebalance: bool = True,
        clahe_clip: float = 2.0,
        bilateral: bool = False,
        unsharp_radius: int = 5,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 1.0,
        gamma: float | None = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    img_bchw  :  (B,3,H,W) float32 / float64 in [0,1]  (RGB order).
    do_whitebalance : apply simple “gray-world” white balance first.
    clahe_clip      : clip-limit for CLAHE (0 = skip).
    bilateral       : apply bilateral blur (edge-preserving denoise).
    unsharp_*       : parameters for un-sharp mask.
    gamma           : final gamma correction (None → skip).

    Returns
    -------
    img_enh : (B,3,H,W)  – same dtype / device as input.
    """
    assert img_bchw.ndim == 4 and img_bchw.shape[1] == 3, \
        "expect (B,3,H,W) RGB tensor"

    x = img_bchw

    # (1) white balance ---------------------------------------------------
    if do_whitebalance:
        x = gray_world_white_balance_no_clip(x)

    # (2) CLAHE on Y channel ---------------------------------------------
    if clahe_clip > 0:
        ycbcr = K.color.rgb_to_ycbcr(x)
        y, cb, cr = ycbcr[:, :1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        y_eq = KE.equalize_clahe(y, clip_limit=clahe_clip)
        x = K.color.ycbcr_to_rgb(torch.cat([y_eq, cb, cr], dim=1))

    # (3) edge-preserving denoise ----------------------------------------
    if bilateral:
        # kernel_diameter=9 , sigma_color=0.1 , sigma_space=2 ‒ tune if needed
        x = KF.bilateral_blur(x, kernel_size=(9, 9),
                              sigma_color=0.1, sigma_space=2.0)

    # (4) un-sharp mask (sharpen) ----------------------------------------
    sharpen = KF.UnsharpMask(
        kernel_size=(unsharp_radius, unsharp_radius),
        sigma=(unsharp_sigma, unsharp_sigma),
    )
    x = sharpen(x)

    # (5) optional gamma --------------------------------------------------
    if gamma is not None:
        x = KE.adjust_gamma(x, gamma)

    # clamp – Kornia filters may produce tiny overshoots
    return x.clamp(0.0, 1.0)

def local_eq(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

@timecall(immediate=False)
def sift_in_mask(
    video_rgb: np.ndarray,
    n_samples: int,
    seg_mask: Optional[np.ndarray] = None,
    sift_kwargs: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract up to `n_samples` SIFT keypoints per frame, using a single pass:
     - If seg_mask is all ones (or None), we enforce half-left/half-right sampling.
     - Otherwise we simply take the top-n responses within the mask.
    """
    assert video_rgb.ndim == 4 and video_rgb.shape[-1] == 3
    N, H, W, _ = video_rgb.shape
    assert n_samples > 0

    # Prepare the mask flag
    if seg_mask is None:
        use_full_mask = True
    else:
        assert seg_mask.shape == (N, H, W)
        use_full_mask = bool(seg_mask.all())

    # Default SIFT args
    default_kwargs = {
        "nfeatures": 0,
        "edgeThreshold": 15,
        "sigma": 1.6,
        "nOctaveLayers": 5,
        "contrastThreshold": 0.015,
    }
    sift = cv2.SIFT_create(**(sift_kwargs or default_kwargs))

    # Outputs
    coords   = np.full((N, n_samples, 2), np.nan, dtype=np.float32)
    sides    = np.zeros((N, n_samples),       dtype=np.float32)
    coverage = np.zeros(N, dtype=int)

    for i in range(N):
        # 1) grayscale + local equalization
        gray = cv2.cvtColor(video_rgb[i], cv2.COLOR_RGB2GRAY)
        gray = local_eq(gray)

        # 2) choose mask
        mask_arg = None if use_full_mask else seg_mask[i].astype(np.uint8)

        # 3) detect & compute
        kps, _ = sift.detectAndCompute(gray, mask_arg)
        if not kps:
            continue

        # 4) pts & responses
        pts       = np.array([kp.pt for kp in kps], dtype=np.float32)      # (M,2)
        responses = np.array([kp.response for kp in kps], dtype=np.float32) # (M,)

        M = pts.shape[0]
        if use_full_mask:
            # ---- split into left/right ----
            left_idx  = np.where(pts[:,0] < (W/2))[0]
            right_idx = np.where(pts[:,0] >= (W/2))[0]

            n_left  = n_samples // 2
            n_right = n_samples - n_left

            # sort each side by response
            left_sorted  = left_idx[np.argsort(responses[left_idx])[::-1]]
            right_sorted = right_idx[np.argsort(responses[right_idx])[::-1]]

            pick_left  = left_sorted[:n_left]
            pick_right = right_sorted[:n_right]

            picked = np.concatenate([pick_left, pick_right])
            # if we still have slots, fill from the remaining best
            if picked.size < n_samples:
                remaining = np.setdiff1d(np.argsort(responses)[::-1], picked, assume_unique=True)
                to_fill   = n_samples - picked.size
                picked    = np.concatenate([picked, remaining[:to_fill]])

        else:
            # ---- unrestricted top-n ----
            order  = np.argsort(responses)[::-1]
            picked = order[:n_samples]

        # 5) write out
        Kkeep = min(picked.size, n_samples)
        selected_pts = pts[picked[:Kkeep]]
        coords[i, :Kkeep]    = selected_pts
        coverage[i]          = Kkeep
        sides[i, :Kkeep]     = np.where(selected_pts[:, 0] < (W/2), -1.0, 1.0)

    return {"keyps": coords, "sides": sides, "coverage": coverage}

def sample_edge_points(mask: np.ndarray,
                       n_samples: int=100,
                       dtype=np.float32) -> np.ndarray:
    """
    Uniformly sample `n_samples` points along the outer contour of a binary mask.

    Parameters
    ----------
    mask : np.ndarray (H, W)
        Binary image (dtype uint8 / bool / int) whose *non-zero* pixels represent the object.
    n_samples : int
        How many points to return.
    dtype : np.dtype, optional
        Data-type of the returned coordinates (default: np.float32).

    Returns
    -------
    pts : np.ndarray, shape (n_samples, 2)
        Sampled contour points **in (x, y) order**.  If `n_samples == 0`
        or the mask is empty, an empty array is returned.

    Notes
    -----
    * The points are sampled **uniformly with respect to arc–length**,
      i.e. equal distance along the contour, not equal angle.
    * If the mask has several disconnected components the largest one is used.
    * Requires OpenCV (≥4.0).
    """
    if n_samples <= 0:
        return np.empty((0, 2), dtype=dtype)

    # 1. find contour(s)
    cnts, _ = cv2.findContours(mask.astype('uint8'),
                               mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_NONE)

    if not cnts:                                           # empty mask
        return np.empty((0, 2), dtype=dtype)

    # choose the longest contour (most points)
    contour = max(cnts, key=lambda c: c.shape[0]).squeeze(1)   # (L, 2)

    if contour.shape[0] < 2:
        return np.repeat(contour.astype(dtype)[None, :], n_samples, axis=0)

    # 2. arc-length parameterisation
    deltas     = np.diff(contour.astype(np.float32), axis=0)
    seg_len    = np.hypot(deltas[:, 0], deltas[:, 1])
    cum_len    = np.concatenate(([0.0], np.cumsum(seg_len)))   # size L
    total_len  = cum_len[-1]

    #  closed contour → include last segment (from last to first)
    last_delta = contour[0] - contour[-1]
    last_len   = float(np.hypot(*last_delta))
    total_len += last_len
    cum_len    = np.append(cum_len, total_len)

    # 3. target arc-lengths equally spaced
    target_s   = np.linspace(0.0, total_len, n_samples, endpoint=False)

    # 4. interpolate each target length
    pts = np.empty((n_samples, 2), dtype=np.float32)
    for i, s in enumerate(target_s):
        idx = np.searchsorted(cum_len, s, side='right') - 1
        s0  = cum_len[idx]
        s1  = cum_len[idx + 1] if idx + 1 < len(contour) else total_len
        p0  = contour[idx % len(contour)].astype(np.float32)
        p1  = contour[(idx + 1) % len(contour)].astype(np.float32)

        # linear interpolation factor (0 ≤ t < 1)
        if s1 == s0:          # degenerate (zero-length) segment
            pts[i] = p0
        else:
            t      = (s - s0) / (s1 - s0)
            pts[i] = (1 - t) * p0 + t * p1

    return pts.astype(dtype)

def smooth_contour(mask: np.ndarray,
                           smooth_lambda: float = 400.0,
                           num_ctrl_pts: int = 1200) -> geom.Polygon:
    """
    Return a Shapely polygon approximating `mask` with a *smooth* closed
    cubic B-spline.

    Parameters
    ----------
    mask           (H,W) np.uint8 / bool
    smooth_lambda  SciPy `splprep` ‘s’ parameter (0 == interpolate exactly)
    num_ctrl_pts   how many samples to output along the smoothed curve

    Returns
    -------
    smooth_mask (H, W) np.uint8 / bool
    """
    if mask.ndim != 2 or not np.any(mask):
        return geom.Polygon()

    # 1) raw marching-squares contour  (row,col) → (x,y)
    contour = measure.find_contours(mask.astype(np.uint8), 0.5, fully_connected='low')[0]
    xy      = contour[:, ::-1].T                       # 2×N  (x row)

    # 2) periodic cubic B-spline fit in *parametric* form x(t),y(t)
    tck, _  = splprep(xy, s=smooth_lambda, per=True, k=3)   # per=True -> closed
    u       = np.linspace(0, 1, num_ctrl_pts, endpoint=False)
    x, y    = splev(u, tck)

    poly    = geom.Polygon(np.c_[x, y]).buffer(0)      # buffer(0) fixes tiny topo errs
    
    smooth_mask = np.zeros(mask.shape, dtype=np.uint8)
    try:
        if not poly.is_empty:
            # rasterise the polygon to a mask
            pts = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(smooth_mask, [pts], 255)
        else:
            smooth_mask = mask.copy()  # return original mask if empty polygon
    except AttributeError:
        logging.warning("Failed to rasterise polygon, returning empty mask.")
        smooth_mask = mask.copy()

    return smooth_mask

def apply_intrinsics(
    video: np.ndarray,
    intrinsics: Dict[str, np.ndarray],
):
    """
    Apply camera intrinsics to a video,converting to undistorted coordinates.
    video: (N, H, W, 3) RGB uint8 ndarray
    intrinsics: Dict with keys 'camera_matrix', 'distortion_coefficients',
                'image_height', 'image_width', 'camera_name'.
    """

    assert video.ndim == 4 and video.shape[-1] == 3, "video must be (N, H, W, 3)"
    assert intrinsics['camera_matrix'].shape == (3, 3), "camera_matrix must be (3, 3)"
    assert intrinsics['distortion_coefficients'].shape == (1, 5), "distortion_coefficients must be (1, 5)"

    camera_matrix = intrinsics['camera_matrix']
    dist_coeffs = intrinsics['distortion_coefficients'].flatten()

    # Create undistort map
    h, w = intrinsics['image_height'], intrinsics['image_width']
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )
    
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Apply undistortion to each frame
    undistorted_video = np.zeros_like(video)
    for i in range(video.shape[0]):
        undistorted_video[i] = cv2.remap(video[i], map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_video
