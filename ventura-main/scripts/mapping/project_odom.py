

import torch
import hickle as hkl
import numpy as np
from PIL import Image
from spinflow.dataset.frodo_helpers import (set_frodo_dir)
from spinflow.util.projection_utils import (project_xyz_to_pixel)
from spinflow.util.vis_utils import (draw_trajectory_on_image)
from spinflow.util.math_utils import (odom_to_local_pose)
from spinflow.util.path_utils import ( blend_mask )
from spinflow.util.action_utils import (compensate_action)
from scripts.utils.log_utils import logging

def project_odom_single(
    root_dir,
    ride_name: str,
    start_frame: int,
    end_frame: int,
    *,
    x_offset: float = 0.27,
    z_offset: float = -0.576,
    horizon: int = 40,
    color: tuple = (51, 255, 255),
    save_image: bool = False
):
    """
    Projects odometry to image and computes alignment
    with path mask
    """
    parts = ride_name.split(' ')
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    root_ride_dir = set_frodo_dir(root_dir, *parts)
    seq_dir = root_ride_dir / f"seq_{start_frame}"

    info_path = seq_dir / f"ride_info.h5"
    odom_path = seq_dir / f"odometry_info.h5"
    mask_path = seq_dir / f"path_tracker.h5"
    for path in [info_path, odom_path, mask_path]:
        if not path.exists():
            logging.error(f"Missing file {path}. Cannot project odometry.")
            return 0

    info_dict = hkl.load(info_path)
    odom_dict = hkl.load(odom_path)
    mask_dict = hkl.load(mask_path)

    mask_np = mask_dict['path_mask'] 
    odom_np = odom_to_local_pose(odom_dict['smoothed_poses'], mode="se3")
    xyz_np = compensate_action(
        odom_np,
        info_dict,
        x_offset=x_offset,
        z_offset=z_offset
    )[:horizon] # [N, 3]

    pixel_uv = project_xyz_to_pixel(
        xyz_np,
        info_dict['intrinsics'],
        info_dict['T_optical_to_base']
    )
    
    H, W = mask_np.shape[:2]
    mask = (mask_np[..., 0] if mask_np.ndim == 3 else mask_np).astype(bool)
    if pixel_uv is None or len(pixel_uv) == 0:
        logging.warning("No valid pixel coordinates found in projection.")
        return 0

    uv = np.asarray(pixel_uv, dtype=np.float64)
    uv = np.rint(uv).astype(np.int32)  # Round to nearest pixel
    u, v = uv[:, 0], uv[:, 1]

    # keep only finite, in-bounds points
    valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
    if not np.any(valid):
        return 0
    u, v = u[valid], v[valid]

    # Optional de-dup (avoids over-counting multiple points landing on same pixel)
    # pairs = np.stack([vi, ui], axis=1)
    # pairs = np.unique(pairs, axis=0)
    # hits  = mask[pairs[:, 0], pairs[:, 1]]

    hits = mask[v, u]
    pct_on_mask = float(np.count_nonzero(hits)) / hits.size

    if save_image:
        # Optional: save debug image
        rgb_np = mask_dict['front_rgb']
        debug_image = blend_mask(rgb_np, mask_np, alpha=0.5)
        debug_image = draw_trajectory_on_image(
            debug_image,
            uv,
            color=color,
            thickness=2,
        )
        # Overlay mask on debug image
        Image.fromarray(debug_image).save("test.jpg")
    return pct_on_mask

