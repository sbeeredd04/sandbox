import cv2
import numpy as np
from typing import Tuple

from scipy.ndimage import distance_transform_edt

from scripts.utils.polyline_utils_smoothed import (
    build_mask_from_crumbs
)

def draw_heading_arrow(
    img_size : Tuple[int, int] | Tuple[int, int, int],
    heading_deg: float,
    *,
    base_px   : Tuple[int, int] = None,     # (x, y) anchor; default center-bottom
    length_px : int           = 150,
    thickness : int           = 15,
    color_bgr : Tuple[int,int,int] = (0, 0, 255),   # red
) -> np.ndarray:
    """
    Draw a planar heading arrow onto a blank RGB image.

    Parameters
    ----------
    img_size    : (H, W) or (H, W, 3)
    heading_deg : float
        0° = straight ahead (upwards in the image), +CCW.
        e.g. +90° points to the image left; −90° to the right.
    base_px     : (x, y) pixel of the arrow tail (shaft base).
                  Default = image centre bottom (W//2, H−1).
    length_px   : arrow length in pixels.
    thickness   : shaft thickness in pixels (cv2 line thickness).
    color_bgr   : RGB/BGR tuple in 0–255.

    Returns
    -------
    np.ndarray : uint8 image shaped (H, W, 3)
    """
    # ------------------------ blank canvas ----------------------------
    if len(img_size) == 2:
        H, W = img_size
    else:
        H, W, _ = img_size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # default base – centre bottom
    if base_px is None:
        base_px = (W // 2, H - 1)

    bx, by = base_px

    # ------------------- convert heading to vector --------------------
    # forward (0°) => (dx, dy) = (0, -length)
    rad = np.deg2rad(heading_deg)
    dx  = -np.sin(rad) * length_px
    dy  = -np.cos(rad) * length_px          # minus because y grows downward

    tip_x = int(round(bx + dx))
    tip_y = int(round(by + dy))

    # clip tip inside image
    tip_x = max(0, min(W-1, tip_x))
    tip_y = max(0, min(H-1, tip_y))

    # ------------------------ draw arrow ------------------------------
    cv2.arrowedLine(
        img,
        (bx, by),
        (tip_x, tip_y),
        color_bgr,
        thickness,
        tipLength=0.2,          # relative tip size
    )

    return img

def sample_path_mask(path_dict, min_windows=-1):
    """
    Build a binary path mask from crumbs inside ``path_dict``.

    Args:
        path_dict : dict
            Contains at least:
               visibility  – [1,T,N] or [T,N]  bool
               tracks      – [1,T,N,2] or [T,N,2] float
               crumbs      – [1,N,?]   with crumbs[...,0] = time-stamp
               sides       – [1,N]     int (-1 / +1)
        min_windows : int | None
            Minimum number of unique time indices to include.

    Returns:
        tuple[np.ndarray, int]
            uint8 mask of shape ``image_dims`` with 0/255 values,
            and number of unique time indices used.
    """
    mask_dims = path_dict['path_mask'].shape[:2]  # [H,W]
    image_dims = (mask_dims[0], mask_dims[1], 3)

    # ---------- unpack tensors (strip batch dim if present) ----------
    vmask = path_dict["visibility"]
    tracks = path_dict["tracks"]
    crumbs = path_dict["crumbs"]
    sides  = path_dict["sides"]

    if tracks.ndim == 4:
        tracks = tracks[0]
    if vmask.ndim == 3:
        vmask = vmask[0]
    if crumbs.ndim == 3:
        crumbs = crumbs[0]
    if sides.ndim == 2:
        sides = sides[0]

    # keep only crumbs visible in the *last* frame --------------------
    last_visible = vmask[-1] > 0
    if not np.any(last_visible):
        return np.zeros(image_dims, dtype=np.uint8), 0

    tracks_vis   = tracks[-1, last_visible, :]
    crumb_times  = crumbs[last_visible, 0]
    crumb_sides  = sides[last_visible]

    unique_ts = -np.sort(-np.unique(crumb_times))  # descending order
    span_len  = len(unique_ts)
    if span_len == 0:
        return np.zeros(image_dims, dtype=np.uint8), 0

    # Determine number of timestamps to use (randomly in [min_windows, span_len])
    min_windows = min_windows if min_windows > 0 else span_len
    max_possible = max(min_windows, span_len)
    num_to_use = np.random.randint(min_windows, max_possible + 1)
    chosen_times = unique_ts[:num_to_use]  # take latest `num_to_use` timestamps
    keep = np.isin(crumb_times, chosen_times)
    if not np.any(keep):
        return np.zeros(image_dims, dtype=np.uint8), 0

    tracks_sub      = tracks_vis[keep, :]
    crumb_times_sub = crumb_times[keep]
    sides_sub       = crumb_sides[keep]

    tmp_img = np.zeros(image_dims, dtype=np.uint8)
    mask_dict = build_mask_from_crumbs(
        tmp_img,
        tracks_sub,
        crumb_times_sub,
        sides_sub,
    )
    assert mask_dict["success"], "Mask building failed."

    return mask_dict["mask"], num_to_use

def get_closest_entity_mask(video_segments: dict[str, np.ndarray],
                             path_mask: np.ndarray) -> str:
    """
    Return the key whose mask is closest to the *front-most* (small-y)
    part of `path_mask`.

    • “Front-most” = pixels of the path that have the **smallest y row**
      (0 = top of image).
    • Distance metric = mean Euclidean distance (in pixels) from every
      ‘on’ pixel of the candidate mask to the nearest ‘end-of-path’
      pixel. Lower is better.
    """
    if path_mask.sum() == 0:
        raise ValueError("path_mask contains no foreground pixels")

    # ---- 1) isolate the path END (smallest y) -------------------------
    ys, _ = np.where(path_mask)
    y_min = ys.min()                     # smallest row index
    end_mask = np.zeros_like(path_mask, dtype=bool)
    end_mask[y_min] = path_mask[y_min]   # only that row

    # distance map to the path end
    dist_map = distance_transform_edt(~end_mask)   # (H,W) float32

    # ---- 2) score every non-empty candidate mask ----------------------
    best_key, best_score = None, np.inf
    for k, m in video_segments.items():
        if m.sum() == 0:
            continue                     # skip empty
        score = dist_map[m[0]].mean()    # m is stored with shape (1,H,W)
        if score < best_score:
            best_key, best_score = k, score

    if best_key is None:
        raise RuntimeError("No non-empty entity masks found")
    return best_key, best_score

# def blend_mask(
#     image: np.ndarray,
#     mask: np.ndarray,
#     color: tuple = (51, 255, 255),
#     alpha: float = 0.5
# ) -> np.ndarray:
#     """
#     Blends a mask onto an image with a specified color and transparency.

#     Args:
#         image:  H×W×3 uint8 RGB image.
#         mask:   H×W bool mask array.
#         color:  RGB color for the mask overlay.
#         alpha:  Transparency level (0–1).

#     Returns:
#         Blended H×W×3 uint8 image.
#     """
#     colored_mask = image.copy()
#     # Use the mask as the alpha value for blending
#     colored_mask[mask] = color
#     blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
#     return blended

def blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (51, 255, 255),
    alpha: float | None = None,
) -> np.ndarray:
    """
    Blend a mask onto an image. If `alpha` is None, use `mask` as a per-pixel alpha map;
    otherwise use constant alpha (original behavior).

    Args:
        image: H×W×3 uint8 RGB image.
        mask:  H×W mask. bool or numeric; if numeric, interpreted in [0,1] (or [0,255]).
        color: RGB color for the overlay.
        alpha: If not None, constant transparency in [0,1]. If None, use per-pixel alpha from `mask`.

    Returns:
        H×W×3 uint8 blended image.
    """
    # Ensure float32 for math
    img_f = image.astype(np.float32)

    if alpha is not None:
        # --- Original constant-alpha path ---
        colored = img_f.copy()
        colored[mask.astype(bool)] = np.array(color, dtype=np.float32)
        out = cv2.addWeighted(img_f, 1 - float(alpha), colored, float(alpha), 0.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- Per-pixel alpha path ---
    # Build alpha map in [0,1]
    if mask.dtype == bool:
        a = mask.astype(np.float32)
    else:
        a = mask.astype(np.float32)
        if a.max() > 1.0:  # accept 0..255 as well
            a = a / 255.0
        a = np.clip(a, 0.0, 1.0)
    a = a[..., None]  # H×W×1 for broadcasting

    # Construct color image
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    color_img = np.broadcast_to(color_arr, img_f.shape)

    # Per-pixel blend: out = (1-a)*img + a*color
    out = img_f * (1.0 - a) + color_img * a
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # out = draw_heading_arrow(
    #     (288, 512),
    #     heading_deg=-90,        # 45° left-forward
    #     length_px=150,
    #     thickness=15,
    #     color_bgr=(0, 0, 255)  # red arrow
    # )
    out = np.zeros((288, 512, 3), dtype=np.uint8)
    status = cv2.imwrite("./scripts/inference/assets/arrow_demo.jpg", out)
    print(f"Arrow image saved: {status}")

