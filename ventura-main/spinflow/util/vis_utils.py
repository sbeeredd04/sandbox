import cv2
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Tuple, Union, Dict

ArrayLike = Union[np.ndarray, torch.Tensor]

from spinflow.util.projection_utils import (
    project_xyz_to_pixel
)

def draw_satellite_on_image(
    img: ArrayLike,
    sat: Union[ArrayLike, Dict[str, ArrayLike]],
    *,
    circle_center: Tuple[float, float] | Tuple[int, int] | None = None,
    circle_scale: float = 1.0,                 # 0‒1, diameter relative to min(H,W)
    color: Tuple[int, int, int] = (51, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Overlay a *circular* satellite thumbnail (with a north-pointing arrow) onto
    the main image.

    Parameters
    ----------
    img            : RGB image – shapes accepted:
                       (H,W,C) / (C,H,W) / (B,1,C,H,W)  (torch or np)
    sat            : satellite RGB image or dict containing key 'satellite_image'
    circle_center  : (x,y) in **pixels** *or* **fractions** (0‒1) relative to
                     satellite image.  None ⇒ centre of the satellite frame.
    circle_scale   : diameter / min(h,w)   (clamped to (0,1])
    color,thickness: arrow style.
    """
    
    # ───────────────────────── helper: to (H,W,C) uint8 ──────────────────
    def to_hwc_uint8(arr: ArrayLike) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu()
            while arr.ndim > 3:
                arr = arr[0]
            if arr.shape[0] in {1, 3}:       # CHW → HWC
                arr = arr.permute(1, 2, 0)
            arr = arr.numpy()

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, 2)      # gray ➜ RGB
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, 2)

        if arr.dtype != np.uint8:
            # Normalize to 0-255 range
            arr = cv2.normalize(arr, None, 0, 1, cv2.NORM_MINMAX)
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr

    # unpack dict input
    if isinstance(sat, dict):
        sat = sat["satellite_image"]

    img = to_hwc_uint8(img).copy()
    sat = to_hwc_uint8(sat)

    # ─── make circular crop & arrow ───────────────────────────────────────
    h, w = sat.shape[:2]
    r = int(min(h, w) * circle_scale / 2)
    if circle_center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = circle_center
        if 0 <= cx <= 1 and 0 <= cy <= 1:       # allow fractions
            cx, cy = int(cx * w), int(cy * h)
        cx, cy = int(cx), int(cy)
    r = min(r, cx, cy, w - 1 - cx, h - 1 - cy)  # clamp to image

    # full-frame mask & circular crop
    mask_full = np.zeros((h, w), np.uint8)
    cv2.circle(mask_full, (cx, cy), r, 255, -1)
    sat_full = cv2.bitwise_and(sat, sat, mask=mask_full)

    tip        = (cx,              int(cy - 0.05*r))          # sharp tip
    left_base  = (cx - int(0.05*r), cy + int(0.05*r))
    right_base = (cx + int(0.05*r), cy + int(0.05*r))

    arrow_pts  = np.array([tip, right_base, left_base], dtype=np.int32)
    cv2.fillConvexPoly(sat_full, arrow_pts, color)

    # ─── crop a square around the circle so resize keeps it round ─────────
    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r
    square     = sat_full[y0:y1, x0:x1]          # (2r, 2r, 3)
    mask_sq    = mask_full[y0:y1, x0:x1]         # matching mask

    # ─── resize thumbnail to fit the main image ───────────────────────────
    H, W = img.shape[:2]
    sH, sW = int(0.6*H), int(0.6*W)
    margin     = 1
    thumb_side = min(square.shape[0], min(sH, sW) - 2 * margin)
    thumb      = cv2.resize(square, (thumb_side, thumb_side), cv2.INTER_AREA)
    mask_thumb = cv2.resize(mask_sq, (thumb_side, thumb_side), cv2.INTER_NEAREST)

    # ─── paste into top-right corner ──────────────────────────────────────
    y, x = margin, W - thumb_side - margin
    roi  = img[y : y + thumb_side, x : x + thumb_side]
    np.copyto(roi, thumb, where=np.repeat(mask_thumb[..., None], 3, 2).astype(bool))
    img[y : y + thumb_side, x : x + thumb_side] = roi

    return img

def densify_points(
    xyz: np.array,
    N: int
):
    if xyz.shape[0] < 2:
        raise ValueError("Need at least two points to interpolate.")

    # Compute segment lengths
    diffs = np.diff(xyz, axis=0)          # (T-1, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # (T-1,)

    # Cumulative arc-length along the path
    cumlen = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # (T,)

    # Target arc-lengths to interpolate at (N equally spaced samples)
    target_lens = np.linspace(0, cumlen[-1], N)

    # Find which segment each target length falls into
    idx = np.searchsorted(cumlen, target_lens, side="right") - 1
    idx = np.clip(idx, 0, len(seg_lengths)-1)  # ensure valid indices

    # Segment-wise interpolation factors
    seg_start = cumlen[idx]
    seg_end = cumlen[idx+1]
    seg_frac = (target_lens - seg_start) / (seg_end - seg_start + 1e-12)

    # Interpolate positions
    start_points = xyz[idx]
    end_points = xyz[idx+1]
    out = start_points + (end_points - start_points) * seg_frac[:, None]

    return out

def draw_xyz_on_image(
    image, 
    xyz, 
    infos,
    num_points=-1,
    color=(51, 255, 255), 
    thickness=2
):
    """
    Draws odometry xyz points on image
    """
    if isinstance(image, torch.Tensor):
        assert image.ndim == 4, "Image should be of shape [B, C, H, W]"
        image = image[0].permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]
    if isinstance(xyz, torch.Tensor):
        assert xyz.ndim == 3, "XYZ should be of shape [B, T, 3]"
        xyz = xyz[0].cpu().numpy()
    assert xyz.ndim == 2 and xyz.shape[-1] == 3, "XYZ should be of shape [T, 3]."
    assert image.ndim == 3 and image.shape[-1] in (1, 3), "Image should be of shape [H, W, C] or [H, W]."

    if num_points > -1:
        # Density the xyz trajectory
        xyz = densify_points(xyz, num_points)

    uv = project_xyz_to_pixel(xyz, infos['intrinsics'], infos['T_optical_to_base'])
    if uv is None:
        annotated_image = image.copy()
    else:
        annotated_image = draw_trajectory_on_image(
            image, 
            uv, 
            color=color, 
            thickness=thickness
        )
    annotated_image = cv2.normalize(annotated_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # import cv2
    # cv2.imwrite("test.jpg", annotated_image)  # Save for debugging
    return annotated_image

def draw_trajectory_on_image(image, trajectory, color=(255, 255, 51), thickness=2):
    """
    Draws a trajectory on an image.

    Args:
        image (torch.Tensor or np.ndarray): [B, C, H, W] or [H, W, C] image tensor/array.
        trajectory (torch.Tensor or np.ndarray): [B, N, 2] or [N, 2] trajectory pixel coords.
        color (tuple): BGR color of the trajectory.
        thickness (int): Line thickness (and circle “size”).

    Returns:
        np.ndarray: Annotated image of shape (H, W, 3).
    """
    # --- prepare the image ---
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
    else:
        img_np = image.copy()

    # handle batch dimension
    if img_np.ndim == 4:
        # [B, C, H, W] → take first
        img_np = img_np[0]

    # now either [C, H, W] or [H, W, C]
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        # channels-first → channels-last
        img_np = np.transpose(img_np, (1, 2, 0))

    # Noramlize image to 0-255 range if needed
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)

    # ensure we have 3-channel BGR
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    annotated = img_np.copy()

    # --- prepare the trajectory ---
    if isinstance(trajectory, torch.Tensor):
        traj = trajectory.cpu().numpy()
    else:
        traj = trajectory

    # handle batch
    if traj.ndim == 3:
        traj = traj[0]

    # cast to int pixel coords
    pts = np.round(traj).astype(int)

    # --- draw circles at each point ---
    radius = max(1, thickness)
    for (x, y) in pts:
        cv2.circle(annotated, (x, y), radius=radius, color=color, thickness=-1)

    # --- draw lines connecting them ---
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        cv2.line(annotated, (x0, y0), (x1, y1), color=color, thickness=thickness)

    return annotated

def _mono_gradient(base_rgb: tuple[float, float, float], n: int) -> np.ndarray:
    """
    Build a monochromatic gradient that starts *light* (t = 0) and
    ends at the original hue (t = 1).

    colour(t) = white · (1-t) + base_rgb · t
    """
    t = np.linspace(0.5, 1.0, n)[:, None]              # (N,1)
    base = np.asarray(base_rgb)[None, :]               # (1,3)
    return 1.0 - (1.0 - base) * t                      # (N,3)

def plot_odometry_topdown(
    data_dict,
    keys,
    labels,
    batch_index: int,
    *,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 150,
    axis_limits= None,  # [xmin, ymin, xmax, ymax]
    fontsize=20,
) -> np.ndarray:
    """
    Plot XY trajectories (top-down) for multiple keys from ``data_dict``.

    Parameters
    ----------
    data_dict : dict
        Must map each key in ``keys`` to an ndarray / tensor of shape (B,T,P≥2).
    keys : list[str]
        Keys to extract from ``data_dict``.
    labels : list[str]
        Legend labels (same length as ``keys``).
    batch_index : int
        Which batch element to display.
    figsize, dpi : see matplotlib.
    axis_limits : optional (xmin, ymin, xmax, ymax)
        If provided, fix the axes to these limits.

    Returns
    -------
    img : (H,W,3) uint8 array
    """
    # ---------- figure ----------------------------------------------------
    mpl.rcParams.update(
        {
            "font.size": fontsize,
            "axes.labelsize": fontsize + 1,
            "axes.titlesize": fontsize + 1,
            "legend.fontsize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    base_cmap = plt.get_cmap("tab10")

    # ---------- trajectories ---------------------------------------------
    for i, (key, label) in enumerate(zip(keys, labels)):
        val = data_dict[key]
        if torch.is_tensor(val):
            val = val.cpu().numpy()
        arr = np.asarray(val, dtype=float)

        if arr.ndim != 3 or batch_index >= arr.shape[0] or arr.shape[2] < 2:
            raise ValueError(
                f"key {key!r} must be shape (B,T,P>=2) and contain batch {batch_index}"
            )

        xy = arr[batch_index, :, :2]  # (T,2)
        x, y = xy[:, 0], xy[:, 1]

        base_colour = base_cmap(i % 10)
        # line for legend
        ax.plot(x, y, lw=1.5, color=base_colour, label=label)

        # gradient markers
        grad_cols = _mono_gradient(mpl.colors.to_rgb(base_colour), len(x))
        ax.scatter(
            x,
            y,
            s=24,
            c=grad_cols,
            edgecolors="none",
            zorder=3,
        )

    # ---------- styling ---------------------------------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="best")

    if axis_limits is not None:
        xmin, ymin, xmax, ymax = map(float, axis_limits)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # ---------- tighter layout & tiny margins ---------------------------
    ax.margins(0.02)          # shrink default 5 % margins to 2 %
    fig.tight_layout(pad=0.1) # minimal padding around axes

    # ---------- render --------------------------------------------------
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img

def draw_text_on_image(img, text, org, font, font_scale, color, thickness, max_width):
    """
    Draw multi-line wrapped text onto an image with OpenCV.

    :param img: Image (BGR or RGB)
    :param text: String to draw
    :param org: (x, y) starting position of the first line
    :param font: OpenCV font (e.g., cv2.FONT_HERSHEY_SIMPLEX)
    :param font_scale: Font scale factor
    :param color: Text color (BGR tuple)
    :param thickness: Line thickness
    :param max_width: Max pixel width before wrapping
    """
    words = text.split(" ")
    wrapped_lines = []
    line = ""

    for word in words:
        test_line = line + word + " "
        # Measure size of this tentative line
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w <= max_width:
            line = test_line
        else:
            wrapped_lines.append(line.strip())
            line = word + " "
    wrapped_lines.append(line.strip())

    x, y = org
    line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 4

    for line in wrapped_lines:
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
        y += line_height