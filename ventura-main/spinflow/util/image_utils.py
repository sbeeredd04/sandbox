# Borrowed from Marigold
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

import matplotlib
import numpy as np

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="turbo", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().cpu().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img

def resize_match_aspect(             # ← drop this next to your dataset utilities
    img: torch.Tensor,      #  (B,C,H,W)  or  (C,H,W)
    target_hw: tuple[int,int],
    mode: str = "crop",     # "crop"  or "pad"
    interp: InterpolationMode = InterpolationMode.BILINEAR,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Resize *img* so that it fits `target_hw` while preserving aspect ratio.

    mode == "crop":  scale so the **smaller** side ≥ target, then centre-crop → no
                     dead pixels, some content lost.
    mode == "pad" :  scale so the **larger** side ≤ target, then centre-pad   → no
                     content lost, dead pixels around the frame (value=pad_value).

    Accepts (C,H,W) or (B,C,H,W).  Returns the same rank tensor.
    """
    assert img.dim() in (3, 4), "img must be (C,H,W) or (B,C,H,W)"
    batched = img.dim() == 4
    if not batched:
        img = img.unsqueeze(0)                   # make B dim

    _, _, H, W = img.shape
    tgt_h, tgt_w = target_hw
    assert tgt_h > 0 and tgt_w > 0

    # --------------------------------------------------- scale factor
    if mode == "crop":
        scale = max(tgt_h / H, tgt_w / W)        # scale up to cover target
    elif mode == "pad":
        scale = min(tgt_h / H, tgt_w / W)        # scale down to fit inside
    else:
        raise ValueError("mode must be 'crop' or 'pad'")

    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    img = resize(img, [new_h, new_w], interpolation=interp, antialias=True)

    # --------------------------------------------------- crop or pad
    if mode == "crop":
        top  = (new_h - tgt_h) // 2
        left = (new_w - tgt_w) // 2
        img = img[..., top : top + tgt_h, left : left + tgt_w]
    else:                                       # mode == "pad"
        pad_v = (tgt_w - new_w) // 2
        pad_h = (tgt_h - new_h) // 2
        pad = (pad_v, tgt_w - new_w - pad_v,     # left, right
               pad_h, tgt_h - new_h - pad_h)     # top,  bottom
        img = F.pad(img, pad, value=pad_value)   # (l,r,t,b)

    return img if batched else img.squeeze(0)