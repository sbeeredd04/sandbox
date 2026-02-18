# Borrowed from Marigold
import numpy as np
import torch

def align_path_mask(
    np_arr: np.ndarray,
    src_range: tuple[float, float] = (-1.0, 1.0),
    dst_range: tuple[float, float] = (0.0, 1.0)
):
    """
    Align a numpy array from one range to another.
    Args:
        np_arr (np.ndarray): Input numpy array to be aligned.
        src_range (tuple[float, float]): Source range (min, max).
        dst_range (tuple[float, float]): Destination range (min, max).
    """
    assert isinstance(np_arr, np.ndarray), "Input must be a numpy array."
    assert len(src_range) == 2 and len(dst_range) == 2, "Ranges must be tuples of length 2."

    src_min, src_max = src_range
    dst_min, dst_max = dst_range

    # Normalize to [0, 1]
    normalized = (np_arr - src_min) / (src_max - src_min)

    # Scale to destination range
    aligned = normalized * (dst_max - dst_min) + dst_min

    return aligned

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred