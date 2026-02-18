import torch
import pandas as pd
import numpy as np
import cv2

from scripts.utils.log_utils import logging
from spinflow.util.metric_utils import (
    intersection_over_union
)


def _angular_error_deg(pred_hdg: torch.Tensor, cur_hdg: torch.Tensor) -> float:
    """
    Smallest absolute angular difference in degrees (0-180].

    Both inputs should be or arrays, returns a Python float.
    """
    # wrap to (-180, 180]
    diff = (pred_hdg - cur_hdg + 180.0) % 360.0 - 180.0
    return float(np.abs(diff))

def ensemble_inputs(inputs, ensemble_size):
    """
    Repeat the inputs for ensemble sampling.
    This is useful when the model expects multiple samples per input.
    """
    for key in inputs:
        if isinstance(inputs[key], list):
            inputs[key] = inputs[key] * ensemble_size
        elif isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].repeat(ensemble_size, *[1] * (inputs[key].ndim - 1))
        else:
            raise ValueError(f"Unsupported input type for key {key}: {type(inputs[key])}")
    return inputs

def goal_ensemble_inputs(inputs, goals, ensemble_size):
    """
    Duplicate inputs so the final batch has
      • len(goals) x ensemble_size items for **every** key.
    Keys present in *goals* are overwritten with the goal-specific values,
    repeated *ensemble_size* times; all other keys are broadcast to the
    same final batch length.
    """
    G = len(next(iter(goals.values())))          # number of goal entries
    GE = G * ensemble_size                       # final batch length

    for k in inputs.keys():
        if k in goals:                           # ---- goal-specific keys ----
            v = goals[k]
            if isinstance(v, list):
                inputs[k] = sum(([item] * ensemble_size for item in v), [])
            elif isinstance(v, torch.Tensor):
                # v.shape[0] == G  → repeat along batch dim
                inputs[k] = v.repeat_interleave(ensemble_size, dim=0)
            else:
                raise TypeError(f"Unsupported type for key {k}: {type(v)}")
        else:                                    # ---- shared keys ----------
            v = inputs[k]
            if isinstance(v, list):
                inputs[k] = v * GE
            elif isinstance(v, torch.Tensor):
                inputs[k] = v.repeat(GE, *([1] * (v.ndim - 1)))
            else:
                raise TypeError(f"Unsupported type for key {k}: {type(v)}")

    return inputs

def _mask_to_centerline(mask: torch.Tensor, n_pts: int = 200) -> np.ndarray:
    """
    Extract a centre-line from a binary path mask.

    Algorithm
    ---------
    • Treat the mask as (1,H,W) or (H,W).  
    • For every image row *y* (starting at the BOTTOM = y=H-1):
        – find all x-coords where mask==1  
        – take x_left  = min(x),  x_right = max(x)  
        – centre-line x = (x_left + x_right) / 2
    • Stop when a row has no positive pixels above – those positions are
      *ignored* (no knee-jerk padding with zeros).
    • Uniformly resample **n_pts** points along that poly-line so every
      centre-line has identical length.

    Returns
    -------
    (n_pts, 2) float32 array  with **(x,y)** pixel coordinates,
    going from bottom (row H-1) upwards.
    """
    # ------------ 1.  flatten & find positive pixels -----------------
    if mask.ndim == 3:                       # [1,H,W]  or  [C,H,W]
        mask2d = mask.squeeze(0)
    else:
        mask2d = mask                       # already (H,W)

    y_idx, x_idx = mask2d.nonzero(as_tuple=True)   # 1-D tensors

    if x_idx.numel() == 0:                   # completely empty mask
        return np.zeros((n_pts, 2), np.float32)

    H = mask2d.shape[0]                      # image height

    # ------------ 2.  per-row (y) min / max x ------------------------
    # convert to CPU numpy arrays once for speed
    y_np = y_idx.cpu().numpy()
    x_np = x_idx.cpu().numpy()

    # Use pandas-like vectorisation via numpy bincount
    # Step 1 – create arrays of +inf and -inf then scatter-reduce.
    max_x = np.full(H, -np.inf, dtype=np.float32)
    min_x = np.full(H,  np.inf, dtype=np.float32)

    np.maximum.at(max_x, y_np, x_np)         # max_x[y] = max(x)
    np.minimum.at(min_x, y_np, x_np)         # min_x[y] = min(x)

    valid_rows = np.isfinite(max_x)          # rows that had ≥1 pixel

    if not valid_rows.any():                 # paranoia
        return np.zeros((n_pts, 2), np.float32)

    ys = np.nonzero(valid_rows)[0]           # ascending 0…H-1
    xs = (min_x[ys] + max_x[ys]) * 0.5       # centre x

    # ------------ 3.  build (x,y) poly-line & resample ---------------
    # flip to bottom→top (y descending)
    xs = xs[::-1]
    ys = ys[::-1]

    if len(xs) == 1:                         # degenerate 1-point path
        xy = np.repeat([[xs[0], ys[0]]], n_pts, axis=0)
        return xy.astype(np.float32)

    # cumulative arc-length along poly-line (euclidean)
    seg_len = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    s /= s[-1]                               # normalise 0-1

    # uniform sampling in s-space
    u = np.linspace(0, 1, n_pts)
    x_resamp = np.interp(u, s, xs)
    y_resamp = np.interp(u, s, ys)

    xy = np.stack([x_resamp, y_resamp], 1).astype(np.float32)
    return xy

def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Directed (asymmetric) Hausdorff distance  H(a → b).

    • *Only* points in **a** are considered; extra points in **b**
      therefore do **not** increase the distance – this avoids
      penalising predictions that are longer than the GT path.
    """
    if a.size == 0 or b.size == 0:
        return np.inf               # undefined – treat as very bad match

    diff   = a[:, None, :] - b[None, :, :]          # (N,M,2)
    d2     = np.sum(diff**2, axis=-1)               # (N,M)
    min_d2 = d2.min(axis=1)                         # (N,)
    return float(np.sqrt(min_d2.max())) 

def _vis_ensemble_preds(
    rgb_batch: torch.Tensor,
    pred_masks: torch.Tensor,       # (G,E,1,H,W)
    pred_lines: np.ndarray,         # (G,E,T,2)
    goals_dict: dict
) -> np.ndarray:
    """
    Visualise ensemble predictions.

    • row  = one goal (G rows)
    • cols = RGB-overlay (+ optional goal-image on the right)
    • Overlays per-row:
        – mean mask   (TURBO)
        – variance    (JET)
        – ensemble centre-lines   (green)
        – mean centre-line        (red)
        – goal string text (if provided)
    """
    assert rgb_batch.ndim == 4 and pred_masks.ndim == 5 and pred_lines.ndim == 4
    G, _, H, W = rgb_batch.shape

    # -------- prep RGB to uint8 -------------------------------------------
    rgb = rgb_batch.clone().cpu()
    if rgb.min() < 0:           # [-1,1] ⇒ [0,1]
        rgb = (rgb + 1) * 0.5
    rgb = (rgb * 255).clamp(0, 255).byte().permute(0, 2, 3, 1)  # (G,H,W,3)

    # helper to convert goal images (if any) → uint8 (H,W,3)
    def _prep_goal_img(gimg):
        if isinstance(gimg, torch.Tensor):
            gimg = gimg.cpu().squeeze()
            if gimg.ndim == 4: gimg = gimg[0]            # drop time dim if (C,H,W) inside list
            if gimg.ndim == 3 and gimg.shape[0] in (1, 3):
                gimg = gimg.permute(1, 2, 0)             # CHW→HWC

            if gimg.min() < 0: gimg = (gimg + 1) * .5
            gimg = (gimg * 255).clamp(0, 255).byte().numpy()
        elif isinstance(gimg, np.ndarray):
            if gimg.ndim == 3 and gimg.shape[0] in (1, 3):
                gimg = np.transpose(gimg, (1, 2, 0))
            gimg = gimg.copy()
        else:
            raise TypeError(f"Unsupported goal_image dtype {type(gimg)}")
        if gimg.ndim == 2:                       # gray → 3-chan
            gimg = np.repeat(gimg[..., None], 3, 2)
        gimg = cv2.resize(gimg, (W, H), interpolation=cv2.INTER_AREA)
        return gimg

    have_goal_img   = "goal_image"   in goals_dict
    have_goal_str   = "goal_command" in goals_dict
    have_gt_lines   = "paths_2d"   in goals_dict    
    have_goal_match = "match_map" in goals_dict

    imgs = []                                                # rows to concat

    GT_COLOR = (51, 255, 255)  # RGB (red)
    PRED_COLOR = (0, 0, 255)  # aqua
    for g in range(G):
        canvas = rgb[g].numpy().copy()                       # RGB uint8

        # ----- mean + variance heat-maps -------------------------------
        mean_m = pred_masks[g].float().mean(0).squeeze().cpu().numpy()  # (H,W)
        var_m  = pred_masks[g].float().var (0).squeeze().cpu().numpy()  # (H,W)

        heat_mean = cv2.applyColorMap((mean_m * 255).astype(np.uint8),
                                      cv2.COLORMAP_TURBO)               # (H,W,3)
        if var_m.max():
            heat_var = cv2.applyColorMap(
                (var_m / var_m.max() * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
        else:
            heat_var = np.zeros_like(heat_mean, dtype=np.uint8)

        # --- pixel-wise alpha so *zero* pixels keep the original RGB ----
        mean_mask = (mean_m > 0).astype(np.float32)[..., None]          # (H,W,1)
        var_mask  = (var_m  > 0).astype(np.float32)[..., None]

        # α = 0.3 for mean, 0.2 for variance; 0 where mask==0
        alpha_m = 0.3 * mean_mask
        alpha_v = 0.2 * var_mask

        canvas_f = canvas.astype(np.float32)
        canvas_f = (
            canvas_f * (1.0 - alpha_m - alpha_v) +
            heat_mean.astype(np.float32) * alpha_m +
            heat_var .astype(np.float32) * alpha_v
        )
        canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)

        # ----- centre-lines --------------------------------------------
        for ln in pred_lines[g]:
            cv2.polylines(canvas, [ln.astype(int)], False, (0,255,0), 1,
                          cv2.LINE_AA)
        mean_ln = pred_lines[g].mean(0).astype(int)
        cv2.polylines(canvas, [mean_ln], False, PRED_COLOR, 2, cv2.LINE_AA)

        # ----- goal string ---------------------------------------------
        if not have_goal_match and have_goal_str:
            goal_txt = str(goals_dict["goal_command"][g])[:40]  # trim long
            cv2.putText(canvas, goal_txt, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,
                        cv2.LINE_AA)

        # ----- optional goal image concat ------------------------------
        if have_gt_lines:
            gt_lines = goals_dict["paths_2d"].cpu().numpy()  # (G,T,2)
            cv2.polylines(canvas, [gt_lines[g].astype(int)], False,
                          GT_COLOR, 2, cv2.LINE_AA)

        if have_goal_img:
            gimg = goals_dict["goal_image"][g]
            gimg = _prep_goal_img(gimg)
            canvas = cv2.hconcat([canvas, gimg])

        if have_goal_match:
            gt_match_label = goals_dict["match_map"][g, 0, 0]
            pred_match_label = goals_dict["match_map"][g, 0, 1]
            cv2.putText(canvas, f"GT: {gt_match_label}",
                        (5, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        GT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Pred: {pred_match_label}",
                        (5, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        PRED_COLOR, 2, cv2.LINE_AA)
                        

        imgs.append(canvas)

    grid = cv2.vconcat(imgs)           # (G*H, W or W*2, 3)  BGR
    return grid[..., ::-1]             # back to RGB for logging

"""END HELPER FUNCTIONS"""

def coverage_metric(dinputs, minputs, model, meta_df, mdl_cfg, eval_cfg,
                    device="cuda", seed=42):
    """
    Coverage metric:
        • G  ground-truth (goal, path) pairs in *dinputs*
        • E  ensemble size  (eval_cfg['metrics']['coverage']['ensemble_size'])
        • For every (g,e) we predict a path mask  → centre-line (T,2)
        • We match each prediction to the closest GT path via Hausdorff distance
        • A *true-positive* is a prediction whose matched GT goal string is
          identical to the goal string that was fed to the model.
    Returns
    -------
    metrics_df : pd.DataFrame  with one row for this sample
    aux        : dict  extra tensors for later inspection
    """
    if model is None:
        logging.error("coverage_metric: `model` must be passed.")
        return None, {}

    # ---------------- 0. sanity & metadata -------------------------------------------------
    ride_name   = minputs["infos"]["sequence"]
    start_frame = int(minputs["infos"]["frame"])
    meta_row = meta_df[
        (meta_df["ride_name"] == ride_name)
        & (meta_df["start_frame"] == start_frame)
    ]
    if meta_row.empty:
        logging.warning(f"No metadata for ride {ride_name} @ frame {start_frame}.")
        return None, {}

    # ---------------- 1.   grab GT goals & paths ------------------------------------------
   
    goals_dict = {}
    gt_goals = minputs["goals"]                 # length = G
    valid_mask = np.array([len(g) > 0 for g in gt_goals], dtype=bool)  # (G,)
    goals_dict["goal_command"] = np.array(gt_goals)[valid_mask].tolist()  # (G,)
    if "target_segment" in minputs:
        gimgs = minputs["target_segment"]           # (G,C,H,W) tensor  *or*  list
        goals_dict["goal_image"] = gimgs[valid_mask].to(device)
        dinputs["goal_image"]   = gimgs[valid_mask].to(device)
    
    if "paths_2d" in minputs:
        gt_paths = minputs["paths_2d"].float()
        goals_dict["paths_2d"] = gt_paths[valid_mask]
    G = len(goals_dict["goal_command"])

    # ---------------- 2.   ensemble predictions  ------------------------------------------
    ens_size   = eval_cfg["metrics"]["coverage"]["ensemble_size"]
    assert ens_size > 0, "ensemble_size must be > 0"
    
    # Generate multiple goals
    inputs = goal_ensemble_inputs(dinputs, goals_dict, ens_size)
    cfg_scale        = eval_cfg["cfg_scale"]
    guidance_rescale = mdl_cfg["validation"]["scheduler"]["kwargs"].get(
        "guidance_rescale", 0.0
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    outs = model(
        inputs,
        cfg_scale        = cfg_scale,
        guidance_rescale = guidance_rescale,
        generator        = generator,
    )
    #  outs["path_mask_pred"] : (G·E , 1 , H , W)
    pred_masks = outs["path_mask_pred"].float() > 0.5
    pred_masks = pred_masks.view(G, ens_size, *pred_masks.shape[1:])          # (G,E,1,H,W)

    # ---------------- 3.   centre-line extraction -----------------------------------------
    pred_lines = []
    for g in range(G):
        lines_g = []
        for e in range(ens_size):
            lines_g.append(_mask_to_centerline(pred_masks[g, e]))             # (T',2)
        pred_lines.append(np.stack(lines_g, 0))                               # (E,T,2)
    pred_lines = np.stack(pred_lines, 0)                                      # (G,E,T,2)
    gt_lines = gt_paths.cpu().numpy().astype(np.float32)                      # (G,T,2)
 
     # ---------------- 4.   matching & true/false-positives -----------------
    match_map   = np.full((G, ens_size, 2), -1, np.int32)
    match_map_str = np.full((G, ens_size, 2), "", dtype=object)

    tp = fp = 0
    hd_sum = 0.0
    n_pred = 0 

    for g in range(G):
        for e in range(ens_size):
            dists = [_hausdorff(pred_lines[g, e], gt_lines[k]) for k in range(G)]
            match = int(np.argmin(dists))
            match_map[g, e] = [g, match]

            # accumulate asymmetric HD of the chosen match -------------
            hd_sum += dists[match]
            n_pred += 1
            # ----------------------------------------------------------

            # TP/FP check ---------------------------------------------
            if gt_goals[match].strip().lower() == gt_goals[g].strip().lower():
                tp += 1
            else:
                fp += 1
            match_map_str[g, e] = [gt_goals[g], gt_goals[match]]

    goals_dict["match_map"] = match_map_str  # (G,E,2)  goal strings

    try:
        H, W = pred_masks.shape[-2:]  # height, width of the mask
        vis_grid = _vis_ensemble_preds(
            inputs['rgb_image'][::ens_size], # (G,E,H,W,3) → (G,H,W,3)
            pred_masks,                    # (G,E,1,H,W)
            pred_lines,                   # (G,E,T,2)  from step 3
            goals_dict
        )
        cv2.imwrite("test.jpg", vis_grid)  # Save the grid image
    except Exception as e:
        logging.warning(f"vis_grid generation failed: {e}")

    # ---------------- 5.   pack results ---------------------------------------------------
    avg_hd = hd_sum / max(1, n_pred)          # ← NEW

    metrics_df = pd.DataFrame({
        "ride_name"   : [ride_name],
        "start_frame" : [start_frame],
        "true_pos"    : [tp],
        "false_pos"   : [fp],
        "total"       : [n_pred],
        "coverage"    : [tp / max(1, tp + fp)],
        "hd_avg"      : [avg_hd],              # ← NEW
    })

    aux = {
        "match_map"        : match_map,                     # (G,E,2)
        "pred_mask"        : pred_masks.cpu().numpy(),      # (G,E,1,H,W)
        "pred_centerlines" : pred_lines,                    # (G,E,T,2)
        "pred_rgb"         : vis_grid
    }
    return metrics_df, aux

def robustness_metric(dinputs, minputs, model, meta_df, mdl_cfg, eval_cfg, device="cuda", seed=42):
    """
    Robustness metric:
      • ensembles `ensemble_size` predictions along the batch dim
      • computes per-member IoU against the GT mask, then averages → mIoU
      • computes absolute heading error (deg) and buckets it
      • returns {mIoU, hdg_error_deg, hdg_bucket}
    """
    ride_name   = minputs["infos"]["sequence"]
    start_frame = int(minputs["infos"]["frame"])

    meta_row = meta_df[
        (meta_df["ride_name"] == ride_name)
        & (meta_df["start_frame"] == start_frame)
    ]
    if meta_row.empty:
        logging.warning(f"No metadata for ride {ride_name} @ frame {start_frame}.")
        return None
    if "goal_heading_deg" not in meta_row.columns or \
        "path_heading_deg" not in meta_row.columns:
        logging.error(f"Missing heading metadata for {ride_name} @ frame {start_frame}.")

    eval_dict      = eval_cfg["metrics"]["robustness"]
    ensemble_size  = eval_dict["ensemble_size"]
    assert ensemble_size > 0, "ensemble_size must be > 0"

    # ----------------------- 1. run the model (ensembled) -------------------
    inputs            = ensemble_inputs(dinputs, ensemble_size)  # (E, …)
    cfg_scale         = eval_cfg["cfg_scale"]
    guidance_rescale  = mdl_cfg["validation"]["scheduler"]["kwargs"].get(
        "guidance_rescale", 0.0
    )
    generator         = torch.Generator(device=device).manual_seed(seed)
    outputs = model(
        inputs,
        cfg_scale=cfg_scale,
        guidance_rescale=guidance_rescale,
        generator=generator,
    )

    # ----------------------- 2. heading error & bucket ----------------------
    range_min, range_max = eval_dict["range"]          # expected (0, 180)
    n_buckets            = eval_dict["buckets"]        # e.g. 18

    # average heading over ensemble if needed
    goal_hdg = meta_row["goal_heading_deg"].iloc[0]
    path_hdg = meta_row["path_heading_deg"].iloc[0]
    hdg_error_deg = _angular_error_deg(goal_hdg, path_hdg)
    hdg_error_deg = max(range_min, min(range_max, hdg_error_deg))  # clip 0-180

    bucket_size = (range_max - range_min) / n_buckets
    hdg_bucket  = min(
        n_buckets - 1, int((hdg_error_deg - range_min) // bucket_size)
    )

    # ----------------------- 3. mIoU over ensemble --------------------------
    pred_masks = outputs["path_mask_pred"].float()                  # [E, 1, H, W]
    gt_mask    = minputs["path_mask"].float().to(pred_masks.device) # [1, 1, H, W]

    iou_vals = []
    for pred in pred_masks:                            # iterate ensemble members
        iou = intersection_over_union(
            pred, gt_mask, valid_mask=None
        )                                              # returns tensor
        iou_vals.append(iou)

    mIoU = torch.stack(iou_vals).mean().item()

    metrics_df = pd.DataFrame({
        "ride_name": [ride_name],
        "start_frame": [start_frame],
        "mIoU": [mIoU],
        "hdg_error_deg": [hdg_error_deg],
        "hdg_bucket": [hdg_bucket],
    })
    # ----------------------- 4. return results ------------------------------
    return metrics_df, { 
        "path_mask_preds": pred_masks.cpu().numpy() 
    }