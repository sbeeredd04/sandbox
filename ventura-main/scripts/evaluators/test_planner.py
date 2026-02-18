# train_depth.py
from __future__ import annotations
import os
import copy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["NUMEXPR_MAX_THREADS"] = "16"   # must be before pandas/numexpr
os.environ["NUMEXPR_NUM_THREADS"] = "16"   # also cap actual thread usage
import torch
from torch import nn
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) 

from tqdm import tqdm
import cv2, numpy as np, pandas as pd
import hickle as hkl
import hydra
import argparse, yaml
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
from typing import Any, Tuple

# --------------------------------------------------------------------------- #
#  Spinflow Modules
# --------------------------------------------------------------------------- #
from scripts.evaluators.test_helpers import (
    robustness_metric,
    coverage_metric
)

from spinflow.dataset.dataloader import SpinFlowDataModule        # created earlier
from spinflow.model.marigold_lightning_planning import LitMarigold 
from scripts.inference.build_dataset import build_dataset
from scripts.inference.build_model import build_model
from spinflow.dataset.frodo_helpers import (
    set_frodo_id,
    get_frodo_id
)


"""
python scripts/inference/test_path.py \
  --dataset_cfg config/dataset/planning/frodo8k_imagegoal.yaml \
  --eval_cfg   config/evaluation/cfg/standard_diffusion.yaml \
  --model_cfg  config/model/planning/cfg/marigold_imagegoal_nosat.yaml \
  --ckpt_path  model_ckpts_awslarge/MarigoldPlanning/cfg_vanilla_sd2_imagegoal_sat/20250617/163442/best-0029-0.7433.ckpt \
  --split_dir  divergence_splits \
  --out_dir    model_evals
"""

DEBUG_MODE = False  # Set to True to enable debug mode

# ───── CLI ─────────────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument("--dataset_cfg", 
                default="config/dataset/planning/frodo8k_turn_goal.yaml", 
                help="Dataset configuration file [default: frodo8k_turn_goal.yaml]")
pa.add_argument("--model_cfg",
                default="config/model/planning/cfg/controlnet_base_spatial.yaml",
                help="Model configuration file [default: controlnet_base_spatial.yaml]")
pa.add_argument("--eval_cfg", required=False, 
                default="config/evaluation/cfg/spatial_diffusion.yaml", 
                help="Evaluation configuration file [default: spatial_diffusion.yaml]")
pa.add_argument("--ckpt_path", required=True, help="Path to the model checkpoint")
pa.add_argument("--out_dir", 
                default="model_evals", 
                help="Output directory for results [default: model_evals]")

# ───── helpers ─────────────────────────────────────────────────────
def load_cfg(p):         # YAML → dict
    with open(p) as f: return yaml.safe_load(f)

def get_key_cfg(cfg_root:dict, name:str): 
    """search top-level & sub-keys (dict type)"""
    for item in cfg_root.get("load_cfgs", []):
        if item.get("name") == name: return item
        if item.get("type") == "dict": 
            for sub in item["kwargs"].get("subkeys", []):
                if sub.get("name") == name: return sub
    return {}

def model_inputs(data, mdl_cfg, device="cuda"):
    out={}
    for d in mdl_cfg["dataloader_inputs"]:
        v=data.get(d["in_key"], None)
        if v is None:
            logging.warning(f"Key {d['in_key']} not found in data, default to None.")
            out[d["out_key"]] = None
            continue
        if isinstance(v, str):
            out[d["out_key"]] = [v]
        elif isinstance(v, list):
            out[d["out_key"]] = v
        else:
            out[d["out_key"]] = torch.as_tensor(v).float().to(device)
    return out

def _fmt_flt(key: str, val: float) -> str:
    """Pretty formatting for a few common metric names."""
    if "deg" in key.lower():
        return f"{val:.1f}deg"
    elif "iou" in key.lower():
        return f"{val:.3f}"
    else:
        return f"{val:.3f}"

def save_robustness_visualizations(
    pred_dict: dict,
    batch: dict,
    img_dir: Path,
    metrics: dict,
    *,
    thr: float = 0.5,
    cmap: int = cv2.COLORMAP_TURBO,
    mask_color: list[int, int, int] = [51, 255, 255]  # Aqua color for predicted path
) -> Path:
    """
    Make a 3-panel figure:
        [ GT mask | Predicted mask | Ensemble std-dev heatmap ]
    and stamp the heading error (deg) on the top-left corner.

    Returns the saved path so the caller can log it if desired.
    """
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- 1. tensors → numpy ------------------------
    preds = pred_dict["path_mask_preds"]               # (E,1,H,W)  float [0..1]
    gt    = batch["path_mask"].cpu().numpy()           # (1,1,H,W)  {0,1}
    front_rgb = batch["front_rgb"].cpu().permute(0, 2, 3, 1).numpy()[0]  # (H,W,3) RGB
    front_rgb = cv2.normalize(front_rgb, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")  # normalize to 0-255

    preds_mean = preds.mean(axis=0)[0]                 # (H,W)
    preds_std  = preds.std(axis=0)[0]                  # (H,W)
    preds_mask  = (preds_mean >= thr)                  # binary mask
    gt_mask     =   gt[0, 0] > thr                     # binary mask

    # ensure 8-bit grayscale for masks
    gt_overlay = front_rgb.copy()  # (H,W,3) RGB
    gt_overlay[gt_mask] = mask_color  # color the path in the RGB image
    pred_overlay = front_rgb.copy()  # (H,W,3) RGB
    pred_overlay[preds_mask] = mask_color  # color the predicted path in the RGB

    # --------------------------- 2. heat-map -------------------------------
    std_norm = preds_std
    if std_norm.max() > 0:
        std_norm = std_norm / std_norm.max()           # 0-1
    std_img  = (std_norm * 255).astype(np.uint8)
    heatmap  = cv2.applyColorMap(std_img, cmap)        # (H,W,3) BGR
    heatmap_overlay = front_rgb.copy()  # (H,W,3) RGB
    heatmap_overlay = cv2.addWeighted(heatmap_overlay, 0.5, heatmap, 0.5, 0)  # blend heatmap with RGB

    # --------------------------- 3. concatenate ---------------------------
    viz = np.concatenate([gt_overlay, pred_overlay, heatmap_overlay], axis=1).astype(np.uint8)  # (H, 3W, 3)

    # --------------------------- 4. annotate ------------------------------
    y0, dy = 35, 30
    text_keys = ['mIoU', 'hdg_error_deg']
    for i, k in enumerate(text_keys):
        if k not in metrics:             # skip missing keys silently
            continue
        txt = _fmt_flt(k, metrics[k])
        cv2.putText(
            viz,
            txt,
            (10, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # --------------------------- 5. save ----------------------------------
    ride   = batch["infos"]["sequence"]
    ride_name = set_frodo_id(*ride.split(" "))
    frame  = int(batch["infos"]["frame"])
    out_fp = img_dir / f"{ride_name}_start_{frame:06d}.jpg"
    cv2.imwrite(str(out_fp), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    return out_fp

def save_coverage_visualizations(
    pred_dict: dict,
    batch: dict,
    img_dir: Path
):
    """Save coverage visualizations"""
    ride   = batch["infos"]["sequence"]
    ride_name = set_frodo_id(*ride.split(" "))
    frame  = int(batch["infos"]["frame"])
    out_fp = img_dir / f"{ride_name}_start_{frame:06d}.jpg"
    cv2.imwrite(str(out_fp), pred_dict["pred_rgb"])
    
    pred_dir = img_dir.parent / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_fp = pred_dir / f"{ride_name}_start_{frame:06d}.h5"
    hkl.dump(pred_dict, pred_fp, mode='w')
    return out_fp

def merge_sample(
    combined: pd.DataFrame, 
    sample: pd.DataFrame, *, 
    key_cols: Tuple[str, ...] = ["ride_name", "start_frame"]
) -> pd.DataFrame:
    """
    Add `sample` (1-row DataFrame) into `combined`.
    """
    if combined.empty:
        return sample.copy()
    return (
        combined.set_index(list(key_cols))
        .combine_first(sample.set_index(list(key_cols)))   # union + overwrite
        .reset_index()
    )

def get_split_dir(ds_cfg: dict, split_key: str) -> str:
    """
    Get the split directory from the dataset configuration.
    """
    for subdir in ds_cfg.get("subdirs", []):
        if subdir.get("name") == split_key:
            assert "path" in subdir, "split_dir must have a 'path' key"
            return subdir["path"]
    raise ValueError(f"No {split_key} found in dataset configuration.")

@torch.no_grad()
def evaluate_checkpoint_single(model, dataset, ds_cfg, mdl_cfg, eval_cfg, out_dir):
    # Iterate through dataset and compute visualizations + eval metrics
    logging.info("Starting evaluation...")
    model.eval()
    split = eval_cfg.get("split", "test")
    split_dir = ds_cfg['split_dir']
    label_dir = ds_cfg['label_dir']

    split_path = Path(ds_cfg['root_dir']) / label_dir / f'{split}.txt'
    meta_path = Path(ds_cfg['root_dir']) / split_dir / "combined_split_divergence.csv"
    split_df = pd.read_csv(split_path, header=0, sep=",")
    meta_df = pd.read_csv(meta_path, header=2, sep=",")
    meta_df = meta_df.drop_duplicates(subset=["ride_name", "start_frame"])
    filtered_df = meta_df.merge(
        split_df[["ride_name", "start_frame"]].drop_duplicates(),   # keys to match
        on=["ride_name", "start_frame"],
        how="inner",           # keep only matching rows
        validate="one_to_one"  # optional safety check
    )

    mask_tp_threshold = eval_cfg["mask_tp_threshold"]
    cfg_scale = mdl_cfg["validation"]["scheduler"]["kwargs"].get("cfg_scale", 1.0)
    guidance_rescale = mdl_cfg["validation"]["scheduler"]["kwargs"].get("guidance_rescale", 0.0)

    metrics_dict = eval_cfg['metrics']

    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)  # Set a fixed start seed for reproducibility

    results_df   = pd.DataFrame()
    try:
        logging.info(f"Using cfg_scale={cfg_scale}, guidance_rescale={guidance_rescale}, ")
        pbar = tqdm(total=len(dataset), desc="Evaluating Checkpoint", unit="batch")
        print("Length of dataset:", len(dataset))
        for i, batch in enumerate(dataset):
            pbar.update(1)
            if DEBUG_MODE and i >= 5:
                logging.info("Debug mode: stopping after 5 batches.")
                break

            inputs = model_inputs(batch, mdl_cfg, device="cuda")
            for metric in metrics_dict:
                if metric == "robustness":
                    metrics_df, pred_dict = robustness_metric(inputs, batch, model, filtered_df, mdl_cfg, eval_cfg)
                    img_dir = out_dir / metric / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    # Convert metrics_df to a dict for visualization
                    row_dict = metrics_df.iloc[0].to_dict()
                    save_robustness_visualizations(pred_dict, batch, img_dir, row_dict)
                elif metric == "coverage":
                    metrics_df, pred_dict = coverage_metric(
                        inputs, batch, model, filtered_df, mdl_cfg, eval_cfg
                    )
                    
                    # Save coverage visualizations
                    img_dir = out_dir / metric / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    save_coverage_visualizations(pred_dict, batch, img_dir)
                else:
                    logging.warning(f"Unknown metric: {metric}")

                if metrics_df is not None:
                    results_df = merge_sample(results_df, metrics_df)
    except KeyboardInterrupt as e:
        logging.warning("Evaluation interrupted by user, saving partial results.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")

    pbar.close()
    csv_path = out_dir / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved merged metrics → {csv_path}")

    # ─────────────────────── NEW: ride-level precision ranking ────────────
    if {"true_pos", "false_pos"}.issubset(results_df.columns):
        # aggregate TP / FP per ride, then precision = TP / (TP+FP)
        ride_prec = (
            results_df.groupby("ride_name")[["true_pos", "false_pos"]].sum()
                      .assign(precision=lambda df:
                              df.true_pos / (df.true_pos + df.false_pos))
                      .sort_values("precision", ascending=True)     # worst first
                      .reset_index()
        )
        worst_path = out_dir / "rides_sorted_by_precision.csv"
        ride_prec.to_csv(worst_path, index=False)
        logging.info(f"Saved rides sorted by precision → {worst_path}")
    # ──────────────────────────────────────────────────────────────────────

    # ---------------- Overall statistics ----------------------------------
    summary_path = out_dir / "evaluation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        if {"true_pos", "false_pos"}.issubset(results_df.columns):
            tp_tot = int(results_df["true_pos"].sum())
            fp_tot = int(results_df["false_pos"].sum())
            acc    = tp_tot / max(1, results_df["total"].sum())
            f.write(f"coverage  |  TP {tp_tot}  FP {fp_tot}  Total {results_df['total'].sum()} \n"
                    f"accuracy {acc:.4f}\n")

        if "hd_avg" in results_df.columns:
            mean_hd = results_df["hd_avg"].mean()
            f.write(f"mean_hausdorff_distance : {mean_hd:.4f}\n")
            f.write(f"std_hausdorff_distance  : {results_df['hd_avg'].std():.4f}\n")

    logging.info(f"Saved summary  →  {summary_path}")

def merge_shared(cfg1: dict, cfg2: dict, *, in_place: bool = False) -> dict:
    """
    Merge *shared* keys of `cfg1` into `cfg2`.
    """
    dst = cfg2 if in_place else copy.deepcopy(cfg2)

    for k in cfg1.keys() & cfg2.keys():
        v1, v2 = cfg1[k], dst[k]
        if isinstance(v1, list) and isinstance(v2, list):
            dst[k] = v2 + v1 
        elif isinstance(v1, dict):
            merged = dict(v2)
            merged.update(v1)
            dst[k] = merged
    return dst

def main(args):
    ds_cfg  = load_cfg(args.dataset_cfg)
    eval_cfg = load_cfg(args.eval_cfg)
    mdl_cfg = load_cfg(args.model_cfg)

    # Override key configurations
    # mdl_cfg['validation']['scheduler']['kwargs']['ensemble_size'] = eval_cfg['num_samples']
    mdl_cfg["validation"]["scheduler"]["kwargs"]["default_denoising_steps"] = eval_cfg.get('default_denoising_steps', 50)
    mdl_cfg["validation"]["scheduler"]["kwargs"]['cfg_scale'] = eval_cfg.get('cfg_scale', 1.0)
    ds_cfg["split_dir"] = get_split_dir(ds_cfg, "split_dir")
    ds_cfg["label_dir"] = eval_cfg['label_dir']
    split = eval_cfg.get('split', 'test')

    ds_cfg = merge_shared(eval_cfg["dataset_cfg_kwargs"], ds_cfg, in_place=False)  # merge dataset config with model config
    # mdl_cfg = merge_shared(eval_cfg["model_cfg_kwargs"], mdl_cfg, in_place=False)  # merge model config with eval config

    dataset = build_dataset(ds_cfg, split_dir=ds_cfg["label_dir"], split=split)
    logging.info(f"Dataset loaded with {len(dataset)} samples.")    
    
    ckpt_path = Path(args.ckpt_path)
    ckpt_path_list = [ ckpt_path ]
    if ckpt_path.is_dir():
        ckpt_path_list = list(ckpt_path.glob("*.ckpt"))[0]
    elif not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist or is not a file.")

    # Create output directory if it does    n't exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpt_path_list:
        model   = build_model(mdl_cfg, ckpt, seed=42, device="cuda").eval()
        logging.info(f"Evaluating checkpoint: {ckpt}")
        evaluate_checkpoint_single(model, dataset, ds_cfg, mdl_cfg, eval_cfg, out_dir)



if __name__ == "__main__":
    args = pa.parse_args()
    main(args)