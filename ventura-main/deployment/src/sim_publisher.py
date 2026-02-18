"""
This script takes static dataset frames and publishes them. Then it listens and saves
the model predictions and computes error metrics.
"""
import torch
import logging
import rospy
import yaml
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
from spinflow.dataset.frodo_dataset import FrodoDataset
from spinflow.dataset.dataloader         import SpinFlowDataModule
from spinflow.model.flowpolicy import WaypointFlowPolicy
# from spinflow.model.flowpolicy_lightning import LitFlowPolicy
from spinflow.util.action_utils import (
    unnormalize_action
)
import spinflow.util.train_utils as tu 
import spinflow.util.metric_utils as mu
from scripts.inference.build_model import build_model
from spinflow.util.log_utils import (
    log_image_actions_to_tb
)


import pytorch_lightning as pl

SEED=23

def load_cfg(p):         # YAML → dict
    with open(p) as f: return yaml.safe_load(f)

def load_model(cfg, seed=42, max_epochs=50):
    weights_ckpt = cfg.model["weights_ckpt"]

    # This loads the vision backbone
    model = WaypointFlowPolicy(cfg.model)
    model.eval()
    # This loads the action head
    sd = torch.load(weights_ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        if miss or unexp:
            logging.warning(f"Missing keys:     {miss}")
            logging.warning(f"Unexpected keys: {unexp}")

    return model

@hydra.main(version_base="1.3", config_path="../../config", config_name="policy")
def main(cfg: DictConfig) -> None:
    rospy.init_node("sim_publisher", anonymous=True)

    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(SEED, workers=True)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    """BEGIN Settings to use GT path conditionining"""
    # cfg['dataset']['normalize_actions'] = False
    # cfg['model']['model_name'] = "WaypointFlowPolicy"
    # cfg['model']['use_gt_feats'] = True
    """END Settings to use GT path conditionining"""
    """BEGIN Settings to use Marigold vision encoder"""
    cfg['dataset']['normalize_actions'] = False
    cfg['model']['model_name'] = "WaypointFlowPolicy"
    cfg['model']['use_gt_feats'] = False
    """END Settings to use Marigold vision encoder"""
    cfg = OmegaConf.create(cfg)

    # Setup configs
    mdl_cfg = cfg.model
    val_cfg = mdl_cfg['validation']
    normalize_actions = mdl_cfg['normalize_actions']

    dm = SpinFlowDataModule(
        cfg.dataset,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.trainer.num_workers,
        # use_distributed_sampler=False,
        # batch_size=
        # num_workers=0,
        # persistent_workers=False
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg)
    model = model.to(device)
    model = model.eval()

    # Build metrics
    metric_specs = val_cfg["evaluation"]["metrics"]
    metrics = mu.MetricManager(*metric_specs)   # single tracker

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", unit="batch",
                    total=len(val_loader), leave=False, dynamic_ncols=True)
        for i, inputs in enumerate(pbar):
            train_inputs = {
                spec["out_key"]: inputs[spec["in_key"]]
                for spec in mdl_cfg["dataloader_inputs"]
            }
            train_inputs = {k: (v.to(device).float() if torch.is_tensor(v) else v)
                            for k, v in train_inputs.items()}

            outputs = model.infer(
                train_inputs,
                **mdl_cfg['validation']['integrator']['kwargs']
            )

            if normalize_actions:
                unnorm_action = unnormalize_action(
                    train_inputs['action_label'],
                    val_cfg['action_range'],
                    val_cfg['action_stats'],
                    mdl_cfg['action_dim']
                )
                train_inputs['action_label'] = unnorm_action

            merged = tu.merge_dict(('inputs', train_inputs), ('outputs', outputs))
            metrics_dict = metrics.update(merged, cur_epoch=0)

            vis_config = val_cfg['evaluation']['vis_config']
            B = inputs['action_label'].shape[0]
            batch_indices = torch.arange(B)
            for vis_spec in vis_config:
                if vis_spec['type'] != 'image_actions':
                    continue
                # Save images to directory
                grid_img_th = log_image_actions_to_tb(
                    None, merged, inputs, vis_spec, 0, 0, batch_indices, prefix='val'
                ) 
                out_path = f"sim_pub_debug/image_actions/{i:04d}.jpg"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                grid_img = (grid_img_th.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(grid_img).save(out_path)

            # Optional: show a running metric in the bar
            if metrics_dict:
                pbar.set_postfix({k.split('/')[-1]: f"{v:.3f}" for k, v in metrics_dict.items()})

    # Print out metrics
    logging.info("Validation metrics:")
    print(metrics._data)

if __name__ == "__main__":
    main()

""" This matches validation well
python deployment/src/sim_publisher.py model=planning/simplepolicy/spinflow_linear_marigold_inference model.weights_ckpt=/data/model_ckpts_awslarge2x/SimplePolicy/spinflow_linear_marigold/20250820/232125/best-0049-0.0684.ckpt model.pretrained.vision_encoder.ckpt=/data/model_ckpts_awslarge2x/ControlNetPlanning/fai_controlnet_base_stlang/20250819/091735/best-0049-0.8484.ckpt
batch_size: 32 (original)
                                    total counts   average
mean_l2_error                 2.648779     72  0.036789
mean_asym_hausdorff_distance  4.912862     72  0.068234
"""

""" Same settings but using predicted masks instead of GT masks
python deployment/src/sim_publisher.py model=planning/simplepolicy/spinflow_linear_marigold_inference model.weights_ckpt=/data/model_ckpts_awslarge2x/SimplePolicy/spinflow_linear_marigold/20250820/232125/best-0049-0.0684.ckpt model.pretrained.vision_encoder.ckpt=/data/model_ckpts_awslarge2x/ControlNetPlanning/fai_controlnet_base_stlang/20250819/091735/best-0049-0.8484.ckpt
batch_size: 32 (original)
                                  total counts   average                 
mean_l2_error                  22.805764     72  0.316747
mean_asym_hausdorff_distance   53.331928     72  0.740721
"""