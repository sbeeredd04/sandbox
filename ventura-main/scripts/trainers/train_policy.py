#!/usr/bin/env python3
"""
Train a diffusion‑policy Lightning model.

Differences from train_depth.py
-------------------------------
* Removed “stage = pretrain / finetune” logic.
* Always builds one DataModule from cfg.dataset.
* Uses LitDiffusionPolicy instead of LitMarigold.
"""

#!/usr/bin/env python3
import os
for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'

# --------------------- MUST BE SET BEFORE ANYTHING ELSE ---------------------
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["GLOG_minloglevel"] = "3"
# os.environ["NUMEXPR_MAX_THREADS"] = "16"
# os.environ["NUMEXPR_NUM_THREADS"] = "16"
# os.environ["DEEPSPEED_LOG_LEVEL"] = "error"

# Must be before *any* other logging config or imports
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import re
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision("high")

import cv2
cv2.setNumThreads(0)

import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from spinflow.dataset.dataloader         import SpinFlowDataModule
from spinflow.model.flowpolicy_lightning import LitFlowPolicy

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _latest_tb_version(tb_root: Path) -> int | None:
    vers = [int(m.group(1)) for p in tb_root.glob("version_*")
            if (m := re.match(r"version_(\d+)", p.name))]
    return max(vers) if vers else None

def get_ckpt_path(cfg: DictConfig) -> tuple[str, str | None]:
    """
    Decide where to save new checkpoints and, optionally, which checkpoint to
    resume from. Returns (ckpt_save_dir, resume_ckpt|None).
    """
    is_zero_rank = (int(os.environ.get("LOCAL_RANK", 0)) == 0
                    and int(os.environ.get("NODE_RANK", 0)) == 0)

    ckpt_path = cfg.model.get("ckpt_path", "")
    if ckpt_path and Path(ckpt_path).is_file():
        # resume training
        ckpt_save_dir = Path(ckpt_path).parent
        return str(ckpt_save_dir), ckpt_path

    # fresh run → make new save dir
    if is_zero_rank:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_save_dir = (Path(cfg.model["save_dir"])
                         / cfg.model["project_name"]
                         / cfg.model["run_name"]
                         / ts[:8] / ts[9:])
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CKPT_SAVE_DIR"] = str(ckpt_save_dir)
    else:
        ckpt_save_dir = Path(os.environ["CKPT_SAVE_DIR"])

    return str(ckpt_save_dir), None

# --------------------------------------------------------------------------- #
# Hydra entry‑point
# --------------------------------------------------------------------------- #
@hydra.main(version_base="1.3", config_path="../../config", config_name="policy")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------- seed & reproducibility ---------------------------
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # ------------------- data --------------------------------------------
    dm = SpinFlowDataModule(
        cfg.dataset,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.trainer.num_workers,
    )

    # ------------------- model -------------------------------------------
    CKPT_SAVE_DIR, resume_ckpt = get_ckpt_path(cfg)
    seed        = 42
    max_epochs  = cfg.model.max_epochs
    weights_ckpt = cfg.model.get("weights_ckpt", "")

    # ensure exclusivity
    assert not (resume_ckpt and weights_ckpt), \
        "Set only one of 'ckpt_path' (resume) or 'weights_ckpt' (warm‑start)."

    model = LitFlowPolicy(cfg.model, seed, max_epochs)
    fit_ckpt_path = None
    if resume_ckpt:
        logging.info(f"Resuming from checkpoint: {resume_ckpt}")
        fit_ckpt_path = resume_ckpt
    elif weights_ckpt and Path(weights_ckpt).is_file():
        logging.info(f"Warm‑starting from weights: {weights_ckpt}")

        sd = torch.load(weights_ckpt, map_location="cpu", weights_only=False)
        # Try loading pytorch lightning state dict if available
        if "state_dict" in sd:
            sd = sd["state_dict"]
            miss, unexp = model.load_state_dict(sd, strict=False)
            if miss or unexp: # Note: does not load vision_encder weights
                logging.warning(f"Missing keys:     {miss}")
                logging.warning(f"Unexpected keys: {unexp}")
        else:
            # Offload model loading to helper function for older models
            model.load_weights(weights_ckpt)

    # ------------------- callbacks ---------------------------------------
    ckpt_best = ModelCheckpoint(
        monitor = cfg.model.monitor_metric.name,
        mode    = cfg.model.monitor_metric.mode,
        dirpath = CKPT_SAVE_DIR,
        save_top_k = 5,
        filename   = "best-{epoch:04d}-{"+cfg.model.monitor_metric.name+":.4f}",
        auto_insert_metric_name=False,
    )
    ckpt_recent = ModelCheckpoint(
        monitor = "epoch",
        mode    = "max",
        dirpath = CKPT_SAVE_DIR,
        save_top_k = 3,
        filename   = "last-{epoch:04d}-{step:07d}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # ------------------- logger ------------------------------------------
    tb_dir = Path(CKPT_SAVE_DIR) / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    version = _latest_tb_version(tb_dir) if resume_ckpt else None
    tb_logger = TensorBoardLogger(save_dir=tb_dir, version=version)

    # ------------------- trainer -----------------------------------------
    trainer = pl.Trainer(
        logger                   = [tb_logger],
        callbacks                = [ckpt_best, ckpt_recent, lr_cb],
        accelerator              = cfg.trainer.accelerator,
        devices                  = cfg.trainer.devices,
        use_distributed_sampler  = cfg.trainer.use_distributed_sampler,
        max_epochs               = cfg.model.max_epochs,
        accumulate_grad_batches  = cfg.model.accumulate_grad_batches,
        log_every_n_steps        = cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch  = cfg.trainer.check_val_every_n_epoch,
        enable_checkpointing     = True,
        deterministic            = False,
        gradient_clip_val        = 1.0,
        gradient_clip_algorithm  = "norm",
        profiler                 = "simple",
        strategy                 = DDPStrategy(find_unused_parameters=True)
    )

    # ------------------- train -------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=fit_ckpt_path)

    # ------------------- (optional) test ---------------------------------
    # if cfg.trainer.run_test:
    #     trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
