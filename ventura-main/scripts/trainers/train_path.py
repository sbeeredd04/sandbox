# train_depth.py
from __future__ import annotations
import os
import re
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["NUMEXPR_MAX_THREADS"] = "16"   # must be before pandas/numexpr
os.environ["NUMEXPR_NUM_THREADS"] = "16"   # also cap actual thread usage
os.environ["DEEPSPEED_LOG_LEVEL"] = "error" 
import logging
for name in list(logging.root.manager.loggerDict):
    if name.startswith("deepspeed"):
        logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR) 


import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  your modules
# --------------------------------------------------------------------------- #
from spinflow.dataset.dataloader import SpinFlowDataModule        # created earlier
from spinflow.model.marigold_lightning_planning import LitMarigold    # wrap MarigoldPlanner in a LightningModule

def _latest_tb_version(tb_root: Path) -> int | None:
    """
    Return the highest integer `version_*` under tb_root, or None if none exist.
    """
    vers = [int(m.group(1)) for p in Path(tb_root).glob("**/version_*")
            if (m := re.match(r"version_(\d+)", p.name))]
    return max(vers) if vers else None

def get_ckpt_path(cfg: DictConfig) -> str | None:
    """
    Get the checkpoint path from the configuration.
    """
    is_zero_rank = os.environ.get('LOCAL_RANK', 0)==0 and os.environ.get('NODE_RANK', 0)==0

    ckpt_path = cfg.model.get('ckpt_path', '')
    is_ckpt_valid = Path(ckpt_path).is_file()
    if not is_ckpt_valid:
        if is_zero_rank:
            from datetime import datetime
            cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            day, time = cur_time.split('_')
            ckpt_save_dir = Path(cfg.model['save_dir']) / \
                cfg.model['project_name'] / \
                cfg.model['run_name'] \
                / day / time
            ckpt_save_dir.mkdir(parents=True, exist_ok=True)
            os.environ['CKPT_SAVE_DIR'] = str(ckpt_save_dir)
        else:
            ckpt_save_dir = os.environ['CKPT_SAVE_DIR']
    else:
        ckpt_save_dir = os.path.dirname(ckpt_path)
    
    if len(ckpt_path) == 0:
        ckpt_path = None
    return ckpt_save_dir, ckpt_path

# --------------------------------------------------------------------------- #
#  Hydra entry-point
# --------------------------------------------------------------------------- #
@hydra.main(version_base="1.3", config_path="../../config", config_name="planning")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------- seed & reproducibility ---------------------------
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # ------------------- data --------------------------------------------
    stage = cfg.model.get("stage", "pretrain").lower()
    assert stage in ["pretrain", "finetune"], \
        f"Invalid stage '{stage}'. Expected 'pretrain' or 'finetune'."

    if stage == "pretrain":
        logging.info("Pre-training with large dataset")
        dm = SpinFlowDataModule(
            cfg.dataset,
            use_distributed_sampler=cfg.trainer.use_distributed_sampler,
            batch_size=cfg.model.batch_size,
            num_workers=cfg.trainer.num_workers,
        )
    elif stage == "finetune":
        logging.info("Fine-tuning with small dataset")
        dm = SpinFlowDataModule(
            cfg.dataset,
            small_cfg=cfg.get("dataset_small", None),
            use_distributed_sampler=cfg.trainer.use_distributed_sampler,
            batch_size=cfg.model.batch_size,
            num_workers=cfg.trainer.num_workers,
        )

        # ↓ override LR *before* the Optimizer is built
        if cfg.model.get("finetune_lr", None):
            cfg.model.optimizer.kwargs["lr"] = cfg.model.finetune_lr

    # ------------------- model -------------------------------------------
    CKPT_SAVE_DIR, resume_ckpt = get_ckpt_path(cfg)
    seed = 42           # Arbitrary seed for reproducibility
    max_epochs = cfg.model.max_epochs 
    weights_ckpt = cfg.model.get('weights_ckpt', '')

    # Make sure only resume or weights ckpt is set
    assert not (resume_ckpt and weights_ckpt), \
        "You can only set one of 'resume_ckpt' or 'weights_ckpt' in the config."

    resume_mode = Path(resume_ckpt).is_file() if resume_ckpt else False
    model = LitMarigold(cfg.model, seed, max_epochs)
    if resume_mode:
        logging.info(f"Resuming training from checkpoint: {resume_ckpt}")
        fit_ckpt_path = resume_ckpt
    else:
        fit_ckpt_path = None
        if weights_ckpt and Path(weights_ckpt).is_file():
            logging.info(f"Warm starting model from weights: {weights_ckpt}")
            state_dict = torch.load(weights_ckpt, weights_only=False, map_location='cpu')['state_dict']
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logging.warning(f"Missing keys: {missing}")
                logging.warning(f"Unexpected keys: {unexpected}")

    # 1) best‐k by your monitor metric
    ckpt_best = ModelCheckpoint(
        monitor=cfg.model.monitor_metric.name,
        mode=cfg.model.monitor_metric.mode,
        dirpath=CKPT_SAVE_DIR,
        save_top_k=5,
        filename="best-{epoch:04d}-{"+cfg.model.monitor_metric.name+":.4f}",
        auto_insert_metric_name=False,
    )

    # 2) most‐recent‐k by “epoch” (the higher the epoch, the more recent)
    ckpt_recent = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        dirpath=CKPT_SAVE_DIR,
        save_top_k=3,
        filename="last-{epoch:04d}-{step:07d}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    
    # ------------------- logger ------------------------------------------
    tb_dir = Path(CKPT_SAVE_DIR) / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    if resume_mode:
        version = _latest_tb_version(CKPT_SAVE_DIR)
        print(f"Resuming TensorBoard logger from version {version} in {tb_dir}")
    else:
        version = None                    # Lightning will create a new one
    tb_logger = TensorBoardLogger(save_dir=tb_dir, version=version)

    # ------------------- trainer -----------------------------------------
    trainer = pl.Trainer(
        logger=[tb_logger],
        callbacks=[ckpt_best, ckpt_recent, lr_cb],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        max_epochs=cfg.model.max_epochs,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_checkpointing=True,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        profiler="simple"
    )

    # ------------------- train / validate --------------------------------
    # TODO: Add ability to resume training from a checkpoint [set ckpt_path in train.fit]
    trainer.fit(model, datamodule=dm, ckpt_path=fit_ckpt_path)

    # ------------------- test (optional) ---------------------------------
    # if cfg.trainer.run_test:
    #     trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
