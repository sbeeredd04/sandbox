# train_depth.py
from __future__ import annotations
import os 

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  your modules
# --------------------------------------------------------------------------- #
from spinflow.dataset.dataloader import SpinFlowDataModule        # created earlier
from spinflow.model.marigold_lightning import LitMarigold    # wrap MarigoldPlanner in a LightningModule

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
@hydra.main(version_base="1.3", config_path="../../config", config_name="depth")
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
    # TODO: Properly figure out the correct way to handle checkpoints vs weights
    CKPT_SAVE_DIR, ckpt_path = get_ckpt_path(cfg)
    model = LitMarigold(cfg)
    # if ckpt_path is not None:
    #     model.load_checkpoint(ckpt_path)

    # ------------------- callbacks ---------------------------------------
    ckpt_cb = ModelCheckpoint(
        monitor=cfg.model.monitor_metric.name,
        mode=cfg.model.monitor_metric.mode,
        dirpath=CKPT_SAVE_DIR,
        save_top_k=5,
        filename="{epoch:04d}-{step:07d}-{"+cfg.model.monitor_metric.name+":.4f}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # ------------------- logger ------------------------------------------
    tb_dir = Path(CKPT_SAVE_DIR) / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=tb_dir)

    # ------------------- trainer -----------------------------------------
    trainer = pl.Trainer(
        logger=[tb_logger],
        callbacks=[ckpt_cb, lr_cb],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=True,
        deterministic=False
    )

    # ------------------- train / validate --------------------------------
    # TODO: Add ability to resume training from a checkpoint [set ckpt_path in train.fit]
    trainer.fit(model, datamodule=dm)

    # ------------------- test (optional) ---------------------------------
    # if cfg.trainer.run_test:
    #     trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
