import os
import torch
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["DIFFUSERS_NO_TF"]       = "1"
# 2) silence the C++ CPU-feature “INFO” spam from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Union, List
import pytorch_lightning as pl

from torch.optim import Adam
from spinflow.util.lr_scheduler import HalfCosineLR
from spinflow.model.lelan.lelan import LeLaN_clip
from spinflow.model.flowpolicy import WaypointFlowPolicy

import spinflow.util.train_utils as tu 
import spinflow.util.loss_utils as lu
import spinflow.util.metric_utils as mu
from spinflow.util.action_utils import (
    unnormalize_action
)

from spinflow.util.log_utils import (
    log_bev_actions_to_tb,
    log_image_actions_to_tb
)

class LitFlowPolicy(pl.LightningModule):
    def __init__(self,
        model_cfg,
        seed: int = 42,
        max_epochs: int = 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.validation_cfg = model_cfg['validation']

        # ---------- randomness ----------
        self.seed = seed
        self.max_epochs = max_epochs
        self.num_freeze_epochs = self.model_cfg.get("num_freeze_epochs", 0)
        self.train_seed_ls: Union[int, None] = (self.seed)
        self.val_seed_ls: Union[int, None] = (self.seed)
        self.global_seed_sequence: List = []

        # ---------- model ----------
        model_project_name = self.model_cfg['project_name']
        if model_project_name == "LeLaNPolicy":
            self.model = LeLaN_clip(self.model_cfg)
        else: # Otherwise assume WaypointFlowPolicy
            self.model = WaypointFlowPolicy(self.model_cfg)

        # ---------- loss & metrics ----------
        self.loss = lu.LossManager(self.model_cfg)
        metric_specs = self.validation_cfg["evaluation"]["metrics"]
        self.metrics = mu.MetricManager(*metric_specs)   # single tracker

        # ---------- misc ----------
        self.log_config =  self.validation_cfg['evaluation']['log_config']

    def load_weights(self, ckpt_path: str):
        """
        Load model weights from a checkpoint.
        """
        self.model._load_pretrained(ckpt_path)

    def training_step(self, inputs, batch_idx):
        """
        Training step for the model.
        """
        #1 Prepare data inputs
        train_inputs = {
            spec["out_key"]: inputs[spec["in_key"]]
            for spec in self.model_cfg["dataloader_inputs"]
        }

        #2 Forward pass with model
        outputs = self.model(train_inputs)
        with torch.no_grad():
            merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        #3 Compute loss
        loss_dict, meta_data = self.loss(merged_dict)
        loss_dict_full, meta_data_full = {}, {}
        meta_data_full = tu.merge_loss_dict(
            meta_data_full, meta_data
        )
        loss_dict_full = tu.merge_loss_dict(
            loss_dict_full, loss_dict
        )
        loss = sum(w*v for w, v in loss_dict.values())

        #4 Log metrics
        loss_dict_full = tu.prefix_dict('train', loss_dict_full)
        meta_data_full = tu.prefix_dict('train', meta_data_full)
        self.log('train/loss', loss, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        
        return loss
    
    def on_after_backward(self):
        # fire right after loss.backward()
        unused = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                unused.append(name)
        if unused:
            print(f"[{self.__class__.__name__}] parameters never used in loss:")
            for name in unused:
                print("   ", name)

    def on_validation_epoch_start(self):
        assert len(self.loggers) > 0, "At least one logger must be defined in the config."
        self.val_step_count = 0
        self.metrics.set_writer(self.loggers[0])

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """
        Validation step for the model.
        """
        train_inputs = {
            spec["out_key"]: inputs[spec["in_key"]]
            for spec in self.model_cfg["dataloader_inputs"]
        }

        normalize_actions = self.model_cfg['normalize_actions']
        if self.model_cfg['vision_encoder']['class_name'] == 'MarigoldModel':
            outputs = self.model.infer(
                train_inputs,
                **self.model_cfg['validation']['integrator']['kwargs']
            )
        else:
            outputs = self.model(train_inputs, unnormalize=normalize_actions)

        # Unnormalize action labels if needed (model outputs already unnormalized)
        if normalize_actions:
            unnorm_action = unnormalize_action(
                train_inputs['action_label'],
                self.validation_cfg['action_range'],
                self.validation_cfg['action_stats'],
                self.model_cfg['action_dim']
            )
            train_inputs['action_label'] = unnorm_action

        # Compute validation metrics
        merged_dict = tu.merge_dict(('inputs', train_inputs), ('outputs', outputs))

        if 'vis_config' in self.validation_cfg['evaluation']:
            vis_config = self.validation_cfg['evaluation']['vis_config']
            B = inputs['action_label'].shape[0]
            max_batch_size = 16
            if B > max_batch_size:
                batch_indices = torch.randperm(B)[:max_batch_size]
            else:
                batch_indices = torch.arange(B)
            for vis_spec in vis_config:
                if vis_spec['type'] == 'bev_actions':
                    log_bev_actions_to_tb(
                        self.loggers[0], merged_dict, vis_spec, self.current_epoch, self.val_step_count, batch_indices, prefix='val'
                    )
                elif vis_spec['type'] == 'image_actions':
                    log_image_actions_to_tb(
                        self.loggers[0], merged_dict, inputs, vis_spec, self.current_epoch, self.val_step_count, batch_indices, prefix='val'
                    )   
            # TODO: Add image view logging here 
        # self.val_step_count += 1

        # Log metrics
        metrics_dict = self.metrics.update(merged_dict, cur_epoch=self.current_epoch)
        metrics_dict = tu.prefix_dict("val", metrics_dict)

        self.log_dict(
            metrics_dict, on_step=False, on_epoch=True, rank_zero_only=False, sync_dist=True
        )

    def on_validation_epoch_end(self):
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.model_cfg.optimizer
        lr_scheduler_cfg = self.model_cfg.lr_scheduler
        if isinstance(optimizer_cfg, DictConfig):
            optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
        if isinstance(lr_scheduler_cfg, DictConfig):
            lr_scheduler_cfg = OmegaConf.to_container(lr_scheduler_cfg, resolve=True)

        try:
            optimizer = globals()[optimizer_cfg['name']](
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **optimizer_cfg['kwargs'],
            )

            # ensure that initial LR is set for resuming training from ckpt
            for pg in optimizer.param_groups:
                pg.setdefault("initial_lr", pg["lr"])

            lr_scheduler = globals()[lr_scheduler_cfg['name']](
                optimizer, **lr_scheduler_cfg['kwargs']
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to configure optimizer or scheduler: {e}"
            )

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }]