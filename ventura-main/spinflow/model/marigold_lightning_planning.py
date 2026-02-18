# spinflow/model/marigold_lightning.py
import os
import torch
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["DIFFUSERS_NO_TF"]       = "1"
# 2) silence the C++ CPU-feature “INFO” spam from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import pytorch_lightning as pl
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Union, List
import logging

from pytorch_lightning.utilities import rank_zero_only

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# Custom imports
from spinflow.model.marigold import MarigoldModel
from spinflow.util.lr_scheduler import IterExponential
from spinflow.util.seeding import generate_seed_sequence
import spinflow.util.loss_utils as lu
import spinflow.util.train_utils as tu
import spinflow.util.metric_utils as mu
from spinflow.util.log_utils import (log_path_mask_to_tb)
from spinflow.util.multi_res_noise import multi_res_noise_like

class LitMarigold(pl.LightningModule):
    """
    Lightning interface around MarigoldModel.

    * Adds a **DDPM** forward-diffusion scheduler for training.
    * Adds a **DDIM** reverse-diffusion scheduler for validation.
    """

    def __init__(self, 
        model_cfg,
        seed: int = 42,
        max_epochs: int = 1000
    ):
        """
        cfg: DictConfig or dict with fields
            .model, .optim.lr, .optim.weight_decay, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.validation_cfg = model_cfg.validation

        # -------------- randomness -----------------------------------
        self.seed = seed
        self.max_epochs = max_epochs
        self.num_freeze_epochs  = self.model_cfg.get("num_freeze_epochs", 0)
        self.train_seed_ls: Union[int, None] = (self.seed)
        self.val_seed_ls: Union[int, None] = (self.seed)
        self.rand_num_generator = None
        self.global_seed_sequence: List = []

        # -------------- backbone -----------------------------------------
        self.model = MarigoldModel(self.model_cfg)
        self.scheduler_cfg = self.model_cfg.diffusion.scheduler
        self.multires_cfg = self.model_cfg.diffusion.multi_res_noise
        self.model.encode_empty_text()
        self.empty_text_embed = None
        self.cond_method = self.model_cfg['backbone'].get('cond_method', 'Marigold')

        if self.num_freeze_epochs > 0:
            logging.info(
                f"⏸  Freezing UNet / ControlNet for the first {self.num_freeze_epochs} epochs"
            )
            for p in self.model.unet.parameters(): 
                p.requires_grad_(False)

        # -------------- noise schedulers ---------------------------------
        self.train_sched = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type=self.model.scheduler.config.prediction_type,
        )
        self.prediction_type = self.train_sched.config.prediction_type
        self.scheduler_timesteps = (
            self.train_sched.num_train_timesteps
        )

        # Multi-resolution noise
        self.apply_multi_res_noise = self.multires_cfg is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.multires_cfg.strength
            self.annealed_mr_noise = self.multires_cfg.annealed
            self.mr_noise_downscale_strategy = (
                self.multires_cfg.downscale_strategy
            )

        # -------------- loss ---------------------------------------------
        self.loss = lu.LossManager(self.model_cfg)

        metric_specs = self.validation_cfg["evaluation"]["metrics"]
        self.use_two_sets = getattr(self.model_cfg, "log_small_dataset", False)
        if self.use_two_sets:
            self.metrics = {
                "big":   mu.MetricManager(*metric_specs),
                "small": mu.MetricManager(*metric_specs),
            }
        else:
            self.metrics = mu.MetricManager(*metric_specs)   # single tracker

        # -------------- misc -------------------------------------------
        self.log_config =  self.validation_cfg['evaluation']['log_config']

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            accumulate_grad_batches = self.model_cfg.accumulate_grad_batches
            max_epochs = self.max_epochs
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=max_epochs * accumulate_grad_batches
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()
        
    def on_train_epoch_start(self):
        if (not self.model_cfg.get("freeze_unet", False) and self.current_epoch >= self.num_freeze_epochs
                and getattr(self, "_modules_unfrozen", False) is False):
            self._modules_unfrozen = True
            self.print(f"⏩  Unfreezing UNet (epoch {self.current_epoch})")
            for p in self.model.unet.parameters():
                p.requires_grad_(True)
            if hasattr(self.model, "controlnet") and self.model.controlnet is not None:
                self.print(f"⏩  Unfreezing ControlNet (epoch {self.current_epoch})")
                for p in self.model.controlnet.parameters():
                    p.requires_grad_(True)

        with torch.no_grad():
            self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(self.device)

        if self.seed is not None:
            local_seed = self._get_next_seed()
            self.rand_num_generator = torch.Generator(device=self.device)
            self.rand_num_generator.manual_seed(local_seed)
        else:
            self.rand_num_generator = None
  
    def sample_noise(self, gt_target_latent, batch_size, device):
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.scheduler_timesteps,
            (batch_size,),
            device=device,
            generator=self.rand_num_generator,
        ).long()  # [B]

        if self.apply_multi_res_noise:
            strength = self.mr_noise_strength
            if self.annealed_mr_noise:
                # calculate strength depending on t
                strength = strength * (timesteps / self.scheduler_timesteps)
            noise = multi_res_noise_like(
                gt_target_latent,
                strength=strength,
                downscale_strategy=self.mr_noise_downscale_strategy,
                generator=self.rand_num_generator,
                device=device,
            )
        else:
            noise = torch.randn(
                gt_target_latent.shape,
                device=device,
                generator=self.rand_num_generator,
            )  # [B, 4, h, w]

        return noise, timesteps

    def training_step(self, batch, batch_idx, dataloader_idx: int | str = 0):
        # 1) CombinedLoader gives dict; single loader gives tuple/dict
        if isinstance(batch, dict) and "big" in batch and "small" in batch:
            loss_big   = self._training_step_single(batch["big"])
            loss_small = self._training_step_single(batch["small"])
            alpha      = getattr(self.hparams, "finetune_loss_weight", 1.0)
            return loss_big + alpha * loss_small
        else:
            return self._training_step_single(batch)

    def _training_step_single(self, inputs):
        train_inputs = {}
        for input_dict in self.model_cfg["dataloader_inputs"]:
            in_key, out_key = input_dict["in_key"], input_dict["out_key"]
            train_inputs[out_key] = inputs[in_key]

        with torch.no_grad():
            # Create per modality shared dropout masks
            droppable_keys = [
                k for k, spec in self.model.unet_inputs.items()
                if spec.get("dropout_prob", 0.0) > 0.0
            ]
            dropout_prob = (
                self.model.unet_inputs[droppable_keys[0]]["dropout_prob"]
                if droppable_keys else 0.0
            )
            batch_size = next(iter(train_inputs.values())).shape[0]
            device      = self.device         # or any input.device
            keep_mask_flat = (
                torch.rand(batch_size, device=device) > dropout_prob
                if dropout_prob > 0.0 else
                torch.ones(batch_size, dtype=torch.bool, device=device)
            )
            keep_img  = keep_mask_flat.view(-1, 1, 1, 1)   # [B,1,1,1]
            keep_txt  = keep_mask_flat.view(-1, 1, 1)      # [B,1,1]

            rgb_latents = []
            ctrl_latents = []
            text_latents = []
            for key, spec in self.model.unet_inputs.items():
                modality        = spec['modality']
                allow_dropout   = spec.get("dropout_prob", 0.0) > 0.0
                x               = train_inputs[key]

                if modality == 'image':
                    if x.ndim == 5:  # B x T x C x H x W
                        x = x[:, 0]  # Take the first frame
                    assert x.ndim == 4, f"Expected 4D input for {key}, got {x.dim()}D"

                    is_control = self.model._is_control_input(spec)
                    if is_control:
                        latent = x
                    else:
                        assert x.shape[1] == 3, f"Expected 3 channels for {key}, got {x.shape[1]} channels"
                        latent = self.model.encode_rgb(x)

                    if allow_dropout:                       # goal_image has prob 0.1
                        latent = torch.where(
                            keep_img, latent,
                            self.model.encode_empty_image_like(latent)
                        )

                    if is_control:
                        ctrl_latents.append(latent)
                    else:
                        rgb_latents.append(latent)
                elif modality == 'text':
                    text_latent = self.model.encode_text(x)         # [B,L,D]
                    if allow_dropout:                       # goal_command has prob 0.1
                        text_latent = torch.where(
                            keep_txt, text_latent,
                            self.model.encode_empty_text_like(text_latent)
                        )
                    text_latents.append(text_latent)
                else:
                    raise ValueError(f"Unknown modality {modality} for key {key}")

            # Remove time dimension
            gt_target = train_inputs["path_mask_label"][:, 0]  # [B, 1, h, w]
            gt_target_latent = self.model.encode_path_mask(gt_target)

        batch_size = rgb_latents[0].shape[0]
        device = rgb_latents[0].device

        noise, timesteps = self.sample_noise(
            gt_target_latent, batch_size, device
        )  # [B, 4, h, w]

        noisy_latents = self.train_sched.add_noise(
            gt_target_latent, noise, timesteps
        )  # [B, 4, h, w]
        
        if len(text_latents) > 0:
            text_embed = torch.cat(text_latents, dim=0) # [B, Len, F]
        else:
            text_embed = self.empty_text_embed.repeat(
                (batch_size, 1, 1)
            )

        rgb_latents = torch.cat(rgb_latents, dim=1)  # [B, 3*T, h, w]
        if self.cond_method == 'Marigold':
            cat_latents = torch.cat(
                [rgb_latents, noisy_latents], dim=1
            ).float()  # [B, 8, h, w]

            model_pred = self.model.unet(
                cat_latents, timesteps, text_embed, 
            ).sample # [B, 4, h, w]
        elif self.cond_method == 'ControlNet':
            cat_latents = torch.cat(
                [rgb_latents, noisy_latents], dim=1
            ).float()  # [B, 7, h, w]

            if ctrl_latents:
                ctrl_latents = torch.cat(ctrl_latents, dim=1)  # [B, 3*T, h, w]
            model_pred = self.model.controlnet(
                cat_latents, timesteps, text_embed, ctrl_latents
            ).sample # [B, 4, h, w]

        if torch.isnan(model_pred).any():
            logging.warning("LitMarigold(): NaN detected in model prediction. Check input data and model parameters.")

        outputs = {}
        if "epsilon" == self.prediction_type:
            outputs["path_mask_noise_pred"] = model_pred
            inputs["path_mask_noise_label"] = noise
        elif "v_prediction" == self.prediction_type:
            target = self.train_sched.get_velocity(
                gt_target_latent, noise, timesteps
            )
            outputs["path_mask_velocity_pred"] = model_pred
            inputs["path_mask_velocity_label"] = target
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Merge and prefix 
        with torch.no_grad():
            merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        # Compute loss
        loss_dict, meta_data = self.loss(merged_dict)

        loss_dict_full, meta_data_full = {}, {}
        meta_data_full = tu.merge_loss_dict(
            meta_data_full, meta_data
        )
        loss_dict_full = tu.merge_loss_dict(
            loss_dict_full, loss_dict
        )
        loss = sum(w*v for w, v in loss_dict.values())

        # Log metrics
        loss_dict_full = tu.prefix_dict('train', loss_dict_full)
        meta_data_full = tu.prefix_dict('train', meta_data_full)
        self.log(
            'train/loss', 
            loss, 
            on_step=True, 
            prog_bar=True, 
            rank_zero_only=True, 
            sync_dist=False,
            batch_size=batch_size
        )
        # self._log_metrics(loss_dict_full, meta_data_full, step=True, sync=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx: int | str = 0):
        if self.use_two_sets and dataloader_idx == 0:
            return
        self._validation_step_single(batch, dataloader_idx)

    @torch.no_grad()
    def _validation_step_single(self, inputs, dataloader_idx):
        train_inputs = {}
        for input_dict in self.model_cfg["dataloader_inputs"]:
            in_key, out_key = input_dict["in_key"], input_dict["out_key"]
            train_inputs[out_key] = inputs[in_key]

        scheduler_kwargs = self.validation_cfg['scheduler']['kwargs']
        ensemble_size = scheduler_kwargs.get('ensemble_size', 1)
        ensemble_outputs = {}
        for _ in range(ensemble_size):
            outputs = self.model(
                train_inputs,
                denoising_steps=scheduler_kwargs['denoising_steps'],
                generator=self.rand_num_generator,
                cfg_scale=scheduler_kwargs.get('cfg_scale', 1.0),
                guidance_rescale=scheduler_kwargs.get('guidance_rescale', 0.0)
            )
            # Stack outputs along new time dimension
            for key in outputs.keys():
                if key not in ensemble_outputs:
                    ensemble_outputs[key] = outputs[key].unsqueeze(1)
                else:
                    ensemble_outputs[key] = torch.cat(
                        [ensemble_outputs[key], outputs[key].unsqueeze(1)],
                        dim=1
                    ) # [B, E, C, H, W]
        outputs = ensemble_outputs

        # Scale output prediction and label ranges
        path_mask_cfg = self.validation_cfg["postprocess"]["path_mask"]        
        in_min, in_max = path_mask_cfg['in_range'][0], path_mask_cfg['in_range'][1]
        out_min, out_max = path_mask_cfg['out_range'][0], path_mask_cfg['out_range'][1]
        train_inputs['path_mask_label'] = torch.clip(
            train_inputs['path_mask_label'].float(), in_min, in_max
        ).squeeze(1)  # [B, 1, H, W]
        train_inputs['path_mask_label'] = (
            train_inputs['path_mask_label'] - in_min
        ) / (in_max - in_min)
        train_inputs['path_mask_label'] = torch.clamp(
            train_inputs['path_mask_label'].float(),
            min=out_min,
            max=out_max
        )
        assert outputs['path_mask_pred'].min() >= out_min and \
                outputs['path_mask_pred'].max() <= out_max, \
            f"Output path mask prediction out of range: {outputs['path_mask_pred'].min()} - {outputs['path_mask_pred'].max()}"
        merged_dict = tu.merge_dict(('inputs', train_inputs), ('outputs', outputs))

        # Evaluate depth prediction accuracy
        if self.val_step_count % 10 and self.global_rank == 0:
            # Save a few outputs for visualization
            log_path_mask_to_tb(
                self.loggers[0], merged_dict, self.log_config, self.current_epoch, self.val_step_count, prefix='val'
            )
        self.val_step_count += 1
    
        # Construct metrics dictionary from self.metrics
        if isinstance(self.metrics, dict):
            key          = "big" if dataloader_idx in (0, "big") else "small"
            tracker      = self.metrics[key]
        else:
            tracker      = self.metrics

        metrics_dict = tracker.update(merged_dict, cur_epoch=self.current_epoch)
        metrics_dict = tu.prefix_dict("val", metrics_dict)
        
        self.log_dict(
            metrics_dict, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=False, sync_dist=True
        )

    def on_validation_epoch_start(self):
        assert len(self.loggers) > 0, "At least one logger must be defined in the config."
        self.val_step_count = 0
        if isinstance(self.metrics, dict):
            for mm in self.metrics.values():
                mm.set_writer(self.loggers[0])
        else:
            self.metrics.set_writer(self.loggers[0])

        # Generate seed
        if self.val_seed_ls is not None:
            local_seed = self._get_next_seed()
            self.rand_num_generator = torch.Generator(device=self.device)
            self.rand_num_generator.manual_seed(local_seed)
        else:
            self.rand_num_generator = None
        
    def on_validation_epoch_end(self):
        """Reset metrics and save results."""
        # TODO: Add some sort of global logging dump here
        if isinstance(self.metrics, dict):
            for mm in self.metrics.values():
                mm.reset()
        else:
            self.metrics.reset()
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer_cfg = self.model_cfg.optimizer
        lr_scheduler_cfg = self.model_cfg.lr_scheduler
        # Convert to dict if it's a DictConfig
        if isinstance(optimizer_cfg, DictConfig):
            optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
        if isinstance(lr_scheduler_cfg, DictConfig):
            lr_scheduler_cfg = OmegaConf.to_container(lr_scheduler_cfg, resolve=True)
        try:
            optimizer = globals()[optimizer_cfg['name']](
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **optimizer_cfg['kwargs'],
            )

            if 'lr_lambda' in lr_scheduler_cfg:
                lr_func = IterExponential(
                    **lr_scheduler_cfg['kwargs'],
                )
                lr_scheduler_cfg['kwargs'] = {
                    'lr_lambda': lr_func,
                }
            
            lr_scheduler = globals()[lr_scheduler_cfg['name']](
                optimizer, **lr_scheduler_cfg['kwargs']
            )
        except KeyError:
            raise ValueError("Optimizer or LR scheduler not found in globals.")

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }]
