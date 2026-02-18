# spinflow/model/marigold_lightning.py
import torch
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
from spinflow.util.log_utils import *
from spinflow.util.multi_res_noise import multi_res_noise_like
from spinflow.util.alignment import (align_depth_least_square)


class LitMarigold(pl.LightningModule):
    """
    Lightning interface around MarigoldModel.

    * Adds a **DDPM** forward-diffusion scheduler for training.
    * Adds a **DDIM** reverse-diffusion scheduler for validation.
    """

    def __init__(self, cfg):
        """
        cfg: DictConfig or dict with fields
            .model, .optim.lr, .optim.weight_decay, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.trainer_cfg = cfg.trainer
        self.validation_cfg = cfg.model.validation

        # -------------- randomness -----------------------------------
        self.seed = self.trainer_cfg.seed
        self.train_seed_ls: Union[int, None] = (self.trainer_cfg.seed)
        self.val_seed_ls: Union[int, None] = (self.trainer_cfg.seed)
        self.rand_num_generator = None
        self.global_seed_sequence: List = []

        # -------------- backbone -----------------------------------------
        self.model = MarigoldModel(self.model_cfg)
        self.scheduler_cfg = self.model_cfg.diffusion.scheduler
        self.multires_cfg = self.model_cfg.diffusion.multi_res_noise
        self.model.encode_empty_text()
        self.empty_text_embed = None

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
        self.metrics = mu.MetricManager(
            *self.validation_cfg['evaluation']['metrics']
        )

        # -------------- misc -------------------------------------------
        self.log_config =  self.validation_cfg['evaluation']['log_config']

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            accumulate_grad_batches = self.trainer_cfg.accumulate_grad_batches
            max_epochs = self.trainer_cfg.max_epochs
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=max_epochs * accumulate_grad_batches
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()
        
    def on_train_epoch_start(self):
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

    def training_step(self, inputs):
        train_inputs = {}
        for input_dict in self.model_cfg["dataloader_inputs"]:
            in_key, out_key = input_dict["in_key"], input_dict["out_key"]
            train_inputs[out_key] = inputs[in_key]

        with torch.no_grad():
            rgb_latent = self.model.encode_rgb(train_inputs["rgb_image"])
            gt_target_latent = self.model.encode_depth(train_inputs["depth_label"])
        batch_size = rgb_latent.shape[0]
        device = rgb_latent.device

        noise, timesteps = self.sample_noise(
            gt_target_latent, batch_size, device
        )  # [B, 4, h, w]

        noisy_latents = self.train_sched.add_noise(
            gt_target_latent, noise, timesteps
        )  # [B, 4, h, w]
        
        text_embed = self.empty_text_embed.repeat(
            (batch_size, 1, 1)
        )

        cat_latents = torch.cat(
            [rgb_latent, noisy_latents], dim=1
        ).float()  # [B, 8, h, w]

        model_pred = self.model.unet(
            cat_latents, timesteps, text_embed
        ).sample # [B, 4, h, w]

        if torch.isnan(model_pred).any():
            logging.warning("LitMarigold(): NaN detected in model prediction. Check input data and model parameters.")

        outputs = {}
        if "epsilon" == self.prediction_type:
            outputs["depth_noise_pred"] = model_pred
            inputs["depth_noise_label"] = noise
        elif "v_prediction" == self.prediction_type:
            target = self.train_sched.get_velocity(
                gt_target_latent, noise, timesteps
            )
            outputs["depth_velocity_pred"] = model_pred
            inputs["depth_velocity_label"] = target
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
        self.log('train/loss', loss, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        # self._log_metrics(loss_dict_full, meta_data_full, step=True, sync=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, inputs):
        train_inputs = {}
        for input_dict in self.model_cfg["dataloader_inputs"]:
            in_key, out_key = input_dict["in_key"], input_dict["out_key"]
            train_inputs[out_key] = inputs[in_key]
        # Normalize depth to 0 to 1
        train_inputs["depth_label"] = (train_inputs["depth_label"] + 1) / 2  # Assuming depth_label is in [0, 1] range

        outputs = self.model(
            train_inputs['rgb_image'],
            generator=self.rand_num_generator,
        )
  
        if self.validation_cfg["evaluation"]["alignment"] == "least_square":
            valid_mask = torch.isfinite(train_inputs["depth_label"]) & (train_inputs["depth_label"] > 0)
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=train_inputs["depth_label"].cpu().numpy(),
                pred_arr=outputs["depth_pred"].cpu().numpy(),
                valid_mask_arr=valid_mask.cpu().numpy(),
                return_scale_shift=True,
                max_resolution=self.validation_cfg['evaluation']['align_max_res']
            )
        else:
            raise RuntimeError(f"Unknown alignment method: {self.validation_cfg['alignment']}")

        depth_cfg = self.dataset_cfg['load_cfgs'][-1]
        assert depth_cfg['type'] == 'depth', "Last load_cfg must be depth type"
        
        # Clip dataset to min max
        depth_pred = torch.clamp(
            torch.from_numpy(depth_pred).float(),
            min=depth_cfg['kwargs']['min_depth'],
            max=depth_cfg['kwargs']['max_depth']
        )
        merged_dict = tu.merge_dict(('inputs', train_inputs), ('outputs', outputs))

        # Evaluate depth prediction accuracy
        metrics_dict = self.metrics.update(merged_dict, cur_epoch=self.current_epoch)

        if self.val_step_count < 10 and self.global_rank == 0:
            # Save a few outputs for visualization
            log_depth_img_to_tb(
                self.loggers[0], merged_dict, self.log_config, self.current_epoch, self.val_step_count, prefix='val'
            )
        self.val_step_count += 1
    
        # Construct metrics dictionary from self.metrics
        metrics_dict = tu.prefix_dict('val', metrics_dict)
        # Normalize gt depth
        # depth_np = merged_dict['inputs/depth_label'][0].permute(1,2,0).cpu().numpy() # [H, W, 1]
        # depth_np = cv2.normalize(
        #     depth_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
        # )
        # Convert to turbo colormap
        # import cv2
        # depth_np = cv2.applyColorMap(
        #     (depth_np * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        # )
        # cv2.imwrite("test.jpg", depth_np)s
        self.log_dict(
            metrics_dict, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True
        )

    def on_validation_epoch_start(self):
        assert len(self.loggers) > 0, "At least one logger must be defined in the config."
        self.val_step_count = 0
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

        return [optimizer], [lr_scheduler]
