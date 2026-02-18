"""
"""

import math, inspect
from typing import Dict, Optional, Sequence, Union
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from torchvision import models
from transformers import CLIPTextModel, CLIPTokenizer

from spinflow.util.log_utils import logging
from spinflow.util.action_utils import (
    unnormalize_action
)
# Vision encoders
from spinflow.model.nomad_vint import NoMaD_ViNT
from spinflow.model.blocks.convnet_spatial import (ConvNetSpatial)
from spinflow.model.marigold import MarigoldModel

# Context aggregators
from spinflow.model.blocks.context_aggregator import ContextAggregator

# Action heads
from spinflow.model.blocks.dense_network import DenseNetwork
from spinflow.model.blocks.conditional_unet1d import ConditionalUnet1D
from spinflow.model.blocks.flow_scheduler import (
    get_uniform_step, 
    get_shifted_beta_step
)

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

# --------------------------------------------------------------------------- #
#                       1‑D conditional diffusion policy                      #
# --------------------------------------------------------------------------- #
class WaypointFlowPolicy(nn.Module):
    """
    Conditional 1-D U-Net that denoises a (T,2) waypoint sequence.
    Conditioning vector = concatenation of feature extractor latents.
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        # Convert DictConfig to dict if necessary
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        self.cfg = cfg
        self.vision_cfg = cfg['vision_encoder']
        self.postvision_cfg = cfg.get('postvision_encoder', None)
        self.sampler_cfg = cfg['flow_sampler']
        self.action_head_cfg = cfg['action_head']
        self.distance_head_cfg = cfg['distance_head']
        self.validation_cfg = cfg['validation']

        try:
            self.vision_encoder = globals()[self.vision_cfg["class_name"]](
                cfg['vision_encoder']
            )

            self.postvision_encoder = None
            if self.postvision_cfg is not None:
                self.postvision_encoder = globals()[self.postvision_cfg["class_name"]](
                    cfg['postvision_encoder']
                )
        except KeyError as e:
            logging.error(f"Model component not found {e}, ensure it is implemented")
            raise

        try:
            sampler_name = self.sampler_cfg['name']
            if sampler_name =='VanillaBC':
                self.fm_sampler = None
            else:
                self.fm_sampler = globals()[sampler_name](**self.sampler_cfg['kwargs'])
        except KeyError as e:
            logging.error(f"Sampler not found {e}, ensure it is implemented")
            raise

        # ---------- vision aggregation adapter ----------
        if self.vision_cfg['aggregation']['strategy'] == 'native':
            self.context_aggregator = nn.Identity()
        elif self.vision_cfg['aggregation']['strategy'] == 'vitsimple':
            self.context_aggregator = ContextAggregator(cfg['vision_encoder']['aggregation'])

        # ---------- action head ---------
        self.action_head = globals()[cfg['action_head']['name']](
            **cfg['action_head']['kwargs']
        )

        # ---------- action params ----------
        self.aggr_cfg = cfg['vision_encoder']['aggregation']

        self.action_dim = cfg['action_dim']
        self.num_actions = cfg['num_actions']
        self.normalize_actions = cfg['normalize_actions']

        self.register_buffer( 'action_range',
            torch.tensor(self.validation_cfg['action_range']).float())
        self.register_buffer('action_stats', 
            torch.tensor(self.validation_cfg['action_stats']).float())
        
        # ---------- load pretrained weights ----------
        self._load_pretrained()

    def _load_pretrained(self):
        """Warm-start individual sub-modules if cfg.pretrained is given."""
        paths = self.cfg.get("pretrained", {})
        if not paths:
            return                                  # nothing to do

        for name, sub_dict in paths.items():
            ckpt_path = sub_dict.get("ckpt", None)
            optimize_inference = sub_dict.get("optimize_inference", False)

            if ckpt_path in (None, "NONE", "", False):
                continue                            # user disabled
            if not hasattr(self, name):
                raise ValueError(f"No sub-module named '{name}'")

            module = getattr(self, name)
            # 1. load raw checkpoint (Lightning or plain)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in sd:                  # Lightning ckpt
                sd = sd["state_dict"]
            # 2. strip prefixes like "module." or "model."
            sd = {k.split(".", 1)[-1]: v for k, v in sd.items()}
            # 3. keep only tensors that belong to THIS sub-module
            prefix = f"{name}."
            filtered = {
                k[len(prefix):]: v
                for k, v in sd.items() if k.startswith(prefix)
            } or sd                                 # fall back: whole dict


            # 4. finally load
            missing, unexpected = module.load_state_dict(
                filtered, strict=False
            )
            # 5. Optimize inference if requested
            if optimize_inference:
                module.optimize_inference()
            logging.info(
                f"Loaded {ckpt_path} → {name}  "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    # --------------------------------------------------------------------- #
    #                            training forward                           #
    # --------------------------------------------------------------------- #
    def get_vision_feats(
        self, 
        inputs,
        **kwargs
    ):
        """Extract vision features from the inputs."""
        #1 Extract vision features
        feats_dict = {}
        if not self.cfg['use_gt_feats']:
            if 'cfg_scale' in kwargs:
                feats_dict = self.vision_encoder(
                    inputs, 
                    denoising_steps=kwargs['default_denoising_steps'],
                    cfg_scale=kwargs['cfg_scale'],
                    guidance_rescale=kwargs['guidance_rescale'],
                    generator=kwargs.get('generator', None),
                    ensemble_size=kwargs.get('ensemble_size', 1)
                )
            else:
                feats_dict = self.vision_encoder(inputs)

        # from PIL import Image
        # import numpy as np
        # Image.fromarray((feats_dict['path_mask_pred'][1,0]>0.5).cpu().numpy().astype(np.uint8)*255).save("test.jpg")
        # Ensure no nans
        for key, value in feats_dict.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in {key} of vision features")

        #2 Aggregate features
        if self.postvision_encoder is not None:
            feats_dict.update({
                key: inputs[key] for key in self.postvision_cfg['encoder']['in_keys'] if key in inputs and key not in feats_dict
            })
            # Add post-vision features
            feats_dict.update(
                self.postvision_encoder(feats_dict)
            )

        context_feats = self.context_aggregator(feats_dict)  # [B, C, H, W] -> [B, hidden_dim]
        assert context_feats.ndim == 2, "Context aggregator must output [B, hidden_dim] tensor"
        feats_dict["ctx_feats"] = context_feats

        # Ensure no nans
        for key, value in feats_dict.items():
            if torch.isnan(value).any():
                import pdb; pdb.set_trace()  # Debugging line to inspect NaNs
                raise ValueError(f"NaN detected in {key} of vision features")

        return feats_dict

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Build a Brownian-bridge training pair with torchcfm and predict the
        drift v_t.

        Expected keys in `inputs`
        -------------------------
        action_label : [B,T,2]   ground-truth trajectory
        plus whatever the vision encoder needs (e.g. rgb_image, state, …)
        """
        gt_traj = inputs["action_label"].float()                 # (B,T,2)
        B, _, _ = gt_traj.shape
        device  = gt_traj.device

        # --- conditioning features --------------------------
        outputs = self.get_vision_feats(inputs, **kwargs)
        assert "ctx_feats" in outputs, "Vision encoder must return 'ctx_feats' key"
        global_cond = outputs["ctx_feats"]  # [B, hidden_dim]

        # --- Switch between different policy learning methods ---------
        if self.sampler_cfg['name'] == 'VanillaBC':
            action_pred = self.action_head(global_cond)
            outputs.update({
                "action_pred": action_pred
            })
        elif self.sampler_cfg['name'] == 'ConditionalFlowMatcher':
            # --- sample Brownian-bridge pair with torchcfm ---------------
            x0 = self.sample_action_prior(B, device=device)         # (B,T,2)
            t, xt, ut = self.fm_sampler.sample_location_and_conditional_flow(
                x0, gt_traj
            )                                                       # t:(B,), xt/ut:(B,T,2)

            # --- network prediction --------------------------------------
            vt_pred = self.action_head(
                xt, t, global_cond=global_cond
            )                                                       # (B,T,2)

            outputs.update({
                "action_vt_pred": vt_pred,
                "action_vt_target": ut,
                "timestep": t,
            })
        else:
            raise ValueError(f"Unknown sampler {self.sampler_cfg['name']}")

        return outputs
    
    def sample_action_prior(self, batch_size, device='cuda'):
        """
        Sample gaussian distributed actions for the given batch size.
        Returns: [batch_size, 2] tensor of random actions.
        """
        # Randomly sample actions in the range [-1, 1]
        # noise = torch.rand((1, self.num_actions, self.action_dim)).to(device)
        # x0 = torch.rand((batch_size, self.num_actions, self.action_dim)).to(device)
        # x0 = torch.normal(mean=0.0, std=1.0, size=(batch_size, self.num_actions, self.action_dim)).to(device)
        x0 = torch.randn((batch_size, self.num_actions, self.action_dim)).to(device)
        # x0 = noise.expand(batch_size, -1, -1)
        return x0

    def infer(self, 
        inputs: Dict[str, torch.Tensor],
        integration_method: str = 'euler',
        dt: float = 1.0,
        total_steps: int = 10,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass
        """
        #1 Extract observation/conditioning features
        B = inputs['rgb_image'].shape[0]
        device = inputs['rgb_image'].device
        vision_feats = self.get_vision_feats(inputs, **kwargs)
        ctx_feats = vision_feats['ctx_feats']

        if self.sampler_cfg['name'] == 'VanillaBC':
            # Action head predicts normalized actions
            action_pred = self.action_head(ctx_feats)  # [B, T, 2]
        elif self.sampler_cfg['name'] == 'ConditionalFlowMatcher':
            #2 Integrate velocity -> actions
            x = self.sample_action_prior(B, device=device) # initially x0 
            for k in range(total_steps):
                # t_k = torch.full((B,1,1), (k + 0.5)/total_steps, device=device)  # midpoint
                t_k = get_uniform_step(k, total_steps, device=device)

                if integration_method == 'euler':
                    v = self.action_head(x, t_k, global_cond=ctx_feats)
                    x = x + v * dt
                elif integration_method == 'heun':
                    v1 = self.action_head(x, t_k, global_cond=ctx_feats)
                    x_e = x + dt * v1
                    t_np1 = get_uniform_step(k + 1, total_steps, device=device)
                    v2 = self.action_head(x_e, t_np1, global_cond=ctx_feats)
                    x = x + 0.5 * dt * (v1 + v2)
                else:
                    raise ValueError(f"Unknown integrator: {integration_method}")
            action_pred = x

        if self.normalize_actions:
            # Unnormalize actions to the original range
            action_pred = unnormalize_action(
                action_pred,
                self.action_range,
                self.action_stats,
                self.action_dim
            )

        #3 Distance head
        # distance_pred = self.distance_head(ctx_feats)
    
        outputs =  {
            "action_pred": action_pred,
            # "distance_pred": distance_pred,
        }
        outputs.update(vision_feats)

        return outputs

