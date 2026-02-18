"""
Pipeline for Marigold inspired image space planning: https://marigoldcomputervision.github.io.

This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
"""
import logging
from tqdm import tqdm

import types
import torch
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
from torch import autocast
from torch import nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from scripts.utils.log_utils import logging

from omegaconf import DictConfig, OmegaConf

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)

from typing import Dict, Optional, Union, Sequence

from spinflow.model.blocks.controlnet import ControlNet
from spinflow.util.image_utils import ( 
    get_tv_resample_method, 
    resize_max_res,
    colorize_depth_maps
)

class MarigoldModel(nn.Module):
    """
    Inputs:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """
    def __init__(
        self,
        model_cfg: Dict
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.text_cfg = model_cfg.get('text', {})
        self.pipeline_cfg = model_cfg['pipeline']
        self.scheduler_cfg = model_cfg['validation']['scheduler']
        self.vae_latent_size = 4
        self.cond_method = model_cfg['backbone']['cond_method']
        self.setup_inputs(model_cfg)
        self.unet_in_channels =  self.vae_latent_size * self.num_unet_inputs  # +1 for depth input

        # Load pretrained model settings
        try:
            if model_cfg['backbone'].get('use_pretrained', True):
                logging.info("Loading pretrained UNet model...")
                self.unet = globals()[model_cfg['backbone']['name']].from_pretrained(
                    model_cfg['backbone']["pretrained_ckpt"], subfolder="unet"
                )
            else:
                logging.info("Loading non-pretrained UNet model from config...")
                self.unet = globals()[model_cfg['backbone']['name']].from_config(
                    model_cfg['backbone']["pretrained_ckpt"], subfolder="unet"
                )

            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.unet.set_attn_processor(AttnProcessor2_0())       # uses torch SDPA
            except Exception:
                self.unet.enable_xformers_memory_efficient_attention() # fallback
            if self.unet.config["in_channels"] !=  self.unet_in_channels:
                self._replace_unet_conv_in()

            self.controlnet = None
            if self.cond_method == "ControlNet":
                self.controlnet = globals()['ControlNet'].from_unet(
                    self.unet, model_cfg['backbone']['kwargs']
                )

            self.vae = globals()[model_cfg['encoder']['name']].from_pretrained(
                model_cfg['encoder']["pretrained_ckpt"], subfolder="vae"
            )
            self.text_encoder = globals()[self.text_cfg['text_encoder']].from_pretrained(
                self.text_cfg['text_encoder_ckpt'], subfolder="text_encoder"
            )
            self.tokenizer = globals()[self.text_cfg['tokenizer']].from_pretrained(
                self.text_cfg['tokenizer_ckpt'], subfolder="tokenizer"
            )
            self.scheduler = globals()[self.scheduler_cfg['name']].from_pretrained(
                self.scheduler_cfg['pretrained_ckpt'], subfolder="scheduler"
            )
        except KeyError as e:
            logging.error(f"Model component not found: {e}. Please ensure the model components are correctly imported.")
            raise

        self.warned_once = False

        self.empty_text_embed = None
        self.setup(model_cfg)

        # Setup shift and scale configs
        self.scale_invariant = self.scheduler_cfg['kwargs'].get('scale_invariant', True)
        self.shift_invariant = self.scheduler_cfg['kwargs'].get('shift_invariant', True)
        self.default_denoising_steps = self.scheduler_cfg['kwargs'].get('default_denoising_steps', 4)
        self.default_processing_resolution = self.scheduler_cfg['kwargs'].get('default_processing_resolution', 768)

        self.latent_scale_factor = model_cfg['encoder']['kwargs']['latent_scale_factor']

    def setup_inputs(self, model_cfg):
        # Setup input -> modality mappings for each model component
        pipeline_flags = model_cfg['pipeline']
        comp_inputs = {}
        num_unet_inputs = 1 # One for the target (always there)

        for comp, comp_dict in pipeline_flags.items():
            input_keys = comp_dict.get('input_keys', [])
            if len(input_keys) == 0:
                continue

            comp_key = f'{comp}_inputs'
            if not comp_key in comp_inputs:
                comp_inputs[comp_key] = {}

            for input_dict in input_keys:
                name = input_dict['name']
                modality = input_dict['modality']
                comp_inputs[comp_key][name] = input_dict.copy()
                if modality == 'image' and comp == 'unet':
                    num_unet_inputs += 1
            setattr(self, comp_key, comp_inputs[comp_key])
            logging.info(f"Input modality mapping for {comp}: {comp_inputs[comp_key]}")
        if model_cfg['backbone']['cond_method'] == "Marigold":
            setattr(self, 'num_unet_inputs', num_unet_inputs)
        else:
            setattr(self, 'num_unet_inputs', 2) # Always include path mask label

    def setup(self, model_cfg):
        # Enable training pipeline
        pipeline_flags = model_cfg['pipeline']
        for comp, comp_dict in pipeline_flags.items():
            requires_grad = not comp_dict.get('frozen', False)
            getattr(self, comp).requires_grad_(requires_grad)

        # Print which modules are frozen
        frozen_modules = [
            comp for comp, comp_dict in pipeline_flags.items() if comp_dict.get('frozen', False)
        ]
        if frozen_modules:
            logging.info(f"Frozen modules: {', '.join(frozen_modules)}")

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 4* #in_channels (8 with RGB + depth)
        _weight = self.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, self.num_unet_inputs, 1, 1))  # Keep selected channel(s)
        _weight *= 1.0 / self.num_unet_inputs # halves activation magnitude if 2 modalities

        # new conv_in channel
        _n_convin_out_channel = self.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            self.unet_in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        # TODO: Check if we need to register this explicitly with pl
        self.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.unet.config["in_channels"] = self.unet_in_channels
        logging.info("Unet config is updated")
        return
    
    def _check_inference_step(self, n_step: int) -> None:
        """Check if denoising step is reasonable"""
        assert n_step > 1, f"Number of denoising steps must be greater than 1, got {n_step}."

        if self.warned_once:
            return
        self.warned_once = True

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}. ")

    def forward(self, 
        inputs: Dict[str, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        cfg_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = None,
        show_progress_bar: bool = False,
        ensemble_kwargs: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model
        Inputs: 
            rgb_in (torch.Tensor): (B, C, H, W), 0-1
        Outputs:
            Dict[str, torch.Tensor]: (B, 1, H, W) depth prediction
        """
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = inputs[next(iter(self.unet_inputs))].shape[-1]

        assert processing_res > 0, "Processing resolution must be greater than 0."
        assert ensemble_size >= 1, "Ensemble size must be at least 1."
        self._check_inference_step(denoising_steps)
        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        for input_key, input_dict in self.unet_inputs.items():
            assert input_key in inputs, f"Input key '{input_key}' not found in unet inputs."
            modality = input_dict['modality']
            if modality == 'image':
                rgb_in = inputs[input_key]
                if rgb_in.ndim == 5: # Squeeze away time dimension if exists
                    rgb_in = rgb_in.squeeze(1)

                assert rgb_in.ndim == 4, "Input RGB tensor must have 4 dimensions (B, C, H, W)."
                # assert rgb_in.shape[1] == 3, "Input RGB tensor must have 3 channels (RGB)."
                input_size = rgb_in.shape

                if processing_res > 0:
                    rgb_in = resize_max_res(
                        rgb_in,
                        max_edge_resolution=processing_res,
                        resample_method=resample_method
                    )
                assert rgb_in.min() >= -1.0 and rgb_in.max() <= 1.0 
                inputs[input_key] = rgb_in
            elif modality == 'text':
                if isinstance(inputs[input_key], str):
                    inputs[input_key] = [inputs[input_key]]
            else:
                raise ValueError(f"Unsupported modality '{modality}' for input key '{input_key}'.")

        # import pdb; pdb.set_trace()
        B = inputs[next(iter(self.unet_inputs))].shape[0]
        if ensemble_size > 1:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v):
                    # Repeat along batch dim
                    inputs[k] = v.repeat_interleave(ensemble_size, dim=0)
                elif isinstance(v, list):
                    # Repeat each element e times while preserving order
                    inputs[k] = [x for x in v for _ in range(ensemble_size)]
                else:
                    # Leave other types untouched
                    inputs[k] = v
        outputs = self.infer(
            inputs=inputs, 
            num_inference_steps=denoising_steps, 
            cfg_scale=cfg_scale,
            guidance_rescale=guidance_rescale,
            generator=generator,
            show_progress_bar=show_progress_bar
        )
        final_pred = outputs['target_pred']  # [B, 3, H, W]

        if ensemble_size > 1:
            C, H, W = final_pred.shape[1:]
            final_pred = final_pred.reshape(B, ensemble_size, C, H, W).mean(dim=1)  # (B, C, H, W)

        if match_input_res:
            final_pred = resize(
                final_pred,
                size=input_size[-2:],
                interpolation=resample_method,
                antialias=True
            )

        outputs.update({
            "path_mask_pred": final_pred,  # [B, 3, H, W]
        })

        return outputs
    
    def _infer_autocast(self, *args, **kwargs):
        # autocast only for GPU tensors, fp16
        with autocast(device_type="cuda", dtype=torch.float16):
            return self._infer_fp32(*args, **kwargs)
        
    def optimize_inference(self):
        logging.info("Optimizing model inference with torch.compile...")
        if getattr(self, "_optimised", False):
            # prevent double-patching in case user calls it twice
            return
        self._optimised = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for module in (self.unet, self.vae, self.controlnet) if self.controlnet else (self.unet, self.vae):
            module.to(device, dtype=torch.float16, memory_format=torch.channels_last)
            module.eval()                                # no dropout / BN updates

        def _infer_fast(self, *args, **kwargs):
            with torch.inference_mode(), \
                torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._infer_fp32(*args, **kwargs)

        if torch.cuda.is_available() and hasattr(torch, "compile"):
            compile_opts = dict(mode="reduce-overhead", fullgraph=True, dynamic=True)
            self.unet = torch.compile(self.unet, **compile_opts)
            if self.controlnet is not None:
                self.controlnet = torch.compile(self.controlnet, **compile_opts)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True

        self._infer_fp32 = self.infer
        self.infer       = types.MethodType(_infer_fast, self)

        msg = ("✓  Inference optimised: fp16 ‖ channels_last ‖ "
            f"{'compile ‖ ' if hasattr(torch, 'compile') else ''}autocast+inference_mode")
        logging.info(msg)
    
    def infer(
        self, 
        inputs: Dict[str, torch.Tensor],
        num_inference_steps: int,
        cfg_scale: float,
        guidance_rescale: float,
        generator: Union[torch.Generator, None],
        show_progress_bar: bool,
        **kwargs
    ) -> torch.Tensor:
        
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.eta = 0.0 # Make DDIM stochastic
        timesteps = self.scheduler.timesteps

        lat_dict = self.prepare_cfg_latents(inputs)
        rgb_cond, rgb_uncond = lat_dict["img_cond"], lat_dict["img_uncond"]
        ctrl_cond, ctrl_uncond = lat_dict.get("ctrl_cond", None), lat_dict.get("ctrl_uncond", None)
        txt_cond, txt_uncond = lat_dict["txt_cond"], lat_dict["txt_uncond"]

        B, _, H, W = rgb_cond.shape

        target_latent = torch.randn(
            (B, self.vae_latent_size, H, W),
            device=rgb_cond.device,
            dtype=rgb_cond.dtype,
            generator=generator
        )

        if show_progress_bar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising"
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            if self.cond_method == "ControlNet":
                # For now assume single frame image. TODO: Fix this later!!!
                unet_input = torch.cat(
                    [torch.cat([rgb_cond[:, :self.vae_latent_size],   target_latent], 1),
                        torch.cat([rgb_uncond[:, :self.vae_latent_size], target_latent], 1)],
                    dim=0
                ) # [2B, 4, h, w]

                ctrl_input = None
                if ctrl_cond is not None and ctrl_uncond is not None:
                    ctrl_input = torch.cat(
                        [ctrl_cond, 
                        ctrl_uncond], 
                        dim=0
                    )
                ctrl_txt = torch.cat([txt_cond, txt_uncond], 0) # [2B, L, D]
                
                eps = self.controlnet(
                    unet_input, t,
                    encoder_hidden_states=ctrl_txt,
                    control_image=ctrl_input
                ).sample
            else:
                unet_input = torch.cat(
                    [torch.cat([rgb_cond,   target_latent], 1),
                        torch.cat([rgb_uncond, target_latent], 1)],
                    dim=0
                ) # [2B, 4, h, w]
                txt_in  = torch.cat([txt_cond, txt_uncond], 0) # [2B, L, D]
                eps = self.unet(
                    unet_input, t,
                    encoder_hidden_states=txt_in
                ).sample
            eps_c, eps_u = eps.chunk(2)

            # Apply classifier-free guidance (1 + w) * eps_c - w * eps_u -> rewritten as 
            # eps_c + scale * (eps_c - eps_u)
            eps_guided = eps_u + cfg_scale * (eps_c - eps_u) if cfg_scale is not None else eps_c

            if cfg_scale is not None and guidance_rescale > 0.0:
                # Rescale noise according to guidance rescale
                eps_guided = self.rescale_noise_cfg(eps_guided, eps_c, guidance_rescale=guidance_rescale)

            target_latent = self.scheduler.step(
                eps_guided, t, target_latent, generator=generator
            ).prev_sample
        
        rescale_target_latent, target_pred = self.decode_target(target_latent)  # [B, 3, H, W]
        target_pred = torch.clip(target_pred, -1.0, 1.0)
        target_pred = (target_pred + 1.0) / 2.0  # Normalize to [0, 1]
 
        output = {
            "target_latent": rescale_target_latent,  # [B, 4, H, W]
            "target_pred_cont": target_pred,  # [B, 3, H, W]
            "target_pred": (target_pred > 0.5).float(),  # [B, 3, H, W]
            "rgb_cond": rgb_cond,  # [B, 4, H, W
        }
        if ctrl_cond is not None:
            output["ctrl_cond"] = ctrl_cond
        if txt_cond is not None:
            output["txt_cond"] = txt_cond

        return output

    def rescale_noise_cfg(self, noise_cfg, noise_pred_cond, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def _is_control_input(self, spec):
        # Use ctrl encoder if ControlNet, x is not input image, and tiny encoder is desired
        return (
            self.cond_method == "ControlNet"
            and spec["name"] != next(iter(self.unet_inputs))
            and self.model_cfg['backbone']['kwargs']["cond_encoder"] == "tiny"
        )
    
    def prepare_cfg_latents(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Return the concatenated image & text latents for conditional and
        unconditional branches, respecting each input's `dropout_prob`.
        Keys with `dropout_prob == 0` are *never* dropped.
        
        Returns
        -------
        dict  with  {
            'img_cond'   : (B, C_img, h, w),
            'img_uncond' : (B, C_img, h, w),
            'txt_cond'   : (B, L,   D),
            'txt_uncond' : (B, L,   D),
        }
        """
        img_cond, img_uncond = [], []
        ctrl_cond, ctrl_uncond = [], []
        txt_cond, txt_uncond = None, None

        device = None
        for key, spec in self.unet_inputs.items():
            x        = inputs[key]
            modality = spec["modality"]
            p_drop   = spec.get("dropout_prob", 0.0)
            if device is None:
                device = x.device

            if self.training:
                # training: use stochastic conditioning dropout
                keep_mask = torch.rand((len(x),), device=device) >= p_drop
            else:
                # (i.e., zeros/empty everywhere)
                keep_mask = torch.zeros((len(x),), device=device, dtype=torch.bool)

            # import pdb; pdb.set_trace()  # Debugging breakpoint
            # rgb = (x[0].permute(1, 2, 0).cpu().numpy() + 1)/2.0
            # rgb = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            # ---------- IMAGE inputs ------------------------------------
            if modality == "image":
                if x.ndim == 5:                       # (B,T,C,H,W) → first frame
                    x = x[:, 0]
                
                # Sample probability to drop out this input
                if self._is_control_input(spec):
                    ctrl_cond.append(x)
                    ctrl_uncond.append(
                        torch.where(keep_mask.view(-1, 1, 1, 1), x, torch.zeros_like(x))
                    )
                else:
                    assert x.shape[1] == 3, f"Expected 3 channels for {key}, got {x.shape[1]} channels"
                    lat = self.encode_rgb(x)
                    img_cond.append(lat)                  # always present
                    img_uncond.append(
                        torch.where(keep_mask.view(-1, 1, 1, 1), lat, torch.zeros_like(lat))
                    )
            # ---------- TEXT inputs -------------------------------------
            elif modality == "text":
                lat = self.encode_text(x)             # (B,L,D)
                txt_cond = lat                        # (only one text key expected)

                if self.empty_text_embed is None:
                    self.encode_empty_text()
                empty = self.empty_text_embed.repeat(lat.shape[0], 1, 1).to(lat)
                txt_uncond = torch.where(
                    keep_mask.view(-1, 1, 1), lat, empty                     # (B,L,D)
                )
            else:
                raise ValueError(f"Unsupported modality '{modality}' for key '{key}'")

        # ---------- Final concat / defaults -----------------------------
        img_cond_cat   = torch.cat(img_cond,   dim=1)           # (B,C_img,h,w)
        img_uncond_cat = torch.cat(img_uncond, dim=1)

        if txt_cond is None:                                    # no text inputs at all
            if self.empty_text_embed is None:
                self.encode_empty_text()
            txt_cond = txt_uncond = self.empty_text_embed.repeat(
                img_cond_cat.shape[0], 1, 1).to(img_cond_cat)

        out = {
            "img_cond"  : img_cond_cat,
            "img_uncond": img_uncond_cat,
            "txt_cond"  : txt_cond,
            "txt_uncond": txt_uncond,
        }

        if ctrl_cond:
            out["ctrl_cond"] = torch.cat(ctrl_cond,   dim=0)           # (B,C_img,h,w)
            out["ctrl_uncond"] = torch.cat(ctrl_uncond, dim=0)
        return out
    
    def encode_text(
        self, 
        text: Union[str, Sequence[str]]
    ) -> torch.Tensor:
        """
        Encodes one or more text prompts into CLIP embeddings in a single batch.

        Args:
            text:  either a single string, or a list/tuple of strings

        Returns:
            last_hidden_state: torch.Tensor of shape (B, L, D)
        """
        # ensure we have a list of strings
        texts = [text] if isinstance(text, str) else list(text)

        # tokenize in batch
        text_tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_token_ids = text_tokens.input_ids
        untruncated_ids = self.tokenizer(texts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_token_ids.shape[-1] and not torch.equal(
            text_token_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # move to same device as text_encoder
        text_tokens = {k: v.to(self.text_encoder.device) for k, v in text_tokens.items()}

        # forward once through CLIPTextModel
        out = self.text_encoder(**text_tokens)
        # last_hidden_state: (B, L, D)
        return out.last_hidden_state.float()
    
    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        empty_embed = self.encode_text(prompt)  # this returns (1, seq_len, hidden_dim)
        
        # sanity‐check the dims
        assert empty_embed.ndim == 3 and empty_embed.shape[0] == 1, (
            f"empty_text_embed must be (1, L, D), got {tuple(empty_embed.shape)}"
        )
        
        self.empty_text_embed = empty_embed

    def encode_empty_text_like(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Return an all-zero text embedding with the same shape & device as `latent`.
        Useful for classifier-free text guidance dropout.
        """
        if self.empty_text_embed is None:
            self.encode_empty_text()

        return self.empty_text_embed.repeat((latent.shape[0], 1, 1)).to(latent.device)
       
    @staticmethod
    def encode_empty_image_like(latent: torch.Tensor) -> torch.Tensor:
        """
        Return an all-zero latent with the same shape & device as `latent`.
        Useful for classifier-free image guidance dropout.
        """
        return torch.zeros_like(latent)

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encodes RGB images to latent representations 

        Args:
            rgb_in (torch.Tensor): Input RGB image tensor.

        Returns:
            torch.Tensor: Encoded latent representation of the RGB image.
        """
        if not isinstance(rgb_in, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        # If time dim exists, ensure it's one
        if rgb_in.ndim == 5:
            rgb_in = rgb_in.squeeze(1)
        assert rgb_in.ndim == 4, "Input RGB tensor must have 4 dimensions (B, C, H, W)."
        
        h = self.vae.encoder(rgb_in)

        # Reparametrization trick for differentiable sampling
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent
    
    @staticmethod
    def stack_mask_images(mask_in):
        assert mask_in.ndim == 4, "Input depth tensor must have 4 dimensions (B, C, H, W)."
        stacked = mask_in.repeat(1, 3, 1, 1)
        return stacked

    def encode_path_mask(self, mask_in: torch.Tensor) -> torch.Tensor:
        """
        Encodes path masks images to latent representations.

        Args:
            depth_in (torch.Tensor): Input depth image tensor.

        Returns:
            torch.Tensor: Encoded latent representation of the depth image.
        """
        if not isinstance(mask_in, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        assert mask_in.ndim == 4, "Input depth tensor must have 4 dimensions (B, C, H, W)."

        stacked = self.stack_mask_images(mask_in)
        depth_latent = self.encode_rgb(stacked)
        return depth_latent
    
    def decode_target(self, target_latent: torch.Tensor, guidance_rescale=0.0) -> torch.Tensor:
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        target_latent = target_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(target_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        target_mean = stacked.mean(dim=1, keepdim=True)
        return z, target_mean
    
if __name__ == "__main__":
    cfg_path = "./config/model/depth/marigold.yaml"

    model_cfg = OmegaConf.load(cfg_path)

    model = MarigoldModel(model_cfg)
    print(model)