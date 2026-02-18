import copy, torch, torch.nn as nn, torch.nn.functional as F
from diffusers import (
    UNet2DConditionModel,
)

# ---------- helpers ----------------------------------------------------
class ZeroConv2d(nn.Conv2d):
    """1 × 1 conv whose weights *start at zero*."""
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, 1)
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)

class TinyConvEncoder(nn.Module):
    """
    4x(conv 4x4, stride 2) → Zero-Conv that maps *any* control image
    (RGB or mask) to the UNet hidden width.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        layers = []
        chans  = [in_ch, 16, 32, 64, 128]
        for c_in, c_out in zip(chans[:-2], chans[1:-1]):
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.SiLU()]

        layers += [nn.Conv2d(chans[-2], chans[-1], 3, 1, 1), nn.SiLU()]

        layers += [ZeroConv2d(chans[-1], out_ch)]    # final projection
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------
class ControlNet(nn.Module):
    """
    Paper-faithful ControlNet wrapper **that returns the same output type
    as `UNet2DConditionModel`**, so `... .sample` works unchanged.
    """
    def __init__(self, unet: UNet2DConditionModel, cfg: dict):
        super().__init__()

        # -------- grey (frozen) decoder ---------------------------------
        self.decoder = unet
        if cfg['freeze_unet']:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

        # ─── copied *encoder* path (trainable) ──────────────────────────
        self.conv_in     = copy.deepcopy(unet.conv_in)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block   = copy.deepcopy(unet.mid_block)

        # ─── conditioning image encoder (1×1 zero-conv is enough for 64×64 latents) ──
        c_in              = cfg.get("cond_channels", 3)
        self.cond_scale   = float(cfg.get("cond_scale", 1.0))
        self.downscale    = cfg.get("downscale_cond", True)
        enc_type = cfg['cond_encoder']
        
        if enc_type == "tiny":
            self.cond_enc = TinyConvEncoder(c_in, self.conv_in.out_channels)
        elif enc_type == "zero":
            self.cond_enc = ZeroConv2d(c_in, self.conv_in.out_channels)
        else:
            raise ValueError(f"Unknown cond_encoder type: {enc_type}")

        # ─── zero-init 1×1 projections for every feature map + mid ──────
        proj = [ZeroConv2d(self.conv_in.out_channels,   # conv_in feature
                           self.conv_in.out_channels)]

        for blk in self.down_blocks:
            # one ZeroConv **per ResNet** in the block
            for resnet in blk.resnets:
                proj.append(ZeroConv2d(resnet.out_channels,
                                       resnet.out_channels))
            # one more for the down-sampler feature (if it exists)
            if blk.downsamplers:
                proj.append(ZeroConv2d(blk.downsamplers[-1].out_channels,
                                       blk.downsamplers[-1].out_channels))

        proj.append(ZeroConv2d(self.mid_block.resnets[-1].out_channels,
                               self.mid_block.resnets[-1].out_channels))
        self.proj = nn.ModuleList(proj)

    @classmethod
    def from_unet(cls, unet: UNet2DConditionModel, cfg: dict):
        """
        Create a ControlNet *after* `unet` was loaded from a checkpoint.
        Useful when you first trained w/o ControlNet and now want to add it.
        """
        return cls(unet, cfg)

    # -------------------------------------------------------------------
    def forward(
        self,
        sample,                         # latent input  (B,4,H,W)
        timestep,                       # scalar or (B,)
        encoder_hidden_states,          # text embeddings
        control_image,                  # conditioning image (B,C,Hc,Wc)
        **kwargs,                       # attention_mask, cross_attn kwargs …
    ):
        """"Fall back to vanilla UNet if no control image is provided."""
        if control_image is None:
            # fall back to vanilla Stable Diffusion behaviour
            return self.decoder(sample, timestep, encoder_hidden_states, **kwargs)

        """Use ControlNet to condition the UNet on `control_image`."""
        if self.downscale:
            control_image = F.interpolate(control_image,
                                          size=sample.shape[-2:],
                                          mode="bilinear",
                                          align_corners=False)

        # -- 1. encoder pass on the *copied* encoder ----------------------
        h = self.conv_in(sample) + self.cond_enc(control_image)
        feats = [h]   

        # Expand tensor if scalar
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
            timesteps = timestep.expand(sample.shape[0])
        else:
            timesteps = timestep.to(sample.device)

        t_emb = self.decoder.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.decoder.time_embedding(t_emb)

        for blk in self.down_blocks:
            h, res = blk(
                h, emb, encoder_hidden_states, **kwargs
            )
            feats.extend(res)                         # resnets + downsample
        h = self.mid_block(h, emb, encoder_hidden_states, **kwargs)
        feats.append(h)                               # mid feature last

        # -- 2. project through zero-convs --------------------------------
        down_proj = [self.cond_scale * p(f) for p, f in zip(self.proj, feats)]
        mid_proj  = down_proj.pop()                   # last one is mid

        # -- 3. decoder pass on the *frozen* UNet with residual injection –
        unet_out = self.decoder(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals = tuple(down_proj),
            mid_block_additional_residual   = mid_proj,
            **kwargs,
        )
 
        return unet_out          # UNet2DConditionOutput ⇒ has `.sample`