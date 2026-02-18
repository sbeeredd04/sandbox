import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from efficientnet_pytorch import EfficientNet
    

class SpatialSoftmax(nn.Module):
    """
    Turns a CxHxW feature map into K key-points (mean, optional covariance).
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],     # (C,H,W) of the feature map
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()
        C, H, W = in_shape
        self.H, self.W, self.K = H, W, num_kp
        self.output_variance   = output_variance
        self.noise_std         = noise_std

        # 1×1 conv to produce K attention maps (identity if K == C)
        self.attn_conv = (
            nn.Identity() if num_kp is None or num_kp == C
            else nn.Conv2d(C, num_kp, kernel_size=1, bias=True)
        )

        t = torch.ones(1) * float(temperature)
        self.temperature = nn.Parameter(t) if learnable_temperature else t

        # pre-compute normalised pixel coordinates
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        )
        self.register_buffer('pos_x', xs.reshape(1, H * W))
        self.register_buffer('pos_y', ys.reshape(1, H * W))

    # ------------------------------------------------------------------ #
    def forward(self, feat: torch.Tensor):
        """
        feat : (B,C,H,W) – feature maps coming from EfficientNet.
        returns
            kp_mean : (B,K,2)   normalised XY in [-1,1]
            kp_cov  : (B,K,2,2) optional full covariance matrix
        """
        B, _, H, W = feat.shape
        assert (H, W) == (self.H, self.W), "input spatial size mismatch"

        attn = self.attn_conv(feat)                       # (B,K,H,W)
        attn = attn.view(B * self.K, -1) / self.temperature.clamp(0.1, 10.0)
        attn = F.softmax(attn, dim=-1)                    # (B*K, H*W)

        mu_x = torch.sum(self.pos_x * attn, dim=1, keepdim=True)
        mu_y = torch.sum(self.pos_y * attn, dim=1, keepdim=True)
        mu   = torch.cat([mu_x, mu_y], dim=1)             # (B*K,2)
        mu   = mu.view(B, self.K, 2)

        if self.training and self.noise_std > 0:
            mu = mu + torch.randn_like(mu) * self.noise_std

        if not self.output_variance:
            return mu                                    # (B,K,2)

        # --- second-order moments ----------------------------------------
        xx = torch.sum(self.pos_x**2 * attn, dim=1, keepdim=True)
        yy = torch.sum(self.pos_y**2 * attn, dim=1, keepdim=True)
        xy = torch.sum(self.pos_x * self.pos_y * attn, dim=1, keepdim=True)

        var_x  = xx - mu_x * mu_x
        var_y  = yy - mu_y * mu_y
        var_xy = xy - mu_x * mu_y
        cov    = torch.stack([var_x, var_xy, var_xy, var_y], dim=-1)
        cov    = cov.view(B, self.K, 2, 2)               # (B,K,2,2)
        return mu, cov

class ConvNetSpatial(nn.Module):
    """
    RGB (+extra channels)  ->  ResNet backbone  ->  1×1 projection  ->  SpatialSoftmax
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg      = cfg
        enc_cfg       = cfg["encoder"]

        self.in_keys  = enc_cfg["in_keys"]      # e.g. ["rgb", "path_mask"]
        self.out_key  = enc_cfg["out_key"]      # e.g. "vis_feats"

        in_ch         = sum(enc_cfg["feat_dims"])
        out_ch        = enc_cfg["out_dim"]
        resnet_name   = enc_cfg["name"]         # "resnet18" | "resnet50" …

        # ------------------------------------------------------------------
        self.obs_net = self._init_resnet(resnet_name, in_ch, out_ch)

        # determine spatial size once to size the SpatialSoftmax layer
        dummy = torch.zeros(1, in_ch, *enc_cfg["input_res"])
        with torch.no_grad():
            C, H, W = self.obs_net(dummy).shape[1:]
        
        ssp_cfg = cfg["spatial_softmax"]
        self.spatial = SpatialSoftmax(
            in_shape              = (C, H, W),
            num_kp                = ssp_cfg.get("num_kp", 32),
            temperature           = ssp_cfg.get("temperature", 1.0),
            learnable_temperature = ssp_cfg.get("learnable_temperature", False),
            output_variance       = ssp_cfg.get("output_variance", False),
            noise_std             = ssp_cfg.get("noise_std", 0.0),
        )

    # ------------------------------------------------------------------ #
    def _init_resnet(self, name: str, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Build a ResNet feature extractor that:
        • accepts `in_ch` channels,
        • outputs a feature map of `out_ch` channels.
        """
        # -------- 1. load ImageNet-pretrained backbone -------------------
        ctor = {
            "resnet18":  models.resnet18,
            "resnet34":  models.resnet34,
            "resnet50":  models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }[name]
        
        _weight_enum = {
            "resnet18" : models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34" : models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50" : models.ResNet50_Weights.IMAGENET1K_V2,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
        }

        net = ctor(weights=_weight_enum[name])

        # -------- 2. patch first conv for extra channels -----------------
        if in_ch != 3:
            w = net.conv1.weight.data
            new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
            if in_ch >= 3:
                new_conv.weight.data[:, :3] = w
                if in_ch > 3:                          # extra channels
                    new_conv.weight.data[:, 3:] = w.mean(1, keepdim=True)
            else:                                      # <3 channels
                new_conv.weight.data = w[:, :in_ch]
            net.conv1 = new_conv

        # -------- 3. keep layers up to layer4 (stride-32) ----------------
        backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )

        # -------- 4. 1×1 projection to out_ch ---------------------------
        proj_in = 512 if "18" in name or "34" in name else 2048
        proj = nn.Sequential(
            nn.Conv2d(proj_in, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        return nn.Sequential(backbone, proj)

    # ------------------------------------------------------------------ #
    def forward(self, inputs: dict) -> dict:
        # Prepare input tensor from multiple keys
        x = [
            inputs[k] if inputs[k].ndim == 4 else inputs[k][:, -1]
            for k in self.in_keys if k in inputs
        ]
        x = torch.cat(x, dim=1)   # (B,C,H,W)
        fmap = self.obs_net(x)                                   # (B,out_ch,h,w)
        kps  = self.spatial(fmap)                                 # (B,K,2) or (μ,Σ)
        B, K, D = kps.shape
        return {
            self.out_key: kps.view(B, K*D)  # [B, K*2]
        }