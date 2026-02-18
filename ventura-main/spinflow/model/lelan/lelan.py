"""
Master file for lelan model backbone. Based on the following code:
https://github.com/NHirose/learning-language-navigation

The lelan architecture consists of a:
1. ResNet backbone with FiLM layers to text conditioning,
2. Frozen CLIP text encoder
3. Transformer encoder
4. Dense MLP head for action prediction (originally for linear v and angular w)
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import clip

from scripts.utils.log_utils import logging
from spinflow.model.lelan.film_network import (
    FiLMNetwork,
)
from spinflow.model.blocks.positional_embedding import (
    SinusoidalPositionalEmb1D
)
from spinflow.model.blocks.dense_network import (
    DenseNetwork_lelan
)
from spinflow.model.lelan.nomad_vint import (
    replace_bn_with_gn
)
from spinflow.util.action_utils import (
    unnormalize_action
)

def build_film_model(num_res_blocks, num_classes, num_channels, question_dim):
    return FiLMNetwork(num_res_blocks, num_classes, num_channels, question_dim)

class LeLaN_clip(nn.Module):
    def __init__(self, cfg):
        super(LeLaN_clip, self).__init__()

        self.cfg = cfg
        self.vision_cfg = cfg['vision_encoder']
        self.text_cfg = cfg['text_encoder']
        self.action_cfg = cfg['action_head']
        self.validation_cfg = cfg['validation']

        logging.info(f"Building LeLaN_clip model with vision encoder {self.vision_cfg['name']}\n"
                     f"Building LeLaN_clip model with text encoder {self.text_cfg['clip_type']}\n"
                     f"Building LeLaN_clip model with action head {self.action_cfg['name']}\n"
                     )
        vision_encoder = globals()[self.vision_cfg['name']](self.vision_cfg)
        self.vision_encoder = replace_bn_with_gn(vision_encoder)
        text_encoder, preprocess = clip.load(self.text_cfg["clip_type"])
        self.text_encoder = text_encoder.float()  # Ensure text encoder is in float mode
        self.dist_pred_net = globals()[self.action_cfg['name']](**self.action_cfg['kwargs'])
        logging.info("Finished building LeLaN_clip model")

        # ---------- action params ----------
        self.action_dim = cfg['action_dim']
        self.register_buffer( "action_range",
            torch.tensor(self.validation_cfg['action_range'], dtype=torch.float32)
        )
        self.register_buffer( "action_stats",
            torch.tensor(self.validation_cfg['action_stats'], dtype=torch.float32)
        )

        self.setup(self.cfg)

    def setup(self, model_cfg):
        # Enable training pipeline
        pipeline_flags = model_cfg['pipeline']
        for comp, comp_dict in pipeline_flags.items():
            requires_grad = not comp_dict.get('frozen', False)
            getattr(self, comp).requires_grad_(requires_grad)
            if requires_grad:
                getattr(self, comp).eval()

        # Print which modules are frozen
        frozen_modules = [
            comp for comp, comp_dict in pipeline_flags.items() if comp_dict.get('frozen', False)
        ]
        if frozen_modules:
            logging.info(f"Frozen modules: {', '.join(frozen_modules)}")

    def _load_pretrained(self, ckpt_path, device="cpu"):
        """Warm-start individual sub-modules if cfg.pretrained is given."""
        paths = self.cfg.get("pretrained", {})
        if not paths:
            return
        
        # Load checkpoint
        pre_sd = torch.load(ckpt_path, map_location=device)

        model_sd = self.state_dict()

        # 3) filter out any keys that either don’t exist in the model
        #    or whose shapes differ
        filtered_sd = {}
        skipped = []
        for k, v in pre_sd.items():
            if k not in model_sd:
                skipped.append(k)
            elif v.shape != model_sd[k].shape:
                skipped.append(k)
            else:
                filtered_sd[k] = v

        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)
        logging.info(
            f"Loaded {ckpt_path} → {self.__class__.__name__}  "
            f"(loaded_keys={len(filtered_sd)}, missing={missing}, "
            f"unexpected={unexpected}, skipped={skipped})"
        )

    # Dummy infer function that passes all args to forward
    def infer(self, inputs, **kwargs):
        """
        Dummy inference function that passes all inputs to the forward method.
        This is useful for compatibility with existing inference pipelines.
        """
        return self.forward(inputs, **kwargs)

    def forward(self, inputs, unnormalize=False): #func_name, **kwargs):
        device = inputs["rgb_image"].device

        with torch.no_grad():
            text_tokens = clip.tokenize(inputs["goal_command"]).to(device)
            text_feats  = self.text_encoder.encode_text(text_tokens).float()

        obs_img = inputs["rgb_image"]
        if obs_img.dim() == 5:  # If batch of videos, take the
            obs_img = obs_img[:, -1, :, :, :]

        vision_feats = self.vision_encoder(obs_img, text_feats)
        action_pred = self.dist_pred_net(vision_feats)

        # Unnormalize action predictions if required
        if unnormalize:
            action_pred = unnormalize_action(
                action_pred,
                self.action_range,
                self.action_stats,
                self.action_dim
            )

        output = {
            "vision_feats": vision_feats,
            "text_feats": text_feats,
            "action_pred": action_pred
        }
        return output

class LeLaN_clip_FiLM(nn.Module):
    def __init__(self, cfg):
        super(LeLaN_clip_FiLM, self).__init__()
        self.vision_cfg = cfg

        if self.vision_cfg['clip_type'] == 'ViT-B/32':
            self.film_model = build_film_model(8, 10, 128, 512)
        else:
            raise NotImplementedError(f"Vision encoder {self.vision_cfg['name']} not implemented")

        self.num_goal_features  = cfg['num_goal_features']
        self.obs_encoding_size  = cfg['obs_encoding_size']
        self.goal_encoding_size = cfg['goal_encoding_size']
        self.context_size       = cfg['context_size']
        self.image_size         = cfg['image_size']
        self.aspect_ratio       = cfg['aspect_ratio']
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = SinusoidalPositionalEmb1D(self.obs_encoding_size, max_len=2) #no context
        sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=cfg['mha_num_attention_heads'], 
            dim_feedforward=cfg['mha_ff_dim_factor']*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(sa_layer, num_layers=cfg['mha_num_attention_layers'])

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

    def transform_images_lelan(self, imgs: torch.Tensor, center_crop: bool = False) -> torch.Tensor:
        """Transforms a list of PIL image to a torch tensor."""
        transform_type = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),            
            ]
        )

        h, w = imgs[-1].shape[-2:]
        if center_crop:
            if w > h:
                imgs = TF.center_crop(imgs, (h, int(h * self.aspect_ratio)))  # crop to the right ratio
            else:
                imgs = TF.center_crop(imgs, (int(w / self.aspect_ratio), w))
            transf_imgs = TF.resize(imgs, self.image_size)  # Resize all images to the same size
        else:
            transf_imgs = TF.resize(imgs, self.image_size)

        transf_imgs = transform_type(transf_imgs)
        
        if transf_imgs.dim() == 5:  # If batch of videos, take the last frame
            last_img = transf_imgs[:, -1, :, :, :]
        else:
            last_img = transf_imgs
        return transf_imgs, last_img

    def forward(self, obs_img: torch.tensor, feat_text: torch.tensor):#inst_ref: torch.tensor
        device = obs_img.device
        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)
        
        # Preprocess input images
        with torch.no_grad():
            hist_img, cur_img = self.transform_images_lelan(obs_img, center_crop=True)
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        # sanity_check_video_th = torch.load("/home/ubuntu/playground/learning-language-navigation/deployment/src/batch_obs_current.pt").to(device)
        # assert torch.allclose(cur_img, sanity_check_video_th, atol=1e-3), "Video tensor mismatch!"
        # Get the goal encoding
        obsgoal_img = cur_img       
        inst_encoding = feat_text
        obsgoal_encoding = self.film_model(obsgoal_img, inst_encoding)
        obsgoal_encoding_cat = obsgoal_encoding.flatten(start_dim=1)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding_cat)        

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        obs_encoding = obsgoal_encoding                

        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens

if __name__ == "__main__":
    import yaml
    import decord
    from torchvision import transforms
    cfg_path = "config/model/planning/lelan/lelan.yaml"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = LeLaN_clip(cfg)
    print(model)

    # Test loading checkpoint
    ckpt_path = "/home/ubuntu/playground/learning-language-navigation/deployment/weights/wo_col_loss_wo_temp.pth"
    model._load_pretrained(ckpt_path, device=device)
    model.to(device)

    ### BEGIN DUMMY FORWARD PASS TEST ###
    video_path = "/home/ubuntu/playground/spinflow/data/fai_processed/output_rides_2025-05-21-16-56-00/front_camera/ride_2025-05-21-16-56-00_ferrite2_2025-05-21-17-09-59_1_frames_6273_6335.mp4"
    decord.bridge.set_bridge('torch')
    video_np = decord.VideoReader(video_path, ctx=decord.cpu(0))
    video_np = video_np.get_batch(range(0, 3)) 
    Htarget, Wtarget = 560, 560

    def resize_and_center_crop(imgs: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        imgs: Tensor of shape [B, H, W, C] (uint8)
        Returns: Tensor [B, target_h, target_w, C] (uint8)
        """
        import math
        import torch.nn.functional as F
        B, H, W, C = imgs.shape
        # Move channels to front and convert to float
        imgs_t = imgs.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Compute scale factor so both dims >= target
        scale = max(target_h / H, target_w / W)
        new_h = math.ceil(H * scale)
        new_w = math.ceil(W * scale)

        # Resize
        resized = F.interpolate(
            imgs_t, size=(new_h, new_w),
            mode='bilinear', align_corners=False
        )

        # Center-crop
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        cropped = resized[:, :, top : top + target_h, left : left + target_w]

        # Move channels back and convert to uint8
        return cropped.permute(0, 2, 3, 1).byte()
    
    crop_video_np = resize_and_center_crop(video_np, Htarget, Wtarget)
    video_th = crop_video_np.permute(0, 3, 1, 2).to(device).float()  # [B, C, H, W]
    video_th = video_th / 255.0  # Normalize to [0, 1] range
    print(f"Video tensor shape: {video_th.shape}")
    # video_th = torch.load("/home/ubuntu/playground/learning-language-navigation/deployment/src/batch_obs_current.pt").to(device)
    # torch.save(video_th, "test_video_tensor.pt")

    # Dummy text input
    text_prompt = ["Go to white car"]
    batch_obj_inst = clip.tokenize(text_prompt).to(device)
    print(f"Text tensor shape: {batch_obj_inst.shape}")

    text_feats = model("text_encoder", inst_ref=batch_obj_inst)
    obsgoal_feats = model("vision_encoder", obs_img=video_th, feat_text=text_feats)
    print(f"ObsGoal features shape: {obsgoal_feats.shape}")

    action_preds = model("dist_pred_net", obsgoal_cond=obsgoal_feats)
    print(f"Action predictions shape: {action_preds.shape}")
    print("linear v0: ", action_preds[0, 0, 0].item())
    print("angular w0: ", action_preds[0, 0, 1].item())
    