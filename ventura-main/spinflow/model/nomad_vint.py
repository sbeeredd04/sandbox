import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable

from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from efficientnet_pytorch import EfficientNet

from scripts.utils.log_utils import logging
from spinflow.model.blocks.positional_embedding import SinusoidalPositionalEmb2D
from spinflow.model.blocks.film_conditioning import FiLMConditioning

from typing import Optional, Union, Sequence
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL
)

from einops import rearrange
from einops.layers.torch import Rearrange

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        model_cfg
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        
        self.model_cfg = model_cfg
        self.obs_encoding_size = model_cfg['obs_encoding_size']
        self.goal_encoding_size = model_cfg['goal_encoding_size']
        self.context_size = model_cfg['context_size']
        self.pipeline_cfg = model_cfg['pipeline']

        # Initialize the obs & goal encoder
        obs_encoder_name = model_cfg['encoder']['name']
        if obs_encoder_name == "AutoencoderKL":
            obs_encoder_ckpt = model_cfg['encoder']['pretrained_ckpt']
            self.obs_encoder = AutoencoderKL.from_pretrained(
                obs_encoder_ckpt, subfolder="vae")
            self.num_obs_features = self.obs_encoder.decoder.conv_in.in_channels
            self.latent_scale_factor = model_cfg['encoder']['kwargs']['latent_scale_factor']

            goal_encoder_ckpt = model_cfg['encoder']['pretrained_ckpt']
            self.goal_encoder = AutoencoderKL.from_pretrained(
                goal_encoder_ckpt, subfolder="vae")
            self.num_goal_features = self.goal_encoder.decoder.conv_in.in_channels

            if model_cfg['encoder']['goal_in_channels'] != self.goal_encoder.in_channels:
                self.goal_encoder.encoder.conv_in = self.replace_vae_conv_in(
                    self.goal_encoder, model_cfg['encoder']['goal_in_channels']
                )

            # self.token_dim = model_cfg['aggregation']['token_dim']
            # self._avg_pooling = nn.AdaptiveAvgPool2d(self.token_dim)
        elif obs_encoder_name.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder_name, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features

            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError

        # Text encoder
        text_encoder_ckpt = model_cfg['text']['text_encoder_ckpt']
        tokenizer_ckpt = model_cfg['text']['tokenizer_ckpt']
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_ckpt, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_ckpt, subfolder="tokenizer")
        
        # Positional embedding for self attention
        # self.positional_encoding = SinusoidalPositionalEmb2D(
        #     d_model=self.obs_encoding_size, 
        # )

        # # Initialize compression layers if necessary
        # if self.num_obs_features != self.obs_encoding_size:
        #     self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        # else:
        #     self.compress_obs_enc = nn.Identity()
        
        # if self.num_goal_features != self.goal_encoding_size:
        #     self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        # else:
        #     self.compress_goal_enc = nn.Identity()

        # if model_cfg['text']['text_embedding_dim'] != self.goal_encoding_size:
        #     self.compress_text_enc = nn.Linear(
        #         model_cfg['text']['text_embedding_dim'], self.goal_encoding_size)
        # else:
        #     self.compress_text_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        # self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        # self.sa_layer = nn.TransformerEncoderLayer(
        #     d_model=self.obs_encoding_size, 
        #     nhead=mha_num_attention_heads, 
        #     dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
        #     activation="gelu", 
        #     batch_first=True, 
        #     norm_first=True
        # )
        # self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

    def replace_vae_conv_in(self, enc: AutoencoderKL, new_in_channels: int) -> AutoencoderKL:
        """"""
        conv = enc.encoder.conv_in                      # nn.Conv2d
        old_in = conv.in_channels
        if new_in_channels == old_in:                   # nothing to do
            return enc

        # Clone original parameters
        _weight = conv.weight.data.clone()                    # [C_out, old_in, k, k]
        _bias = conv.bias.data.clone()                      # [C_out]

        # --- build new weight tensor ------------------------------------------------
        # Scale so activations keep the same order of magnitude
        num_unet_inputs = new_in_channels // old_in
        _weight = _weight.repeat((1, num_unet_inputs, 1, 1))  # [C_out, new_in_channels, k, k]
        _weight *= 1.0 / num_unet_inputs

        # --- create & assign the new conv ------------------------------------------
        _n_convin_out_channel = enc.encoder.conv_in.out_channels
        _new_conv_in = Conv2d(
            new_in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        logging.info(f"VAE conv_in switched from {old_in}→{new_in_channels} channels")

        return _new_conv_in

    def encode_rgb(self, rgb: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        Encode RGB images using the obs encoder.
        """
        if rgb.ndim == 5:
            rgb = rgb.squeeze(1)
        assert rgb.ndim == 4, "Input RGB tensor must be 4D (B, C, H, W)"

        if isinstance(enc, EfficientNet):
            h = enc.extract_features(rgb)
            # h = enc._avg_pooling(h)
            if enc._global_params.include_top:
                h = h.flatten(start_dim=1)
                h = enc._dropout(h)
            h = h.squeeze(1)  # Remove the channel dimension
        else:
            h = enc.encoder(rgb)
            moments = enc.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)

            h = mean * self.latent_scale_factor

            # Pool the features to a single vector
            # h = self._avg_pooling(h)

        return h
    
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

    def forward(self, 
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_image = inputs['rgb_image']  # [B, C, H, W]
        goal_image = inputs.get('goal_image', None)  # [B, C,
        goal_command = inputs.get('goal_command', None)  # List[str]
        input_goal_mask = inputs.get('goal_mask', None)  # [B, T
        device = rgb_image.device
        obs_img = rgb_image.squeeze(1) if rgb_image.ndim == 5 else rgb_image
        goal_img = goal_image.squeeze(1) if goal_image is not None and goal_image.ndim == 5 else goal_image

        #1 Encoding the obsgoal image tokens
        goal_mask = None
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        if goal_img is None:
            goal_img = torch.zeros_like(obs_img[:, :3, :, :])

        obsgoal_img = torch.cat([obs_img[:, :3*self.context_size, :, :], goal_img], dim=1) # concatenate the obs image/context and goal image --> non image goal?
        goal_encoding = self.encode_rgb(obsgoal_img, self.goal_encoder) # [B, F, H, W]
        H, W = goal_encoding.shape[2:]
        # goal_encoding = rearrange(goal_encoding, 'b f h w -> b (h w) f')  # [B, H*W, F]
        # goal_encoding = rearrange(self.compress_goal_enc(goal_encoding), 'b (h w) f -> b f h w', h=H, w=W)
        # assert goal_encoding.shape[1] == self.goal_encoding_size # [B, N, F]

        #2 Encode the language goal tokens
        goal_cmd_encoding = None
        if goal_command is not None:
            goal_cmd_encoding = self.encode_text(goal_command) # [B, L, D]
            # goal_cmd_encoding = self.compress_text_enc(goal_cmd_encoding)

        #3 Encode the observation tokens
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)
        obs_encoding = self.encode_rgb(obs_img, self.obs_encoder)
        H, W = obs_encoding.shape[2:]
        # obs_encoding = rearrange(obs_encoding, 'b f h w -> b (h w) f')  # [B, H*W, F]
        # obs_encoding = rearrange(self.compress_obs_enc(obs_encoding), 'b (h w) f -> b f h w', h=H, w=W)
        
        #4 Mask goal tokens if provided
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        #5 Apply positional encoding to obs and goal encoding if provided 
        # if self.positional_encoding:
        #     obs_encoding = self.positional_encoding(obs_encoding)
        #     goal_encoding = self.positional_encoding(goal_encoding) if goal_encoding is not None else goal_encoding
        
        return {
            "rgb_cond": obs_encoding,  # [B, F, H, W]
            "goal_cond": goal_encoding,  # [B, F, H, W]
            "txt_cond": goal_cmd_encoding,  # [B, L, D]
        }

        # #6 Flatten and concatenate all tokens for self-attention
        # state_encoding = rearrange(obs_encoding, 'b f h w -> b (h w) f')  # [B, H*W, F]
        # if goal_encoding is not None:
        #     goal_tokens = rearrange(goal_encoding, 'b f h w -> b (h w) f')  # [B, H*W, F]
        #     state_encoding = torch.cat([state_encoding, goal_tokens], dim=1)

        # if goal_cmd_encoding is not None:
        #     goal_cmd_tokens = rearrange(goal_cmd_encoding, 'b l d -> b l d')
        #     state_encoding = torch.cat([state_encoding, goal_cmd_tokens], dim=1)

        # state_encoding_tokens = self.sa_encoder(state_encoding, src_key_padding_mask=src_key_padding_mask)
        # if src_key_padding_mask is not None:
        #     avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
        #     state_encoding_tokens = state_encoding_tokens * avg_mask
        # avg_state_encoding = torch.mean(state_encoding_tokens, dim=1) # [B, T, F]
        
        # return {
        #     "obs_feats": obs_encoding,   # [B, F, H, W]
        #     "goal_feats": goal_encoding, # [B, F, H, W]
        #     "state_cond": avg_state_encoding,
        # }

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

if __name__ == "__main__":
    from omegaconf import DictConfig, OmegaConf
    cfg_path = "config/model/planning/e2e/nomad_spatial.yaml"
    model_cfg = OmegaConf.load(cfg_path)
    model = NoMaD_ViNT(model_cfg)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    dummy_obs = torch.randn(B, 3, 240, 320).to(device)
    dummy_goal = torch.randn(B, 3, 240, 320).to(device)
    dummy_text = ["This is a test goal"] * B
    model = model.to(device)

    outputs = model(dummy_obs, dummy_goal) # [B, #patches, F]
    print("Observation Encoding Shape:", outputs["goal_encoding"].shape)
    print("Average Observation Encoding Shape:", outputs["avg_goal_encoding"].shape)

