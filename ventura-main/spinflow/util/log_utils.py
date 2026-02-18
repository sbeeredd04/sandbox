import logging
import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict, List, Tuple

from spinflow.util.image_utils import (colorize_depth_maps)
from spinflow.util.vis_utils import (
    plot_odometry_topdown,
    draw_xyz_on_image
)

from einops import rearrange

# Set up basic logging configuration.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# ----- TENSORBOARD LOGGER UTILS ----- #

def _overlay_mask(img, mask, colour, alpha=0.4):
    mask = mask.clip(0, 1) * alpha
    return img * (1 - mask) + colour * mask

@rank_zero_only
@torch.no_grad()
def log_image_actions_to_tb(
    tb_writer,
    tensor_dict: Dict[str, torch.Tensor],
    inputs_dict: Dict[str, torch.Tensor],
    vis_spec: Dict,
    epoch: int,
    global_step: int,
    batch_indices: List[int] = None,
    *,
    prefix: str = "val",
):
    """
    Log a visualization of actions to TensorBoard.
    
    Parameters
    ----------
    tb_writer : TensorBoard logger
        The TensorBoard logger to write the image to.
    tensor_dict : dict
        Dictionary containing tensors for visualization.
    vis_spec : dict
        Specification for the visualization.
    epoch : int
        Current epoch number.
    global_step : int
        Global step count.
    prefix : str, optional
        Prefix for the log tag, by default "val".
    """
    actions_pred = tensor_dict[vis_spec["pred_key"]].float()
    actions_gt = tensor_dict[vis_spec["lab_key"]].float()
    obs = tensor_dict[vis_spec["obs_key"]].float()  # B x T x 3 x H x W
    path_mask = tensor_dict.get(vis_spec.get("path_mask_key", ""), None)

    if obs.ndim == 5:
        B, T, C, H, W = obs.shape
        cur_obs = obs[:, -1]  # [B, C, H, W] last observation
    else:
        assert obs.ndim == 4, f"Expected obs to be 4D or 5D, got {obs.ndim}D"
        B, C, H, W = obs.shape
        cur_obs = obs
    if path_mask is not None and path_mask.ndim == 5:
        path_mask = path_mask[:, -1]  # [B, C, H, W] last path mask

    images_per_row = vis_spec.get("images_per_row", 8)
    log_keys = [vis_spec["pred_key"], vis_spec["lab_key"]]
    labels = ["Predicted Actions", "Ground Truth Actions"]
    assert all([k in tensor_dict for k in log_keys]), \
        f"Missing keys in tensor_dict: {log_keys}"
    
    # Rescale obs to desired range
    in_range = vis_spec.get("in_range", [-1, 1])
    out_range = vis_spec.get("out_range", [0, 255])
    cur_obs = ((cur_obs - in_range[0]) / (in_range[1] - in_range[0])).clamp(0, 1)
    cur_obs = (cur_obs * (out_range[1] - out_range[0]) + out_range[0]).clamp(0, 255).to(torch.uint8)

    pred_tiles = []
    for b in batch_indices:
        # Prepare the observation tile
        cur_obs_b = cur_obs[b].unsqueeze(0)  # [1, C, H, W]
        gt_xyz_b = actions_gt[b].unsqueeze(0)  # [1, T, 3]
        pred_xyz_b = actions_pred[b].unsqueeze(0)  # [1, T, 3]
        infos_b = inputs_dict['infos'][b]

        # Draw ground truth on image in blue
        ann_obs = draw_xyz_on_image(
            cur_obs_b, gt_xyz_b, infos_b, color=(0, 0, 255)
        )
        # Draw predicted actions on image in aqua 
        ann_obs = draw_xyz_on_image(
            ann_obs, pred_xyz_b, infos_b, color=(51, 255, 255)
        )

        # Draw path_mask if available
        if path_mask is not None:
            path_mask_b = path_mask[b].permute(1, 2, 0)  # [H, W, C] -> [C, H, W]
            path_mask_b = (path_mask_b > 0.5).cpu().numpy()
            ann_obs = _overlay_mask(
                ann_obs, 
                path_mask_b,
                colour=np.array([51, 255, 255])  # Aqua color
            )

        ann_obs = rearrange(
            torch.from_numpy(ann_obs), 
            'h w c -> 1 c h w'
        )  # [1, C, H, W]
        pred_tiles.append(ann_obs)

    # Log all predicted tiles as a grid
    grid_img = make_grid(
        torch.cat(pred_tiles, dim=0).float(),
        nrow=images_per_row,
        normalize=True
    )
    if tb_writer is not None:
        tb_writer.experiment.add_image(
            f"{prefix}/image_actions/epoch_{epoch}",
            grid_img,
            global_step=global_step,
        )   
    return grid_img
        
@rank_zero_only
@torch.no_grad()
def log_bev_actions_to_tb(
    tb_writer,
    tensor_dict: Dict[str, torch.Tensor],
    vis_spec: Dict,
    epoch: int,
    global_step: int,
    batch_indices: List[int] = None,
    *,
    prefix: str = "val",
):
    """
    Log a BEV visualization of actions to TensorBoard.
    
    Parameters
    ----------
    tb_writer : TensorBoard logger
        The TensorBoard logger to write the image to.
    tensor_dict : dict
        Dictionary containing tensors for visualization.
    vis_spec : dict
        Specification for the visualization.
    epoch : int
        Current epoch number.
    global_step : int
        Global step count.
    prefix : str, optional
        Prefix for the log tag, by default "val".
    """
    actions_pred = tensor_dict[vis_spec["pred_key"]].float()  # B x T x 3
    B, T, C = actions_pred.shape
    images_per_row = vis_spec.get("images_per_row", 8)
    
    log_keys = [vis_spec["pred_key"], vis_spec["lab_key"]]
    labels = ["Predicted Actions", "Ground Truth Actions"]
    assert all([k in tensor_dict for k in log_keys]), \
        f"Missing keys in tensor_dict: {log_keys}"

    # max_batch_size = 16
    # if B > max_batch_size:
    #     batch_indices = torch.randperm(B)[:max_batch_size]
    # else:
    #     batch_indices = torch.arange(B)

    pred_tiles = []
    for b in batch_indices:
        odom_plot_np = plot_odometry_topdown(
            tensor_dict,
            keys=log_keys,
            labels=labels,
            batch_index=b,
            axis_limits=[0.0, -4.8, 4.8, 4.8],
        ).copy()  # H x W x 3
        odom_plot_th = torch.from_numpy(odom_plot_np).permute(2, 0, 1)  # H x W x C -> C x H x W
        odom_plot_th = resize(odom_plot_th, (256, 256))
        odom_plot_th = odom_plot_th.unsqueeze(0).float()

        pred_tiles.append(odom_plot_th)
    # Log all predicted tiles as a grid
    grid_img = make_grid(
        torch.cat(pred_tiles, dim=0), 
        nrow=images_per_row, 
        normalize=True
    )
    tb_writer.experiment.add_image(
        f"{prefix}/bev_actions/epoch_{epoch}",
        grid_img,
        global_step=global_step,
    )

@rank_zero_only
@torch.no_grad()
def log_path_mask_to_tb(
    tb_writer,
    tensor_dict: Dict[str, torch.Tensor],
    log_config : List[Dict],
    epoch      : int,
    global_step: int,
    *,
    prefix: str = "val",
    mask_color_rgb: Tuple[float, float, float] = (0.2, 1.0, 1.0),
):
    """
    Log a compact grid where *each row* groups:
        ┌─ RGB⊙GT ──┬─ RGB⊙pred₀ ─┬─ … ┬─ RGB⊙pred_E-1 ─┐
    All tiles keep their native H×W aspect.

    Call signature and cfg unchanged → drop-in replacement.
    """
    if not log_config:
        return

    rgb      = tensor_dict[log_config[0]["name"]].float()        # B×3×H×W
    mask_gt  = tensor_dict[log_config[1]["name"]].float()        # B×1×H×W
    mask_pred = tensor_dict[log_config[2]["name"]]               # B×E×1×H×W or B×1×H×W
    if mask_pred.ndim == 4:
        mask_pred = mask_pred.unsqueeze(1)                       # add ensemble dim

    B, _, H, W = rgb.shape
    E          = mask_pred.shape[1]
    device     = rgb.device
    colour     = torch.tensor(mask_color_rgb, device=device).view(3, 1, 1)

    # map rgb to [0,1]
    rgb = ((rgb + 1) / 2).clamp(0, 1) if rgb.min() < 0 else rgb.clamp(0, 1)

    rows = []
    for b in range(B):
        img_b   = rgb[b]                       # 3×H×W
        gt_b    = mask_gt[b]                   # 1×H×W
        preds_b = mask_pred[b]                 # E×1×H×W

        # ---- ground-truth tile ---------------------------------------
        row_tiles = [_overlay_mask(img_b, gt_b, colour)]     # 3×H×W

        # ---- prediction tiles ---------------------------------------
        # broadcast rgb_b to match E predictions WITHOUT repeat-repeat bug
        img_stack = img_b.unsqueeze(0).expand(E, 3, H, W)    # share storage
        pred_tiles = _overlay_mask(img_stack, preds_b, colour)             # E×3×H×W

        # lay them horizontally (no reshape tricks)
        row_tiles += [t for t in pred_tiles]                 # list of E+1 tensors
        row = torch.cat(row_tiles, dim=2)                    # 3×H×((E+1)·W)
        rows.append(row)

    grid = torch.cat(rows, dim=1)                            # 3×(B·H)×((E+1)·W)
    tb_writer.experiment.add_image(
        f"{prefix}/path_mask_overlay/epoch_{epoch}",
        grid.cpu(),
        global_step=global_step,
    )

@rank_zero_only
def log_depth_img_to_tb(
    tb_writer,                   # lightning TensorBoardLogger
    tensor_dict: dict[str, torch.Tensor],
    log_config: list[dict],      # [{"name": "...", "type": "rgb" | "depth"}, ...]
    epoch: int,
    global_step: int,
    prefix: str = "val",
):
    """
    Build a side-by-side collage for every sample in a mini-batch and push it to
    TensorBoard.

    * RGB tensors are expected as  B x T x 3 x H x W  in [0,1] **or** [-1,1]; they
      are rescaled to [0,1].
    * Depth tensors are        B x 1 x H x W ; for every sample `b` we take the
      **minimum and maximum *across all depth maps*** in that sample and linearly
      map them to [0,1] before visualising them as greyscale.
    """
    if not len(log_config):
        return  # nothing to do
    
    # Strip the time dimension from tensor_dict items
    tensor_dict = {k: v.squeeze(1) if isinstance(v, torch.Tensor) and v.ndim == 5 else v for k, v in tensor_dict.items()}

    # ------------- collect the batch size & sanity-check shapes -------------
    first_t = tensor_dict[log_config[0]["name"]]
    assert first_t.ndim == 4, f"Expected 4D tensor, got {first_t.ndim}D tensor for {log_config[0]['name']}"
    B, _, H, W = first_t.shape

    # ------------- pre-compute normalised depth tensors ---------------------
    depth_tensors = []                       # keeps (name, tensor_norm)
    for cfg in log_config:
        if cfg["type"] != "depth":
            continue
        t = tensor_dict[cfg["name"]]         # B×1×H×W
        depth_tensors.append((cfg["name"], t))

    if depth_tensors:
        pass
        # Normalize depth together across the batch
        # # stack into B × D × H × W for per-sample min/max
        # stacked_depth = torch.stack([t for _, t in depth_tensors], dim=1)  # B×D×1×H×W
        # d_min = stacked_depth.amin(dim=[1, 3, 4], keepdim=True)
        # d_max = stacked_depth.amax(dim=[1, 3, 4], keepdim=True).clamp_min(1e-8)

        # # linear map to [0,1]
        # stacked_depth = (stacked_depth - d_min) / (d_max - d_min)

        # # split back into individual tensors
        # for idx, (name, _) in enumerate(depth_tensors):
        #     tensor_dict[name] = stacked_depth[:, idx]       # B×1×H×W  (normalised)
    
    # ------------- build a horizontal strip for every sample ---------------
    strips = []
    for b in range(B):
        imgs_b = []
        for cfg in log_config:
            t = tensor_dict[cfg["name"]][b]                 # C×H×W
            if cfg["type"] == "rgb":
                # bring to [0,1] and ensure 3 channels
                if t.min() < 0:         # assume [-1,1]
                    assert t.min() >= -1 and t.max() <= 1, \
                        f"Expected RGB tensor in [-1,1], got min {t.min()} and max {t.max()}"
                    t = (t + 1) / 2
                imgs_b.append(t.clamp(0, 1))
            elif cfg["type"] == "depth":
                # Normalize depth tensor for each item in batch
                t_rgb = colorize_depth_maps(
                    t.unsqueeze(0),  # add batch dimension
                    t.min().item(),  # min value for normalization
                    t.max().item(),  # max value for normalization
                ).to(t.device)[0]  # 3×H×W RGB

                # Change to turbo colormap
                # t_rgb = depth_to_turbo(t.unsqueeze(0))[0]# 3×H×W
                imgs_b.append(t_rgb)
            else:
                raise ValueError(f"Unknown type {cfg['type']}")

        # concat left-to-right :  C×H×(n*W)
        strip = torch.cat(imgs_b, dim=-1)
        strips.append(strip)

    # ------------- grid : (rows = batch) ------------------------------------
    grid = make_grid(torch.stack(strips), nrow=1)           # 3×H×(n*W)

    tag = f"{prefix}/depth_grid/epoch_{epoch}"
    tb_writer.experiment.add_image(tag, grid, global_step=global_step)

# def log_depth_img_to_tb(tb_writer, tensor_dict, log_config, epoch, global_step, prefix='val'):
#     """
#     Logs predicted depth images to tensorboard
#     """
#     import pdb; pdb.set_trace()


    # for key in keys:
    #     if key in tensor_dict:
    #         depth_img = tensor_dict[key]
    #     else:
    #         print(f"Key {key} not found in tensor_dict!")
    #         continue
    #     import pdb; pdb.set_trace()
    #     if not ("depth" in key and "metric" in key):
    #         continue

    #     # Normalize depth image and convert to grayscale
    #     depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    #     tb_writer.experiment.add_image(f'{prefix}/{key}/{epoch}', make_grid(depth_img.unsqueeze(1), nrow=8), global_step=global_step)