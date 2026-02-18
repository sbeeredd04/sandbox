import torch
from typing import Sequence, Union, Tuple
import numpy as np

from spinflow.util.math_utils import se3_matrix

def unnormalize_action(
    action: torch.Tensor,
    action_range: Sequence[float],
    action_stats: torch.Tensor,
    action_dim: int
) -> torch.Tensor:
    """
    Take an action in [-1,1] and un‐normalize it back into [min,max] per axis,
    where min = action_stats[:action_dim], max = action_stats[action_dim:].
    """
    if not isinstance(action_stats, torch.Tensor):
        action_stats = torch.tensor(action_stats, dtype=action.dtype, device=action.device)

    # 1) clamp to [-1, 1]
    action = action.clip(action_range[0], action_range[1])
    # 2) map [-1,1] → [0,1]
    action = (action - action_range[0]) / (action_range[1] - action_range[0])
    #    (for [-1,1], that is (action + 1) / 2)

    # 3) get your dims
    D   = action_dim                        # number of channels
    mins = action_stats[:D].view(1,1,-1)     # [1,1,D]
    maxs = action_stats[D:].view(1,1,-1)     # [1,1,D]

    # 4) scale [0,1] → [min,max]
    action = action * (maxs - mins) + mins

    # 5) final clamp in case of tiny numerical overshoot
    action = action.clamp(mins, maxs)

    return action

def compensate_action(
    actions: np.ndarray,
    infos: dict,
    x_offset: float = 0.27,
    z_offset: float = -0.576,
):
    assert actions.ndim == 2 and actions.shape[1] == 8, \
        f"Expected actions shape (N, 8), got {actions.shape}"
    assert 'T_optical_to_base' in infos, \
        "infos must contain 'T_optical_to_base' for camera to base transformation."
    p_cam_base = np.concatenate([infos['T_optical_to_base'][:3, 3], [1]])
    T_base_odom = se3_matrix(
        actions[:, 1],
        actions[:, 2],
        actions[:, 3],
        actions[:, 4:8]  # qw, qx, qy, qz
    )
    p_cam_odom = T_base_odom @ p_cam_base.T
    p_cam_odom[:, 2] += z_offset
    p_cam_odom[:, 0] += x_offset

    return p_cam_odom[:, :3]  # Return only x, y, z coordinates