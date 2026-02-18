import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import numpy as np

DEBUG_LOSS = 0

class Loss(nn.Module):
    def __init__(self, name, config):
        super(Loss, self).__init__()

        self.config = config
        self._name = name + config.get('tag', '')
        self.weight = config.get('weight', 1.0)
        self.task = config.get('task', None)

    def forward(self, tensor_dict):
        loss_dict, meta_data = self.loss(tensor_dict)

        ret_loss_dict = dict()

        # Apply the learnable loss weights if present
        logvar_key = self.config.get('logvar_key', None)
        if logvar_key is not None:
            log_var = tensor_dict[logvar_key]
            w = 1.0 / (2.0 * torch.exp(log_var))

            if logvar_key is not None:
                ret_loss_dict['log_std'] = (1.0, 0.5 * log_var)
        else:
            w = 1.0
        
        ret_loss_dict.update({k: (self.weight * w, v)
                             for k, v in loss_dict.items()})
        return ret_loss_dict, meta_data

    def loss(self, tensor_dict):
        raise Exception('Not Implemented!')

    @property
    def name(self):
        return self._name


class LossManager(nn.Module):
    def __init__(self, config):
        super(LossManager, self).__init__()

        self.config = config
        self.losses = nn.ModuleList()  # auotmatically moved to correct device
        for lc in config.loss:
            print(f'Adding loss {lc.name}')
            loss = self.get_loss(lc)
            self.losses.append(loss)

    def forward(self, tensor_dict):
        loss_dict, meta_data = {}, {}
        for loss in self.losses:
            loss_task = loss.task

            # Useful for multi-task learning
            if loss_task is None or len(loss_task)==0 or loss_task == tensor_dict['task']:
                ld, md = loss(tensor_dict)
                md = {f'{loss.name}/{key}': val for key, val in md.items()}
                ld = {f'{loss.name}/{key}': val for key, val in ld.items()}
                meta_data.update(md)
                loss_dict.update(ld)

        return loss_dict, meta_data

    def get_loss(self, config):
        g = self.config
        task = config.get('task', None)
        return globals()[config['name']](config)
    
class PathMaskMSELoss(Loss):
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    def __init__(self, config):
        super(PathMaskMSELoss, self).__init__(config.name, config)
        self.config = config
        self.reduction = config.get('reduction', 'mean')
        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.log_name = config.get('log_name', 'path_mask_mse_loss')

        self.loss_fn = nn.MSELoss(reduction=self.reduction)

    def loss(self, tensor_dict):
        """
        Args:
            tensor_dict (dict): Dictionary containing 'predicted_depth' and 'target_depth'.
        
        Returns:
            dict: Loss value.
        """
        preds = tensor_dict[self.pred_key]
        targets = tensor_dict[self.lab_key]

        # TODO: Implement masking in the future
        assert preds.shape == targets.shape, \
            f"Predicted shape {preds.shape} does not match target shape {targets.shape}."
        loss = self.loss_fn(preds, targets)

        loss_dict = {
            self.log_name: loss
        }

        return loss_dict, {}

class ActionMatchingLoss(Loss):
    """
    Squared‐Euclidean action matching, implemented via built-in MSELoss.
    For each (b,t) we compute ||pred[b,t] − tgt[b,t]||², then reduce over
    batch and time.
    """
    def __init__(self, config):
        super().__init__(config.name, config)
        self.pred_key  = config['pred_key']
        self.lab_key   = config['lab_key']
        self.log_name  = config.get('log_name', 'action_matching_loss')
        self.reduction = config.get('reduction', 'mean')  # 'mean' or 'sum'
        # we only need MSELoss to get per-element squared errors
        self.mse_loss = nn.MSELoss(reduction='none')

    def loss(self, tensor_dict):
        preds   = tensor_dict[self.pred_key]   # [B, T, D]
        targets = tensor_dict[self.lab_key]    # [B, T, D]
        assert preds.shape == targets.shape, \
            f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"

        # 1) per‐element squared error [B,T,D]
        se = self.mse_loss(preds, targets)
        # 2) sum over action dims → [B, T]
        per_step = se.sum(dim=-1)

        # 3) reduce over batch & time
        if self.reduction == 'mean':
            loss = per_step.mean()
        elif self.reduction == 'sum':
            loss = per_step.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        loss_dict = {self.log_name: loss}
        return loss_dict, {}