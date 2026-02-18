import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -------------------- Depth Metrics --------------------
def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()

def intersection_over_union(pred, target, valid_mask=None):
    """Calculate Intersection over Union (IoU) for segmentation tasks."""
    if valid_mask is not None:
        pred = pred[valid_mask]
        target = target[valid_mask]
    pred = pred > 0.5  
    target = target > 0.5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection

    if union == 0:
        return torch.tensor(0.0, device=pred.device)

    iou = intersection / union
    return iou

def mean_l2_error(output, target, valid_mask=None):
    """Compute average L2 error between output and target."""
    l2_error = torch.norm(output - target, p=2, dim=(-1))
    if valid_mask is not None:
        valid_mask = valid_mask.all(dim=-1)
        l2_error = l2_error[valid_mask]
    return torch.mean(l2_error)

def mean_asym_hausdorff_distance(
    output: torch.Tensor,          # (B, T_pred, 2)
    target: torch.Tensor,          # (B, T_gt,   2)
    valid_mask: torch.Tensor | None = None   # (B, T_gt)  True = keep
) -> torch.Tensor:
    """
    Mean asymmetric Hausdorff distance   H(target → output)
    Do *not* penalise predictions that lie beyond all GT points.

    Returns
    -------
    torch.Tensor
        Scalar tensor (0-D) with the batch-mean asymmetric HD.
    """
    # pairwise distances & closest pt (B, T_gt, T_pred)
    dists = torch.cdist(target, output, p=2.0)
    min_dists = dists.min(dim=2).values       # (B, T_gt)

    if valid_mask is not None:
        # mask out invalid GT points
        valid_mask = valid_mask.all(dim=-1)  # (B,)
        min_dists = min_dists.masked_fill(~valid_mask.bool(), float("-inf"))

    # asymmetric HD per sample: max over GT points
    h_per_sample = min_dists.max(dim=1).values.clamp_min(0.0)   # (B,)
    
    return h_per_sample.mean()

def precision_recall(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask: np.ndarray = None,
    num_bins: int = 100,
    gt_threshold: float = 0.5,
    return_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a precision–recall curve between ground-truth and predicted masks.

    Args:
        gt_arr:     (B,1,H,W) array of ground-truth values in [0,1].
        pred_arr:   (B,1,H,W) array of predicted probabilities in [0,1].
        num_bins:   number of thresholds to sweep between 0 and 1.
        gt_threshold: threshold to binarize ground truth into positives.
        return_counts: if True, return (tp, fp, fn) counts instead of precision/recall.

    Returns:
        If return_counts=False:
            thresholds: (num_bins,) array of thresholds used.
            precision:  (num_bins,) array of precision at each threshold.
            recall:     (num_bins,) array of recall at each threshold.
        If return_counts=True:
            thresholds: (num_bins,) array of thresholds used.
            tp_counts:  (num_bins,) array of true positive counts.
            fp_counts:  (num_bins,) array of false positive counts.
            fn_counts:  (num_bins,) array of false negative counts.
    """
    # flatten to 1D
    gt_flat   = gt_arr.reshape(-1)
    pred_flat = pred_arr.reshape(-1)

    if valid_mask is not None:
        # apply valid mask if provided
        valid_mask = valid_mask.reshape(-1)
        gt_flat   = gt_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]

    # binarize ground truth
    gt_pos = gt_flat >= gt_threshold
    n_pos  = gt_pos.sum()

    thresholds = np.linspace(0.0, 1.0, num_bins)
    
    if return_counts:
        tp_counts = np.empty(num_bins, dtype=int)
        fp_counts = np.empty(num_bins, dtype=int)
        fn_counts = np.empty(num_bins, dtype=int)
        
        for i, thr in enumerate(thresholds):
            pred_pos = pred_flat >= thr
            tp = int(np.logical_and(pred_pos, gt_pos).sum())
            pp = int(pred_pos.sum())
            fp = pp - tp
            fn = int(n_pos - tp)
            
            tp_counts[i] = tp
            fp_counts[i] = fp
            fn_counts[i] = fn
            
        return thresholds, tp_counts, fp_counts, fn_counts
    else:
        precision = np.empty(num_bins, dtype=float)
        recall    = np.empty(num_bins, dtype=float)
        for i, thr in enumerate(thresholds):
            pred_pos = pred_flat >= thr
            tp = int(np.logical_and(pred_pos, gt_pos).sum())
            pp = int(pred_pos.sum())
            fn = int(n_pos - tp)

            precision[i] = tp / pp if pp > 0 else 1.0
            recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return thresholds, precision, recall

# Borrowed from https://github.com/prs-eth/Marigold#-citation
class MetricManager(nn.Module):
    def __init__(self, *configs, writer=None):
        super(MetricManager, self).__init__()
        self.writer = writer

        try:
            self.configs = {
                config['name']: config for config in configs if globals()[config['name']] is not None
            }
        except KeyError as e:
            raise ValueError(f"Metric function {e} not found in globals()") from e

        self._data = pd.DataFrame(index=list(self.configs.keys()), columns=["total", "counts", "average"])
        self._streaming_data = {}  # For streaming counts
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        self._streaming_data.clear()

    def set_writer(self, writer):
        self.writer = writer

    def update(self, tensor_dict, cur_epoch, n=1):
        """Return dictionary with current metrics"""
        metrics_dict = {}
        for key, metric in self.configs.items():
            pred = tensor_dict.get(metric['pred_key'])
            target = tensor_dict.get(metric['lab_key'])

            # Average over ensemble if it exists
            if pred.ndim == 5:  # Assuming shape is (B, E, C, H, W)
                pred = pred.mean(dim=1)  # Average over ensemble dimension

            valid_mask = torch.isfinite(pred) & torch.isfinite(target)
            if valid_mask.sum() == 0:
                continue

            metric_fn = globals()[metric['name']]
            kwargs = metric.get('kwargs', {})

            # Check if this is precision_recall and if streaming is enabled
            if metric['name'] == 'precision_recall' and kwargs.get('return_counts', False):                
                result = metric_fn(pred.cpu().numpy(), target.cpu().numpy(), 
                                 valid_mask=valid_mask.cpu().numpy(), **kwargs)
                thresholds, tp_counts, fp_counts, fn_counts = result
                
                # Initialize or accumulate streaming data
                if key not in self._streaming_data:
                    self._streaming_data[key] = {
                        'thresholds': thresholds,
                        'tp_total': tp_counts.copy(),
                        'fp_total': fp_counts.copy(), 
                        'fn_total': fn_counts.copy()
                    }
                else:
                    self._streaming_data[key]['tp_total'] += tp_counts
                    self._streaming_data[key]['fp_total'] += fp_counts
                    self._streaming_data[key]['fn_total'] += fn_counts
                
                # Compute current precision/recall for logging
                tp_total = self._streaming_data[key]['tp_total']
                fp_total = self._streaming_data[key]['fp_total']
                fn_total = self._streaming_data[key]['fn_total']
                
                precision = np.where(tp_total + fp_total > 0, tp_total / (tp_total + fp_total), 0.0)
                recall = np.where(tp_total + fn_total > 0, tp_total / (tp_total + fn_total), 0.0)
                
                # Use mean precision as scalar value for compatibility
                value = np.mean(precision)
            else:
                # Regular metric computation
                value = metric_fn(
                    pred, 
                    target, 
                    valid_mask=valid_mask, 
                    **kwargs
                )
                value = value.cpu().item() if isinstance(value, torch.Tensor) else value

            metrics_dict[key] = value

            # Update running averages
            self._data.loc[key, "total"] += value * n
            self._data.loc[key, "counts"] += n
            self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

        if self.writer is not None:
            for key, value in metrics_dict.items():
                self.writer.experiment.add_scalar(key, value, global_step=cur_epoch)

        return metrics_dict

    def get_streaming_curves(self, key):
        """Get accumulated precision-recall curves for streaming metrics."""
        if key not in self._streaming_data:
            return None
            
        data = self._streaming_data[key]
        tp_total = data['tp_total']
        fp_total = data['fp_total'] 
        fn_total = data['fn_total']
        
        precision = np.where(tp_total + fp_total > 0, tp_total / (tp_total + fp_total), 1.0)
        recall = np.where(tp_total + fn_total > 0, tp_total / (tp_total + fn_total), 0.0)
        
        return data['thresholds'], precision, recall

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)