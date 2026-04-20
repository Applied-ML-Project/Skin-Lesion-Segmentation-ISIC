import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, target, smooth=1.0):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def hausdorff_distance(pred, target):
    """Directed Hausdorff distance between binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)

    hd_pred_to_target = dt_target[pred].max()
    hd_target_to_pred = dt_pred[target].max()

    return max(hd_pred_to_target, hd_target_to_pred)
