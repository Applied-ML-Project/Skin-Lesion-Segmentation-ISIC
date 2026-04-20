import time
import numpy as np
import torch
from tqdm import tqdm
import config
from metrics import dice_score, iou_score, hausdorff_distance


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_model(model, test_loader, model_name="model"):
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()

    dice_scores = []
    iou_scores = []
    hd_scores = []
    inference_times = []

    for images, masks in tqdm(test_loader, desc=f"Evaluating {model_name}"):
        images, masks = images.to(device), masks.to(device)

        start = time.time()
        preds = model(images)
        torch.cuda.synchronize() if device.type == "cuda" else None
        inference_times.append(time.time() - start)

        preds_bin = (preds > 0.5).float()

        for i in range(images.size(0)):
            p = preds_bin[i, 0].cpu().numpy()
            t = masks[i, 0].cpu().numpy()

            dice_scores.append(dice_score(p, t))
            iou_scores.append(iou_score(p, t))

            hd = hausdorff_distance(p, t)
            if hd != float("inf"):
                hd_scores.append(hd)

    results = {
        "model": model_name,
        "params": count_parameters(model),
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "iou_mean": np.mean(iou_scores),
        "iou_std": np.std(iou_scores),
        "hd_mean": np.mean(hd_scores) if hd_scores else float("nan"),
        "hd_std": np.std(hd_scores) if hd_scores else float("nan"),
        "avg_inference_time": np.mean(inference_times),
    }

    return results, dice_scores, iou_scores


def print_comparison(results_list):
    print(f"\n{'='*70}")
    print(f"{'METRIC':<25} | {'Attention U-Net':>18} | {'TransUNet':>18}")
    print(f"{'-'*70}")

    r1, r2 = results_list[0], results_list[1]

    print(f"{'Parameters':<25} | {r1['params']:>18,} | {r2['params']:>18,}")
    print(f"{'Dice Score':<25} | {r1['dice_mean']:.4f} ± {r1['dice_std']:.4f}  | {r2['dice_mean']:.4f} ± {r2['dice_std']:.4f}")
    print(f"{'IoU Score':<25} | {r1['iou_mean']:.4f} ± {r1['iou_std']:.4f}  | {r2['iou_mean']:.4f} ± {r2['iou_std']:.4f}")
    print(f"{'Hausdorff Distance':<25} | {r1['hd_mean']:.2f} ± {r1['hd_std']:.2f}    | {r2['hd_mean']:.2f} ± {r2['hd_std']:.2f}")
    print(f"{'Avg Inference Time (s)':<25} | {r1['avg_inference_time']:.4f}            | {r2['avg_inference_time']:.4f}")
    print(f"{'='*70}\n")
