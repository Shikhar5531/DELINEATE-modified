import os
import cv2
import numpy as np
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from skimage import morphology
from scipy import ndimage

# =========================
# USER SETTINGS
# =========================
INPUT_FOLDER = r"E:\DELINEATE\dil-Unet\checkpoints_fine_tune_combined_data_dice_loss_batch_dice\model-20500"
GT_FOLDER = r"E:\DELINEATE\dil-Unet\datasets\test\val\gt\0"
OUTPUT_CSV = r"E:\DELINEATE\dil-Unet\all_postprocessing_combinations_results.csv"

# Parameters from your previous experiments
RSO_MIN_SIZE = 2550
RSO_CONNECTIVITY = 1
RSH_AREA_THRESHOLD = 51100
RSH_CONNECTIVITY = 1

def apply_operation(mask, op):
    if op == "RSO":
        return morphology.remove_small_objects(
            mask, min_size=RSO_MIN_SIZE, connectivity=RSO_CONNECTIVITY)
    if op == "RSH":
        return morphology.remove_small_holes(
            mask, area_threshold=RSH_AREA_THRESHOLD,
            connectivity=RSH_CONNECTIVITY)
    if op == "BFH":
        return ndimage.binary_fill_holes(mask)
    raise ValueError(f"Unknown operation: {op}")

def apply_pipeline(mask, pipeline):
    out = mask.copy()
    for op in pipeline:
        out = apply_operation(out, op)
    return out

def calculate_metrics(pred, gt):
    # Background
    pred_bg, gt_bg = pred == 0, gt == 0
    TP = np.sum(pred_bg & gt_bg)
    FP = np.sum(pred_bg & (~gt_bg))
    FN = np.sum((~pred_bg) & gt_bg)
    denom = 2 * TP + FP + FN
    bg_dice = (2 * TP) / denom if denom else 1.0
    bg_precision = TP / (TP + FP) if (TP + FP) else 1.0
    bg_recall = TP / (TP + FN) if (TP + FN) else 1.0

    # Fat
    pred_fg, gt_fg = pred == 1, gt == 1
    TP = np.sum(pred_fg & gt_fg)
    FP = np.sum(pred_fg & (~gt_fg))
    FN = np.sum((~pred_fg) & gt_fg)
    denom = 2 * TP + FP + FN
    fat_dice = (2 * TP) / denom if denom else 1.0
    fat_precision = TP / (TP + FP) if (TP + FP) else 1.0
    fat_recall = TP / (TP + FN) if (TP + FN) else 1.0

    return {
        "background_dice": bg_dice,
        "fat_dice": fat_dice,
        "overall_dice": (bg_dice + fat_dice) / 2,
        "background_precision": bg_precision,
        "background_recall": bg_recall,
        "fat_precision": fat_precision,
        "fat_recall": fat_recall
    }

# =========================
# LOAD DATA ONCE
# =========================
files = sorted(f for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))

images = []
missing_gt = 0

print(f"Prediction images found: {len(files)}")

for name in tqdm(files, desc="Loading images"):
    pred_path = os.path.join(INPUT_FOLDER, name)
    gt_path = os.path.join(GT_FOLDER, name)

    if not os.path.exists(gt_path):
        missing_gt += 1
        continue

    pred = cv2.imread(pred_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        print(f"Skipping unreadable file: {name}")
        continue

    if pred.ndim == 3:
        pred = pred[:, :, 2]  # red channel, matching your metrics.py

    pred = (pred > 127).astype(bool)
    gt = (gt > 0).astype(np.uint8)

    images.append((name, pred, gt))

print(f"Images evaluated: {len(images)}")
print(f"Missing GT: {missing_gt}")

# =========================
# ALL POSSIBLE ORDERED PIPELINES
# =========================
ops = ("RSO", "RSH", "BFH")

# Include raw prediction as baseline.
pipelines = [()]
for n in range(1, len(ops) + 1):
    pipelines.extend(permutations(ops, n))

# 1 baseline + 3 singles + 6 pairs + 6 triples = 16
print(f"Total pipelines: {len(pipelines)}")

results = []

for pipeline in pipelines:
    name = "NONE" if not pipeline else " -> ".join(pipeline)

    metric_lists = {k: [] for k in [
        "background_dice", "fat_dice", "overall_dice",
        "background_precision", "background_recall",
        "fat_precision", "fat_recall"
    ]}

    for _, pred, gt in tqdm(images, desc=name):
        processed = apply_pipeline(pred, pipeline).astype(np.uint8)
        m = calculate_metrics(processed, gt)
        for k, v in m.items():
            metric_lists[k].append(v)

    row = {
        "pipeline": name,
        "number_of_images": len(images),
        "missing_gt": missing_gt,
        "fat_dice": np.mean(metric_lists["fat_dice"]),
        "background_dice": np.mean(metric_lists["background_dice"]),
        "overall_dice": np.mean(metric_lists["overall_dice"]),
        "fat_precision": np.mean(metric_lists["fat_precision"]),
        "fat_recall": np.mean(metric_lists["fat_recall"]),
        "background_precision": np.mean(metric_lists["background_precision"]),
        "background_recall": np.mean(metric_lists["background_recall"]),
        "RSO_min_size": RSO_MIN_SIZE if "RSO" in pipeline else "",
        "RSO_connectivity": RSO_CONNECTIVITY if "RSO" in pipeline else "",
        "RSH_area_threshold": RSH_AREA_THRESHOLD if "RSH" in pipeline else "",
        "RSH_connectivity": RSH_CONNECTIVITY if "RSH" in pipeline else ""
    }
    results.append(row)

results_df = pd.DataFrame(results).sort_values("fat_dice", ascending=False)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 70)
print("ALL POST-PROCESSING RESULTS")
print("=" * 70)
print(results_df[[
    "pipeline", "fat_dice", "fat_precision",
    "fat_recall", "overall_dice"
]].to_string(index=False))

best = results_df.iloc[0]
print("\nBEST PIPELINE:")
print(best["pipeline"])
print(f"Fat Dice:      {best['fat_dice']:.4f}")
print(f"Fat Precision: {best['fat_precision']:.4f}")
print(f"Fat Recall:    {best['fat_recall']:.4f}")
print(f"Overall Dice:  {best['overall_dice']:.4f}")
print(f"\nSaved to: {OUTPUT_CSV}")
