##
import os
import cv2
import numpy as np
from tqdm import tqdm

pred_dir = r"E:\DELINEATE\dil-Unet\checkpoints_fine_tune_combined_data\model-20500_new_data"
gt_dir   = r"E:\DELINEATE\dil-Unet\datasets\test\val\gt\0"

missing = 0

mean_dice_scores = []
bg_dice_scores = []
fat_dice_scores = []

bg_precision_scores = []
bg_recall_scores = []

fat_precision_scores = []
fat_recall_scores = []

print("Running evaluation...\n")

gt_files = sorted(os.listdir(gt_dir))

for i, gt_name in enumerate(tqdm(gt_files)):

    pred_path = os.path.join(pred_dir, gt_name)

    if not os.path.exists(pred_path):
        missing += 1
        continue

    gt_path = os.path.join(gt_dir, gt_name)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path)

    if len(pred.shape) == 3:
        pred = pred[:, :, 2]

    pred = (pred > 127).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    # ==========================
    # Background Dice
    # ==========================
    pred_bg = (pred == 0)
    gt_bg   = (gt == 0)

    TP = np.sum(pred_bg & gt_bg)
    FP = np.sum(pred_bg & (~gt_bg))
    FN = np.sum((~pred_bg) & gt_bg)

    denom = 2 * TP + FP + FN
    bg_dice = (2 * TP) / denom if denom != 0 else 1.0

    # ==========================
    # Fat Dice
    # ==========================
    pred_fg = (pred == 1)
    gt_fg   = (gt == 1)

    TP = np.sum(pred_fg & gt_fg)
    FP = np.sum(pred_fg & (~gt_fg))
    FN = np.sum((~pred_fg) & gt_fg)

    denom = 2 * TP + FP + FN
    fat_dice = (2 * TP) / denom if denom != 0 else 1.0

    mean_dice = (bg_dice + fat_dice) / 2

    bg_dice_scores.append(bg_dice)
    fat_dice_scores.append(fat_dice)
    mean_dice_scores.append(mean_dice)

    # ==========================
    # Background Precision & Recall
    # ==========================

    pred_bg = (pred == 0)
    gt_bg   = (gt == 0)

    TP = np.sum(pred_bg & gt_bg)
    FP = np.sum(pred_bg & (~gt_bg))
    FN = np.sum((~pred_bg) & gt_bg)

    bg_precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    bg_recall    = TP / (TP + FN) if (TP + FN) != 0 else 1.0

    bg_precision_scores.append(bg_precision)
    bg_recall_scores.append(bg_recall)

    # ==========================
    # Fat Precision & Recall
    # ==========================

    pred_fg = (pred == 1)
    gt_fg   = (gt == 1)

    TP = np.sum(pred_fg & gt_fg)
    FP = np.sum(pred_fg & (~gt_fg))
    FN = np.sum((~pred_fg) & gt_fg)

    fat_precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    fat_recall    = TP / (TP + FN) if (TP + FN) != 0 else 1.0

    fat_precision_scores.append(fat_precision)
    fat_recall_scores.append(fat_recall)

# ==========================
# Final Results
# ==========================

print("\n=================================")
print(f"Missing predictions: {missing}")
print("=================================")

print(f"Mean Dice (overall):      {np.mean(mean_dice_scores):.4f}")
print(f"Mean Dice (background):   {np.mean(bg_dice_scores):.4f}")
print(f"Mean Dice (fat):          {np.mean(fat_dice_scores):.4f}")

print(f"Mean Background Precision: {np.mean(bg_precision_scores):.4f}")
print(f"Mean Background Recall:    {np.mean(bg_recall_scores):.4f}")

print(f"Mean Fat Precision:        {np.mean(fat_precision_scores):.4f}")
print(f"Mean Fat Recall:           {np.mean(fat_recall_scores):.4f}")

print("=================================")