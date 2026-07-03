import os
import cv2
import numpy as np
from tqdm import tqdm

pred_dir = r"E:\DELINEATE\dil-Unet\checkpoints_training_from_scratch\model-1000"
gt_dir   = r"E:\DELINEATE\dil-Unet\datasets\test_64\val\gt\0"

missing = 0

mean_dice_scores = []
bg_dice_scores = []
fat_dice_scores = []

print("Indexing prediction files...")

pred_list = []

for f in os.listdir(pred_dir):
    try:
        idx = int(f.split("_")[0])
        pred_list.append((idx, f))
    except:
        continue

pred_list = sorted(pred_list, key=lambda x: x[0])

pred_files = {str(i): f for i, (_, f) in enumerate(pred_list)}

print("Running evaluation...\n")

gt_files = sorted(os.listdir(gt_dir))

for i, gt_name in enumerate(tqdm(gt_files)):

    pred_file = pred_files.get(str(i))

    if pred_file is None:
        missing += 1
        continue

    gt_path = os.path.join(gt_dir, gt_name)
    pred_path = os.path.join(pred_dir, pred_file)

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
# Final Results
# ==========================
print("\n=================================")
print(f"Missing predictions: {missing}")
print("=================================")

print(f"Mean Dice (overall):      {np.mean(mean_dice_scores):.4f}")
print(f"Mean Dice (background):   {np.mean(bg_dice_scores):.4f}")
print(f"Mean Dice (fat):          {np.mean(fat_dice_scores):.4f}")

print("=================================")