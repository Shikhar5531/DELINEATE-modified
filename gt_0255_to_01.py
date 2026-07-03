import os
import cv2
import numpy as np

# -----------------------------
# Base path
# -----------------------------
base_dir = r"E:\DELINEATE\dil-Unet\datasets"

# -----------------------------
# GT folders (explicit paths)
# -----------------------------
gt_paths = [
    os.path.join(base_dir, "train_original", "gt", "0"),
    os.path.join(base_dir, "val_original", "gt", "0"),
    os.path.join(base_dir, "test_original", "val", "gt", "0"),
]

# -----------------------------
# Convert function
# -----------------------------
def convert_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"❌ Failed to read: {path}")
        return

    # Convert to 0/1
    new_mask = (mask > 0).astype(np.uint8)

    # Overwrite
    cv2.imwrite(path, new_mask)

# -----------------------------
# Process all GT folders
# -----------------------------
total = 0

for gt_dir in gt_paths:
    print(f"\n🔄 Processing: {gt_dir}")

    if not os.path.exists(gt_dir):
        print("⚠️ Path not found, skipping")
        continue

    for file in os.listdir(gt_dir):
        file_path = os.path.join(gt_dir, file)

        convert_mask(file_path)
        total += 1

print(f"\n✅ Done! Processed {total} masks.")