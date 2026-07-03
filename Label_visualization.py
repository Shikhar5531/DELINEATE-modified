import os
import cv2
import numpy as np
from tqdm import tqdm

# -------- PATHS --------
input_dir  = r"C:\Users\shikh\DELINEATE\dil-Unet\datasets\test\val\gt\0"
output_dir = r"C:\Users\shikh\DELINEATE\dil-Unet\datasets\test\val\gt_visible"

os.makedirs(output_dir, exist_ok=True)

print("Converting masks (0/1 → 0/255)...\n")

for file in tqdm(os.listdir(input_dir)):

    if not file.endswith(".png"):
        continue

    input_path  = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)

    # read grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # convert: 0/1 → 0/255
    visible = (img > 0).astype(np.uint8) * 255

    # save
    cv2.imwrite(output_path, visible)

print("\nDone! Saved to:", output_dir)