import os
import cv2
import numpy as np

# Paths
img_dir = r"E:\DELINEATE\dil-Unet\datasets\test_64\val\img\0"
mask_dir = r"E:\DELINEATE\dil-Unet\datasets\test_64\val\gt_smooth_255_9"
output_dir = r"E:\DELINEATE\dil-Unet\datasets\test_64\val\overlay_green_apr1"

os.makedirs(output_dir, exist_ok=True)

# Overlay settings
alpha = 0.4  # transparency of mask

# Loop through images
for filename in os.listdir(img_dir):
    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {filename}")
        continue

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {filename}")
        continue

    # Read mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize mask if needed
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Convert mask to binary
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create green overlay
    green_mask = np.zeros_like(img)
    green_mask[:, :, 1] = mask_bin  # Green channel

    # Blend image and mask
    overlay = cv2.addWeighted(img, 1.0, green_mask, alpha, 0)

    # Save result
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, overlay)

print("Overlay generation complete!")