import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import morphology
from scipy import ndimage

# ==========================================================
# INPUT / OUTPUT
# ==========================================================

input_folder = r"E:\DELINEATE\dil-Unet\checkpoints_our_data_their_data_downsampled\model-4920"
parent_output = r"E:\DELINEATE\dil-Unet\checkpoints_our_data_their_data_downsampled"

# ==========================================================
# OPERATION
#operation = "remove_small_objects"
#operation = "remove_small_holes"
operation = "binary_fill_holes"
# ==========================================================

# ==========================================================
# IMAGE LIST
# ==========================================================

image_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

print(f"\nFound {len(image_list)} images.\n")

# ==========================================================
# LOOP OVER KERNEL SIZES
# ==========================================================
output_folder = os.path.join(parent_output, f"{operation}")

os.makedirs(output_folder, exist_ok=True)

    # ======================================================
    # PROCESS ALL IMAGES
    # ======================================================

for image_name in tqdm(image_list):

    image_path = os.path.join(input_folder, image_name)

    # --------------------------------------------------
    # Read prediction image (RGB)
    # --------------------------------------------------

    img = cv2.imread(image_path)

    # Extract RED channel
    img = img[:, :, 2]

    # Convert to binary (0 / 255)
    img = (img > 0)

    # --------------------------------------------------
    # Morphological operation
    # --------------------------------------------------

    #processed = morphology.remove_small_objects(img, min_size=2550, connectivity=2)
    #processed = morphology.remove_small_holes(img, area_threshold=51100, connectivity=1)
    processed = ndimage.binary_fill_holes(img)

    # --------------------------------------------------
    # Convert back to DELINEATE colour format
    # --------------------------------------------------

    output = np.zeros((processed.shape[0], processed.shape[1], 3), dtype=np.uint8)

    # Red channel = 128
    output[:, :, 2] = np.where(processed > 0, 128, 0).astype(np.uint8)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    cv2.imwrite(os.path.join(output_folder, image_name), output)

print("\n")
print("=" * 60)
print("All post-processing experiments completed!")
print("=" * 60)