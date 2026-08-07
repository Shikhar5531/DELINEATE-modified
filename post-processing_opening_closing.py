import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# INPUT / OUTPUT
# ==========================================================

input_folder = r"E:\DELINEATE\dil-Unet\checkpoints_fine_tune_combined_data_dice_loss_batch_dice\model-20500"
parent_output = r"E:\DELINEATE\dil-Unet\checkpoints_fine_tune_combined_data_dice_loss_batch_dice"

# ==========================================================
# OPERATION
# ==========================================================

# Choose ONE:
# "closing"
# "opening"
# "opening_closing"

operation = "opening_closing"

# ==========================================================
# PARAMETERS
# ==========================================================

kernel_sizes = [50]

# ==========================================================
# IMAGE LIST
# ==========================================================

image_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

print(f"\nFound {len(image_list)} images.\n")

# ==========================================================
# LOOP OVER KERNEL SIZES
# ==========================================================

for kernel_size in kernel_sizes:

    print("=" * 60)
    print(f"Running {operation} | Kernel = {kernel_size}")
    print("=" * 60)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    output_folder = os.path.join(parent_output, f"{operation}_{kernel_size}")

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
        img = (img > 0).astype(np.uint8) * 255

        # --------------------------------------------------
        # Morphological operation
        # --------------------------------------------------

        if operation == "closing":

            processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        elif operation == "opening":

            processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        elif operation == "opening_closing":

            processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        else:

            raise ValueError(
                "Unknown operation.\n"
                "Choose one of:\n"
                "closing\n"
                "opening\n"
                "opening_closing"
            )

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