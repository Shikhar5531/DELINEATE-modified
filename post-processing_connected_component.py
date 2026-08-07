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
#
# "remove_small_blobs"
# "fill_small_holes"
# "fill_remove"

operation = "fill_remove"

# ==========================================================
# THRESHOLDS
# ==========================================================

thresholds = [5,10,20,50,100,200,500,1000]

# ==========================================================
# IMAGE LIST
# ==========================================================

image_list = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith(".png")
])

print(f"\nFound {len(image_list)} images.\n")

# ==========================================================
# FUNCTIONS
# ==========================================================

def remove_small_blobs(binary, area_threshold):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    output = np.zeros_like(binary)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= area_threshold:
            output[labels == i] = 255

    return output


def fill_small_holes(binary, area_threshold):

    inv = cv2.bitwise_not(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inv,
        connectivity=8
    )

    output = binary.copy()

    h, w = binary.shape

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        mask = (labels == i)

        # Ignore components touching image border
        ys, xs = np.where(mask)

        if (
            np.any(ys == 0) or
            np.any(xs == 0) or
            np.any(ys == h-1) or
            np.any(xs == w-1)
        ):
            continue

        if area <= area_threshold:
            output[mask] = 255

    return output

# ==========================================================
# LOOP
# ==========================================================

for threshold in thresholds:

    print("="*60)
    print(f"{operation} | Threshold = {threshold}")
    print("="*60)

    output_folder = os.path.join(
        parent_output,
        f"{operation}_{threshold}"
    )

    os.makedirs(output_folder, exist_ok=True)

    for image_name in tqdm(image_list):

        image_path = os.path.join(input_folder, image_name)

        img = cv2.imread(image_path)

        img = img[:,:,2]

        img = (img>0).astype(np.uint8)*255

        # ------------------------------------------

        if operation == "remove_small_blobs":

            processed = remove_small_blobs(
                img,
                threshold
            )

        elif operation == "fill_small_holes":

            processed = fill_small_holes(
                img,
                threshold
            )

        elif operation == "fill_remove":

            processed = fill_small_holes(
                img,
                threshold
            )

            processed = remove_small_blobs(
                processed,
                threshold
            )

        else:

            raise ValueError("Unknown operation.")

        # ------------------------------------------
        # DELINEATE format
        # ------------------------------------------

        output = np.zeros(
            (processed.shape[0],
             processed.shape[1],
             3),
            dtype=np.uint8
        )

        output[:,:,2] = np.where(
            processed>0,
            128,
            0
        ).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_folder,image_name),
            output
        )

print("\nDone!")
