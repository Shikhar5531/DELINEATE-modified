import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# PATHS
# ==========================================================

reference_image = r"E:\UPF\Actual_work\1f423f40-46ab-4676-91bc-3812ac993cec_160642b0-53d9-433e-a134-772190d05807_1original.jpg"

input_folder = r"E:\UPF\Actual_work\Histology_images"

output_folder = r"E:\UPF\Actual_work\Histology_images_reinhard"

comparison_folder = r"E:\UPF\Actual_work\Reinhard_Comparison"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)

# ==========================================================
# READ REFERENCE
# ==========================================================

print("Loading reference image...")

reference = cv2.imread(reference_image)

reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

# ==========================================================
# REFERENCE STATISTICS
# ==========================================================

ref_mean = np.zeros(3)
ref_std = np.zeros(3)

for c in range(3):

    ref_mean[c] = np.mean(reference_lab[:,:,c])

    ref_std[c] = np.std(reference_lab[:,:,c])

# ==========================================================
# IMAGE LIST
# ==========================================================

image_list = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))
])

print(f"Found {len(image_list)} images.\n")

# ==========================================================
# FONT
# ==========================================================

font = cv2.FONT_HERSHEY_SIMPLEX

def add_title(img,text):

    img = img.copy()

    cv2.putText(
        img,
        text,
        (15,35),
        font,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    return img

# ==========================================================
# NORMALIZATION LOOP
# ==========================================================

for image_name in tqdm(image_list):

    image_path = os.path.join(input_folder,image_name)

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image_lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    normalized_lab = np.zeros_like(image_lab)

    # ----------------------------------------------
    # Channel-wise Reinhard normalization
    # ----------------------------------------------

    for c in range(3):

        src = image_lab[:,:,c]

        src_mean = np.mean(src)

        src_std = np.std(src)

        if src_std < 1e-8:
            src_std = 1.0

        normalized_lab[:,:,c] = (
            (src - src_mean)
            / src_std
            * ref_std[c]
            + ref_mean[c]
        )

    # ----------------------------------------------

    normalized_lab = np.clip(
        normalized_lab,
        0,
        255
    ).astype(np.uint8)

    normalized_rgb = cv2.cvtColor(
        normalized_lab,
        cv2.COLOR_LAB2RGB
    )

    # ======================================================
    # SAVE NORMALIZED IMAGE
    # ======================================================

    save_path = os.path.join(
        output_folder,
        image_name
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            normalized_rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    # ======================================================
    # COMPARISON IMAGE
    # ======================================================

    ref_vis = cv2.resize(
        reference_rgb,
        (image_rgb.shape[1],image_rgb.shape[0])
    )

    original_vis = add_title(image_rgb,"Original")

    reference_vis = add_title(ref_vis,"Reference")

    normalized_vis = add_title(normalized_rgb,"Normalized")

    comparison = np.hstack([
        original_vis,
        reference_vis,
        normalized_vis
    ])

    comparison_path = os.path.join(
        comparison_folder,
        image_name
    )

    cv2.imwrite(
        comparison_path,
        cv2.cvtColor(
            comparison,
            cv2.COLOR_RGB2BGR
        )
    )

print()

print("="*60)

print("Finished Reinhard normalization!")

print("="*60)

print(f"\nNormalized images saved to:\n{output_folder}")

print(f"\nComparison images saved to:\n{comparison_folder}")
