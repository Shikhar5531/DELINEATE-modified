import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# PARAMETERS
# ==========================================================

BACKGROUND_THRESHOLD = 220

# ==========================================================
# PATHS
# ==========================================================

reference_image = r"E:\UPF\Actual_work\32f98933-e724-4160-a42f-d152c9f94741_8db995e8-d15f-4c2a-be10-5c964025a402_1original.jpg"

input_folder = r"E:\UPF\Actual_work\Histology_images"

output_folder = r"E:\UPF\Actual_work\Histology_images_reinhard"

comparison_folder = r"E:\UPF\Actual_work\Reinhard_Comparison"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)

# ==========================================================
# LOAD REFERENCE
# ==========================================================

print("Loading reference image...")

reference = cv2.imread(reference_image)

reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

reference_lab = cv2.cvtColor(
    reference_rgb,
    cv2.COLOR_RGB2LAB
).astype(np.float32)

# ==========================================================
# TISSUE MASK
# ==========================================================

mask_ref = reference_lab[:,:,0] < BACKGROUND_THRESHOLD

print("Reference tissue pixels :", np.sum(mask_ref))

# ==========================================================
# REFERENCE STATISTICS
# ==========================================================

ref_mean = np.zeros(3)

ref_std = np.zeros(3)

for c in range(3):

    pixels = reference_lab[:,:,c][mask_ref]

    ref_mean[c] = np.mean(pixels)

    ref_std[c] = np.std(pixels)

print("\nReference Statistics")

print("Mean :", ref_mean)

print("Std  :", ref_std)

# ==========================================================
# IMAGE LIST
# ==========================================================

image_list = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        )
    )
])

print(f"\nFound {len(image_list)} images.\n")

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
# NORMALIZATION
# ==========================================================

for image_name in tqdm(image_list):

    image_path = os.path.join(
        input_folder,
        image_name
    )

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # ------------------------------------------------------
    # Tissue mask
    # ------------------------------------------------------

    mask_src = image_lab[:,:,0] < BACKGROUND_THRESHOLD

    normalized_lab = image_lab.copy()

    # ------------------------------------------------------
    # Channel-wise Reinhard
    # ------------------------------------------------------

    for c in range(3):

        src_channel = image_lab[:,:,c]

        pixels = src_channel[mask_src]

        src_mean = np.mean(pixels)

        src_std = np.std(pixels)

        if src_std < 1e-8:
            src_std = 1.0

        normalized_lab[:,:,c] = (
            (src_channel - src_mean)
            / src_std
            * ref_std[c]
            + ref_mean[c]
        )

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
        (image_rgb.shape[1], image_rgb.shape[0])
    )

    mask_vis = np.zeros_like(image_rgb)

    mask_vis[mask_src] = (255,255,255)

    original_vis = add_title(
        image_rgb,
        "Original"
    )

    reference_vis = add_title(
        ref_vis,
        "Reference"
    )

    normalized_vis = add_title(
        normalized_rgb,
        "Normalized"
    )

    mask_vis = add_title(
        mask_vis,
        "Mask"
    )

    top = np.hstack([
        original_vis,
        reference_vis
    ])

    bottom = np.hstack([
        normalized_vis,
        mask_vis
    ])

    comparison = np.vstack([
        top,
        bottom
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

print("\n"+"="*60)

print("Finished Reinhard Normalization!")

print("="*60)

print(f"\nNormalized images : {output_folder}")

print(f"Comparison images : {comparison_folder}")
