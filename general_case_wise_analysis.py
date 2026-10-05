import os
import cv2
import pandas as pd
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# Change ONLY this path for each experiment
PRED_DIR = r"E:\DELINEATE\dil-Unet\checkpoints\model-4920"

# Ground-truth / hospital steatosis values
EXCEL_PATH = r"E:\UPF\Actual_work\Steatosis_values_of_cases.csv"

# Where to save the case-wise results
OUTPUT_CSV = r"E:\DELINEATE\dil-Unet\checkpoints\model-4920\experiment_2_casewise_steatosis.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def extract_case_id(filename):
    """
    Extract case ID from prediction filename.

    Expected filename structure:
    caseid_anotherid_1original_patch_X_Y.png
    """

    return filename.split("_")[0]


def calculate_patch_steatosis(pred_path):
    """
    Calculate predicted steatosis percentage for one patch.
    """

    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

    if pred is None:
        raise ValueError(f"Could not read: {pred_path}")

    # --------------------------------------------------------
    # Handle prediction format
    # --------------------------------------------------------

    # If prediction is RGB/BGR, use red channel.
    if len(pred.shape) == 3:

        # OpenCV loads images as BGR.
        # Steatosis is encoded in the red channel.
        pred = pred[:, :, 2]

    # Convert prediction to binary:
    # 1 = steatosis
    # 0 = background
    pred_binary = (pred > 127).astype(np.uint8)

    # --------------------------------------------------------
    # Calculate percentage
    # --------------------------------------------------------

    total_pixels = pred_binary.size
    fat_pixels = np.sum(pred_binary)

    steatosis_percentage = (fat_pixels / total_pixels) * 100

    return steatosis_percentage


# ============================================================
# READ HOSPITAL STEATOSIS VALUES
# ============================================================

df_gt = pd.read_csv(EXCEL_PATH)

# Make sure the required columns exist
required_columns = {"case_id", "steatosis"}

if not required_columns.issubset(df_gt.columns):
    raise ValueError(
        f"Excel/CSV must contain columns: {required_columns}. "
        f"Found: {list(df_gt.columns)}"
    )

# Make case IDs strings
df_gt["case_id"] = df_gt["case_id"].astype(str)


# Create lookup dictionary
gt_dict = dict(
    zip(
        df_gt["case_id"],
        df_gt["steatosis"]
    )
)


# ============================================================
# PROCESS PREDICTIONS
# ============================================================

case_predictions = {}

prediction_files = [
    f for f in os.listdir(PRED_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
]

print(f"Prediction files found: {len(prediction_files)}")


for filename in prediction_files:

    case_id = extract_case_id(filename)

    pred_path = os.path.join(PRED_DIR, filename)

    try:
        patch_steatosis = calculate_patch_steatosis(pred_path)

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
        continue

    # Store prediction for this case
    if case_id not in case_predictions:
        case_predictions[case_id] = []

    case_predictions[case_id].append(patch_steatosis)


# ============================================================
# CASE-WISE AGGREGATION
# ============================================================

results = []

for case_id, patch_values in case_predictions.items():

    patch_values = np.array(patch_values)

    # Mean percentage of fat across all available patches
    predicted_steatosis = np.mean(patch_values)

    number_of_patches = len(patch_values)

    # --------------------------------------------------------
    # Ground-truth hospital value
    # --------------------------------------------------------

    if case_id in gt_dict:
        original_steatosis = gt_dict[case_id]

        absolute_error = abs(
            predicted_steatosis - original_steatosis
        )

    else:
        original_steatosis = np.nan
        absolute_error = np.nan

        print(
            f"WARNING: No hospital value found for case {case_id}"
        )

    results.append({
        "case_id": case_id,
        "number_of_patches": number_of_patches,
        "predicted_steatosis_percent": predicted_steatosis,
        "original_steatosis_percent": original_steatosis,
        "absolute_error": absolute_error
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Sort by case ID
results_df = results_df.sort_values("case_id")


# ============================================================
# SUMMARY METRICS
# ============================================================

valid = results_df.dropna(
    subset=[
        "predicted_steatosis_percent",
        "original_steatosis_percent"
    ]
)

if len(valid) > 0:

    mae = np.mean(
        np.abs(
            valid["predicted_steatosis_percent"]
            - valid["original_steatosis_percent"]
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                valid["predicted_steatosis_percent"]
                - valid["original_steatosis_percent"]
            ) ** 2
        )
    )

    print("\n==============================")
    print("CASE-WISE STEATOSIS RESULTS")
    print("==============================")

    print(
        results_df.to_string(index=False)
    )

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print(f"Number of cases: {len(valid)}")
    print(f"MAE:  {mae:.4f} percentage points")
    print(f"RMSE: {rmse:.4f} percentage points")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_CSV)