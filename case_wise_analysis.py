import os
import re
import csv
from collections import defaultdict

# ============================================================
# INPUT FOLDER
# ============================================================

data_folder = r"E:\DELINEATE\dil-Unet\datasets\test\val\gt\0"

# ============================================================
# FIND ALL IMAGE FILES
# ============================================================

image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

files = [
    f for f in os.listdir(data_folder)
    if os.path.splitext(f)[1].lower() in image_extensions
]

print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)

print(f"\nTotal image files found: {len(files)}")

# ============================================================
# EXPECTED NAME FORMAT
#
# caseid_anotherid_1original_patch_0_53
#
#                         ^    ^
#                  original   final patch
#
# ============================================================

pattern = re.compile(
    r"^(.+?)_(.+?)_1original_patch_(0|1)_(\d+)"
)

# ============================================================
# DATA STRUCTURES
# ============================================================

# case_id -> number of final patches
case_patch_count = defaultdict(int)

# (case_id, another_id, original_patch_number)
# -> set of final patch numbers
original_patch_contents = defaultdict(set)

# Keep track of parsing failures
unparsed_files = []

# ============================================================
# PARSE FILE NAMES
# ============================================================

for filename in files:

    name = os.path.splitext(filename)[0]

    match = pattern.match(name)

    if not match:
        unparsed_files.append(filename)
        continue

    case_id = match.group(1)
    another_id = match.group(2)
    original_patch = int(match.group(3))
    final_patch = int(match.group(4))

    # Count final patches belonging to this case
    case_patch_count[case_id] += 1

    # Store final patch number
    original_patch_contents[
        (case_id, another_id, original_patch)
    ].add(final_patch)

# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET SUMMARY")
print("=" * 70)

print(f"Total image files       : {len(files)}")
print(f"Successfully parsed     : {len(files) - len(unparsed_files)}")
print(f"Could not parse         : {len(unparsed_files)}")
print(f"Unique cases            : {len(case_patch_count)}")

# ============================================================
# PATCH COUNT PER CASE
# ============================================================

print("\n" + "=" * 70)
print("2. PATCHES PER CASE")
print("=" * 70)

for case_id, count in sorted(case_patch_count.items()):
    print(f"{case_id} : {count} patches")

# ============================================================
# ORIGINAL PATCH COMPLETENESS
# ============================================================

complete_original_patches = []
incomplete_original_patches = []

for key, patch_numbers in sorted(original_patch_contents.items()):

    case_id, another_id, original_patch = key

    missing = sorted(
        set(range(64)) - patch_numbers
    )

    record = {
        "case_id": case_id,
        "another_id": another_id,
        "original_patch": original_patch,
        "number_of_final_patches": len(patch_numbers),
        "missing_patches": missing
    }

    if len(patch_numbers) == 64:
        complete_original_patches.append(record)
    else:
        incomplete_original_patches.append(record)

# ============================================================
# ORIGINAL PATCH SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("3. ORIGINAL PATCH COMPLETENESS")
print("=" * 70)

print(
    f"Total original patches found : "
    f"{len(original_patch_contents)}"
)

print(
    f"Complete original patches    : "
    f"{len(complete_original_patches)}"
)

print(
    f"Incomplete original patches  : "
    f"{len(incomplete_original_patches)}"
)

# ============================================================
# COMPLETE ORIGINAL PATCHES
# ============================================================

print("\n" + "=" * 70)
print("4. COMPLETE ORIGINAL PATCHES (64/64)")
print("=" * 70)

for record in complete_original_patches:

    print(
        f"Case: {record['case_id']} | "
        f"Original patch: {record['original_patch']} | "
        f"64/64"
    )

# ============================================================
# INCOMPLETE ORIGINAL PATCHES
# ============================================================

print("\n" + "=" * 70)
print("5. INCOMPLETE ORIGINAL PATCHES")
print("=" * 70)

for record in incomplete_original_patches:

    print(
        f"Case: {record['case_id']} | "
        f"Original patch: {record['original_patch']} | "
        f"{record['number_of_final_patches']}/64 | "
        f"Missing: {record['missing_patches']}"
    )

# ============================================================
# CASE-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("6. CASE-LEVEL SUMMARY")
print("=" * 70)

case_original_patch_counts = defaultdict(int)
case_complete_counts = defaultdict(int)
case_incomplete_counts = defaultdict(int)

for key, patch_numbers in original_patch_contents.items():

    case_id = key[0]

    case_original_patch_counts[case_id] += 1

    if len(patch_numbers) == 64:
        case_complete_counts[case_id] += 1
    else:
        case_incomplete_counts[case_id] += 1

for case_id in sorted(case_original_patch_counts):

    print(
        f"{case_id} | "
        f"Final patches: {case_patch_count[case_id]} | "
        f"Original patches: {case_original_patch_counts[case_id]} | "
        f"Complete: {case_complete_counts[case_id]} | "
        f"Incomplete: {case_incomplete_counts[case_id]}"
    )

# ============================================================
# SAVE DETAILED CSV
# ============================================================

csv_path = os.path.join(
    data_folder,
    "patch_completeness_analysis.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "case_id",
        "another_id",
        "original_patch",
        "number_of_final_patches",
        "complete_64_of_64",
        "missing_patches"
    ])

    for key, patch_numbers in sorted(original_patch_contents.items()):

        case_id, another_id, original_patch = key

        missing = sorted(
            set(range(64)) - patch_numbers
        )

        writer.writerow([
            case_id,
            another_id,
            original_patch,
            len(patch_numbers),
            len(patch_numbers) == 64,
            ",".join(map(str, missing))
        ])

# ============================================================
# UNPARSED FILES
# ============================================================

if unparsed_files:

    print("\n" + "=" * 70)
    print("7. FILES THAT COULD NOT BE PARSED")
    print("=" * 70)

    for filename in unparsed_files:
        print(filename)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nDetailed CSV saved to:")
print(csv_path)
