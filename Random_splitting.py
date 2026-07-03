import os
import random
import shutil

# -----------------------------
# Paths
# -----------------------------
base_dir = r"E:\DELINEATE\dil-Unet\datasets\test_64\val"

# Source (all data in one place BEFORE split)
img_dir = os.path.join(base_dir, "img", "0")
gt_dir  = os.path.join(base_dir, "gt", "0")

# Desired split sizes
splits = {
    "train": 1000,
    "test": 500,
    "val": 284
}

# -----------------------------
# Pairing check function
# -----------------------------
def check_pairing(img_dir, gt_dir):
    img_set = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
    gt_set  = set(os.path.splitext(f)[0] for f in os.listdir(gt_dir))

    if img_set == gt_set:
        print(f"✅ Perfect match in {img_dir}")
    else:
        print(f"❌ Mismatch in {img_dir}")
        print("Missing GT:", list(img_set - gt_set)[:10])
        print("Extra GT:", list(gt_set - img_set)[:10])

# -----------------------------
# Initial sanity check (before split)
# -----------------------------
print("🔍 Checking original dataset...")
check_pairing(img_dir, gt_dir)

# -----------------------------
# Create output folders
# -----------------------------
for split in splits:
    os.makedirs(os.path.join(base_dir, split, "img", "0"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "gt", "0"), exist_ok=True)

# -----------------------------
# Get and shuffle files
# -----------------------------
all_files = [f for f in os.listdir(img_dir)]
random.seed(42)
random.shuffle(all_files)

# Safety check
total_required = sum(splits.values())
if len(all_files) < total_required:
    raise ValueError(f"Not enough images! Found {len(all_files)}, need {total_required}")

# -----------------------------
# Split assignment
# -----------------------------
train_files = all_files[:splits["train"]]
test_files  = all_files[splits["train"]:splits["train"] + splits["test"]]
val_files   = all_files[splits["train"] + splits["test"]:total_required]

split_map = {
    "train": train_files,
    "test": test_files,
    "val": val_files
}

# -----------------------------
# Move files (img + gt)
# -----------------------------
for split, files in split_map.items():
    print(f"\n📦 Processing {split} ({len(files)} files)...")

    for file in files:
        name, _ = os.path.splitext(file)

        img_src = os.path.join(img_dir, file)

        # find matching GT
        gt_src = None
        for gt_file in os.listdir(gt_dir):
            if os.path.splitext(gt_file)[0] == name:
                gt_src = os.path.join(gt_dir, gt_file)
                break

        if gt_src is None:
            print(f"⚠️ Missing GT for {file}")
            continue

        img_dst = os.path.join(base_dir, split, "img", "0", file)
        gt_dst  = os.path.join(base_dir, split, "gt", "0", os.path.basename(gt_src))

        shutil.move(img_src, img_dst)
        shutil.move(gt_src, gt_dst)

print("\n✅ Splitting complete!")

# -----------------------------
# Final verification
# -----------------------------
print("\n🔍 Verifying all splits...\n")

for split in ["train", "test", "val"]:
    check_pairing(
        os.path.join(base_dir, split, "img", "0"),
        os.path.join(base_dir, split, "gt", "0")
    )