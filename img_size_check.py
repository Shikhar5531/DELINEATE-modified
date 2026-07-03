import os
from PIL import Image

img_dir = r"E:\DELINEATE\dil-Unet\datasets\val\img\0"
gt_dir = r"E:\DELINEATE\dil-Unet\datasets\val\gt\0"

expected_size = (512, 512)

def check_images(directory, label):
    print(f"\nChecking {label} directory: {directory}")
    issues = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        try:
            with Image.open(file_path) as img:
                if img.size != expected_size:
                    print(f"[SIZE ERROR] {filename}: {img.size}")
                    issues += 1
        except Exception as e:
            print(f"[READ ERROR] {filename}: {e}")
            issues += 1

    if issues == 0:
        print("All images are 512x512 ✅")
    else:
        print(f"Total issues found: {issues}")

def check_matching(img_dir, gt_dir):
    img_files = set(os.listdir(img_dir))
    gt_files = set(os.listdir(gt_dir))
    
    missing_in_gt = img_files - gt_files
    missing_in_img = gt_files - img_files

    if missing_in_gt:
        print("\n[WARNING] Files present in IMG but missing in GT:")
        for f in missing_in_gt:
            print(f"  {f}")

    if missing_in_img:
        print("\n[WARNING] Files present in GT but missing in IMG:")
        for f in missing_in_img:
            print(f"  {f}")

    if not missing_in_gt and not missing_in_img:
        print("\nFilenames match perfectly between IMG and GT ✅")

# Run checks
check_images(img_dir, "IMG")
check_images(gt_dir, "GT")
check_matching(img_dir, gt_dir)