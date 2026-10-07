# DELINEATE-modified

This repository contains the modified **dil-Unet** scripts used for steatosis segmentation of H&E-stained liver histopathology images.

The code is based on the original [DELINEATE](https://github.com/StonyBrookDB/DELINEATE) dil-Unet implementation, with modifications for binary segmentation, the current dataset organization, Dice-loss experiments, post-processing, case-wise steatosis estimation, and Reinhard stain normalization.

## 1. Repository overview

### Core dil-Unet files

| Script | Purpose |
|---|---|
| `model.py` | Defines the dil-Unet architecture used for pixel-wise binary segmentation. The network contains the U-Net encoder/decoder and a dilated-convolution bottleneck. |
| `loader.py` | Loads images and ground-truth masks from the dataset folders, applies normalization (`mean=0.5`, `std=0.5`), converts masks to binary 0/1 labels, and applies horizontal/vertical flips during training. |
| `opts.py` | Defines command-line training/evaluation parameters such as batch size, learning rate, number of epochs, image size, dataset path, and checkpoint path. |
| `utils.py` | Contains visualization and segmentation metric utilities, including mean IoU and Dice coefficient functions. |
| `train.py` | Main training script using sparse softmax cross-entropy loss. It trains dil-Unet, runs validation periodically, saves checkpoints, and writes TensorBoard logs. |
| `train_dice_loss.py` | Alternative training script using foreground (steatosis) Dice loss instead of cross-entropy. It is intended for experiments where improving the fat/foreground segmentation is the main objective. |
| `eval.py` | Loads a trained checkpoint and predicts segmentation masks for a test/validation folder. It also calculates mean IoU and saves the predicted masks. |

### Analysis / post-processing scripts

| Script | Purpose |
|---|---|
| `metrics.py` | Calculates binary segmentation metrics from an existing prediction folder and its corresponding ground-truth masks: background Dice, fat Dice, overall Dice, precision, and recall. |
| `post_processing.py` | Applies one selected morphological post-processing operation to predicted masks. The current active operation is `binary_fill_holes`; the `remove_small_objects` and `remove_small_holes` options are present but commented out. |
| `all_postprocessing_combinations.py` | Tests all ordered combinations of three post-processing operations: Remove Small Objects (RSO), Remove Small Holes (RSH), and Binary Fill Holes (BFH). It evaluates every pipeline against the ground truth and writes the results to a CSV file. |
| `general_case_wise_analysis.py` | Converts patch-level predictions into case-level steatosis percentages. It extracts the case ID from each filename, calculates the fat percentage of each patch, averages patches belonging to the same case, and compares the prediction with hospital steatosis values from a CSV file. It reports MAE and RMSE. |
| `random_splitting.py` | Performs a random patch-level train/test/validation split. This was used for earlier exploratory experiments. It should **not** be used for the final patient/case-wise split because patches from the same case can be distributed across different splits. |

### Reinhard normalization scripts

| Script | Purpose |
|---|---|
| `reinhard_norm.py` | Performs standard channel-wise Reinhard stain normalization in LAB colour space using a reference histopathology image. It also creates side-by-side comparison images. |
| `masked_reinhard_norm.py` | Performs tissue-masked Reinhard normalization. Background pixels are excluded when calculating the source/reference LAB statistics using a luminance threshold. It also saves comparison images. |

---

# 2. Dataset structure

The data loader expects the image and ground-truth directories to have the same filenames.

For `train.py`, the current script expects:

```text
datasets/
├── train_64/
│   ├── img/
│   │   └── 0/
│   │       ├── image_001.png
│   │       └── ...
│   └── gt/
│       └── 0/
│           ├── image_001.png
│           └── ...
│
└── val_64/
    ├── img/
    │   └── 0/
    └── gt/
        └── 0/
```

For `train_dice_loss.py`, the current script expects:

```text
datasets/
├── train_combined/
│   ├── img/0/
│   └── gt/0/
│
└── val_combined/
    ├── img/0/
    └── gt/0/
```

For `eval.py`, the supplied `--data_path` must contain a `val/` directory:

```text
datasets/
└── test/
    └── val/
        ├── img/0/
        └── gt/0/
```

The image and corresponding mask must have matching filenames.

---

# 3. Environment

The modified scripts use the older TensorFlow 1.x-style API (`tf.Session`, `tf.ConfigProto`, `tensorflow.contrib.keras`, etc.). Therefore, the environment should be compatible with the TensorFlow version used by the original DELINEATE implementation.

The original DELINEATE repository provides its dependency file and describes the dil-Unet training/evaluation workflow here:

https://github.com/StonyBrookDB/DELINEATE

Additional packages used by the modified analysis scripts include:

```text
numpy
opencv-python
pandas
tqdm
scikit-image
scipy
Pillow
```

---

# 4. Training with cross-entropy loss

The standard training script is:

```text
train.py
```

It uses:

- sparse softmax cross-entropy loss
- Adam optimizer
- exponential learning-rate decay
- horizontal and vertical flip augmentation
- periodic validation
- checkpoint saving
- TensorBoard logging

### Example

From the repository directory:

```bash
python train.py ^
  --data_path E:/DELINEATE/dil-Unet/datasets ^
  --checkpoint_path E:/DELINEATE/dil-Unet/checkpoints/ ^
  --imSize 512 ^
  --batch_size 5 ^
  --epoch 50
```

For Windows PowerShell, the same command can be written on one line:

```powershell
python train.py --data_path E:/DELINEATE/dil-Unet/datasets --checkpoint_path E:/DELINEATE/dil-Unet/checkpoints/ --imSize 512 --batch_size 5 --epoch 50
```

The script automatically uses:

```text
data_path/train_64/
data_path/val_64/
```

for training and validation.

### Important

The checkpoint path should preferably end with `/` (or `\`) because the script constructs the checkpoint prefix using:

```python
opt.checkpoint_path + 'model'
```

Checkpoints are saved periodically during training.

---

# 5. Training with Dice loss

The alternative training script is:

```text
train_dice_loss.py
```

This version replaces the cross-entropy loss with a **foreground Dice loss**, where the foreground class corresponds to steatosis/fat.

It uses:

```text
data_path/train_combined/
data_path/val_combined/
```

### Example

```powershell
python train_dice_loss.py --data_path E:/DELINEATE/dil-Unet/datasets --checkpoint_path E:/DELINEATE/dil-Unet/checkpoints_dice/ --imSize 512 --batch_size 5 --epoch 50
```

This script is useful when comparing cross-entropy training with Dice-loss training, particularly because the main target of interest is the steatosis/fat class.

---

# 6. Evaluating a trained model

The evaluation script is:

```text
eval.py
```

It:

1. loads the test/validation images;
2. restores a trained TensorFlow checkpoint;
3. generates a pixel-wise segmentation;
4. converts the prediction into binary background/fat classes;
5. calculates mean IoU;
6. saves the predicted segmentation masks.

### Example

If the test data are located at:

```text
E:/DELINEATE/dil-Unet/datasets/test/val/
```

and the checkpoint is:

```text
E:/DELINEATE/dil-Unet/checkpoints/model-4920
```

run:

```powershell
python eval.py --data_path E:/DELINEATE/dil-Unet/datasets/test/ --load_from_checkpoint E:/DELINEATE/dil-Unet/checkpoints/model-4920 --batch_size 1 --imSize 512
```

### Output

The prediction masks are written under the checkpoint path supplied to `--load_from_checkpoint`.

The original DELINEATE repository follows the same general pattern: train a dil-Unet checkpoint first and then use `eval.py` to evaluate the trained checkpoint. See the original repository for the baseline workflow.

---

# 7. Calculating detailed segmentation metrics

After predictions have been generated, use:

```text
metrics.py
```

This script compares prediction masks with the corresponding ground-truth masks.

It reports:

- Overall Dice
- Background Dice
- Fat Dice
- Background precision
- Background recall
- Fat precision
- Fat recall
- Number of missing predictions

### Important

The input and ground-truth directories are currently specified directly inside `metrics.py`:

```python
pred_dir = r"..."
gt_dir   = r"..."
```

Change these two paths before running.

Then run:

```powershell
python metrics.py
```

---

# 8. Post-processing individual predictions

The script:

```text
post_processing.py
```

applies morphological post-processing to already-generated segmentation masks.

Three operations were considered in the experiments:

### Remove Small Objects (RSO)

Removes small disconnected foreground components.

```python
morphology.remove_small_objects(...)
```

Current experimental parameters:

```text
min_size = 2550
connectivity = 2
```

### Remove Small Holes (RSH)

Fills enclosed background regions smaller than a specified area.

Current experimental parameter:

```text
area_threshold = 51100
```

### Binary Fill Holes (BFH)

Fills enclosed holes in the binary foreground mask.

The current version of `post_processing.py` has BFH enabled:

```python
processed = ndimage.binary_fill_holes(img)
```

The other two operations are currently commented out.

Change the input/output paths at the top of the script before running it.

---

# 9. Testing all post-processing combinations

For the systematic post-processing experiment, use:

```text
all_postprocessing_combinations.py
```

This script evaluates:

- no post-processing
- RSO
- RSH
- BFH
- all ordered pairs
- all ordered triples

For three operations, this gives:

```text
1 baseline
+ 3 single operations
+ 6 two-operation pipelines
+ 6 three-operation pipelines
= 16 pipelines
```

For every pipeline, the script calculates:

- fat Dice
- background Dice
- overall Dice
- fat precision
- fat recall
- background precision
- background recall

The results are sorted by fat Dice and saved as a CSV file.

### Run

Update these paths first:

```python
INPUT_FOLDER = r"..."
GT_FOLDER = r"..."
OUTPUT_CSV = r"..."
```

Then:

```powershell
python all_postprocessing_combinations.py
```

---

# 10. Case-wise steatosis estimation

Use:

```text
general_case_wise_analysis.py
```

This script is used after segmentation to estimate the final steatosis percentage for each case.

For each prediction patch:

```text
steatosis % =
(number of predicted fat pixels / total pixels) × 100
```

Patches are grouped using the case ID contained in the filename.

For each case:

```text
predicted case steatosis =
mean steatosis percentage across its patches
```

The predicted value is then compared with the hospital value from:

```text
Steatosis_values_of_cases.csv
```

The CSV is expected to contain:

```text
case_id
steatosis
```

The script reports:

- case ID
- number of patches
- predicted steatosis %
- original/hospital steatosis %
- absolute error
- MAE
- RMSE

### Run

Update:

```python
PRED_DIR = r"..."
EXCEL_PATH = r"..."
OUTPUT_CSV = r"..."
```

Then:

```powershell
python general_case_wise_analysis.py
```

---

# 11. Reinhard stain normalization

Two scripts are included for stain-normalization experiments.

## `reinhard_norm.py`

This performs standard Reinhard normalization:

1. reads a reference image;
2. converts images from RGB to LAB;
3. calculates the mean and standard deviation of each LAB channel;
4. matches each input image to the reference statistics;
5. converts the normalized image back to RGB;
6. saves normalized images;
7. saves comparison images.

Update:

```python
reference_image
input_folder
output_folder
comparison_folder
```

Then run:

```powershell
python reinhard_norm.py
```

## `masked_reinhard_norm.py`

This is a tissue-masked version of Reinhard normalization.

It excludes background pixels when calculating the LAB statistics using:

```python
BACKGROUND_THRESHOLD = 220
```

This allows the normalization statistics to be based primarily on tissue rather than blank/background regions.

Update the paths and run:

```powershell
python masked_reinhard_norm.py
```

---

# 12. Random patch-level splitting

```text
random_splitting.py
```

This script randomly divides a combined dataset into:

```text
train = 1511
test  = 750
val   = 430
```

using random seed 42.

It also checks that image/ground-truth pairs match.

### Important

This script was used for earlier exploratory experiments.

It performs a **patch-level random split**, so patches originating from the same histopathology case can appear in different splits. Therefore, it should **not** be used for the final case-wise evaluation where patient/case leakage must be avoided.

---

# 13. Recommended workflow

For the current case-wise experiments, the main workflow is:

```text
Dataset
   │
   ▼
train.py
or
train_dice_loss.py
   │
   ▼
Saved checkpoint
   │
   ▼
eval.py
   │
   ▼
Predicted segmentation masks
   │
   ├───────────────► metrics.py
   │                  │
   │                  ▼
   │              Pixel-level metrics
   │
   ├───────────────► post_processing.py
   │                  │
   │                  ▼
   │              Processed masks
   │
   ├───────────────► all_postprocessing_combinations.py
   │                  │
   │                  ▼
   │              Best post-processing pipeline
   │
   └───────────────► general_case_wise_analysis.py
                      │
                      ▼
                 Case-wise steatosis %
                      │
                      ▼
                Hospital value comparison
```

The Reinhard scripts are separate preprocessing experiments and are not required for the basic train → evaluate → post-process → case-wise analysis pipeline.

---

# 14. Important path configuration note

Most of the analysis scripts were written for the experiments on the local Windows machine and therefore currently contain absolute paths such as:

```text
E:\DELINEATE\...
E:\UPF\...
```

Before sharing the repository with another user, these paths should be changed to the paths on their machine.

The core training/evaluation scripts (`train.py`, `train_dice_loss.py`, and `eval.py`) accept paths through command-line arguments, so these are more portable.

