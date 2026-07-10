'''
Fixed Data Loader for Binary Segmentation (0/1 masks + deprocess)
'''

from data_generator.image import ImageDataGenerator
import numpy as np
import itertools
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Preprocess
# -----------------------------
def preprocess(img, mean, std, label):
    # ---- IMAGE ----
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean).reshape(1,1,3)) / np.array(std).reshape(1,1,3)

    # ---- LABEL ----
    # Keras gives (B,H,W,1)
    if len(label.shape) == 4:
        label = label[:,:,:,0]

    # Force binary mask
    label = (label > 0.5).astype(np.int32)

    return img, label


# -----------------------------
# Deprocess (for visualization)
# -----------------------------
def deprocess(img, mean, std, label=None):
    # Reverse normalization
    img = (img * np.array(std).reshape(1,1,3)) + np.array(mean).reshape(1,1,3)
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)

    if label is not None:
        label = (label > 0).astype(np.uint8)
        return img, label

    return img


# -----------------------------
# Data Loader
# -----------------------------
def dataLoader(path, batch_size, imSize, train_mode=True,
               mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], return_filenames=False):

    def merge_generators(img_gen, mask_gen):
        for img, label in itertools.zip_longest(img_gen, mask_gen):
            img, label = preprocess(img, mean, std, label)
            yield img, label

    # -----------------------------
    # Augmentation
    # -----------------------------
    if train_mode:
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True
        )
    else:
        data_gen_args = dict()

    seed = 1

    # -----------------------------
    # Image generator
    # -----------------------------
    image_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        path + 'img',
        class_mode=None,
        target_size=imSize,
        batch_size=batch_size,
        seed=seed,
        shuffle=train_mode
    )

    filenames = image_generator.filenames

    # -----------------------------
    # Mask generator
    # -----------------------------
    mask_datagen = ImageDataGenerator(**data_gen_args)

    mask_generator = mask_datagen.flow_from_directory(
        path + 'gt',
        class_mode=None,
        target_size=imSize,
        batch_size=batch_size,
        color_mode='grayscale',
        seed=seed,
        shuffle=train_mode
    )

    # -----------------------------
    # Merge generators
    # -----------------------------
    generator = merge_generators(image_generator, mask_generator)
    samples = image_generator.samples

    if return_filenames:
        return generator, samples, filenames
    else:
        return generator, samples