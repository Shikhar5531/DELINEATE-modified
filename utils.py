'''
Fixed Utils for Binary Segmentation (0/1 classes only)
'''

import numpy as np
from PIL import Image
import os, copy, shutil, json


# -----------------------------
# VISUALIZATION + METRICS
# -----------------------------
class VIS:
    def __init__(self, save_path):

        self.path = save_path

        # safer folder handling
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)

        self.mean_iu = []
        self.cls_iu = []
        self.score_history = {}
        self.suffix = str(np.random.randint(1000))

        self.palette = Image.open('palette_refer.png').palette

    # -----------------------------
    # Save segmentation image
    # -----------------------------
    def save_seg(self, label_im, name, im=None, gt=None):

        seg = Image.fromarray(label_im.astype(np.uint8), mode='P')
        seg.palette = copy.copy(self.palette)

        if gt is not None or im is not None:
            gt_img = Image.fromarray(gt.astype(np.uint8), mode='P')
            gt_img.palette = copy.copy(self.palette)

            im_img = Image.fromarray(im.astype(np.uint8), mode='RGB')

            I = Image.new('RGB', (label_im.shape[1]*3, label_im.shape[0]))
            I.paste(im_img, (0, 0))
            I.paste(gt_img, (label_im.shape[1], 0))
            I.paste(seg, (label_im.shape[1]*2, 0))

            I.save(os.path.join(self.path, name))
        else:
            seg.save(os.path.join(self.path, name))

    # -----------------------------
    # Add sample for evaluation
    # -----------------------------
    def add_sample(self, pred, gt):
        # FORCE BINARY
        pred = (pred > 0).astype(np.int32)
        gt   = (gt > 0).astype(np.int32)

        score_mean, score_cls = mean_IU(pred, gt)

        self.mean_iu.append(score_mean)
        self.cls_iu.append(score_cls)

        return score_mean

    # -----------------------------
    # Compute final scores
    # -----------------------------
    def compute_scores(self, suffix=0):
        meanIU = np.mean(self.mean_iu)
        meanIU_per_cls = np.mean(self.cls_iu, axis=0)

        print ('-'*20)
        print ('overall mean IU: {}'.format(meanIU))
        print ('mean IU per class')

        for i, c in enumerate(meanIU_per_cls):
            print ('\t class {}: {}'.format(i,c))

        print ('-'*20)

        data = {
            'mean_IU': '%.4f' % meanIU,
            'mean_IU_cls': ['%.4f' % a for a in meanIU_per_cls]
        }

        self.score_history['%.10d' % suffix] = data

        with open(os.path.join(self.path, 'meanIU{}.json'.format(self.suffix)), 'w') as f:
            json.dump(self.score_history, f, indent=2, sort_keys=True)


# -----------------------------
# FIXED IoU (ONLY 2 CLASSES)
# -----------------------------
def mean_IU(pred, gt):
    pred = pred.flatten()
    gt   = gt.flatten()

    ious = []

    for cls in [0, 1]:
        pred_inds = (pred == cls)
        gt_inds   = (gt == cls)

        intersection = np.sum(pred_inds & gt_inds)
        union = np.sum(pred_inds | gt_inds)

        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)

    return np.mean(ious), ious


# -----------------------------
# Dice (optional, unchanged)
# -----------------------------
def dice_coef(y_true, y_pred):
    import tensorflow.keras.backend as K
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)