'''
 * Fixed training script for correct binary evaluation
'''

from __future__ import absolute_import, division, print_function

SEED = 0
import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

import os

from model import UNet
from loader import dataLoader
from utils import VIS, mean_IU
from opts import *
from opts import dataset_mean, dataset_std

# -----------------------------
# Session
# -----------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

vis = VIS(save_path=opt.checkpoint_path)

# -----------------------------
# Data
# -----------------------------
img_shape = [opt.imSize, opt.imSize]

train_generator, train_samples = dataLoader(
    opt.data_path + '/train_combined/',
    opt.batch_size,
    img_shape,
    mean=dataset_mean,
    std=dataset_std
)

val_generator, val_samples = dataLoader(
    opt.data_path + '/val_combined/',
    1,
    img_shape,
    train_mode=False,
    mean=dataset_mean,
    std=dataset_std
)

opt.iter_epoch = int(train_samples)

# -----------------------------
# Model
# -----------------------------
label = tf.placeholder(tf.int32, shape=[None] + img_shape)

with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape + [3], num_class=opt.num_class)
    img = model.input
    pred = model.output

# -----------------------------
# Loss
# -----------------------------
#loss = tf.reduce_mean(
#    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred)
#)

# -----------------------------
# DICE Loss
# -----------------------------

def dice_loss(labels, logits, smooth=1e-6):

    # Convert logis --> probabilities 
    probs = tf.nn.softmax(logits, axis=-1)

    # Convert labels --> one-hot encoded form
    labels = tf.one_hot(labels, depth=2)

    labels = tf.cast(labels, tf.float32)

    # Using only foreground (fat) channel 
    y_true = labels[:, :, :, 1]
    y_pred = probs[:, :, :, 1]

    # Computing DICE for each image in given batch size
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2] + tf.reduce_sum(y_pred, axis=[1, 2]))
    dice = (2 * intersection + smooth) / (union + smooth)

    # Averaging over batch
    loss = 1.0 - tf.reduce_mean(dice)

    return loss

loss = dice_loss(label, pred)

# -----------------------------
# Optimizer
# -----------------------------
global_step = tf.Variable(0, name='global_step', trainable=False)

learning_rate = tf.train.exponential_decay(
    opt.learning_rate,
    global_step,
    opt.iter_epoch,
    opt.lr_decay,
    staircase=True
)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(
    loss,
    global_step=global_step
)

# -----------------------------
# Tensorboard
# -----------------------------
tf.summary.scalar('loss', loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(opt.checkpoint_path, sess.graph)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# -----------------------------
# Training loop
# -----------------------------
with sess.as_default():

    if opt.load_from_checkpoint != '':
        try:
            saver.restore(sess, opt.load_from_checkpoint)
            print('--> Loaded from checkpoint:', opt.load_from_checkpoint)
        except Exception as e:
            print('❌ Failed to load checkpoint:', e)

    start = global_step.eval()
    total_iter = opt.iter_epoch * opt.epoch

    for it in range(start, total_iter):

        # -------------------------
        # Validation
        # -------------------------
        if it % 820 == 0 or it == start:

            saver.save(sess, opt.checkpoint_path + 'model', global_step=global_step)

            print(f"\n💾 Saved checkpoint at iteration {it}")
            print(f"🔍 Running validation on {val_samples} samples...")

            for _ in range(val_samples):
                x_batch, y_batch = next(val_generator)

                feed_dict = {img: x_batch, label: y_batch}

                pred_logits = sess.run(pred, feed_dict=feed_dict)

                # CRITICAL FIX
                pred_map = np.argmax(pred_logits, axis=3)
                pred_map = (pred_map > 0).astype(np.int32)
                y_batch = (y_batch > 0).astype(np.int32)

                for p, y in zip(pred_map, y_batch):
                    vis.add_sample(p, y)

            vis.compute_scores(suffix=it)

        # -------------------------
        # Train step
        # -------------------------
        x_batch, y_batch = next(train_generator)

        feed_dict = {
            img: x_batch,
            label: y_batch
        }

        _, l, summary, lr, pred_logits = sess.run(
            [train_step, loss, summary_merged, learning_rate, pred],
            feed_dict=feed_dict
        )

        train_writer.add_summary(summary, it)

        # 🔥 FIX HERE TOO
        pred_map = np.argmax(pred_logits[0], axis=2)
        pred_map = (pred_map > 0).astype(np.int32)
        y_batch[0] = (y_batch[0] > 0).astype(np.int32)

        score, _ = mean_IU(pred_map, y_batch[0])

        if it % 20 == 0:
            print(f"[iter {it}, epoch {it/opt.iter_epoch:.3f}] "
                  f"lr={lr:.6f}, loss={l:.6f}, mean_IU={score:.6f}")