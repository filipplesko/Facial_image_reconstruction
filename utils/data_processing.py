"""
Data preprocess functions

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
import os
import tensorflow as tf
import numpy as np
import random
import cv2
import keras.backend as backend
from config import config
from PIL import Image, ImageDraw
from utils.polygon_gen import generate_polygon


def read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3, dtype=tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def preprocessing_slice(d, r):
    return read_img(d), read_img(r)


def create_gen_from_slices(path_r, path_d):
    train_images = sorted(os.listdir(f"{path_d}"))
    train_images_paths_d = [f"{path_d}/" + x for x in train_images]
    train_images_paths_r = [f"{path_r}/" + x for x in train_images]

    dataset = tf.data.Dataset.from_tensor_slices((train_images_paths_d, train_images_paths_r))
    dataset = dataset.map(preprocessing_slice, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(4096, reshuffle_each_iteration=False, seed=42)
    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
