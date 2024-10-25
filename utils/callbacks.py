"""
Custom callbacks during training

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from keras.callbacks import Callback
from config import config
from pathlib import Path
from utils.data_processing import read_img
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class EpochResults(Callback):
    def __init__(self):
        super().__init__()
        if os.path.exists(f"{config.HOME_DIR}/output/results/train"):
            self.prev_epoch_cnt = len(
                os.listdir(
                    f"{config.HOME_DIR}/output/results/train")) - 1
        else:
            self.prev_epoch_cnt = 0
        self.images = sorted(os.listdir(f"{config.VALID_IMG_PATH_DMG}"))[:10]

    def on_epoch_end(self, epoch, logs=None):
        images_paths = [os.path.join(config.VALID_IMG_PATH_DMG, x) for x in self.images]
        epoch = epoch + self.prev_epoch_cnt

        ckpt_path = f"{config.HOME_DIR}/output/checkpoints"
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(
            f"{ckpt_path}/ep_{epoch:02d}_g_loss_{logs['val_g_loss']:.5f}_d_loss_{logs['val_d_loss']:.5f}.h5")

        imgs = []
        for f in images_paths:
            img = read_img(f)
            imgs.append(img)

        path = os.path.join(config.HOME_DIR, 'output', 'examples', f"{epoch:02d}")
        Path(path).mkdir(parents=True, exist_ok=True)
        out = self.model.generator(np.array(imgs))
        for i, generated_image in enumerate(out):
            image = np.uint8(generated_image * 255)
            cv2.imwrite(
                os.path.join(path, f'{i:02d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
