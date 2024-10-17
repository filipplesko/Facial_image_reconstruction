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

        path = os.path.join(config.HOME_DIR, 'output', 'results', 'train', f"{epoch:02d}")
        Path(path).mkdir(parents=True, exist_ok=True)
        out = self.model.generator(np.array(imgs))
        for i, generated_image in enumerate(out):
            image = np.uint8(generated_image * 255)
            cv2.imwrite(
                os.path.join(path, f'{i:02d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class LossPlot(Callback):
    def __init__(self):
        super().__init__()
        self.d_final_loss = []
        self.advers_loss = []
        self.app_loss = []
        if os.path.exists(f"{config.HOME_DIR}/output/results/train"):
            self.prev_epoch_cnt = len(
                os.listdir(
                    f"{config.HOME_DIR}/output/results/train")) - 1
        else:
            self.prev_epoch_cnt = 0

    def on_epoch_end(self, epoch, logs=None):
        self.d_final_loss.append(logs.get("d_final_loss"))
        self.advers_loss.append(logs.get("advers_loss"))
        self.app_loss.append(logs.get("app_loss"))
        epoch = epoch + self.prev_epoch_cnt
        path = os.path.join(config.HOME_DIR, 'output', 'results', 'train', f"{epoch:02d}")
        Path(path).mkdir(parents=True, exist_ok=True)

        # summarize history for accuracy
        plt.plot(self.d_final_loss)
        plt.title('Discriminator loss')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['Discriminant'], loc='upper right')
        plt.savefig(os.path.join(path, f"Discriminator_loss_{epoch:02d}.png"), bbox_inches='tight')
        plt.clf()

        plt.plot(self.advers_loss)
        plt.title('Adversarial loss')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['Adversarial'], loc='upper right')
        plt.savefig(os.path.join(path, f"Adversarial_loss_{epoch:02d}.png"), bbox_inches='tight')
        plt.clf()

        plt.plot(self.app_loss)
        plt.title('Application loss')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['Application'], loc='upper right')
        plt.savefig(os.path.join(path, f"Application_loss_{epoch:02d}.png"), bbox_inches='tight')
        plt.clf()

        plt.plot(self.app_loss)
        plt.plot(self.advers_loss)
        plt.plot(self.d_final_loss)
        plt.title('Losses')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['Application', 'Adversarial', 'Discriminator'], loc='upper right')
        plt.savefig(os.path.join(path, f"Losses_{epoch:02d}.png"), bbox_inches='tight')
        plt.clf()
