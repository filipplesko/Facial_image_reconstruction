"""
Functions for loading model, or checkpoints

Author: Pleško Filip <xplesk02@stud.fit.vutbr.cz>
Date: May 14, 2023
"""

from config import config
from models.discriminator import discriminator
from models.gan import MyGan
from models.generator import autoencoder
from keras.optimizers import Adam

import tensorflow_addons.optimizers as optimizers
import keras.losses as losses
import glob


def init_model(gan):
    d_optimizer = Adam(learning_rate=0.00025, beta_1=0.5)
    d_loss = [losses.BinaryCrossentropy(label_smoothing=0.25)]
    discriminator_loss_w = 0.5
    d_loss_w = [discriminator_loss_w]

    # g_optimizer = Adam(learning_rate=0.00025, beta_1=0.5)
    g_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
    g_loss = [losses.BinaryCrossentropy(label_smoothing=0.25), losses.mse]
    adversarial_loss_w = 1
    application_loss_w = 100
    g_loss_w = [adversarial_loss_w, application_loss_w]

    gan.compile(
        d_optimizer=d_optimizer,
        d_loss=d_loss,
        d_loss_w=d_loss_w,
        g_optimizer=g_optimizer,
        g_loss=g_loss,
        g_loss_w=g_loss_w,
    )
    w, h, d = config.IMAGE_SIZE
    gan.build((None, w, h, d))
    return gan


def load_model(ckpt):
    g = autoencoder(config.IMAGE_SIZE)
    d = discriminator(config.IMAGE_SIZE)
    gan = MyGan(config.IMAGE_SIZE, config.DISC_EXTRA_STEP, g, d)
    gan = init_model(gan)

    if ckpt is not None:
        print(f"Loading checkpoint...")
        model_path = glob.glob(
            f"{config.HOME_DIR}/output/checkpoints/ep_{ckpt}*"
        )[0]
        gan.load_weights(model_path)
        print("Checkpoint loaded")
    return gan