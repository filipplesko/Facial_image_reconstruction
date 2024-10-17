"""
Discriminator for binary classification

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from keras.layers import *
from keras.models import Model


def discriminator(input_dim):
    inputs = Input(input_dim)

    # x = GaussianNoise(stddev=0.2)(merged_inputs)

    x = Conv2D(filters=64, kernel_size=3, padding='same')(inputs)
    x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
    x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, out, name="Discriminator")
    return model
