"""
Vnet_2d architecture for 30m param version

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from keras_unet_collection import models


def autoencoder(input_dim):
    return models.vnet_2d(input_dim, filter_num=[64, 128, 256, 512, 512], n_labels=3,
                          res_num_ini=2, res_num_max=2,
                          activation='ReLU', output_activation="Sigmoid",
                          batch_norm=True, pool=False, unpool=False, name='vnet')
