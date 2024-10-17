"""
Main train script

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from config import config
from utils.data_processing import create_gen_from_slices
from utils.callbacks import EpochResults, LossPlot
from utils.load_model import load_model
import glob
import time
import numpy as np
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Running defined models')
    parser.add_argument('-t', action='store_true', dest="trained",
                        help='Use pre-trained model', required=False, default=False)
    parser.add_argument('-c', action='store', dest='checkpoint',
                        help='Checkpoint folder', required=False, default=None)
    parser.add_argument('-e', action='store', dest='epoch_cnt',
                        help='Number epochs to train', required=False, default=20, type=int)

    return parser.parse_args()

def main():
    args = parse_args()

    config.EPOCH = args.epoch_cnt

    gan = load_model(args.checkpoint)

    train_data_gen = create_gen_from_slices(config.TRAIN_IMG_PATH_REAL, config.TRAIN_IMG_PATH_DMG)
    train_step_len = int(len(glob.glob(f"{config.TRAIN_IMG_PATH_REAL}/*.png")) / config.BATCH_SIZE)
    print(f"Train dataset ready. {train_step_len} x {config.BATCH_SIZE}")

    validation_data_gen = create_gen_from_slices(config.VALID_IMG_PATH_REAL, config.VALID_IMG_PATH_DMG)
    val_step_len = int(len(glob.glob(f"{config.VALID_IMG_PATH_REAL}/*.png")) / config.BATCH_SIZE)
    print(f"Validation dataset ready. {val_step_len} x {config.BATCH_SIZE}")
    
    history = gan.fit(
        train_data_gen,
        epochs=config.EPOCH,
        steps_per_epoch=train_step_len,
        validation_data=validation_data_gen,
        validation_steps=val_step_len,
        callbacks=[EpochResults(),
                   LossPlot()]
    )


if __name__ == "__main__":
    main()
