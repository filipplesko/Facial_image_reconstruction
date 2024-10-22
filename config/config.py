"""
Global configurations

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""

# SETTINGS
HOME_DIR = "."

EPOCH = 20
BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256
DIMENSIONS = 3
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT, DIMENSIONS)
DISC_EXTRA_STEP = 1  # 1 means no extra steps

# TRAINING images path
TRAIN_IMG_PATH_REAL = f"{HOME_DIR}/dataset/celeba/train/aligned"
TRAIN_IMG_PATH_DMG = f"{HOME_DIR}/dataset/celeba/train/damaged"
# VALIDATION images path
VALID_IMG_PATH_REAL = f"{HOME_DIR}/dataset/celeba/validate/aligned"
VALID_IMG_PATH_DMG = f"{HOME_DIR}/dataset/celeba/validate/damaged"

TEST_OUTPUT = f"{HOME_DIR}/test_output"
