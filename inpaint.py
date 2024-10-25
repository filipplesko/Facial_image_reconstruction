"""
Main inference script with optional mask and custom damage generation

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from config import config
from utils.load_model import load_model
from utils.image_processing import load_image, save_image, generate_noise_image, create_custom_damage, apply_mask_to_image, process_image
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm  # Import the tqdm progress bar
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Inference using trained model')
    parser.add_argument('-c', action='store', dest='checkpoint',
                        help='Checkpoint folder', required=True)
    parser.add_argument('-s', action='store', dest='source',
                        help='Source images (file or directory)', required=True)
    parser.add_argument('-m', action='store', dest='mask',
                        help='Mask images (file or directory)', required=False)
    parser.add_argument('-o', action='store', dest='output',
                        help='Output directory for results', required=True)

    return parser.parse_args()


def check_supported_extension(file_path):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file extension: {ext}. Supported formats are: {supported_extensions}")


def check_input_files(source, mask=None):
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp')

    def get_image_files(path):
        files = []
        for ext in supported_formats:
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)

    if os.path.isfile(source):
        check_supported_extension(source)
        if mask:
            check_supported_extension(mask)
        return [(source, mask)]  # Single file pair or source only
    
    elif os.path.isdir(source):
        source_files = get_image_files(source)
        if mask:
            mask_files = get_image_files(mask)
            if len(source_files) != len(mask_files):
                raise ValueError("Source and mask directories must contain the same number of files.")
            return list(zip(source_files, mask_files))  # Paired source and mask files
        else:
            return [(src, None) for src in source_files]  # Source files only

    else:
        raise ValueError("Source must be either a file or directory.")


def main():
    args = parse_args()

    # Load model from the checkpoint
    gan = load_model(args.checkpoint)
    
    # Validate and retrieve input image pairs (source, mask)
    image_pairs = check_input_files(args.source, args.mask)

    with tqdm(total=len(image_pairs), desc="Processing Images", unit="image") as pbar:
        for source_path, mask_path in image_pairs:
            source_img = load_image(source_path)

            # Align the source image and check yaw rotation
            aligned_image = process_image(source_img, os.path.basename(source_path), args.output)

            if aligned_image is None:
                print(f"Skipping image {source_path} due to yaw issues or failed alignment.")
                pbar.update(1)
                continue

            mask_img = None

            # If no mask is provided, create custom damage
            if not mask_path:
                mask_img = create_custom_damage(aligned_image)
                mask_file_path = os.path.join(args.output, "mask", os.path.basename(source_path))
                os.makedirs(mask_file_path, exist_ok=True)
                mask_img.save(mask_file_path)
            else:
                mask_img = Image.open(mask_path)

            damaged_image = apply_mask_to_image(aligned_image, mask_img)

            damaged_image_np = np.expand_dims(damaged_image, axis=0) / 255.0
            result = gan.generator.predict(damaged_image_np)

            result_img = np.squeeze(result) * 255.0
            result_img = result_img.astype(np.uint8)

            save_image(source_path, result_img, os.path.join(args.output, "inpainted"))

            # If no mask was provided, save the combined image
            if not mask_path:
                save_image(os.path.basename(source_path), damaged_image, os.path.join(args.output, "damaged"))

            pbar.update(1)


if __name__ == "__main__":
    main()
