# [Facial Image Reconstruction and its Influence to Face Recognition](https://ieeexplore.ieee.org/document/10346000)

[Read the Full Paper on IEEE Xplore</a>](https://ieeexplore.ieee.org/document/10346000)

This repository contains the implementation of **Facial Image Reconstruction using Generative Adversarial Networks (GANs)** to improve the performance of face recognition systems. The code is based on the research presented in *"Facial Image Reconstruction and its Influence on Face Recognition"* by **Filip Pleško, Tomáš Goldmann, and Kamil Malinka**.

## Abstract

This paper focuses on reconstructing damaged facial images using GAN neural networks. In addition, the effect of generating the missing part of the face on face recognition is investigated. The main objective of this work is to observe whether it is possible to increase the accuracy of face recognition by generating missing parts while maintaining a low false accept rate (FAR). A new model for generating the missing parts of a face has been proposed. For face-based recognition, state-of-the-art solutions from the DeepFace library and the QMagFace solution have been used.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experiments](#experiments)
  - [Performance Metrics](#performance-metrics)
  - [Reconstruction Results](#reconstruction-results)
  - [Recognition Results](#recognition-results)
- [Installation](#installation)
- [Citations](#citations)

## Introduction

Facial image reconstruction is a challenging task, especially when dealing with unique facial features like the eyes, nose, or mouth. This project explores the reconstruction of damaged facial images and evaluates whether these reconstructions can enhance the performance of face recognition algorithms such as **ArcFace**, **SFace**, and **QMagFace**.

## Dataset

The **CelebA-C** dataset was created by modifying the original CelebA dataset to simulate corrupted face images. Each image is occluded with random Gaussian noise. This modified dataset is used for training the GAN model.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/filipplesko/Facial_image_reconstruction.git
cd Facial_image_reconstruction
```

### 2. Download the Dataset

```bash
wget https://nextcloud.fit.vutbr.cz/s/6GDFBo4ky24ZpeS/download/celeba.tar.gz -P ./dataset/celeba/
cd ./dataset/celeba/
tar --use-compress-program=pigz -xvf celeba.tar.gz
rm -rf celeba.tar.gz
cd ../../
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Training Process

The training process in this repository involves training a Generative Adversarial Network (GAN) to restore damaged facial images. The GAN is trained on a dataset of real (undamaged) images and corresponding damaged versions. The model learns to reconstruct the missing or damaged parts of the face. The training can be done from scratch or continued from a saved checkpoint.

#### To train the model from scratch, use:

```bash
python train.py -e 20
```

#### To resume training from a checkpoint, use:

```bash
python train.py -c /path/to/checkpoint -e 20
```

#### Options:
- `-c`: Path to the checkpoint folder (optional). If omitted, training starts from scratch.
- `-e`: Number of epochs to train (default is 20).

During training, the model processes batches of real and damaged images, and validation is performed on a separate dataset. Epoch results and loss plots are generated as callbacks.

### Inpainting Process

The inpainting process in this repository utilizes a GAN model to restore damaged facial images. The script takes as input a source image (with or without a mask), aligns the face, and checks for proper yaw rotation. If no mask is provided, a custom damage mask is generated. The damaged image is then passed through the GAN model, which reconstructs the missing parts of the face, enhancing its quality and structure.

#### To perform inpainting on a set of images, run the following command:

```bash
python inpaint.py -c /path/to/checkpoint -s /path/to/source_images -m /path/to/mask_images -o /path/to/output_dir
```

#### Options:
- `-c`: Path to the checkpoint folder containing the trained model.
- `-s`: Path to the source images (single file or directory).
- `-m`: Path to the mask images (optional; single file or directory). If not provided, a mask is generated automatically.
- `-o`: Output directory where the results will be saved.



### Reconstruction Results

A comparison of different architectures is included in the paper. Our model demonstrated superior performance in terms of PSNR and SSIM compared to other state-of-the-art solutions.

### Recognition Results

The face recognition accuracy is evaluated using **ArcFace**, **SFace**, and **QMagFace** algorithms on both the original and reconstructed CelebA-C datasets.

### This repository specifics

This repository, in compare to paper, uses only frontal images up to 20 degrees yaw rotation for training.

## Citations

If you use this code or dataset in your research, please cite the following paper:

```
@INPROCEEDINGS{10346000,
  author={Pleško, Filip and Goldmann, Tomáš and Malinka, Kamil},
  booktitle={2023 International Conference of the Biometrics Special Interest Group (BIOSIG)}, 
  title={Facial Image Reconstruction and its Influence to Face Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Image recognition;Face recognition;Biological system modeling;Neural networks;Generative adversarial networks;Libraries;Image reconstruction;Face reconstruction;Face recognition;GAN;ArcFace;SFace;QMagFace},
  doi={10.1109/BIOSIG58226.2023.10346000}}
```
