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
wget https://nextcloud.fit.vutbr.cz/s/JfbSBqzzzWZbqsJ/download/celeba.tar.gz -P ./dataset/celeba/
cd ./dataset/celeba/
tar --use-compress-program=pigz -xvf celeba.tar.gz
rm -rf celeba.tar.gz
cd ../../
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Training the Model

To train the GAN for facial reconstruction, run:

```bash
python train.py --dataset /path/to/celeba-c --epochs 100
```

## Experiments

### Performance Metrics

The following metrics are used to evaluate the reconstruction quality:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

### Reconstruction Results

A comparison of different architectures is included in the paper. Our model demonstrated superior performance in terms of PSNR and SSIM compared to other state-of-the-art solutions.

### Recognition Results

The face recognition accuracy is evaluated using **ArcFace**, **SFace**, and **QMagFace** algorithms on both the original and reconstructed CelebA-C datasets.

## Installation

Ensure you have Python 3.x installed. Then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

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