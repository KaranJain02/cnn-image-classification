# Image Classification Using Convolutional Neural Networks (CNN)

## Overview
This project implements an image classification system using Convolutional
Neural Networks (CNNs) to classify images into multiple categories.

## Problem
Image datasets often require careful preprocessing and model tuning to
achieve good accuracy and generalization.

## Solution
- Preprocessed and resized image data
- Applied data augmentation techniques
- Designed and trained a CNN model
- Tuned hyperparameters to improve performance
- Evaluated the model using validation metrics

## Tools & Technologies
- Python
- PyTorch / TensorFlow
- NumPy
- Matplotlib

## Results
Achieved strong classification performance on the validation dataset,
demonstrating effective model training and generalization.

## Pretrained Model
This repository includes a pretrained CNN model saved as `bird.pth`.
The model can be directly used for inference without retraining.

## Running Inference

```bash
python bird.py sample_data test bird.pth

