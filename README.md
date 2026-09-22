# 🧠 Brain Tumor Segmentation

A deep learning-based medical image segmentation project that identifies and segments tumor regions from brain MRI images.

The project uses **U-Net** and **SegNet** deep learning architectures to perform pixel-level segmentation of brain tumor regions. The predicted segmentation masks are used to visualize the location and shape of the tumor in MRI scans.

---

## 📌 Project Overview

Brain tumor segmentation is an important task in medical image analysis.

Instead of only predicting whether a tumor is present, this project performs **pixel-level segmentation**, allowing the system to identify the specific region of the MRI image corresponding to the tumor.

Two deep learning segmentation models are implemented and compared:

- **U-Net**
- **SegNet**

The models learn from MRI images and their corresponding tumor masks to predict the tumor region in unseen MRI scans.

---

## ✨ Features

- 🧠 Brain MRI image processing
- 🎯 Pixel-level tumor segmentation
- 🤖 U-Net segmentation model
- 🤖 SegNet segmentation model
- 🖼️ Tumor mask generation
- 📊 Model performance comparison
- 🔍 Segmentation visualization
- 📈 Evaluation using segmentation metrics

---

## 🧠 Models Used

### 1. U-Net

U-Net is a convolutional neural network architecture widely used for biomedical image segmentation.

It consists of:

text
Input Image
     ↓
Encoder
     ↓
Feature Extraction
     ↓
Bottleneck
     ↓
Decoder
     ↓
Segmentation Mask

2. SegNet
SegNet is a deep learning architecture designed for semantic image segmentation.

It uses an encoder-decoder architecture to perform pixel-level classification and generate the final segmentation mask.

Input MRI Image
       ↓
    Encoder
       ↓
Feature Extraction
       ↓
    Decoder
       ↓
Pixel-Level Prediction
       ↓
Tumor Segmentation Mask

3 System Workflow
             Brain MRI Image
                    ↓
             Image Preprocessing
                    ↓
             Resize / Normalize
                    ↓
             ┌──────┴──────┐
             ↓             ↓
           U-Net         SegNet
             ↓             ↓
       Predicted Mask  Predicted Mask
             ↓             ↓
             └──────┬──────┘
                    ↓
          Segmentation Evaluation
                    ↓
             Model Comparison
                    ↓
          Final Tumor Visualization

