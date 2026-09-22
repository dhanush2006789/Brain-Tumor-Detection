# 🧠 Brain Tumor Segmentation

A deep learning-based medical image segmentation project that identifies and segments tumor regions from brain MRI images using **U-Net** and **SegNet** deep learning architectures.

The system performs pixel-level segmentation to identify the tumor region in MRI scans and compares the performance of both segmentation models.

---

## 📌 Project Overview

Brain tumor segmentation is an important task in medical image analysis. Instead of only determining whether a tumor is present, segmentation identifies the exact region of the MRI image that corresponds to the tumor.

This project uses two deep learning segmentation models:

- **U-Net**
- **SegNet**

Both models are trained using brain MRI images and their corresponding tumor segmentation masks. The trained models generate predicted tumor masks, which are then evaluated and compared using segmentation metrics.

---

## ✨ Features

- 🧠 Brain MRI image processing
- 🎯 Pixel-level tumor segmentation
- 🤖 U-Net model
- 🤖 SegNet model
- 🖼️ Tumor segmentation mask generation
- 📊 U-Net and SegNet performance comparison
- 🔍 Segmentation result visualization
- 📈 Model evaluation using segmentation metrics

---

## 🧠 Models Used

### U-Net and SegNet

The project uses **U-Net and SegNet**, two encoder-decoder based deep learning architectures for image segmentation.

**U-Net** is designed especially for biomedical image segmentation. It uses an encoder to extract important features and a decoder to reconstruct the segmentation output. Skip connections between the encoder and decoder help preserve important spatial information from the MRI image.

**SegNet** is an encoder-decoder architecture designed for semantic segmentation. It uses the pooling information from the encoder during decoding to reconstruct the segmentation map and perform pixel-level classification.

Both models take an MRI image as input and generate a segmentation mask representing the predicted tumor region.

text
                    Brain MRI Image
                           ↓
                    Image Preprocessing
                           ↓
                     Feature Extraction
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
           U-Net                     SegNet
       Encoder-Decoder            Encoder-Decoder
              ↓                         ↓
       Tumor Prediction           Tumor Prediction
              ↓                         ↓
       Segmentation Mask          Segmentation Mask
              └────────────┬────────────┘
                           ↓
                  Model Evaluation
                           ↓
                  Model Comparison
                           ↓
                Tumor Visualization
