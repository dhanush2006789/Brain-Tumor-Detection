<div align="center">

# 🧠 Brain Tumor Segmentation

### AI-Powered Medical Imaging System for Automated Brain Tumor Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

[Features](#-features) • [Architecture](#-architecture) • [Getting Started](#-getting-started) • [API Documentation](#-api-documentation) • [Model Comparison](#-model-comparison)

</div>

---

## 📖 Overview

**Brain Tumor Segmentation** is a web-based medical imaging system that leverages advanced deep learning models to automatically detect and segment brain tumors from MRI scans. The system implements two state-of-the-art segmentation architectures—**UNet** and **SegNet**—to provide accurate tumor localization with comparative analysis.

### 🎯 What Makes This Project Special?

- 🤖 **Dual Deep Learning Models** - Compare UNet and SegNet predictions side-by-side
- 📤 **Web-Based Interface** - Easy-to-use Flask application for MRI uploads
- 📊 **Accuracy Tracking** - Real-time performance metrics and visualization
- 🎨 **Interactive Results** - Visual segmentation overlays and comparative analysis
- 🚀 **GPU Optimized** - CUDA support for fast inference and training
- 📈 **Model Performance Dashboard** - Monitor accuracy trends across predictions
- 🔧 **Modular Architecture** - Clean separation of model loading, training, and prediction

---

## ✨ Features

### 🧬 Advanced Segmentation Models
- **UNet Architecture** - ResNet34 encoder with skip connections for precise tumor localization
- **SegNet Architecture** - Custom encoder-decoder with max pooling indices for efficient segmentation
- **Comparative Analysis** - Side-by-side model predictions on the same input

### 📤 Image Upload & Processing
- Secure file upload handling with Flask
- Automatic image normalization and preprocessing
- Support for grayscale MRI images (DICOM-compatible formats)
- Real-time processing with progress tracking

### 📊 Accuracy & Performance Metrics
- **Automatic Accuracy Calculation** - Compare model outputs against ground truth
- **Performance Visualization** - Dynamic plotting of accuracy trends
- **JSON-Based Tracking** - Persistent storage of historical accuracy data
- **Live Dashboard** - View model performance over time

### 🎯 Tumor Segmentation
- Binary segmentation (tumor vs. non-tumor)
- High-resolution output images
- Confidence scoring and probability maps
- Batch processing capabilities

### 📁 File Management
- Organized folder structure (uploads, outputs)
- Automatic output generation
- Original image preservation
- Easy result retrieval

---

## 🏗️ Architecture

### Tech Stack

#### Backend
- **Framework**: Flask 3.0+
- **Deep Learning**: PyTorch 2.0+
- **Image Processing**: OpenCV 4.8+, NumPy
- **Model Library**: segmentation-models-pytorch
- **Data Augmentation**: Albumentations
- **Visualization**: Matplotlib

#### Frontend
- **Interface**: HTML5/CSS3/JavaScript
- **Display**: Interactive image viewer
- **Charting**: matplotlib for accuracy plots
- **Client**: Static assets with responsive design

#### Infrastructure
- **GPU Support**: CUDA-enabled PyTorch
- **CPU Fallback**: Automatic device detection
- **Memory Management**: Optimized batch processing
- **File System**: Local storage with secure filename handling

### 📂 Project Structure

```
brain-tumor-segmentation/
│
├── 📊 Core Models
│   ├── UNet.py                         # UNet model definition (ResNet34 encoder)
│   ├── SegNet.py                       # Custom SegNet implementation
│   ├── LoadUNet.py                     # UNet model loader
│   └── LoadSegNet.py                   # SegNet model loader
│
├── 🚀 Backend Application
│   ├── app.py                          # Flask web application
│   ├── train.py                        # Model training script
│   ├── testunet.py                     # UNet inference module
│   ├── testsegnet.py                   # SegNet inference module
│   └── accuracy.py                     # Accuracy calculation module
│
├── 🎨 Frontend
│   ├── templates/
│   │   └── index.html                  # Web interface
│   └── static/
│       ├── css/                        # Styling
│       └── js/                         # Client-side logic
│
├── 💾 Data Management
│   ├── accuracy_data.json              # Historical accuracy metrics
│   ├── uploads/                        # User-uploaded MRI images
│   └── outputs/                        # Generated segmentation results
│
├── 📦 Configuration
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore
│
└── 📚 Documentation
    └── README.md                       # This file

```

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                         │
│         Upload MRI → View Results → Compare Models             │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP POST/GET
┌────────────────▼────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │  /predict    │  /accuracy   │  /uploads    │  /outputs    │ │
│  │ (inference)  │  (metrics)   │  (serve)     │  (serve)     │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└─────┬────────┬────────────────────────────┬────────────────────┘
      │        │                            │
      │        │                            └──────► File System
      │        │                                     (uploads/outputs)
      │        └──────────────────┬──────────────────► accuracy_data.json
      │                           │
      └────────────────┬──────────┴───────────────────► Deep Learning Models
                       │
       ┌───────────────┴───────────────┐
       │                               │
    UNet                            SegNet
    (ResNet34)                    (Custom Arch)
    PyTorch                        PyTorch
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **PyTorch 2.0+** (with CUDA 11.8+ optional for GPU acceleration)
- **pip** (Python package manager)
- **Git**

### Installation

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Killer-94/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

#### 2️⃣ Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU Support (Optional):**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

#### 4️⃣ Project Structure Setup

Create the necessary folders:

```bash
mkdir -p uploads outputs
```

### Running the Application

#### Option A: Development Mode

```bash
# Start the Flask development server
python app.py

# The application will be available at:
# http://127.0.0.1:5000
```

#### Option B: Production Mode

```bash
# Install Gunicorn (production WSGI server)
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Training Models

To train models on your own dataset:

#### 1️⃣ Prepare Dataset

Organize your MRI images and masks in the following structure:

```
dataset/
├── images/
│   ├── brain_001.png
│   ├── brain_002.png
│   └── ...
└── masks/
    ├── brain_001.png
    ├── brain_002.png
    └── ...
```

#### 2️⃣ Run Training Script

```bash
python train.py
```

**Training Configuration** (edit `train.py`):
- **Batch Size**: 4 (line 48)
- **Epochs**: 30 (line 63)
- **Learning Rate**: 1e-4 (line 60)
- **Image Size**: 128x128 (line 41)

**Output**: Models are saved as:
- `unet_brain_segmentation.pth` - Trained UNet weights

---

## 📚 API Documentation

### Endpoints

#### Health Check
**GET `/`**
```
Returns: HTML interface for the web application
```

#### Image Prediction
**POST `/predict`**

Request:
```
Content-Type: multipart/form-data
- image: <MRI image file>
```

Response (Success - 200):
```json
{
  "original": "/uploads/brain_scan.png",
  "unet_output": "/outputs/unet_brain_scan.png",
  "segnet_output": "/outputs/segnet_brain_scan.png",
  "unet_acc": 0.87,
  "segnet_acc": 0.89
}
```

Response (Error - 400/500):
```json
{
  "error": "No image provided"
}
```

#### Accuracy Data
**GET `/accuracy_data`**

Response:
```json
{
  "unet": [0.85, 0.87, 0.86, 0.88],
  "segnet": [0.86, 0.89, 0.88, 0.90]
}
```

#### Accuracy Plot
**GET `/accuracy_plot.png`**

Returns: PNG image with accuracy trend visualization

#### File Serving
**GET `/uploads/<filename>`** - Serve uploaded images
**GET `/outputs/<filename>`** - Serve segmentation results

---

## 🔬 Model Comparison

### UNet
- **Architecture**: Encoder-Decoder with skip connections
- **Encoder**: ResNet34 pre-trained on ImageNet
- **Strengths**:
  - High accuracy with moderate computational cost
  - Skip connections preserve fine-grained details
  - Pre-trained weights accelerate convergence
- **Input**: Grayscale images (1 channel)
- **Output**: Binary segmentation mask

### SegNet
- **Architecture**: Custom encoder-decoder with max pooling indices
- **Characteristics**:
  - 4 encoding blocks with max pooling
  - 4 decoding blocks with max unpooling
  - Batch normalization at each layer
- **Strengths**:
  - Efficient memory usage (recovers pooling indices)
  - Lightweight and fast inference
  - Good for real-time applications
- **Input**: Grayscale images (1 channel)
- **Output**: Binary segmentation mask

### Performance Metrics

| Metric | UNet | SegNet |
|--------|------|--------|
| Accuracy | High | High |
| Inference Speed | Moderate | Fast |
| Memory Usage | Higher | Lower |
| Training Time | Moderate | Faster |
| Fine-Grain Details | Excellent | Good |

---

## 📊 Core Modules

### `UNet.py` - UNet Model Definition
```python
# Architecture: ResNet34 encoder with 1 input channel
# Pretrained weights from ImageNet
# Suitable for medical imaging tasks
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)
```

### `SegNet.py` - SegNet Implementation
- Custom encoder with 4 blocks (64, 128, 256, 512 filters)
- Max pooling with indices for unpooling
- Symmetric decoder structure
- Batch normalization for stability

### `train.py` - Training Pipeline
- **Dataset**: Custom `BrainTumorDataset` class
- **Augmentation**: Albumentations (resize, normalization)
- **Loss Function**: Dice Loss (medical imaging standard)
- **Optimizer**: Adam with learning rate 1e-4
- **Device**: Automatic CUDA/CPU detection

### `app.py` - Flask Web Application
- **POST /predict**: Handle image upload and inference
- **GET /accuracy_data**: Retrieve historical metrics
- **GET /accuracy_plot.png**: Generate accuracy visualization
- File management (uploads, outputs folders)
- Error handling and logging

### `accuracy.py` - Metrics Calculation
- Compares predicted segmentation with ground truth
- Calculates accuracy as intersection over union (IoU) or similar metric
- Returns percentage accuracy for each model

### `testunet.py` & `testsegnet.py` - Inference Modules
- Load pre-trained model weights
- Preprocess input image (normalization, resizing)
- Run inference on image
- Post-process output (thresholding, resizing to original size)
- Save segmentation result

---

## 🔧 Configuration & Customization

### Folder Configuration (app.py)
```python
UPLOAD_FOLDER = 'uploads'      # Where uploaded images are stored
OUTPUT_FOLDER = 'outputs'      # Where segmentation results are saved
STATIC_FOLDER = 'static'       # Static files location
```

### Model Training Parameters (train.py)

```python
# Dataset
image_dir = "dataset/images"
mask_dir = "dataset/masks"

# Hyperparameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 30
image_size = 128

# Loss Function
loss_fn = smp.losses.DiceLoss(mode='binary')

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Image Preprocessing (train.py)
```python
transform = A.Compose([
    A.Resize(128, 128),                    # Resize to 128x128
    A.Normalize(mean=0.0, std=1.0),       # Normalize
    ToTensorV2()                           # Convert to tensor
])
```

---

## 🧪 Testing & Validation

### Test Predictions

#### Test UNet Model
```bash
python testunet.py
```

#### Test SegNet Model
```bash
python testsegnet.py
```

### Manual Testing with cURL

```bash
# Upload and predict
curl -X POST -F "image=@path/to/brain_scan.png" http://localhost:5000/predict

# Get accuracy data
curl http://localhost:5000/accuracy_data

# View accuracy plot
curl http://localhost:5000/accuracy_plot.png > plot.png
```

---

## 📈 Performance Optimization

### GPU Acceleration
The system automatically detects and uses GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Batch Processing
For multiple images, batch inference improves throughput:
```python
batch_loader = DataLoader(dataset, batch_size=8)
```

### Model Quantization (Optional)
Reduce model size for deployment:
```python
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
```bash
Solution: pip install -r requirements.txt
```

**Issue**: `CUDA out of memory`
```bash
Solution: Reduce batch_size in train.py (line 48)
```

**Issue**: Images not uploading
```bash
Solution: Ensure 'uploads' and 'outputs' folders exist
mkdir -p uploads outputs
```

**Issue**: Model not loading
```bash
Solution: Ensure .pth model files are in the correct directory
Check LoadUNet.py and LoadSegNet.py paths
```

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ UNet and SegNet models
- ✅ Web-based image upload
- ✅ Real-time segmentation
- ✅ Accuracy tracking
- ✅ Comparative analysis

### Upcoming Features (v1.1)
- 🔄 **3D Volume Segmentation** - Support for volumetric MRI data
- 📊 **Advanced Metrics** - DSC, Hausdorff distance, sensitivity/specificity
- 🎯 **ROI Analysis** - Tumor location and size extraction
- 📁 **Batch Processing** - Process multiple scans in parallel
- 🔐 **User Authentication** - Secure access to predictions

### Future Vision (v2.0)
- 🧠 **Multi-Modal Support** - CT, PET, fMRI integration
- 🤖 **Ensemble Models** - Combine predictions from multiple architectures
- 📱 **Mobile App** - iOS/Android application
- ☁️ **Cloud Deployment** - AWS/GCP/Azure integration
- 🔬 **Research Dashboard** - Statistical analysis and reporting
- 🌐 **DICOM Support** - Full DICOM image handling
- 🏥 **Hospital Integration** - PACS system connectivity

---

## 📝 Dataset Format

### Image Requirements
- **Format**: PNG, JPEG, or TIFF
- **Color Space**: Grayscale (1 channel)
- **Resolution**: Minimum 128x128 pixels
- **File Size**: < 50 MB per image

### Mask Requirements (for training)
- **Format**: PNG, JPEG, or TIFF (binary mask)
- **Values**: 0 (background) and 255 (tumor)
- **Resolution**: Must match corresponding image
- **Filename**: Must match image filename

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

- **Author**: vamshi-005
- **GitHub**: [@vamshi-005](https://github.com/vamshi-005)
- **Repository**: [brain-tumor-segmentation](https://github.com/Killer-94/brain-tumor-segmentation)

---

<div align="center">

**Built with ❤️ for Medical Imaging Research**

[⭐ Star this repo](https://github.com/Killer-94/brain-tumor-segmentation) | [🐛 Report Bug](https://github.com/Killer-94/brain-tumor-segmentation/issues) | [✨ Request Feature](https://github.com/Killer-94/brain-tumor-segmentation/issues)

</div>
