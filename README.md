# 🤟 Sign Language Detection with Faster R-CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Real-time American Sign Language (ASL) alphabet detection using a custom-trained Faster R-CNN model with a FastAPI web interface.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Training](#-model-training) • [API Reference](#-api-reference) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

This project implements a complete pipeline for detecting and recognizing American Sign Language (ASL) alphabet gestures (A-Z) using deep learning. The system uses a **Faster R-CNN** object detection model with a **ResNet-50 FPN** backbone, fine-tuned for 27 classes (26 letters + background).

### Key Components

| Component | Description |
|-----------|-------------|
| `model_pipeline.ipynb` | Jupyter notebook with complete ML pipeline (dataset, training, evaluation, visualization) |
| `app.py` | FastAPI application with real-time webcam inference and word building |
| `requirements.txt` | Project dependencies |

---

## ✨ Features

### 🎯 Model Training Pipeline
- **Custom Dataset Class**: Supports both PASCAL VOC XML and YOLO TXT annotation formats
- **Data Augmentation**: Random horizontal flip, color jitter, photometric distortion using `torchvision.transforms.v2`
- **Transfer Learning**: Pre-trained Faster R-CNN with modified classification head
- **Training Loop**: AdamW optimizer, StepLR scheduler, gradient clipping, model checkpointing
- **Evaluation**: Mean Average Precision (mAP) using `torchmetrics`
- **Feature Map Visualization**: Forward hooks for inspecting backbone, FPN, and RPN features

### 🌐 Web Application
- **Real-time Inference**: MJPEG video stream with live bounding box visualization
- **Word Builder**: Automatically accumulates letters when held for consecutive frames
- **Modern UI**: Responsive dark-themed interface with live statistics
- **REST API**: Endpoints for inference, word management, and status updates

### 📊 Visualization Tools
- Backbone feature maps (layer1-layer4)
- Feature Pyramid Network (FPN) multi-scale outputs
- Region Proposal Network (RPN) activations
- Bounding box predictions with confidence scores

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Webcam (for real-time inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmmarRiaz123/SignLanguage-FasterRCNN-FastAPI.git
cd SignLanguage-FasterRCNN-FastAPI
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### Step 3: Install PyTorch

Install PyTorch with the appropriate CUDA version for your system:

```bash
# With CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# With CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
SignLanguage-FasterRCNN-FastAPI/
│
├── 📓 model_pipeline.ipynb    # Complete ML training pipeline
├── 🌐 app.py                  # FastAPI web application
├── 📋 requirements.txt        # Project dependencies
├── 📖 README.md               # This file
├── 🚫 .gitignore              # Git ignore rules
│
├── 📂 data/                   # Dataset directory (create this)
│   ├── images/                # Training images
│   └── annotations/           # XML/TXT annotations
│
├── 📂 checkpoints/            # Model checkpoints (created during training)
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
│
├── 📂 test_images/            # Test images for inference
│
└── 🤖 model_deployment.pth    # Final trained model for deployment
```

---

## 📚 Dataset Preparation

### Expected Format

The dataset should be organized as follows:

```
data/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── annotations/
    ├── img_001.xml
    ├── img_002.xml
    └── ...
```

### PASCAL VOC XML Format

Each annotation file should follow this structure:

```xml
<annotation>
    <filename>img_001.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>A</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>50</ymin>
            <xmax>200</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
```

### YOLO TXT Format (Alternative)

```
# class_id center_x center_y width height (normalized 0-1)
0 0.5 0.5 0.2 0.3
```

### Recommended Datasets

- [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Custom dataset with bounding box annotations

---

## 🎓 Model Training

### 1. Open the Jupyter Notebook

```bash
jupyter notebook model_pipeline.ipynb
```

### 2. Configure Dataset Paths

Update the configuration cell:

```python
IMAGES_DIR = "./data/images"
ANNOTATIONS_DIR = "./data/annotations"
BATCH_SIZE = 4
NUM_EPOCHS = 25
```

### 3. Run the Training Pipeline

Execute cells sequentially:

| Section | Description |
|---------|-------------|
| 1 | Import libraries and set device |
| 2 | Define custom dataset class |
| 3 | Create augmentation pipeline |
| 4 | Set up data loaders |
| 5 | Create and configure Faster R-CNN model |
| 6 | Train the model |
| 7 | Evaluate with mAP metrics |
| 8 | Run inference visualization |
| 9-11 | Visualize feature maps |
| 12 | Save model for deployment |

### 4. Training Output

```
Epoch 1/25 (LR: 0.001000)
  Epoch 1, Batch 10/50, Loss: 1.2345, Avg Loss: 1.3456
  ...
Epoch 1 - Train Loss: 1.1234, Val Loss: 0.9876
  ✓ Saved best model (val_loss: 0.9876)
```

### 5. Evaluation Metrics

```
==================================================
Mean Average Precision (mAP) Results
==================================================
  mAP (IoU=0.50:0.95): 0.7523
  mAP_50 (IoU=0.50):   0.9234
  mAP_75 (IoU=0.75):   0.8012
==================================================
```

---

## 🌐 Web Application Usage

### Starting the Server

```bash
# Method 1: Direct Python execution
python app.py

# Method 2: Using uvicorn with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Accessing the Interface

Open your browser and navigate to:

```
http://localhost:8000
```

### Web Interface Features

| Feature | Description |
|---------|-------------|
| **Live Video Feed** | Real-time webcam stream with bounding box overlays |
| **Current Detection** | Shows the letter being detected with confidence progress |
| **Word Builder** | Accumulates letters when held for 15+ consecutive frames |
| **Controls** | Add space, backspace, clear word |
| **Statistics** | Detection count, confidence, FPS |

### Word Builder Logic

1. Show a sign to the camera
2. Hold it steady with confidence > 70%
3. After 15 consecutive frames, the letter is added
4. Use "Add Space" between words
5. Use "Clear All" to start over

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main HTML interface |
| `GET` | `/video_feed` | MJPEG video stream |
| `GET` | `/status` | Current detection status (JSON) |
| `POST` | `/clear_word` | Clear accumulated word |
| `POST` | `/add_space` | Add space to word |
| `POST` | `/backspace` | Remove last character |
| `GET` | `/predict?image_path=<path>` | Single image inference |

### Status Response Example

```json
{
    "word": "HELLO",
    "current_letter": "O",
    "consecutive_frames": 12,
    "frames_required": 15,
    "num_detections": 1,
    "top_confidence": 0.89
}
```

### Single Image Prediction

```bash
curl "http://localhost:8000/predict?image_path=./test_images/sign_a.jpg&threshold=0.5"
```

Response:
```json
{
    "image_path": "./test_images/sign_a.jpg",
    "num_detections": 1,
    "detections": [
        {
            "box": [120.5, 80.2, 340.8, 380.1],
            "label": 1,
            "class_name": "A",
            "score": 0.95
        }
    ]
}
```

---

## 🏗 Architecture

### Faster R-CNN Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Faster R-CNN Architecture                 │
├─────────────────────────────────────────────────────────────┤
│  Input Image (H × W × 3)                                    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Backbone (ResNet-50)                                │    │
│  │  - Conv1 → Layer1 → Layer2 → Layer3 → Layer4        │    │
│  │  - Extracts hierarchical feature maps               │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Feature Pyramid Network (FPN)                       │    │
│  │  - P2 (1/4), P3 (1/8), P4 (1/16), P5 (1/32)        │    │
│  │  - Multi-scale feature representation               │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Region Proposal Network (RPN)                       │    │
│  │  - Generates ~300 object proposals per image        │    │
│  │  - Objectness scores + bounding box deltas          │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ROI Heads                                           │    │
│  │  - ROI Pooling (7×7 features per proposal)          │    │
│  │  - Box Predictor (27 classes + bbox regression)     │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  Output: boxes, labels, scores                               │
└─────────────────────────────────────────────────────────────┘
```

### Class Mapping

| Index | Class | Index | Class | Index | Class |
|-------|-------|-------|-------|-------|-------|
| 0 | Background | 10 | J | 19 | S |
| 1 | A | 11 | K | 20 | T |
| 2 | B | 12 | L | 21 | U |
| 3 | C | 13 | M | 22 | V |
| 4 | D | 14 | N | 23 | W |
| 5 | E | 15 | O | 24 | X |
| 6 | F | 16 | P | 25 | Y |
| 7 | G | 17 | Q | 26 | Z |
| 8 | H | 18 | R | | |
| 9 | I | | | | |

---

## 🔧 Configuration

### Model Configuration (`app.py`)

```python
MODEL_PATH = "./model_deployment.pth"  # Trained model path
CONFIDENCE_THRESHOLD = 0.5             # Detection threshold
WORD_BUILDER_THRESHOLD = 0.7           # Word building threshold
WORD_BUILDER_FRAMES = 15               # Frames to confirm letter
CAMERA_INDEX = 0                       # Webcam index
FRAME_WIDTH = 640                      # Video width
FRAME_HEIGHT = 480                     # Video height
```

### Training Configuration (`model_pipeline.ipynb`)

```python
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 25
LR_STEP_SIZE = 7
LR_GAMMA = 0.1
BATCH_SIZE = 4
TRAIN_VAL_SPLIT = 0.8
```

---

## 📈 Feature Map Visualization

The notebook includes tools to visualize intermediate representations:

### Backbone Features

```python
# Extract and visualize ResNet backbone features
backbone_features = visualize_backbone_features(
    model=model,
    image_path="./test_images/sample.jpg",
    device=DEVICE
)
```

### FPN & RPN Features

```python
# Visualize multi-scale FPN outputs and RPN activations
visualize_rpn_features(
    model=model,
    image_path="./test_images/sample.jpg",  
    device=DEVICE
)
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` or use MobileNetV3 backbone |
| Camera not detected | Check `CAMERA_INDEX` or webcam permissions |
| Model file not found | Ensure `model_deployment.pth` exists after training |
| Low FPS | Use GPU, reduce `FRAME_WIDTH`/`FRAME_HEIGHT` |
| Poor detection accuracy | Train longer, use more data, adjust thresholds |

### Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed
2. **Batch inference**: For video, process multiple frames together
3. **Model optimization**: Use TorchScript or ONNX for production
4. **Reduce resolution**: Lower video resolution for faster inference

---

## 📊 Expected Performance

| Metric | Target Value |
|--------|--------------|
| mAP@0.5 | > 0.85 |
| mAP@0.5:0.95 | > 0.65 |
| Inference FPS (GPU) | > 20 |
| Inference FPS (CPU) | > 3 |

*Actual performance depends on dataset quality and hardware.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [TorchVision](https://pytorch.org/vision/) for Faster R-CNN implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision utilities

---

## 📧 Contact

**Ammar Riaz** - [@AmmarRiaz123](https://github.com/AmmarRiaz123)

Project Link: [https://github.com/AmmarRiaz123/SignLanguage-FasterRCNN-FastAPI](https://github.com/AmmarRiaz123/SignLanguage-FasterRCNN-FastAPI)

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

</div>
