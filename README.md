# HCVT: Hierarchical Cascaded Vision Transformers for Medical Image Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-informational)](https://doi.org/)

## Overview

**HCVT (Hierarchical Cascaded Vision Transformers)** is a novel deep learning framework designed for medical image analysis tasks, particularly focusing on chest X-ray image classification and segmentation. This repository contains the official implementation of our paper introducing a hierarchical cascaded architecture that combines the strengths of Vision Transformers with multi-scale feature extraction and intelligent model combination techniques.

## 🏥 Datasets

### ChestX-ray Dataset
- **Size**: 112,120 frontal X-ray images from 30,805 different patients
- **Labels**: 14 disease categories automatically generated from radiology reports
- **Diseases**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- **Split**: 
  - Training: 86,524 images
  - Test: 25,596 images (benchmark subset)

### VinDr-CXR Dataset
- **Size**: 18,000 carefully radiologist-annotated frontal chest X-ray images
- **Labels**: 28 disease categories with expert annotations
- **Diseases**: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Clavicle fracture, Consolidation, Edema, Emphysema, Enlarged pulmonary artery, Interstitial lung disease, Infiltration, Lung opacity, Lung cavity, Lung cyst, Mediastinal shift, Nodule/Mass, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis, Rib fracture, Other lesion, Chronic obstructive pulmonary disease, Lung tumor, Pneumonia, Tuberculosis, Other disease
- **Split**:
  - Training: 15,000 images
  - Test: 3,000 images (benchmark subset)

## 🧠 Proposed Method

### Architecture Overview

HCVT introduces a **Hierarchical Cascaded Vision Transformer** architecture that processes medical images through multiple stages:

```
Input Image → Stage 1 (SWIN Transformers) → Stage 2 (CSWIN Transformers) → ... → Stage N → Final Prediction
                    ↓                              ↓                              ↓
            Multi-scale Features           Enhanced Features           Refined Features
```

### Key Components

#### 1. Hierarchical Feature Extraction
- **Multi-stage Processing**: Progressive feature extraction with alternating SWIN and CSWIN Transformer blocks
- **Progressive Downsampling**: Reduces computational complexity while maintaining representational power
- **Multi-scale Capture**: Extracts features from fine-grained details to larger contextual information

#### 2. Custom SWIN Transformers (CSWIN)
- **Enhanced Attention Mechanisms**: Modified self-attention with local enhancement layers
- **Window-based Processing**: Efficient computation through shifted window multi-head self-attention (SW-MSA)
- **Local Enhancement**: Additional convolutional layers for improved local feature capture

#### 3. Hierarchical Cascading Framework
- **Stage-wise Training**: Each stage refines predictions from the previous stage
- **Cross-validation Partitioning**: Robust training through k-fold cross-validation
- **Prediction Matrix Generation**: Probability matrices for data augmentation

#### 4. Weight-based Model Combination
- **Linear Regression Optimization**: Minimizes squared error ||A*_j * F_j - M_j||²
- **Constrained Weights**: Weights bounded between 0 and 1 for stability
- **Multi-model Fusion**: Intelligent combination of predictions from multiple segmentation models

### Mathematical Framework

#### Weight Matrix Optimization
```
min_{F_j} ||A*_j * F_j - M_j||²
```
Where:
- A*_j: Prediction matrix for class j
- F_j: Weight vector for class j  
- M_j: Ground truth label vector for class j

#### Prediction Fusion
```
U_Qq(X_t(i,j)) = Σ_{s=1}^{S} f_{s,q} * O_s(L_j | X*_t(i,j))
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/hcvt.git
cd hcvt

# Create virtual environment (recommended)
python -m venv hcvt_env
source hcvt_env/bin/activate  # On Windows: hcvt_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Requirements
```txt
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
pydicom>=2.1.0
matplotlib>=3.3.0
tqdm>=4.50.0
Pillow>=8.0.0
```

## 📦 Dataset Preparation

### ChestX-ray Dataset
1. Download from [NIH ChestX-ray website](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. Extract to `/datasets/chest_xray/`
3. Expected structure:
```
/datasets/chest_xray/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── labels/
│   └── Data_Entry_2017.csv
└── train_val_list.txt
```

### VinDr-CXR Dataset
1. Download from [VinDr-CXR website](https://vindr.ai/datasets/cxr)
2. Extract to `/datasets/vindr_cxr/`
3. Expected structure:
```
/datasets/vindr_cxr/
├── train/
│   ├── images/
│   └── annotations/
├── test/
│   ├── images/
│   └── annotations/
└── train.csv
```

## 🏃 Usage

### Training HCVT Model

```bash
# Train on ChestX-ray dataset
python train.py --dataset chest_xray --batch_size 32 --epochs 100 --lr 1e-4

# Train on VinDr-CXR dataset
python train.py --dataset vindr_cxr --batch_size 16 --epochs 150 --lr 5e-5

# Resume training from checkpoint
python train.py --resume --checkpoint_path checkpoints/best_model.pth
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --dataset chest_xray --model_path checkpoints/hcvt_chest.pth

# Cross-dataset evaluation
python evaluate.py --source_dataset chest_xray --target_dataset vindr_cxr --model_path checkpoints/hcvt_chest.pth
```

### Inference on New Images

```bash
# Single image inference
python predict.py --image_path path/to/chest_xray.jpg --model_path checkpoints/hcvt_best.pth

# Batch inference
python predict.py --image_dir path/to/images/ --model_path checkpoints/hcvt_best.pth
```

## 🔧 Configuration

### Training Configuration (configs/chest_xray_config.yaml)
```yaml
model:
  name: "HCVT"
  img_size: 512
  patch_size: 4
  embed_dim: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]
  num_classes: 14
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.05
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 10

data:
  dataset_path: "/datasets/chest_xray"
  image_size: 512
  normalize: true
  augment: true
```

## 📈 Visualization and Analysis

### Training Monitoring
```bash
# Launch TensorBoard for training visualization
tensorboard --logdir=runs/

# Generate attention maps
python visualize_attention.py --image_path sample.jpg --model_path checkpoints/hcvt.pth
```

### Performance Analysis
- ROC curves for each disease category
- Confusion matrices
- Attention visualization
- Feature map analysis

## 🤝 Contributing

We welcome contributions to improve HCVT! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Include docstrings for all functions and classes
- Write unit tests for new functionality
- Update documentation as needed

## 📚 Citation

If you use HCVT in your research, please cite our paper:

```bibtex
@article{hcvt2024,
  title={Hierarchical Cascaded Vision Transformers for Medical Image Analysis},
  author={Your Name et al.},
  journal={Medical Image Analysis},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIH for providing the ChestX-ray dataset
- VinDr for the VinDr-CXR dataset
- Microsoft for SWIN Transformer inspiration
- PyTorch team for the excellent deep learning framework

## 📞 Contact

For questions, issues, or collaborations:
- **Email**: saeediqbalkhattak@gmail.com
---

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**
