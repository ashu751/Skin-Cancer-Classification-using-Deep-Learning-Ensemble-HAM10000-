# Skin Cancer Classification — HAM10000

> Deep learning ensemble for 7-class dermoscopic lesion classification on the HAM10000 dataset, achieving **99.2% accuracy** with a multi-model ensemble.

---

## Overview

This project builds and benchmarks a suite of deep learning models to classify skin lesions from dermoscopic images into 7 diagnostic categories. Five architectures — a custom DCNN, ResNet50, EfficientNet-B0, MobileNetV2, and DenseNet121 — are trained, evaluated, and fused into a final ensemble that achieves state-of-the-art accuracy on the HAM10000 dataset.

The pipeline covers class imbalance correction, augmentation, training with label smoothing, Grad-CAM visualization, feature-space analysis (PCA, t-SNE), and comprehensive statistical evaluation.

---

## Results

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Custom DCNN | 98.10% | 0.985 | 0.995 |
| ResNet50 | 98.29% | 0.978 | 0.991 |
| EfficientNet-B0 | 98.86% | 0.989 | 0.996 |
| MobileNetV2 | 98.72% | 0.967 | 0.988 |
| DenseNet121 | 99.05% | 0.987 | 0.995 |
| **Ensemble** | **99.20%** | **0.992** | **0.998** |

---

## Dataset

**HAM10000** (Human Against Machine with 10000 training images) — a large public dataset of dermoscopic skin lesion images with 7 classes:

| Code | Condition |
|---|---|
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `bkl` | Benign keratosis-like lesions |
| `bcc` | Basal cell carcinoma |
| `akiec` | Actinic keratoses / Intraepithelial carcinoma |
| `vasc` | Vascular lesions |
| `df` | Dermatofibroma |

The dataset is severely imbalanced (`nv` dominates). **RandomOverSampler** is applied to balance all classes before training.

**Download:**
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip -q skin-cancer-mnist-ham10000.zip -d HAM10000
```

---

## Architecture

### Custom DCNN
A bespoke 4-block convolutional network built from scratch:
- 4 × `Conv2d → BatchNorm → ReLU → MaxPool` blocks (32 → 64 → 128 → 256 channels)
- Fully connected head: 256 → 128 → 64 → 7 with BatchNorm and Dropout (0.5)
- Input resolution: 64×64

### Transfer Learning Models
All pretrained on ImageNet via `torchvision` / `timm`:
- **ResNet50** — fine-tuned FC replaced with Linear(2048 → 512 → 7)
- **EfficientNet-B0** — loaded via `timm`, classifier head replaced
- **MobileNetV2** — lightweight model for speed/accuracy trade-off
- **DenseNet121** — dense skip connections, strong on small datasets

### Ensemble
Soft voting over predicted class probabilities from all five models.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Image size | 64 × 64 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-3 (DCNN), 3e-4 (pretrained) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Epochs | 50 |
| Loss | Label Smoothing (smoothing=0.1) |
| Mixed precision | `torch.cuda.amp` |
| Weight decay | 1e-4 |

### Augmentation (training only)
- Horizontal & vertical flip
- ShiftScaleRotate (shift 0.1, scale 0.1, rotate ±20°)
- RandomBrightnessContrast
- CoarseDropout (8 holes, 8×8 max)
- Normalize (mean=0.5, std=0.5)

---

## Evaluation Suite

Beyond accuracy, the notebook computes:

- **Confusion matrix** (raw and normalized)
- **Classification report** (precision, recall, F1 per class)
- **ROC curves** and multi-class AUC (OvR)
- **Precision-recall curves** per class with Average Precision
- **Calibration curves** and **Brier scores** per class
- **Cohen's Kappa** score
- **Sensitivity and specificity** per class
- **Statistical significance** — paired t-test and McNemar's test comparing ResNet50 vs EfficientNet

---

## Interpretability

**Grad-CAM** is applied to the EfficientNet model to generate class activation heatmaps, overlaid on original dermoscopic images to show which regions drove each prediction.

---

## Feature Space Analysis

Latent features extracted from EfficientNet's global pooling layer are visualized using:

- **PCA** (2D projection of validation set embeddings)
- **t-SNE** (perplexity=30, 1000 iterations)
- **Hierarchical clustermap** of top-50 most discriminative ResNet50 features across 7 classes (Ward linkage, Euclidean distance)

---

## Setup & Requirements

```bash
pip install torch torchvision timm albumentations imbalanced-learn \
            scikit-learn scipy statsmodels matplotlib seaborn pandas pillow tqdm
```

```bash
# Kaggle API required for dataset download
pip install kaggle
```

### Run

Open the notebook in Kaggle or Jupyter:

```bash
jupyter notebook skin-cancer-99-20.ipynb
```

A GPU is strongly recommended (the notebook auto-detects CUDA via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## File Structure

```
├── skin-cancer-99-20.ipynb         # Full pipeline notebook
├── best_model.pth                  # Best DCNN checkpoint (saved during training)
└── HAM10000/                       # Dataset directory (after download)
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/
```

---

## Key Design Choices

- **Label Smoothing Loss** instead of standard cross-entropy — reduces overconfidence, improves calibration
- **RandomOverSampler** on file paths (not images) — memory-efficient balancing without augmenting the whole dataset
- **Mixed precision training** (`torch.cuda.amp`) — faster training on GPU with no accuracy loss
- **Stratified train/val split** — ensures class distribution is preserved in both sets
- **Soft ensemble voting** — averages raw probabilities rather than hard votes, more robust to individual model errors

---

## Author

**Ashutosh** — [github.com/ashu751](https://github.com/ashu751)

B.Tech Computer Science, Bennett University  
Published researcher in ML-based medical image classification (Procedia Computer Science, Elsevier)

---

## License

This project is for academic and research purposes. Dataset usage is subject to the [HAM10000 terms](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
