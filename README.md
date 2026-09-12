<img width="1400" height="400" alt="metrics_exp1_dense_labels" src="https://github.com/user-attachments/assets/722d1345-524b-4186-91da-591996cefdc5" />


```markdown
# 🛰️ Semi-Supervised Remote Sensing Image Segmentation

> A PyTorch implementation of semi-supervised semantic segmentation for aerial and satellite imagery using **Partial Cross-Entropy Loss** and a **ResNet34 U-Net** backbone.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Pixel-level annotation of remote sensing imagery is expensive and time-consuming. This project implements a **semi-supervised learning** approach that trains segmentation models using only a small fraction of labeled pixels — drastically reducing annotation cost while maintaining competitive performance.

The core idea: **Partial Cross-Entropy Loss** computes gradients only on labeled pixels and ignores the unlabeled ones (masked with `ignore_index=255`).

---

## ✨ Key Features

- 🎯 **Partial Cross-Entropy Loss** — trains only on labeled pixels
- 🧠 **ResNet34 U-Net** — transfer learning from ImageNet
- 🏷️ **Point Label Simulation** — tests 1%, 5%, 10%, and 100% label density
- 📊 **Full Experiment Pipeline** — training, validation, and comparison
- 💾 **Model Checkpointing** — saves best model per experiment
- 📈 **Metrics Visualization** — IoU, F1, Accuracy, Loss curves

---

## 🏗️ Architecture

```
Input Image (3×256×256)
        │
        ▼
┌───────────────────┐
│  ResNet34 Encoder │  ← Pretrained on ImageNet
│  (Transfer Learn) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  U-Net Decoder    │  ← Skip connections
│  (5 up-blocks)    │
└───────────────────┘
        │
        ▼
Output Mask (2×256×256)
```

---

## 📂 Project Structure

```
meriti_semi_supervised/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── masks/           # Training masks
│   └── val/
│       ├── images/          # Validation images
│       └── masks/           # Validation masks
├── checkpoints/             # Saved models
├── results/                 # Plots & metrics
├── requirements.txt
├── README.md
└── semi_supervised_segmentation.py
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd meriti_semi_supervised

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---
<img width="1920" height="1080" alt="Screenshot (101)" src="https://github.com/user-attachments/assets/815b9298-9dea-4888-9a32-568f51d78d96" />

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
Pillow>=8.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

## 🚀 Usage
<img width="1500" height="1500" alt="22828990_15" src="https://github.com/user-attachments/assets/4850c178-1cab-41c3-87a0-14fb93f401f3" />

### 1. Prepare Your Dataset

Place your images and masks in the appropriate folders:

```
data/train/images/image_001.png
data/train/masks/mask_001.png
data/val/images/image_001.png
data/val/masks/mask_001.png
```

**Recommended datasets:**
- [WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
- [DeepGlobe Land Cover](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
- [LandCover.ai](https://landcover.ai/)

### 2. Run Training

```bash
python semi_supervised_segmentation.py
```

### 3. Output

- ✅ Trained models in `checkpoints/`
- ✅ Metrics plots in `results/`
- ✅ Comparison chart across experiments

---

## 🧪 Experiments

Four configurations are evaluated to study the effect of label density:

| # | Experiment | Label Type | Point Ratio |
|---|-----------|-----------|-------------|
| 1 | `exp1_dense_labels` | Dense | 100% |
| 2 | `exp2_point_labels_5pct` | Point | 5% |
| 3 | `exp3_point_labels_1pct` | Point | 1% |
| 4 | `exp4_point_labels_10pct` | Point | 10% |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **IoU** | Intersection over Union (Jaccard Index) |
| **F1 Score** | Harmonic mean of precision & recall |
| **Accuracy** | Pixel-wise classification accuracy |
| **Loss** | Partial Cross-Entropy on labeled pixels |

---

## 🔬 How Partial Cross-Entropy Works

```python
class PartialCrossEntropyLoss(nn.Module):
    def forward(self, predictions, targets):
        # Standard cross-entropy, computed per-pixel
        loss_per_pixel = self.ce_loss(predictions, targets)
        
        # Mask for labeled pixels only (targets != 255)
        labeled_mask = targets != self.ignore_index
        
        # Apply mask — unlabeled pixels contribute ZERO loss
        masked_loss = loss_per_pixel * labeled_mask.float()
        
        return masked_loss.sum() / labeled_mask.sum()
```

**Why this matters:**
- Unlabeled pixels are marked with `255` in masks
- Loss is averaged only over labeled pixels
- Model learns from sparse supervision without being penalized for unlabeled regions

---

## 📈 Results



Training curves and IoU comparisons are saved to `results/`:

- `metrics_exp1_dense_labels.png`
- `metrics_exp2_point_labels_5pct.png`
- `metrics_exp3_point_labels_1pct.png`
- `metrics_exp4_point_labels_10pct.png`
- `experiment_comparison.png`

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Framework:** PyTorch
- **Models:** torchvision (ResNet34)
- **Metrics:** scikit-learn
- **Visualization:** Matplotlib
- **Image Processing:** PIL / NumPy

---

## 💡 Key Insights

1. **Transfer learning is critical** — pretrained ResNet34 enables strong features with few labels
2. **Partial loss = scalable annotation** — point labels reduce annotation time by 90%+
3. **Semi-supervised learning works** — meaningful segmentation from minimal supervision

---

## 🔮 Future Work

- [ ] Mean Teacher / consistency regularization
- [ ] Pseudo-labeling with confidence thresholds
- [ ] Multi-class land cover segmentation
- [ ] Test-time augmentation
- [ ] ONNX export for deployment

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---


## 🙏 Acknowledgments

- ResNet34 weights from [torchvision](https://pytorch.org/vision/stable/models.html)
- U-Net architecture inspired by [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Partial Cross-Entropy concept from semi-supervised segmentation literature

---

⭐ **If you found this project useful, give it a star!**
```
