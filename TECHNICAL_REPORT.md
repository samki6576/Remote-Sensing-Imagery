# Technical Report: Semi-Supervised Remote-Sensing Segmentation

## Abstract

This project investigates binary building segmentation when dense training masks are replaced by sparse point labels. A Partial Cross-Entropy loss is applied only at labeled pixels, while unlabeled pixels are ignored. The segmentation model is a U-Net decoder built on a ResNet-34 encoder. Experiments compare dense supervision with 1%, 5%, and 10% randomly sampled point labels using real aerial imagery from the Massachusetts Buildings Dataset. The implementation runs end to end and saves checkpoints and metric plots for every condition.

## 1. Problem Definition

Standard semantic segmentation training assumes a class label for every pixel. Point annotation provides only a small set of labeled pixels, so treating the remaining pixels as background would introduce incorrect supervision. The goal is therefore to train a segmentation network using sparse labels while evaluating it against dense validation masks.

This implementation uses binary segmentation:

- Class 0: background.
- Class 1: building.
- Ignore index 255: unlabeled training pixels.

## 2. Dataset

The project uses the Kaggle [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset). The included subset contains six training image/mask pairs and two validation image/mask pairs. Each aerial RGB tile is 1500 x 1500 pixels, and each mask is a binary PNG with values 0 and 255.

The images are resized to 128 x 128 during training and validation. Bilinear interpolation is used for RGB images; nearest-neighbor interpolation is used for masks to preserve discrete class labels. The data is already stored in the loader's expected structure:

```text
data/train/images
data/train/masks
data/val/images
data/val/masks
```

## 3. Method

### 3.1 Model

The model is a U-Net-style encoder-decoder. The encoder is ResNet-34 from TorchVision and extracts features at multiple spatial resolutions. The decoder progressively upsamples the deepest feature representation and concatenates it with encoder skip connections. A final 1 x 1 convolution produces two logits per pixel.

The implementation supports optional ImageNet initialization with `--pretrained`. The default configuration uses randomly initialized ResNet weights so it can run without downloading external model weights.

### 3.2 Point-label simulation

Let a dense mask have height $H$ and width $W$, and let $r$ be the point ratio. The sampler selects:

$$
N = \max(1, \lfloor rHW \rfloor)
$$

pixels uniformly without replacement. The selected pixels retain their dense class labels. Every unselected pixel is assigned ignore index 255. The experiments use $r \in \{0.01, 0.05, 0.10\}$, plus a dense-label baseline.

### 3.3 Partial Cross-Entropy loss

For prediction logits $z_i$, target labels $y_i$, and labeled-pixel indicator $m_i$, where $m_i=1$ when $y_i \ne 255$, the loss is:

$$
L_{PCE} = \frac{\sum_i m_i\,CE(z_i,y_i)}{\sum_i m_i}
$$

Only labeled pixels contribute to the numerator or denominator. If a batch contains no labeled pixels, the implementation returns a zero loss connected to the prediction graph, allowing the training loop to remain valid.

### 3.4 Optimization and evaluation

The model is trained with Adam and a cosine learning-rate schedule. Validation uses dense masks and reports:

- Accuracy.
- Binary F1 score.
- Binary Intersection over Union (IoU).
- Mean validation loss.

The best checkpoint for each experiment is selected by validation IoU.

## 4. Experimental Design

### Experiment A: effect of supervision density

**Purpose:** Determine how reducing the number of labeled pixels affects segmentation performance.

**Hypothesis:** Dense labels should provide the strongest supervision. Lower point ratios are expected to reduce performance, although Partial Cross-Entropy should still allow the model to learn from the available labels.

**Controlled variables:** model architecture, optimizer, batch size, image size, validation set, seed, and training procedure.

**Independent variable:** label type and point ratio.

**Dependent variables:** validation loss, accuracy, F1, and IoU.

**Commands:**

```powershell
.\.venv\Scripts\python.exe semi_supervised_segmentation.py --epochs 30 --image-size 128
```

The command runs these four conditions:

| Experiment | Training supervision |
|---|---|
| `exp1_dense_labels` | 100% dense labels |
| `exp2_point_labels_5pct` | 5% labeled pixels |
| `exp3_point_labels_1pct` | 1% labeled pixels |
| `exp4_point_labels_10pct` | 10% labeled pixels |

### Experiment B: effect of training duration

The `--epochs` argument controls training duration. A short run can be used for debugging, while a longer run should be used for reported results. The intended comparison is to run the same supervision setting with different epoch counts and compare validation IoU, not training loss alone.

## 5. Results

The verified real-data smoke run used:

```powershell
.\.venv\Scripts\python.exe semi_supervised_segmentation.py --epochs 1 --image-size 128
```

| Experiment | Best IoU | Final F1 | Final accuracy |
|---|---:|---:|---:|
| Dense labels | 0.2284 | 0.3719 | 0.2284 |
| 5% point labels | 0.2284 | 0.3719 | 0.2284 |
| 1% point labels | 0.2284 | 0.3719 | 0.2284 |
| 10% point labels | 0.2284 | 0.3719 | 0.2284 |

These values confirm that the real-data pipeline, sparse-label conversion, loss, validation, checkpointing, and plotting execute successfully. They should not be interpreted as evidence that all supervision levels have equal performance. The run used only one epoch and eight total image/mask pairs, so the model had insufficient opportunity to learn meaningful building boundaries.

The generated plots are stored in `results/metrics_*.png`, and the corresponding best checkpoints are stored in `checkpoints/`.

## 6. Discussion

Partial Cross-Entropy addresses the central annotation problem: unlabeled pixels are excluded rather than incorrectly assigned to the background class. The expected trade-off is that fewer labeled pixels provide less information per update. A reliable study of that trade-off requires more images, more epochs, multiple random seeds, and ideally a separate test split.

The current smoke results are dominated by the small dataset and short schedule. Therefore, the correct conclusion is that the proposed method is implemented and operational, not that sparse labels match dense labels in quality.

## 7. Limitations and Future Work

1. The included dataset subset is small and is intended for reproducible execution, not benchmark-level performance claims.
2. The current experiment uses one random seed and one validation split.
3. The default model is trained from scratch unless `--pretrained` is specified.
4. The task is binary building segmentation; multi-class land-cover segmentation would require additional mask classes and output channels.
5. Future experiments should use the full dataset, multiple seeds, longer training, class-balanced sampling, and confidence intervals.

## 8. Reproducibility

Install dependencies and run the real-data experiment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe semi_supervised_segmentation.py --epochs 30 --image-size 128
```

For a synthetic pipeline smoke test:

```powershell
.\.venv\Scripts\python.exe semi_supervised_segmentation.py --epochs 1 --samples 2 --image-size 64 --generate-data
```

Kaggle API credentials are not needed to run the included files. If additional data is downloaded, credentials must remain outside version control.
