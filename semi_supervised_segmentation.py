"""Semi-supervised remote-sensing segmentation with sparse point labels."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


class PartialCrossEntropyLoss(nn.Module):
    """Cross-entropy evaluated only at non-ignore-index pixels."""

    def __init__(self, ignore_index: int = 255, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        valid = targets != self.ignore_index
        losses = F.cross_entropy(
            predictions, targets, ignore_index=self.ignore_index, reduction="none"
        )
        if self.reduction == "none":
            return losses * valid
        if self.reduction == "sum":
            return (losses * valid).sum()
        if not valid.any():
            return predictions.sum() * 0.0
        return losses[valid].mean()


class ResNetUNet(nn.Module):
    """U-Net decoder over a ResNet-34 encoder."""

    def __init__(self, n_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc2 = backbone.layer1
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4

        self.dec5 = self._block(512 + 256, 256)
        self.dec4 = self._block(256 + 128, 128)
        self.dec3 = self._block(128 + 64, 64)
        self.dec2 = self._block(64 + 64, 64)
        self.dec1 = self._block(64, 32)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.dec5(torch.cat([F.interpolate(e5, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1))
        d4 = self.dec4(torch.cat([F.interpolate(d5, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        d1 = self.dec1(F.interpolate(d2, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.final_conv(d1)


class RemoteSensingDataset(Dataset):
    """Load RGB images and binary masks, optionally sparsifying mask labels."""

    extensions = (".png", ".jpg", ".jpeg")

    def __init__(self, image_dir: str | Path, mask_dir: str | Path | None = None,
                 label_type: str = "dense", point_ratio: float = 0.05,
                 image_size: int | None = 256, train: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.label_type = label_type
        self.point_ratio = point_ratio
        self.image_size = image_size
        self.train = train
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in self.extensions)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        if self.mask_dir:
            self.masks = sorted(p for p in self.mask_dir.iterdir() if p.suffix.lower() in self.extensions)
            if len(self.images) != len(self.masks):
                raise ValueError("Image and mask counts must match")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        if self.mask_dir:
            mask = Image.open(self.masks[index]).convert("L")
            mask_array = (np.asarray(mask) > 127).astype(np.int64)
        else:
            mask_array = np.zeros((image.height, image.width), dtype=np.int64)

        if self.image_size:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask_array = np.asarray(Image.fromarray(mask_array.astype(np.uint8)).resize(size, Image.Resampling.NEAREST), dtype=np.int64)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_array = np.fliplr(mask_array).copy()
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask_array = np.flipud(mask_array).copy()

        if self.label_type == "point":
            mask_array = self._simulate_point_labels(mask_array)
        elif self.label_type == "mixed" and random.random() < 0.5:
            mask_array = self._simulate_point_labels(mask_array)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return image_tensor, torch.from_numpy(mask_array.copy()).long()

    def _simulate_point_labels(self, mask: np.ndarray) -> np.ndarray:
        sparse = np.full_like(mask, 255, dtype=np.int64)
        count = max(1, int(mask.size * self.point_ratio))
        indices = np.random.choice(mask.size, min(count, mask.size), replace=False)
        sparse.flat[indices] = mask.flat[indices]
        return sparse


class RemoteSensingDataGenerator:
    def __init__(self, image_size: int = 256, num_samples: int = 30):
        self.image_size = image_size
        self.num_samples = num_samples

    def generate_data(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        (save_dir / "images").mkdir(parents=True, exist_ok=True)
        (save_dir / "masks").mkdir(parents=True, exist_ok=True)
        for index in range(self.num_samples):
            image, mask = self._generate_sample()
            image.save(save_dir / "images" / f"image_{index:03d}.png")
            mask.save(save_dir / "masks" / f"mask_{index:03d}.png")

    def _generate_sample(self) -> tuple[Image.Image, Image.Image]:
        image = Image.new("RGB", (self.image_size, self.image_size), (135, 206, 235))
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        image_draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        for _ in range(np.random.randint(5, 10)):
            x, y = np.random.randint(0, self.image_size - 40, 2)
            width, height = np.random.randint(20, 80, 2)
            color = tuple(np.random.randint(20, 220, 3).tolist())
            label = int(np.random.choice([0, 255], p=[0.4, 0.6]))
            image_draw.rectangle((x, y, x + width, y + height), fill=color)
            if label:
                mask_draw.rectangle((x, y, x + width, y + height), fill=label)
        return image, mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float, float]:
    model.eval()
    loss_total, predictions, targets = 0.0, [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss_total += criterion(logits, masks).item()
            predictions.append(logits.argmax(1).cpu().numpy())
            targets.append(masks.cpu().numpy())
    predictions = np.concatenate(predictions).ravel()
    targets = np.concatenate(targets).ravel()
    valid = targets != 255
    predictions, targets = predictions[valid], targets[valid]
    return (loss_total / max(1, len(loader)), accuracy_score(targets, predictions),
            f1_score(targets, predictions, zero_division=0),
            jaccard_score(targets, predictions, zero_division=0))


def run_experiment(config: dict) -> dict:
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = RemoteSensingDataset(config["train_image_dir"], config["train_mask_dir"], config["label_type"], config["point_ratio"], config.get("image_size", 128), True)
    val_set = RemoteSensingDataset(config["val_image_dir"], config["val_mask_dir"], "dense", 1.0, config.get("image_size", 128), False)
    train_loader = DataLoader(train_set, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.get("batch_size", 4), shuffle=False, num_workers=0)
    model = ResNetUNet(pretrained=config.get("pretrained", False)).to(device)
    criterion = PartialCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["result_dir"]).mkdir(parents=True, exist_ok=True)
    history = {key: [] for key in ("train_loss", "val_loss", "accuracy", "f1", "iou")}
    best_iou = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"{config['name']} epoch {epoch + 1}/{config['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss, accuracy, f1, iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(running_loss / max(1, len(train_loader)))
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)
        history["f1"].append(f1)
        history["iou"].append(iou)
        print(f"Epoch {epoch + 1}: loss={history['train_loss'][-1]:.4f}, val_iou={iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            torch.save({"model_state_dict": model.state_dict(), "config": config, "iou": iou}, Path(config["save_dir"]) / f"best_model_{config['name']}.pth")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(history["accuracy"], label="accuracy")
    axes[1].plot(history["f1"], label="F1")
    axes[1].plot(history["iou"], label="IoU")
    axes[1].set_title("Metrics")
    axes[2].plot(history["iou"], label="IoU")
    axes[2].set_title("IoU")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(Path(config["result_dir"]) / f"metrics_{config['name']}.png")
    plt.close(figure)
    return {"name": config["name"], "label_type": config["label_type"], "point_ratio": config["point_ratio"], "best_iou": best_iou, "final_accuracy": history["accuracy"][-1], "final_f1": history["f1"][-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--generate-data", action="store_true", help="Replace data/train and data/val with synthetic samples")
    parser.add_argument("--pretrained", action="store_true", help="Download and use ResNet-34 ImageNet weights")
    args = parser.parse_args()
    root = Path(__file__).parent
    if args.generate_data:
        generator = RemoteSensingDataGenerator(args.image_size, args.samples)
        generator.generate_data(root / "data" / "train")
        generator.generate_data(root / "data" / "val")
    base = {"train_image_dir": root / "data/train/images", "train_mask_dir": root / "data/train/masks", "val_image_dir": root / "data/val/images", "val_mask_dir": root / "data/val/masks", "save_dir": root / "checkpoints", "result_dir": root / "results", "epochs": args.epochs, "batch_size": 4, "image_size": args.image_size, "pretrained": args.pretrained}
    experiments = [("exp1_dense_labels", "dense", 1.0), ("exp2_point_labels_5pct", "point", 0.05), ("exp3_point_labels_1pct", "point", 0.01), ("exp4_point_labels_10pct", "point", 0.10)]
    results = [run_experiment({**base, "name": name, "label_type": label_type, "point_ratio": ratio}) for name, label_type, ratio in experiments]
    print("\nExperiment comparison")
    for result in results:
        print(f"{result['name']}: IoU={result['best_iou']:.4f}, F1={result['final_f1']:.4f}, accuracy={result['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
