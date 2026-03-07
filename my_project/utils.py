import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    return x.float().to(device, non_blocking=True)


class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        bce = self.bce(y_pred, y_true)

        probs = torch.sigmoid(y_pred)
        numerator = 2 * torch.sum(probs * y_true, dim=(2, 3, 4))
        denominator = (
            torch.sum(probs, dim=(2, 3, 4))
            + torch.sum(y_true, dim=(2, 3, 4))
            + 1e-6
        )
        dice_loss = 1 - torch.mean(numerator / denominator)

        return self.alpha * dice_loss + (1 - self.alpha) * bce


def _binary_prediction(y_pred, threshold=0.5):
    probs = torch.sigmoid(y_pred)
    preds = (probs > threshold).float()
    return probs, preds


def batch_dice(y_pred, y_true, threshold=0.5):
    _, preds = _binary_prediction(y_pred, threshold)

    numerator = 2 * torch.sum(preds * y_true, dim=(2, 3, 4))
    denominator = (
        torch.sum(preds, dim=(2, 3, 4))
        + torch.sum(y_true, dim=(2, 3, 4))
        + 1e-6
    )
    return torch.mean(numerator / denominator)


def batch_iou(y_pred, y_true, threshold=0.5):
    _, preds = _binary_prediction(y_pred, threshold)

    intersection = torch.sum(preds * y_true, dim=(2, 3, 4))
    union = (
        torch.sum(preds, dim=(2, 3, 4))
        + torch.sum(y_true, dim=(2, 3, 4))
        - intersection
        + 1e-6
    )
    return torch.mean(intersection / union)


def batch_precision(y_pred, y_true, threshold=0.5):
    _, preds = _binary_prediction(y_pred, threshold)

    tp = torch.sum(preds * y_true, dim=(2, 3, 4))
    fp = torch.sum(preds * (1 - y_true), dim=(2, 3, 4))
    return torch.mean(tp / (tp + fp + 1e-6))


def batch_recall(y_pred, y_true, threshold=0.5):
    _, preds = _binary_prediction(y_pred, threshold)

    tp = torch.sum(preds * y_true, dim=(2, 3, 4))
    fn = torch.sum((1 - preds) * y_true, dim=(2, 3, 4))
    return torch.mean(tp / (tp + fn + 1e-6))


def save_history_csv(history, out_path):
    fieldnames = ["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "lr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_case_metrics_csv(case_metrics, out_path):
    fieldnames = ["case_id", "dice", "iou", "precision", "recall"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_metrics:
            writer.writerow(row)


def save_test_summary(case_metrics, out_path):
    dice = np.array([row["dice"] for row in case_metrics], dtype=np.float32)
    iou = np.array([row["iou"] for row in case_metrics], dtype=np.float32)
    precision = np.array([row["precision"] for row in case_metrics], dtype=np.float32)
    recall = np.array([row["recall"] for row in case_metrics], dtype=np.float32)

    with open(out_path, "w") as f:
        f.write(f"Num cases: {len(case_metrics)}\n")
        f.write(f"Dice mean: {dice.mean():.4f}\n")
        f.write(f"Dice std:  {dice.std():.4f}\n")
        f.write(f"IoU mean:  {iou.mean():.4f}\n")
        f.write(f"IoU std:   {iou.std():.4f}\n")
        f.write(f"Precision mean: {precision.mean():.4f}\n")
        f.write(f"Precision std:  {precision.std():.4f}\n")
        f.write(f"Recall mean: {recall.mean():.4f}\n")
        f.write(f"Recall std:  {recall.std():.4f}\n")


def plot_history(history, out_dir):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_dice = [row["train_dice"] for row in history]
    val_dice = [row["val_dice"] for row in history]
    lrs = [row["lr"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=200)
    plt.close()


def save_prediction_figure(image, y_true, y_pred, out_path, case_id):
    image = image.detach().cpu().numpy()[0]
    y_true = y_true.detach().cpu().numpy()[0, 0]
    probs = torch.sigmoid(y_pred).detach().cpu().numpy()[0, 0]
    pred = (probs > 0.5).astype(np.float32)

    gt_sum = y_true.sum(axis=(1, 2))
    pred_sum = pred.sum(axis=(1, 2))

    if gt_sum.max() > 0:
        z = int(np.argmax(gt_sum))
    else:
        z = int(np.argmax(pred_sum))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(image[0, z], cmap="gray")
    axes[0, 0].set_title("T2")

    axes[0, 1].imshow(image[1, z], cmap="gray")
    axes[0, 1].set_title("ADC")

    axes[0, 2].imshow(image[2, z], cmap="gray")
    axes[0, 2].set_title("DWI")

    axes[1, 0].imshow(y_true[z], cmap="gray")
    axes[1, 0].set_title("Ground Truth")

    axes[1, 1].imshow(probs[z], cmap="gray")
    axes[1, 1].set_title("Predicted Probability")

    axes[1, 2].imshow(pred[z], cmap="gray")
    axes[1, 2].set_title("Predicted Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle(f"{case_id} | Slice {z}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()