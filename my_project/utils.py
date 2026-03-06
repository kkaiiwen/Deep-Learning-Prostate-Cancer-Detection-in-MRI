import torch
import torch.nn as nn


def to_tensor(x, device):
    return x.float().to(device)


class SegmentationLoss(nn.Module):
    """
    Weighted Dice + BCE loss for binary segmentation.
    """

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        bce = self.bce(y_pred, y_true)

        probs = torch.sigmoid(y_pred)
        numerator = 2 * torch.sum(probs * y_true, dim=(2, 3, 4))
        denominator = (
            torch.sum(probs, dim=(2, 3, 4)) +
            torch.sum(y_true, dim=(2, 3, 4)) +
            1e-6
        )
        dice_loss = 1 - torch.mean(numerator / denominator)

        return self.alpha * dice_loss + (1 - self.alpha) * bce


def bin_dice(y_pred, y_true, threshold=0.5):
    probs = torch.sigmoid(y_pred)
    preds = (probs > threshold).float()

    numerator = 2 * torch.sum(preds * y_true, dim=(2, 3, 4))
    denominator = (
        torch.sum(preds, dim=(2, 3, 4)) +
        torch.sum(y_true, dim=(2, 3, 4)) +
        1e-6
    )
    return torch.mean(numerator / denominator)


def soft_dice(y_pred, y_true):
    probs = torch.sigmoid(y_pred)

    numerator = 2 * torch.sum(probs * y_true, dim=(2, 3, 4))
    denominator = (
        torch.sum(probs, dim=(2, 3, 4)) +
        torch.sum(y_true, dim=(2, 3, 4)) +
        1e-6
    )
    return torch.mean(numerator / denominator)