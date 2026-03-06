# utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(x, device):
    return x.float().to(device)


class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):

        # BCE
        bce = self.bce(y_pred, y_true)

        # Dice
        probs = torch.sigmoid(y_pred)
        numerator = 2 * torch.sum(probs * y_true, dim=(2,3,4))
        denominator = torch.sum(probs, dim=(2,3,4)) + torch.sum(y_true, dim=(2,3,4)) + 1e-6
        dice = 1 - torch.mean(numerator / denominator)

        return self.alpha * dice + (1 - self.alpha) * bce


def bin_dice(y_pred, y_true):
    probs = torch.sigmoid(y_pred)
    numerator = 2 * torch.sum(probs * y_true, dim=(2,3,4))
    denominator = torch.sum(probs, dim=(2,3,4)) + torch.sum(y_true, dim=(2,3,4)) + 1e-6
    return torch.mean(numerator / denominator)
