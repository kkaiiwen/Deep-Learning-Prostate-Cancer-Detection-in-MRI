import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def _load_nii(path, add_channel=True):
    data = nib.load(path).get_fdata().astype(np.float32)
    if add_channel:
        data = data[np.newaxis, ...]  # (1, X, Y, Z)
    return data


class mydataloader(Dataset):
    """
    Reads split files formatted as:
    t2 adc dwi zone lesion

    Uses:
      - first 3 as input image channels
      - last as binary lesion label
    """

    def __init__(self, data_path, phase="train", transform=None):
        self.data_path = data_path
        self.phase = phase
        self.transform = transform

        split_file = os.path.join(data_path, f"{phase}.txt")
        with open(split_file, "r") as f:
            self.lines = [line.strip() for line in f if line.strip()]

        self.base_dir = os.path.dirname(data_path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].split()

        if len(parts) < 5:
            raise ValueError(
                f"Expected at least 5 paths in line {idx}, got {len(parts)}: {self.lines[idx]}"
            )

        img_paths = parts[:3]
        mask_path = parts[-1]

        img_paths_full = []
        for p in img_paths:
            full_p = os.path.normpath(os.path.join(self.base_dir, p))
            img_paths_full.append(full_p)

        mask_path_full = os.path.normpath(os.path.join(self.base_dir, mask_path))

        imgs = []
        for p in img_paths_full:
            imgs.append(_load_nii(p))

        image = np.concatenate(imgs, axis=0)  # (3, X, Y, Z)

        mask = _load_nii(mask_path_full)
        mask = (mask > 0).astype(np.float32)

        # Convert to PyTorch 3D order: (C, X, Y, Z) -> (C, Z, Y, X)
        image = np.transpose(image, (0, 3, 2, 1))
        mask = np.transpose(mask, (0, 3, 2, 1))

        sample = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(mask).float(),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_train_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    ])


def get_val_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])