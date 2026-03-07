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
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
)


def _load_nii(path, add_channel=True):
    data = nib.load(path).get_fdata().astype(np.float32)
    if add_channel:
        data = data[np.newaxis, ...]  # (1, X, Y, Z)
    return data


class MyDataLoader(Dataset):
    """
    Reads split files formatted as:
    t2 adc dwi zone lesion

    Uses:
      - first 3 as input image channels
      - 4th as prostate zone mask for ROI cropping
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
        zone_path = parts[3]
        label_path = parts[4]

        img_paths_full = []
        for p in img_paths:
            full_p = os.path.normpath(os.path.join(self.base_dir, p))
            img_paths_full.append(full_p)

        zone_path_full = os.path.normpath(os.path.join(self.base_dir, zone_path))
        label_path_full = os.path.normpath(os.path.join(self.base_dir, label_path))

        imgs = []
        for p in img_paths_full:
            imgs.append(_load_nii(p))
        image = np.concatenate(imgs, axis=0)  # (3, X, Y, Z)

        zone = _load_nii(zone_path_full)
        zone = (zone > 0).astype(np.float32)

        label = _load_nii(label_path_full)
        label = (label > 0).astype(np.float32)

        # Convert to PyTorch 3D order: (C, X, Y, Z) -> (C, Z, Y, X)
        image = np.transpose(image, (0, 3, 2, 1))
        zone = np.transpose(zone, (0, 3, 2, 1))
        label = np.transpose(label, (0, 3, 2, 1))

        case_id = os.path.basename(os.path.dirname(label_path_full))

        sample = {
            "image": torch.from_numpy(image).float(),
            "zone": torch.from_numpy(zone).float(),
            "label": torch.from_numpy(label).float(),
            "case_id": case_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_train_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "zone", "label"],
            source_key="zone",
            margin=8,
        ),
        SpatialPadd(
            keys=["image", "zone", "label"],
            spatial_size=(64, 128, 128),
        ),
        RandCropByPosNegLabeld(
            keys=["image", "zone", "label"],
            label_key="label",
            spatial_size=(64, 128, 128),
            pos=2,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "zone", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "zone", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "zone", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    ])


def get_eval_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "zone", "label"],
            source_key="zone",
            margin=8,
        ),
    ])