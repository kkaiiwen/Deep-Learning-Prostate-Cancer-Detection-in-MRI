# lesion_dataloaders.py

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, NormalizeIntensityd


def _load_nii(path, add_channel=True):
    data = nib.load(path).get_fdata().astype(np.float32)  # (X,Y,Z)
    if add_channel:
        data = data[np.newaxis, ...]  # (1,X,Y,Z)
    return data


class mydataloader(Dataset):
    """
    Reads split files formatted as:

    t2 adc dwi zone lesion

    Uses:
      - first 3 as input
      - last as label
    """

    def __init__(self, data_path, phase="train", transform=None):
        self.data_path = data_path
        self.phase = phase
        self.transform = transform

        split_file = os.path.join(data_path, f"{phase}.txt")
        with open(split_file, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        parts = [p for p in self.lines[idx].split(" ") if p != ""]

        img_paths = parts[:3]       # T2, ADC, DWI
        mask_path = parts[-1]       # lesion_mask

        # Load images
        imgs = [_load_nii(p) for p in img_paths]
        image = np.concatenate(imgs, axis=0)  # (3,X,Y,Z)

        # Load mask
        mask = _load_nii(mask_path)
        mask = (mask > 0).astype(np.float32)

        # Transpose to PyTorch format
        # (C,X,Y,Z) -> (C,Z,Y,X)
        image = np.transpose(image, (0, 3, 2, 1))
        mask  = np.transpose(mask,  (0, 3, 2, 1))

        sample = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(mask).float()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_train_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])
