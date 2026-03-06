# main.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from lesion_dataloaders import mydataloader, get_train_transforms
from utils import to_tensor, SegmentationLoss, bin_dice


# 3D UNet

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Conv3d(in_f, out_f, 3, padding=1),
                nn.BatchNorm3d(out_f),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_f, out_f, 3, padding=1),
                nn.BatchNorm3d(out_f),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = block(base, base*2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = block(base*2, base*4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = block(base*4, base*8)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.dec3 = block(base*8, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = block(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.dec1 = block(base*2, base)

        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)  # logits


# Training

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "./1-uclH-data_ratio0.8"

    train_ds = mydataloader(data_path, "train", get_train_transforms())
    val_ds   = mydataloader(data_path, "val", None)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SegmentationLoss(alpha=0.5)

    best_dice = 0

    for epoch in range(2):

        # train
        model.train()
        total_loss = 0

        for batch in train_loader:
            x = to_tensor(batch["image"], device)
            y = to_tensor(batch["label"], device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Train loss: {total_loss/len(train_loader):.4f}")

        # validation
        model.eval()
        dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                x = to_tensor(batch["image"], device)
                y = to_tensor(batch["label"], device)
                out = model(x)
                dice_scores.append(bin_dice(out, y).item())

        mean_dice = sum(dice_scores)/len(dice_scores)
        print(f"Validation Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

    print("Training completed.")


if __name__ == "__main__":
    main()
