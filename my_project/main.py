import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import mydataloader, get_train_transforms, get_val_transforms
from utils import to_tensor, SegmentationLoss, bin_dice


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self._pad_if_needed(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_if_needed(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_if_needed(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

    def _pad_if_needed(self, x, ref):
        diff_d = ref.size(2) - x.size(2)
        diff_h = ref.size(3) - x.size(3)
        diff_w = ref.size(4) - x.size(4)

        if diff_d == 0 and diff_h == 0 and diff_w == 0:
            return x

        pad = [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ]
        return nn.functional.pad(x, pad)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(loader):
        print(f"Training batch {i + 1}/{len(loader)}", end="\r")

        x = to_tensor(batch["image"], device)
        y = to_tensor(batch["label"], device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Validation batch {i + 1}/{len(loader)}", end="\r")

            x = to_tensor(batch["image"], device)
            y = to_tensor(batch["label"], device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            dice_scores.append(bin_dice(out, y).item())

    print()

    mean_loss = total_loss / len(loader)
    mean_dice = sum(dice_scores) / len(dice_scores)
    return mean_loss, mean_dice


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "1-uclH-data_ratio0.8")
    save_path = os.path.join(base_dir, "best_model.pth")

    train_ds = mydataloader(data_path, phase="train", transform=get_train_transforms())
    val_ds = mydataloader(data_path, phase="val", transform=get_val_transforms())

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_ch=3, out_ch=1, base=16).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )
    criterion = SegmentationLoss(alpha=0.7)

    num_epochs = 50
    best_dice = 0.0
    epochs_without_improvement = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")
        print(f"LR:         {current_lr:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")
        else:
            epochs_without_improvement += 1

        print(f"Best Dice:  {best_dice:.4f}")

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print("Training completed.")


if __name__ == "__main__":
    main()