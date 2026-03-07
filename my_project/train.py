import os
import json
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate

from data_loader import MyDataLoader, get_train_transforms, get_eval_transforms
from model import UNet
from utils import (
    set_seed,
    to_tensor,
    SegmentationLoss,
    batch_dice,
    save_history_csv,
    plot_history,
)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for i, batch in enumerate(loader):
        print(f"Training batch {i + 1}/{len(loader)}", end="\r")

        x = to_tensor(batch["image"], device)
        y = to_tensor(batch["label"], device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += batch_dice(out.detach(), y).item()

    print()
    return total_loss / len(loader), total_dice / len(loader)


def evaluate_epoch(model, loader, criterion, device, roi_size, sw_batch_size):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Validation batch {i + 1}/{len(loader)}", end="\r")

            x = to_tensor(batch["image"], device)
            y = to_tensor(batch["label"], device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = sliding_window_inference(
                    inputs=x,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.25,
                )
                loss = criterion(out, y)

            total_loss += loss.item()
            total_dice += batch_dice(out, y).item()

    print()
    return total_loss / len(loader), total_dice / len(loader)


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "1-uclH-data_ratio0.8")
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "seed": 42,
        "batch_size": 1,
        "num_workers": 4,
        "base_channels": 16,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "loss_alpha": 0.7,
        "num_epochs": 50,
        "early_stop_patience": 10,
        "roi_size": [64, 128, 128],
        "sw_batch_size": 2,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_ds = MyDataLoader(data_path, phase="train", transform=get_train_transforms())
    val_ds = MyDataLoader(data_path, phase="val", transform=get_eval_transforms())

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    num_workers = config["num_workers"]
    pin_memory = device.type == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    model = UNet(in_ch=3, out_ch=1, base=config["base_channels"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    criterion = SegmentationLoss(alpha=config["loss_alpha"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice = -1.0
    epochs_without_improvement = 0
    history = []
    save_path = os.path.join(out_dir, "best_model.pth")

    roi_size = tuple(config["roi_size"])
    sw_batch_size = config["sw_batch_size"]

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        val_loss, val_dice = evaluate_epoch(
            model, val_loader, criterion, device, roi_size, sw_batch_size
        )

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_dice": round(train_dice, 6),
            "val_loss": round(val_loss, 6),
            "val_dice": round(val_dice, 6),
            "lr": current_lr,
        }
        history.append(row)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Train Dice: {train_dice:.4f}")
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

        print(f"Best Val Dice: {best_dice:.4f}")

        save_history_csv(history, os.path.join(out_dir, "training_history.csv"))
        plot_history(history, out_dir)

        if epochs_without_improvement >= config["early_stop_patience"]:
            print("Early stopping triggered.")
            break

    print("Training completed.")
    print(f"Best model saved to: {save_path}")


if __name__ == "__main__":
    main()