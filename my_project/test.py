import os
import json
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from data_loader import MyDataLoader, get_eval_transforms
from model import UNet
from utils import (
    set_seed,
    to_tensor,
    batch_dice,
    batch_iou,
    batch_precision,
    batch_recall,
    save_case_metrics_csv,
    save_test_summary,
)


def run_test_set(model, loader, device, roi_size, sw_batch_size):
    model.eval()
    case_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Testing case {i + 1}/{len(loader)}", end="\r")

            x = to_tensor(batch["image"], device)
            y = to_tensor(batch["label"], device)
            case_id = batch["case_id"][0]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = sliding_window_inference(
                    inputs=x,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.25,
                )

            case_metrics.append({
                "case_id": case_id,
                "dice": batch_dice(out, y).item(),
                "iou": batch_iou(out, y).item(),
                "precision": batch_precision(out, y).item(),
                "recall": batch_recall(out, y).item(),
            })

    print()
    return case_metrics


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results")
    data_path = os.path.join(base_dir, "1-uclH-data_ratio0.8")

    with open(os.path.join(out_dir, "config.json"), "r") as f:
        config = json.load(f)

    test_ds = MyDataLoader(data_path, phase="test", transform=get_eval_transforms())

    num_workers = config["num_workers"]
    pin_memory = device.type == "cuda"
    persistent = num_workers > 0

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    model = UNet(in_ch=3, out_ch=1, base=config["base_channels"]).to(device)
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device)
    )

    case_metrics = run_test_set(
        model=model,
        loader=test_loader,
        device=device,
        roi_size=tuple(config["roi_size"]),
        sw_batch_size=config["sw_batch_size"],
    )

    save_case_metrics_csv(case_metrics, os.path.join(out_dir, "test_case_metrics.csv"))
    save_test_summary(case_metrics, os.path.join(out_dir, "test_summary.txt"))

    sorted_cases = sorted(case_metrics, key=lambda d: d["dice"])
    with open(os.path.join(out_dir, "worst_cases.txt"), "w") as f:
        for row in sorted_cases[:5]:
            f.write(
                f"{row['case_id']}, Dice={row['dice']:.4f}, "
                f"IoU={row['iou']:.4f}, Precision={row['precision']:.4f}, "
                f"Recall={row['recall']:.4f}\n"
            )

    print("Test evaluation completed.")
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()