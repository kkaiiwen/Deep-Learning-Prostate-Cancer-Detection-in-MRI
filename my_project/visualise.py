import os
import json
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from data_loader import MyDataLoader, get_eval_transforms
from model import UNet
from utils import set_seed, to_tensor, batch_dice, save_prediction_figure


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results")
    vis_dir = os.path.join(out_dir, "test_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

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
    model.eval()

    saved = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = to_tensor(batch["image"], device)
            y = to_tensor(batch["label"], device)
            case_id = batch["case_id"][0]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(config["roi_size"]),
                    sw_batch_size=config["sw_batch_size"],
                    predictor=model,
                    overlap=0.25,
                )

            dice = batch_dice(out, y).item()
            filename = f"{case_id}_dice_{dice:.4f}.png"
            save_prediction_figure(
                image=x,
                y_true=y,
                y_pred=out,
                out_path=os.path.join(vis_dir, filename),
                case_id=case_id,
            )

            saved += 1
            print(f"Saved figure {saved}/{len(test_loader)}", end="\r")

    print()
    print(f"Visualisations saved in: {vis_dir}")


if __name__ == "__main__":
    main()