import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data.colondb_dataset import IMAGE_EXTENSIONS
from train import get_device
from UNet.unet_model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run UNet segmentation inference.")
    parser.add_argument("--checkpoint", default="runs/semisup_unet/best.pt")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-prob", action="store_true", help="Also save probability maps.")
    return parser.parse_args()


def list_input_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(
        [item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: item.name,
    )


def preprocess(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    original_h, original_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.float(), (original_w, original_h)


def main():
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    image_size = args.image_size or checkpoint_args.get("image_size", 256)
    base_channels = args.base_channels or checkpoint_args.get("base_channels", 32)

    model = UNet(n_channels=3, n_classes=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_input_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    with torch.no_grad():
        for image_path in image_paths:
            image, original_size = preprocess(image_path, image_size)
            logits = model(image.to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob >= args.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

            mask_path = output_dir / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            if args.save_prob:
                prob_map = cv2.resize((prob * 255).astype(np.uint8), original_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(output_dir / f"{image_path.stem}_prob.png"), prob_map)

    print(f"Saved {len(image_paths)} prediction mask(s) to {output_dir}")


if __name__ == "__main__":
    main()
