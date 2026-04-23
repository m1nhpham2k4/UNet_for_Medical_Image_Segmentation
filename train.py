import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from data.colondb_dataset import ColonDBDataset
from UNet.unet_model import UNet
from utils.losses import consistency_loss, supervised_loss
from utils.metrics import batch_dice_iou_from_logits


def parse_args():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Semi-supervised UNet training for CVC-ColonDB.")
    parser.add_argument("--data-dir", default=cfg.data_dir, help="Root folder containing labeled/unlabeled/val/TestDataset.")
    parser.add_argument("--output-dir", default=cfg.output_dir, help="Folder for checkpoints and logs.")
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size, help="Labeled batch size.")
    parser.add_argument("--unlabeled-batch-size", type=int, default=cfg.unlabeled_batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--base-channels", type=int, default=cfg.base_channels)
    parser.add_argument("--unsup-weight", type=float, default=cfg.unsup_weight)
    parser.add_argument("--unsup-start-epoch", type=int, default=cfg.unsup_start_epoch)
    parser.add_argument("--confidence-threshold", type=float, default=cfg.confidence_threshold)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--amp", action="store_true", default=cfg.amp, help="Use CUDA mixed precision.")
    parser.add_argument("--device", default=cfg.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=cfg.threshold, help="Evaluation probability threshold.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--no-test-after-train", dest="test_after_train", action="store_false")
    parser.set_defaults(test_after_train=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def save_checkpoint(path, model, optimizer, epoch, best_val_dice, args):
    checkpoint = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer, scaler, device, args, epoch):
    model.train()
    labeled_iter = itertools.cycle(labeled_loader)
    unlabeled_iter = itertools.cycle(unlabeled_loader)
    steps = max(len(labeled_loader), len(unlabeled_loader))
    use_amp = args.amp and device.type == "cuda"
    use_unsup = args.unsup_weight > 0 and epoch >= args.unsup_start_epoch

    totals = {
        "loss": 0.0,
        "sup_loss": 0.0,
        "unsup_loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "valid_ratio": 0.0,
        "samples": 0,
    }

    progress = tqdm(range(steps), desc=f"Epoch {epoch:03d} train", leave=False)
    for _ in progress:
        labeled_batch = next(labeled_iter)
        unlabeled_batch = next(unlabeled_iter)

        images_l = labeled_batch["image"].to(device, non_blocking=True)
        masks_l = labeled_batch["mask"].to(device, non_blocking=True)
        weak_u = unlabeled_batch["image_weak"].to(device, non_blocking=True)
        strong_u = unlabeled_batch["image_strong"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if use_unsup:
                with autocast(device_type=device.type, enabled=use_amp):
                    weak_logits = model(weak_u)
                    weak_probs = torch.sigmoid(weak_logits)
                pseudo_targets = (weak_probs >= 0.5).float()
                confidence = torch.maximum(weak_probs, 1.0 - weak_probs)
                valid_mask = (confidence >= args.confidence_threshold).float()
            else:
                pseudo_targets = torch.zeros_like(strong_u[:, :1])
                valid_mask = torch.zeros_like(strong_u[:, :1])

        with autocast(device_type=device.type, enabled=use_amp):
            logits_l = model(images_l)
            sup = supervised_loss(logits_l, masks_l)

            if use_unsup:
                logits_u = model(strong_u)
                unsup = consistency_loss(logits_u, pseudo_targets.detach(), valid_mask.detach())
            else:
                unsup = logits_l.new_tensor(0.0)

            loss = sup + args.unsup_weight * unsup

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images_l.size(0)
        dice, iou = batch_dice_iou_from_logits(logits_l.detach(), masks_l.detach(), args.threshold)
        valid_ratio = valid_mask.mean().item()

        totals["loss"] += loss.item() * batch_size
        totals["sup_loss"] += sup.item() * batch_size
        totals["unsup_loss"] += unsup.item() * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size
        totals["valid_ratio"] += valid_ratio * batch_size
        totals["samples"] += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            sup=f"{sup.item():.4f}",
            unsup=f"{unsup.item():.4f}",
            dice=f"{dice:.4f}",
        )

    count = max(1, totals.pop("samples"))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, args, split_name):
    model.eval()
    use_amp = args.amp and device.type == "cuda"
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "samples": 0}

    for batch in tqdm(loader, desc=f"{split_name} eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = supervised_loss(logits, masks)

        dice, iou = batch_dice_iou_from_logits(logits, masks, args.threshold)
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size
        totals["samples"] += batch_size

    count = max(1, totals.pop("samples"))
    return {key: value / count for key, value in totals.items()}


def append_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Data root: {Path(args.data_dir).resolve()}")

    pin_memory = device.type == "cuda"
    labeled_dataset = ColonDBDataset(args.data_dir, "labeled", args.image_size, augment=True)
    unlabeled_dataset = ColonDBDataset(args.data_dir, "unlabeled", args.image_size, augment=False)
    val_dataset = ColonDBDataset(args.data_dir, "val", args.image_size, augment=False)
    test_dataset = ColonDBDataset(args.data_dir, "test", args.image_size, augment=False)

    labeled_loader = make_loader(labeled_dataset, args.batch_size, True, args.num_workers, pin_memory)
    unlabeled_loader = make_loader(
        unlabeled_dataset, args.unlabeled_batch_size, True, args.num_workers, pin_memory
    )
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, pin_memory)

    model = UNet(n_channels=3, n_classes=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_val_dice = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_dice = checkpoint.get("best_val_dice", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history_path = output_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, labeled_loader, unlabeled_loader, optimizer, scaler, device, args, epoch
        )
        val_metrics = evaluate(model, val_loader, device, args, "val")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_sup_loss": train_metrics["sup_loss"],
            "train_unsup_loss": train_metrics["unsup_loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "pseudo_valid_ratio": train_metrics["valid_ratio"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_history(history_path, row)

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_dice, args)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_dice, args)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"pseudo_valid={train_metrics['valid_ratio']:.3f}"
        )

    if args.test_after_train:
        best_path = output_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, model, optimizer=None, device=device)
            print(f"Loaded best checkpoint for test: {best_path}")
        test_metrics = evaluate(model, test_loader, device, args, "test")
        print(
            f"Test: loss={test_metrics['loss']:.4f} "
            f"dice={test_metrics['dice']:.4f} "
            f"iou={test_metrics['iou']:.4f}"
        )


if __name__ == "__main__":
    main()
