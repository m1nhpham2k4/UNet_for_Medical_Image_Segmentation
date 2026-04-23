import argparse
import math
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split CVC-ColonDB into SampleData with labeled/val/TestDataset/unlabeled."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\HK2 2025-2026\KLTN\SemiSAM\UNet_for_Medical_Image\CVC_ColonDB"),
        help="Source dataset directory containing images/ and masks/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\HK2 2025-2026\KLTN\SemiSAM\UNet_for_Medical_Image\dataset"),
        help="Output directory for the split dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before splitting.",
    )
    parser.add_argument(
        "--labeled-ratio",
        type=float,
        default=0.1,
        help="Ratio for labeled data.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio for validation data.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio for test data.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Enabled by default if --move is not set.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned split without copying or moving files.",
    )
    return parser.parse_args()


def collect_pairs(source_dir: Path):
    image_dir = source_dir / "images"
    mask_dir = source_dir / "masks"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Source directory must contain 'images' and 'masks' folders.")

    image_map = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    mask_map = {path.name: path for path in mask_dir.iterdir() if path.is_file()}

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))

    if missing_masks or missing_images:
        problems = []
        if missing_masks:
            problems.append(f"missing masks for: {', '.join(missing_masks[:5])}")
        if missing_images:
            problems.append(f"missing images for: {', '.join(missing_images[:5])}")
        raise ValueError("; ".join(problems))

    def sort_key(name: str):
        stem = Path(name).stem
        return (0, int(stem), name) if stem.isdigit() else (1, stem, name)

    file_names = sorted(image_map.keys(), key=sort_key)
    return [(image_map[name], mask_map[name]) for name in file_names]


def build_split_plan(pairs, labeled_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total = len(pairs)

    if labeled_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("All ratios must be >= 0.")

    if labeled_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("labeled_ratio + val_ratio + test_ratio must be <= 1.0")

    labeled_count = math.floor(total * labeled_ratio)
    val_count = math.floor(total * val_ratio)
    test_count = math.floor(total * test_ratio)

    if labeled_count == 0:
        raise ValueError("labeled_ratio is too small for this dataset size.")

    shuffled_pairs = pairs[:]
    random.Random(seed).shuffle(shuffled_pairs)

    labeled_end = labeled_count
    val_end = labeled_end + val_count
    test_end = val_end + test_count

    return {
        "labeled": shuffled_pairs[:labeled_end],
        "val": shuffled_pairs[labeled_end:val_end],
        "TestDataset": shuffled_pairs[val_end:test_end],
        "unlabeled": shuffled_pairs[test_end:],
    }

def ensure_split_dirs(output_dir: Path, split_name: str):
    if split_name == "TestDataset":
        base_dir = output_dir / "TestDataset" / "CVC-ColonDB"
    else:
        base_dir = output_dir / split_name

    image_dir = base_dir / "image"
    mask_dir = base_dir / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, mask_dir


def clear_split_dirs(output_dir: Path, split_name: str, dry_run: bool):
    image_dir, mask_dir = ensure_split_dirs(output_dir, split_name)
    if dry_run:
        return

    for directory in (image_dir, mask_dir):
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def transfer_file(src: Path, dst: Path, move: bool):
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_split(output_dir: Path, split_name: str, pairs, move: bool, dry_run: bool):
    image_dir, mask_dir = ensure_split_dirs(output_dir, split_name)

    for image_path, mask_path in pairs:
        image_target = image_dir / image_path.name
        mask_target = mask_dir / mask_path.name

        if dry_run:
            continue

        transfer_file(image_path, image_target, move=move)
        transfer_file(mask_path, mask_target, move=move)


def main():
    args = parse_args()
    move = args.move and not args.copy

    pairs = collect_pairs(args.source)
    plan = build_split_plan(
        pairs,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Found {len(pairs)} paired samples in {args.source}")
    for split_name, split_pairs in plan.items():
        print(f"{split_name}: {len(split_pairs)}")

    for split_name in plan:
        clear_split_dirs(args.output, split_name, dry_run=args.dry_run)

    for split_name, split_pairs in plan.items():
        write_split(args.output, split_name, split_pairs, move=move, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run only. No files were copied or moved.")
    else:
        action = "moved" if move else "copied"
        print(f"Files successfully {action} to {args.output}")


if __name__ == "__main__":
    main()
