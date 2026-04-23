from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _sort_key(path: Path):
    stem = path.stem
    return (0, int(stem), path.name) if stem.isdigit() else (1, stem, path.name)


def _find_split_root(data_dir, split):
    data_dir = Path(data_dir)
    candidates = {
        "labeled": [data_dir / "labeled"],
        "unlabeled": [data_dir / "unlabeled"],
        "val": [data_dir / "val"],
        "test": [
            data_dir / "TestDataset" / "CVC-ColonDB",
            data_dir / "test",
            data_dir / "TestDataset",
        ],
    }
    for root in candidates.get(split, [data_dir / split]):
        if (root / "image").is_dir():
            return root
    expected = " or ".join(str(path / "image") for path in candidates.get(split, [data_dir / split]))
    raise FileNotFoundError(f"Could not find image folder for split '{split}'. Expected {expected}.")


def _list_images(image_dir):
    files = [path for path in Path(image_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=_sort_key)


def _load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def _resize_pair(image, mask, image_size):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _paired_geometric_augment(image, mask):
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() < 0.25:
        k = random.randint(1, 3)
        image = np.rot90(image, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))
    return image.copy(), mask.copy()


def _photometric_augment(image, strong=False):
    image = image.astype(np.float32) / 255.0

    if random.random() < (0.8 if strong else 0.35):
        contrast = random.uniform(0.65, 1.35) if strong else random.uniform(0.85, 1.15)
        brightness = random.uniform(-0.18, 0.18) if strong else random.uniform(-0.06, 0.06)
        image = image * contrast + brightness

    if strong and random.random() < 0.35:
        noise = np.random.normal(0.0, 0.04, image.shape).astype(np.float32)
        image = image + noise

    if strong and random.random() < 0.25:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    if strong and random.random() < 0.5:
        h, w = image.shape[:2]
        cut_h = random.randint(max(1, h // 16), max(2, h // 6))
        cut_w = random.randint(max(1, w // 16), max(2, w // 6))
        y = random.randint(0, max(0, h - cut_h))
        x = random.randint(0, max(0, w - cut_w))
        image[y : y + cut_h, x : x + cut_w] = 0.0

    return np.clip(image, 0.0, 1.0)


def _image_to_tensor(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float()


def _mask_to_tensor(mask):
    mask = (mask.astype(np.float32) / 255.0) >= 0.5
    mask = np.ascontiguousarray(mask.astype(np.float32)[None, ...])
    return torch.from_numpy(mask).float()


class ColonDBDataset(Dataset):
    """Dataset for the prepared 10/70/10/10 CVC-ColonDB split.

    Expected layout:
      dataset/labeled/image, dataset/labeled/mask
      dataset/unlabeled/image
      dataset/val/image, dataset/val/mask
      dataset/TestDataset/CVC-ColonDB/image, dataset/TestDataset/CVC-ColonDB/mask
    """

    def __init__(self, data_dir, split, image_size=256, augment=False, return_name=True):
        if split not in {"labeled", "unlabeled", "val", "test"}:
            raise ValueError("split must be one of: labeled, unlabeled, val, test")

        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.return_name = return_name
        self.root = _find_split_root(data_dir, split)
        self.image_dir = self.root / "image"
        self.mask_dir = self.root / "mask"
        self.image_paths = _list_images(self.image_dir)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.mask_paths = None
        if split != "unlabeled":
            if not self.mask_dir.is_dir():
                raise FileNotFoundError(f"Missing mask folder for split '{split}': {self.mask_dir}")
            self.mask_paths = []
            for image_path in self.image_paths:
                mask_path = self.mask_dir / image_path.name
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
                self.mask_paths.append(mask_path)

        if split == "unlabeled":
            print(f"unlabeled total {len(self.image_paths)} samples")
        else:
            print(f"{split} total {len(self.image_paths)} paired samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = _load_rgb(image_path)
        original_size = torch.tensor([image.shape[0], image.shape[1]], dtype=torch.long)

        if self.split == "unlabeled":
            image, _ = _resize_pair(image, None, self.image_size)
            weak = _photometric_augment(image, strong=False)
            strong = _photometric_augment(image, strong=True)
            sample = {
                "image_weak": _image_to_tensor(weak),
                "image_strong": _image_to_tensor(strong),
            }
        else:
            mask = _load_mask(self.mask_paths[idx])
            image, mask = _resize_pair(image, mask, self.image_size)
            if self.augment:
                image, mask = _paired_geometric_augment(image, mask)
                image = _photometric_augment(image, strong=False)
            sample = {
                "image": _image_to_tensor(image),
                "mask": _mask_to_tensor(mask),
            }

        if self.return_name:
            sample["name"] = image_path.name
            sample["original_size"] = original_size
        return sample
