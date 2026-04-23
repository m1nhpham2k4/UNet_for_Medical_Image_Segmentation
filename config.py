from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "dataset"
    output_dir: str = "runs/semisup_unet"
    image_size: int = 256
    epochs: int = 50
    batch_size: int = 4
    unlabeled_batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    base_channels: int = 32
    unsup_weight: float = 0.5
    unsup_start_epoch: int = 1
    confidence_threshold: float = 0.7
    seed: int = 42
    amp: bool = False
    device: str = "auto"
    threshold: float = 0.5
