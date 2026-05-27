"""DataLoader utilities for processed chest X-ray images."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.augment import get_train_augmentation_steps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, pretrained: bool, augment: bool) -> tuple:
    """Build train and eval transforms; use ImageNet stats for pretrained models."""
    if pretrained:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    shared = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    train_steps = list(shared)
    if augment:
        train_steps.extend(get_train_augmentation_steps())
    train_steps.extend([transforms.ToTensor(), normalize])
    eval_steps = [*shared, transforms.ToTensor(), normalize]

    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def get_dataloaders(
    data_dir: str | Path = "data/processed",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
    pretrained: bool = False,
    augment: bool = True,
):
    """Return train, validation, and test dataloaders plus class names."""
    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms(
        img_size=img_size,
        pretrained=pretrained,
        augment=augment,
    )

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
