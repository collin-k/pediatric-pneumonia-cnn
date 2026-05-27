"""Train-time augmentation for grayscale pediatric chest X-ray images.

Ranges follow prompts/augment.md: small rotation, brightness/contrast jitter,
and mild zoom. Applied only to the training split before tensor conversion.
"""

from __future__ import annotations

from torchvision import transforms

MAX_ROTATION_DEGREES = 15
BRIGHTNESS_JITTER = 0.15
CONTRAST_JITTER = 0.15
SCALE_RANGE = (0.9, 1.1)


def get_train_augmentation_steps() -> list:
    """Return PIL transforms for train-only augmentation."""
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=BRIGHTNESS_JITTER,
                    contrast=CONTRAST_JITTER,
                )
            ],
            p=0.5,
        ),
        transforms.RandomAffine(
            degrees=MAX_ROTATION_DEGREES,
            scale=SCALE_RANGE,
            fill=0,
        ),
    ]


def build_train_augmentations() -> transforms.Compose:
    """Compose train-only augmentations into a single transform."""
    return transforms.Compose(get_train_augmentation_steps())
