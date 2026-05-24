"""Prepare stratified train/val/test splits for chest X-ray images."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


RAW_DATA_DIR = Path("data/raw/chest_xray")
PROCESSED_DATA_DIR = Path("data/processed")
ORIGINAL_SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SEED = 42


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Collect all image paths from the original train/val/test folders."""
    images_by_class: dict[str, list[Path]] = {class_name: [] for class_name in CLASSES}

    for original_split in ORIGINAL_SPLITS:
        for class_name in CLASSES:
            class_dir = raw_dir / original_split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing expected directory: {class_dir}")

            images = [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            images_by_class[class_name].extend(sorted(images))

    return images_by_class


def split_class_images(
    images_by_class: dict[str, list[Path]],
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, dict[str, list[Path]]]:
    """Create a stratified split by splitting each class independently."""
    rng = random.Random(seed)
    split_paths = {
        split: {class_name: [] for class_name in CLASSES}
        for split in ("train", "val", "test")
    }

    for class_name in CLASSES:
        images = list(images_by_class[class_name])
        rng.shuffle(images)

        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        split_paths["train"][class_name] = images[:train_end]
        split_paths["val"][class_name] = images[train_end:val_end]
        split_paths["test"][class_name] = images[val_end:]

    return split_paths


def copy_split_images(
    split_paths: dict[str, dict[str, list[Path]]],
    output_dir: Path,
    clear_output: bool = True,
    use_symlinks: bool = False,
) -> None:
    """Copy or symlink split images into ImageFolder-compatible directories."""
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    for split, classes in split_paths.items():
        for class_name, image_paths in classes.items():
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for source_path in image_paths:
                destination_path = class_dir / source_path.name
                if destination_path.exists():
                    destination_path = unique_destination_path(class_dir, source_path)
                if use_symlinks:
                    os.symlink(source_path.resolve(), destination_path)
                else:
                    shutil.copy2(source_path, destination_path)


def unique_destination_path(class_dir: Path, source_path: Path) -> Path:
    """Avoid overwriting files when different original splits share a name."""
    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1

    while True:
        candidate = class_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_split_summary(
    split_paths: dict[str, dict[str, list[Path]]],
    output_dir: Path,
    filename: str = "split_summary.csv",
) -> Path:
    """Write counts by split and class to a CSV file."""
    summary_path = output_dir / filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "class", "count"])
        writer.writeheader()

        for split in ("train", "val", "test"):
            for class_name in CLASSES:
                writer.writerow(
                    {
                        "split": split,
                        "class": class_name,
                        "count": len(split_paths[split][class_name]),
                    }
                )

    return summary_path


def prepare_data(
    raw_dir: str | Path = RAW_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    seed: int = DEFAULT_SEED,
    clear_output: bool = True,
    use_symlinks: bool = False,
) -> Path:
    """Merge original splits and create an 80/10/10 stratified split."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    images_by_class = collect_images(raw_dir)
    split_paths = split_class_images(images_by_class, seed=seed)
    copy_split_images(
        split_paths,
        output_dir,
        clear_output=clear_output,
        use_symlinks=use_symlinks,
    )
    return write_split_summary(split_paths, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare processed chest X-ray splits.")
    parser.add_argument("--raw-dir", default=RAW_DATA_DIR, type=Path)
    parser.add_argument("--output-dir", default=PROCESSED_DATA_DIR, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the output directory before copying files.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (faster; raw data must stay in place).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_csv = prepare_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        clear_output=not args.no_clear,
        use_symlinks=args.symlink,
    )
    print(f"Saved split summary to {summary_csv}")
