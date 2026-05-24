"""Evaluate a trained checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.dataloader import build_transforms
from src.metrics import collect_predictions, compute_metrics, format_metrics, save_metrics
from src.models import create_model, uses_pretrained_backbone


DEFAULT_RESULTS_DIR = Path("results")


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_eval_loader(
    data_dir: Path,
    split: str,
    batch_size: int,
    img_size: int,
    num_workers: int,
    pretrained: bool = False,
) -> tuple[DataLoader, list[str]]:
    _, eval_transform = build_transforms(
        img_size=img_size,
        pretrained=pretrained,
        augment=False,
    )

    dataset = datasets.ImageFolder(data_dir / split, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, dataset.classes


def resolve_model_name(checkpoint: dict) -> str:
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint.get("args", {}).get("model", "baseline")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model_name = resolve_model_name(checkpoint)
    model = create_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["model"] = model_name
    return model, class_names, checkpoint


def plot_confusion_matrix(
    confusion_matrix_values: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_training_history(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        return

    with history_path.open(encoding="utf-8") as history_file:
        history_data = json.load(history_file)

    epochs = [record["epoch"] for record in history_data["history"]]
    train_loss = [record["train_loss"] for record in history_data["history"]]
    val_loss = [record["val_loss"] for record in history_data["history"]]
    val_f1 = [record["val_f1_macro"] for record in history_data["history"]]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Val macro F1", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1")
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device()
    print(f"Using device: {device}")

    model, class_names, checkpoint = load_checkpoint(args.checkpoint, device)
    model_name = resolve_model_name(checkpoint)
    loader, loader_class_names = get_eval_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        pretrained=uses_pretrained_backbone(model_name),
    )

    if loader_class_names != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match dataset classes {loader_class_names}."
        )

    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.split

    metrics_path = args.output_dir / f"{prefix}_metrics.txt"
    json_path = args.output_dir / f"{prefix}_metrics.json"
    confusion_matrix_path = args.output_dir / f"{prefix}_confusion_matrix.png"

    save_metrics(metrics, class_names, metrics_path)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, confusion_matrix_path)

    if args.history_path.exists():
        plot_training_history(
            args.history_path,
            args.output_dir / "training_curves.png",
        )

    print(format_metrics(metrics, class_names))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {confusion_matrix_path}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pneumonia CNN checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        type=Path,
        help="Path to checkpoint (e.g. checkpoints/resnet18/best_model.pt)",
    )
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, type=Path)
    parser.add_argument("--history-path", default=Path("results/training_history.json"), type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
