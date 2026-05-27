"""Evaluate a trained checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)
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
    normalize: bool = False,
    title: str | None = None,
) -> None:
    matrix = np.asarray(confusion_matrix_values)
    if normalize:
        matrix = matrix.astype(float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.divide(matrix, row_sums, where=row_sums > 0)
        fmt = ".2f"
        cmap = "Purples"
        default_title = "Confusion Matrix (Row-normalized)"
    else:
        matrix = matrix.astype(int)
        fmt = "d"
        cmap = "Blues"
        default_title = "Confusion Matrix"

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
        cbar=True,
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title or default_title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _binary_positive_index(class_names: list[str]) -> int:
    """Return the index of the positive class for binary tasks.

    Prefers a class containing 'pneumonia' so plots remain interpretable across
    capitalization changes; otherwise defaults to index 1 (sklearn convention).
    """
    for idx, name in enumerate(class_names):
        if "pneumonia" in name.lower():
            return idx
    return 1


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> dict:
    """Plot the ROC curve(s) and return per-class AUCs."""
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")

    auc_scores: dict[str, float] = {}

    if len(class_names) == 2:
        positive_idx = _binary_positive_index(class_names)
        positive_name = class_names[positive_idx]
        y_true_binary = (y_true == positive_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, positive_idx])
        roc_auc = auc(fpr, tpr)
        auc_scores[positive_name] = float(roc_auc)
        axis.plot(fpr, tpr, label=f"{positive_name} (AUC = {roc_auc:.3f})")
    else:
        for class_idx, class_name in enumerate(class_names):
            y_true_binary = (y_true == class_idx).astype(int)
            if y_true_binary.sum() == 0 or y_true_binary.sum() == y_true_binary.size:
                continue
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            auc_scores[class_name] = float(roc_auc)
            axis.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return auc_scores


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> dict:
    """Plot the precision-recall curve(s) and return per-class average precision."""
    figure, axis = plt.subplots(figsize=(6, 5))
    ap_scores: dict[str, float] = {}

    if len(class_names) == 2:
        positive_idx = _binary_positive_index(class_names)
        positive_name = class_names[positive_idx]
        y_true_binary = (y_true == positive_idx).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, positive_idx])
        ap_value = float(auc(recall, precision))
        ap_scores[positive_name] = ap_value
        baseline = float(y_true_binary.mean())
        axis.plot(recall, precision, label=f"{positive_name} (AP ≈ {ap_value:.3f})")
        axis.hlines(
            baseline,
            xmin=0.0,
            xmax=1.0,
            colors="grey",
            linestyles="--",
            label=f"Baseline (prevalence = {baseline:.3f})",
        )
    else:
        for class_idx, class_name in enumerate(class_names):
            y_true_binary = (y_true == class_idx).astype(int)
            if y_true_binary.sum() == 0 or y_true_binary.sum() == y_true_binary.size:
                continue
            precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, class_idx])
            ap_value = float(auc(recall, precision))
            ap_scores[class_name] = ap_value
            axis.plot(recall, precision, label=f"{class_name} (AP ≈ {ap_value:.3f})")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend(loc="lower left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return ap_scores


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

    y_true, y_pred, y_probs = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names, y_probs=y_probs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.split

    metrics_path = args.output_dir / f"{prefix}_metrics.txt"
    json_path = args.output_dir / f"{prefix}_metrics.json"
    confusion_matrix_path = args.output_dir / f"{prefix}_confusion_matrix.png"
    normalized_confusion_matrix_path = (
        args.output_dir / f"{prefix}_confusion_matrix_normalized.png"
    )
    roc_curve_path = args.output_dir / f"{prefix}_roc_curve.png"
    pr_curve_path = args.output_dir / f"{prefix}_precision_recall_curve.png"

    save_metrics(metrics, class_names, metrics_path)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, confusion_matrix_path)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        normalized_confusion_matrix_path,
        normalize=True,
    )
    plot_roc_curve(y_true, y_probs, class_names, roc_curve_path)
    plot_precision_recall_curve(y_true, y_probs, class_names, pr_curve_path)

    if args.history_path.exists():
        plot_training_history(
            args.history_path,
            args.output_dir / "training_curves.png",
        )

    print(format_metrics(metrics, class_names))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {confusion_matrix_path}")
    print(f"Saved normalized confusion matrix to {normalized_confusion_matrix_path}")
    print(f"Saved ROC curve to {roc_curve_path}")
    print(f"Saved Precision-Recall curve to {pr_curve_path}")

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
