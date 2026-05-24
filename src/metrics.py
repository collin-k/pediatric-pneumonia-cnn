"""Shared classification metrics for training and evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def class_weights_from_dataset(dataset) -> torch.Tensor:
    """Return inverse-frequency class weights for a labeled dataset."""
    counts = Counter(label for _, label in dataset.samples)
    num_classes = len(dataset.classes)
    class_counts = [counts.get(class_idx, 0) for class_idx in range(num_classes)]

    if any(count == 0 for count in class_counts):
        raise ValueError("Each class must have at least one training example.")

    total = sum(class_counts)
    weights = [total / (num_classes * count) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return ground-truth and predicted labels."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute accuracy, per-class, and summary classification metrics."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }


def format_metrics(metrics: dict, class_names: list[str]) -> str:
    """Return a readable metrics summary."""
    lines = [
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['f1_macro']:.4f}",
        f"Weighted F1: {metrics['f1_weighted']:.4f}",
        "",
        "Per-class metrics:",
    ]

    report = metrics["classification_report"]
    for class_name in class_names:
        class_metrics = report[class_name]
        lines.append(
            f"  {class_name}: "
            f"precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, "
            f"f1={class_metrics['f1-score']:.4f}, "
            f"support={int(class_metrics['support'])}"
        )

    return "\n".join(lines)


def save_metrics(metrics: dict, class_names: list[str], output_path: Path) -> None:
    """Write metrics summary and classification report to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_metrics(metrics, class_names) + "\n", encoding="utf-8")
