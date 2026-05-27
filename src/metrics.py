"""Shared classification metrics for training and evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return ground-truth labels, predicted labels, and class probabilities."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    probs_array = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    return np.array(all_labels), np.array(all_preds), probs_array


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_probs: np.ndarray | None = None,
) -> dict:
    """Compute accuracy, per-class, and summary classification metrics.

    When ``y_probs`` is provided (shape ``[N, num_classes]``), ROC AUC and
    average precision are also computed (binary: positive-class scalar;
    multiclass: one-vs-rest per class plus macro average).
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
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

    if y_probs is not None and y_probs.size > 0:
        metrics.update(_probability_metrics(y_true, y_probs, class_names))

    return metrics


def _probability_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute ROC AUC and average precision from predicted probabilities."""
    num_classes = len(class_names)
    present_labels = np.unique(y_true)

    if num_classes == 2:
        if present_labels.size < 2:
            return {}
        positive_scores = y_probs[:, 1]
        return {
            "roc_auc": float(roc_auc_score(y_true, positive_scores)),
            "average_precision": float(average_precision_score(y_true, positive_scores)),
        }

    per_class_roc: dict[str, float] = {}
    per_class_ap: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_idx).astype(int)
        if y_true_binary.sum() == 0 or y_true_binary.sum() == y_true_binary.size:
            continue
        per_class_roc[class_name] = float(roc_auc_score(y_true_binary, y_probs[:, class_idx]))
        per_class_ap[class_name] = float(
            average_precision_score(y_true_binary, y_probs[:, class_idx])
        )

    macro_roc = float(np.mean(list(per_class_roc.values()))) if per_class_roc else 0.0
    macro_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {
        "roc_auc_macro": macro_roc,
        "average_precision_macro": macro_ap,
        "roc_auc_per_class": per_class_roc,
        "average_precision_per_class": per_class_ap,
    }


def format_metrics(metrics: dict, class_names: list[str]) -> str:
    """Return a readable metrics summary."""
    lines = [
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['f1_macro']:.4f}",
        f"Weighted F1: {metrics['f1_weighted']:.4f}",
    ]

    if "roc_auc" in metrics:
        lines.append(f"ROC AUC: {metrics['roc_auc']:.4f}")
        lines.append(f"Average Precision: {metrics['average_precision']:.4f}")
    elif "roc_auc_macro" in metrics:
        lines.append(f"Macro ROC AUC (OvR): {metrics['roc_auc_macro']:.4f}")
        lines.append(f"Macro Average Precision (OvR): {metrics['average_precision_macro']:.4f}")

    lines.extend(["", "Per-class metrics:"])

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
