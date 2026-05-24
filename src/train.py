"""Train the baseline CNN with class-weighted loss and checkpointing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline_model import create_model
from src.dataloader import get_dataloaders
from src.metrics import class_weights_from_dataset, collect_predictions, compute_metrics


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_HISTORY_PATH = Path("results/training_history.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    return avg_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = resolve_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    class_weights = class_weights_from_dataset(train_loader.dataset).to(device)
    print(f"Class weights ({', '.join(class_names)}): {class_weights.tolist()}")

    model = create_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    checkpoint_path = args.checkpoint_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, criterion, device, class_names
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, class_names, epoch, val_metrics, args)
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {args.patience} epochs without val loss improvement."
                )
                break

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    with args.history_path.open("w", encoding="utf-8") as history_file:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            history_file,
            indent=2,
        )

    print(f"Training finished. Best epoch: {best_epoch}. History saved to {args.history_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline pneumonia CNN.")
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, type=Path)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH, type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
