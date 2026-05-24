# Pediatric Pneumonia CNN

## Sanity Checks

Prepare the processed split:

```bash
python -m src.data_prep
```

Verify dataloader and model shapes:

```bash
python - <<'PY'
from src.baseline_model import BaselineCNN
from src.dataloader import get_dataloaders

batch_size = 32
train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=batch_size, num_workers=0)
images, labels = next(iter(train_loader))
print(images.shape)  # torch.Size([batch_size, 3, 224, 224])

model = BaselineCNN(num_classes=len(class_names))
logits = model(images)
print(logits.shape)  # torch.Size([batch_size, 2])
PY
```

## Train

```bash
python -m src.train --epochs 20 --patience 5
```

This uses class-weighted cross-entropy, tracks validation metrics each epoch, saves the best checkpoint to `checkpoints/best_model.pt`, and writes `results/training_history.json`.

## Evaluate

Run on the held-out test split:

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt --split test
```

This writes metrics to `results/test_metrics.txt`, `results/test_metrics.json`, and a confusion matrix plot to `results/test_confusion_matrix.png`.
