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

Baseline CNN:

```bash
python -m src.train --model baseline --epochs 20 --patience 5 --num-workers 0
```

Fine-tune ImageNet pretrained models (lower backbone LR, ImageNet normalization):

```bash
python -m src.train --model resnet18 --epochs 15 --lr 1e-3 --num-workers 0
python -m src.train --model efficientnet_b0 --epochs 15 --lr 1e-3 --num-workers 0
```

Checkpoints are saved under `checkpoints/<model>/best_model.pt`. Training history goes to `results/<model>/training_history.json`.

## Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/resnet18/best_model.pt --split test --num-workers 0
```

Writes metrics to `results/` (use `--output-dir results/resnet18` to keep models separate).
