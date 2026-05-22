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
