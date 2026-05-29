# EDA Insights — Pediatric Chest X-Ray Dataset

Summary of findings from `eda.ipynb`. Source: [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## Dataset Overview

| Metric | Value |
|---|---|
| Total images | 5,856 |
| Classes | NORMAL, PNEUMONIA |
| Current Splits | `train`, `val`, `test` |

---

## 1. Class Distribution

### Overall

| Class | Count | Share |
|---|---|---|
| NORMAL | 1,583 | 27.0% |
| PNEUMONIA | 4,273 | 73.0% |

The dataset is **heavily skewed toward pneumonia**. A naive classifier that always predicts pneumonia would achieve ~73% accuracy, so accuracy alone is a misleading metric. Precision, recall, F1, and a confusion matrix are essential.

### Per split

| Split | NORMAL | PNEUMONIA | Total | Pneumonia % |
|---|---|---|---|---|
| train | 1,341 | 3,875 | 5,216 | 74.3% |
| val | 8 | 8 | 16 | 50.0% |
| test | 234 | 390 | 624 | 62.5% |

**Key takeaways:**

- The official **validation set has only 16 images** — far too small for reliable hyperparameter tuning or early stopping. As planned in the project scope, merge `val` into `train` and create a new stratified 80/20 train/validation split.
- Imbalance persists across all splits; no split is balanced.
- The test set (624 images) is the only split large enough for meaningful final evaluation.

---

## 2. Image Properties

### Dimensions

| Stat | Width (px) | Height (px) | Aspect ratio | File size (KB) |
|---|---|---|---|---|
| Mean | 1,328 | 971 | 1.44 | 206 |
| Median | 1,281 | 888 | 1.42 | 94 |
| Min | 384 | 127 | 0.84 | 5 |
| Max | 2,916 | 2,713 | 3.38 | 2,358 |

- **4,803 unique (width, height) pairs** — images vary widely in resolution and acquisition format.
- Median dimensions (1,281 × 888) are much larger than typical CNN inputs (224 × 224), so **downsampling is required** before training.
- Aspect ratios cluster near 1.4 but range from 0.84 to 3.38, so consider whether to resize with distortion, pad, or center-crop.

### Color mode

| Mode | Count | Share |
|---|---|---|
| Grayscale (`L`) | 5,573 | 95.2% |
| RGB | 283 | 4.8% |

Most images are grayscale as expected for chest X-rays, but **283 RGB-encoded files** exist. The preprocessing pipeline should convert all images to a consistent channel format (grayscale or 3-channel replication for transfer learning).

---

## 3. Pixel-Level Analysis

### Per-class intensity statistics

| Class | Mean pixel intensity | Std (within-image) |
|---|---|---|
| NORMAL | 122.6 | 61.3 |
| PNEUMONIA | 122.8 | 55.4 |

Mean brightness is nearly identical between classes, so a model cannot rely on global brightness alone. Pneumonia images tend to have **slightly lower within-image contrast** (std 55.4 vs. 61.3), which may reflect denser opacities reducing intensity variation.

Histogram overlays and mean composite images (see notebook) show subtle class-level differences, but no single pixel statistic cleanly separates the classes — spatial patterns learned by a CNN will matter more.

---

## 4. Data Quality

| Check | Result |
|---|---|
| Corrupted / unreadable files | 0 |
| Exact duplicate groups (MD5) | 30 groups (62 files) |
| Cross-split exact duplicates | 0 |
| Perceptual duplicate groups (identical average hash) | 218 groups |
| Size outliers (outside 1st–99th percentile) | 148 images |
| Intensity outliers (outside 1st–99th percentile) | 118 images |

**Key takeaways:**

- No corrupted files — the dataset is fully loadable.
- **No exact duplicates span train/test splits**, so there is no obvious byte-level data leakage.
- 30 exact-duplicate groups (62 files total) within splits should be reviewed; keeping duplicates in both train and validation could inflate metrics.
- 218 perceptual-duplicate groups suggest many visually similar images; a full near-duplicate scan (Hamming distance ≤ 5) is available in the notebook for deeper investigation.
- Size and intensity outliers are not necessarily errors but warrant spot-checking via the notebook's image viewer.
