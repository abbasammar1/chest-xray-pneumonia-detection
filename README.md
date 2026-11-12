# Chest X-Ray Pneumonia Detection (CNN from scratch)

A convolutional neural network built from scratch in TensorFlow/Keras that classifies chest X-rays as **NORMAL** or **PNEUMONIA**.

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle (5,216 training and 624 test images). It isn't included in this repo.

## Pipeline

1. **Data:** 80/20 train/validation split of the training folder. The test folder is kept completely separate.
2. **Augmentation on training images only** (rotation, zoom, horizontal flip). Validation and test images are only rescaled, so their scores reflect real, undistorted X-rays.
3. **Model:** 2 × (Conv2D + MaxPooling) → Dense(128) → Dropout(0.5) → sigmoid output, trained for 5 epochs with Adam and binary cross-entropy.
4. **Evaluation on the unseen test set:** accuracy, confusion matrix, and precision / recall / F1 per class.

## Results

| Metric | Value |
|---|---|
| Validation accuracy (no augmentation) | 92.0% |
| **Test accuracy** | **74.4%** |
| **Pneumonia recall (sensitivity)** | **98.7%** (missed 5 of 390 cases) |
| Normal recall (specificity) | 33.8% (155 of 234 healthy X-rays flagged) |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| PNEUMONIA | 0.713 | 0.987 | 0.828 | 390 |
| NORMAL | 0.940 | 0.338 | 0.497 | 234 |

**What this means**

- The model **almost never misses pneumonia**, which is the most important property for a screening tool, since a missed case is the most dangerous error.
- But it is **biased towards "pneumonia"**: two out of three healthy patients get flagged. The training data is 74% pneumonia, so the model learned to lean that way. The 74% accuracy hides how lopsided it is, which is why accuracy alone isn't enough for a medical task.
- **Validation (92%) vs test (74%):** the test set is distributed differently from the training folder (38% NORMAL vs 26%), so validation scores from the training folder are optimistic.

## Next steps

- Class weights or oversampling NORMAL to reduce the bias
- Tune the decision threshold on validation data instead of using 0.5
- Transfer learning (MobileNetV2 / DenseNet121) and EarlyStopping

## Ethics

This is a learning project, not a diagnostic tool. A real system would need validation across hospitals and equipment, clinician oversight, and clear accountability. See the conclusion in [`pneumonia.ipynb`](pneumonia.ipynb).

## Run it

```bash
pip install tensorflow scikit-learn matplotlib
# download the Kaggle dataset into chest_xray/ (train/ and test/ folders), then:
jupyter notebook pneumonia.ipynb
```

Trained on an Apple M-series laptop (about 35 s per epoch).

## Changelog

- **Nov 2026 (after mentor review):** validation now uses its own generator without augmentation (before, validation images were augmented like training images); added a confusion matrix and per-class precision/recall/F1 on the test set; fixed random seeds. Re-training gave 74.4% test accuracy (previously reported 79%, measured with the older setup).
