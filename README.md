# Forest vs Desert Classifier

Computer Vision practice task — image classification using a custom PyTorch pipeline, a shallow CNN trained from scratch, and transfer learning with ResNet18.

## Dataset
[Forest vs Desert](https://www.kaggle.com/datasets/akhiljethwa/forest-vs-desert) — 802 images, 2 classes (Forest, Desert), 80/20 train/val split.

## What's inside
- **Custom Dataset & DataLoader** — parses the image directory from scratch, no `ImageFolder` shortcut
- **Albumentations augmentations** — random crop, horizontal flip, color jitter, Gaussian noise
- **Shallow CNN** — trained from scratch (BatchNorm + AdaptiveAvgPool), 8 epochs
- **Transfer learning** — ResNet18 (frozen backbone, fine-tuned classification head), 10 epochs
- **Prediction demo** — visualizes model predictions across sample validation images

## Results
| Model | Best Validation Accuracy |
|---|---|
| Shallow CNN (scratch) | 99.38% |
| ResNet18 (transfer learning) | 98.75% |

## Run it
Open `forestvsdesert.ipynb` in Kaggle (with the dataset attached and GPU enabled) or Colab, and run top to bottom.
