# Semiconductor Image Restoration

Deep-learning based restoration of degraded semiconductor microscopy images.

The project restores low-resolution/noisy semiconductor images from
128×128 inputs to 256×256 outputs using a residual convolutional neural
network.

---

## 1. Problem

Semiconductor microscopy images can contain noise and resolution
degradation that make fine structures difficult to analyze.

The objective of this project is to learn a mapping from degraded
128×128 images to high-quality 256×256 ground-truth images.

### Input

- Size: 128×128
- Channels: 1
- Format: `.npy`

### Output

- Size: 256×256
- Channels: 1
- Format: `.npy`

---

## 2. Dataset

The dataset contains:

- 3,200 ground-truth images
- 3,200 corresponding NoisyLR images

The available split is:

- Training: 2,880 images
- Validation: 320 images

Expected dataset structure:

```text
kaggle/
├── splits/
│   ├── train.txt
│   └── val.txt
│
└── train/
    ├── GT/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    │
    └── NoisyLR/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
