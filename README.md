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
````

The dataset itself is not included in this repository.

---

## 3. Model

The final model is called **V1 Extended**.

Architecture:

```text
128×128 input
      │
      ▼
64-channel convolution
      │
      ▼
8 Residual Blocks
      │
      ▼
Body Convolution
      │
      ▼
Global Residual Connection
      │
      ▼
PixelShuffle ×2
      │
      ▼
Output Convolution
      │
      ▼
256×256 restored image
```

### Model details

* Input channels: 1
* Feature channels: 64
* Residual blocks: 8
* Upscaling factor: 2×
* Upsampling: PixelShuffle
* Loss: L1
* Optimizer: AdamW
* Total parameters: **776,705**

---

## 4. Training

The model was initially trained for 10 epochs.

A further 5 epochs of fine-tuning were performed using a lower learning rate.

The final checkpoint corresponds to:

```text
Epoch: 15
Validation L1: 0.031402
```

The final checkpoint is:

```text
checkpoints/FINAL_v1_epoch15.pth
```

---

## 5. Results

Final validation results on 320 validation images:

| Metric     |         Result |
| ---------- | -------------: |
| L1 Loss    |   **0.031402** |
| PSNR       | **28.1759 dB** |
| SSIM       |     **0.7548** |
| Parameters |    **776,705** |

The final predictions were independently saved and evaluated to
verify that the reported metrics are reproducible.

---

## 6. Repository Structure

```text
Semiconductor-Image-Restoration/
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── inference.py
│   └── evaluate.py
│
├── notebooks/
│   └── training.ipynb
│
├── results/
│   └── validation_comparison.png
│
├── checkpoints/
│   └── FINAL_v1_epoch15.pth
│
├── README.md
└── requirements.txt
```

---

## 7. Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 8. Inference

The inference script accepts a folder containing `.npy` NoisyLR
images and produces restored 256×256 `.npy` images.

From the `src` directory:

```bash
python inference.py \
    --input /path/to/NoisyLR \
    --output /path/to/restored \
    --checkpoint /path/to/FINAL_v1_epoch15.pth
```

The script automatically uses CUDA when an NVIDIA GPU is available.

---

## 9. Evaluation

When ground-truth images are available, evaluate the predictions using:

```bash
python evaluate.py \
    --predictions /path/to/restored \
    --targets /path/to/GT
```

The evaluation reports:

* Mean PSNR
* Mean SSIM
* Minimum PSNR
* Maximum PSNR

---

## 10. Validation

The final model was evaluated on 320 held-out validation images.

The saved predictions reproduced:

```text
PSNR: 28.175881503981067 dB
SSIM: 0.7548356621041301
L1:   0.03140215165913105
```

---

## 11. Limitations

The currently available dataset contains training and validation data.
A separate test directory was not included in the uploaded dataset.

The inference pipeline is therefore designed so that the actual test
images can be supplied later without changing the model architecture.

---

## 12. Future Work

Possible future improvements include:

* perceptual loss
* stronger attention mechanisms
* additional degradation modelling
* larger training schedules
* multi-scale restoration
* additional quantitative evaluation

---

## 13. Final Model

The final selected model is the **V1 Extended residual restoration
network**, selected after comparing multiple controlled variants.

The final model prioritizes a balance of reconstruction quality,
simplicity, and computational efficiency.


```
