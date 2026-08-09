from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):
    def __init__(self, data_root, split_file):
        self.data_root = Path(data_root)

        self.gt_dir = self.data_root / "train" / "GT"
        self.noisy_dir = self.data_root / "train" / "NoisyLR"

        split_file = Path(split_file)

        with open(split_file, "r") as f:
            self.ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        noisy = np.load(
            self.noisy_dir / f"{sample_id}.npy"
        ).astype(np.float32)

        gt = np.load(
            self.gt_dir / f"{sample_id}.npy"
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy)
        gt = torch.from_numpy(gt)

        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(0)

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)

        return noisy, gt
