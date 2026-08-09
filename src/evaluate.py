import argparse
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def evaluate_predictions(pred_dir, target_dir):
    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)

    prediction_files = sorted(
        pred_dir.glob("*.npy")
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"No .npy predictions found in {pred_dir}"
        )

    psnr_values = []
    ssim_values = []

    for pred_file in prediction_files:

        target_file = (
            target_dir / pred_file.name
        )

        if not target_file.exists():
            print(
                f"Skipping {pred_file.name}: "
                "target not found"
            )
            continue

        prediction = np.load(
            pred_file
        ).astype(np.float32)

        target = np.load(
            target_file
        ).astype(np.float32)

        prediction = np.clip(
            prediction,
            0.0,
            1.0
        )

        target = np.clip(
            target,
            0.0,
            1.0
        )

        psnr = peak_signal_noise_ratio(
            target,
            prediction,
            data_range=1.0
        )

        ssim = structural_similarity(
            target,
            prediction,
            data_range=1.0
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    if not psnr_values:
        raise RuntimeError(
            "No matching prediction/target pairs found."
        )

    print(
        "Images evaluated:",
        len(psnr_values)
    )

    print(
        "Mean PSNR:",
        np.mean(psnr_values)
    )

    print(
        "Mean SSIM:",
        np.mean(ssim_values)
    )

    print(
        "Min PSNR:",
        np.min(psnr_values)
    )

    print(
        "Max PSNR:",
        np.max(psnr_values)
    )


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate restored semiconductor images"
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Folder containing predicted .npy files"
    )

    parser.add_argument(
        "--targets",
        required=True,
        help="Folder containing ground-truth .npy files"
    )

    args = parser.parse_args()

    evaluate_predictions(
        args.predictions,
        args.targets
    )


if __name__ == "__main__":
    main()
