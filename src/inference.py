import argparse
from pathlib import Path

import numpy as np
import torch

from model import BaselineRestoration


def load_model(checkpoint_path, device):
    model = BaselineRestoration().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def restore_images(
    model,
    input_dir,
    output_dir,
    device
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:
        raise FileNotFoundError(
            f"No .npy files found in {input_dir}"
        )

    print("Input images:", len(input_files))

    with torch.no_grad():

        for index, input_file in enumerate(input_files):

            image = np.load(
                input_file
            ).astype(np.float32)

            image = torch.from_numpy(image)

            if image.ndim == 2:
                image = image.unsqueeze(0)

            image = image.unsqueeze(0).to(device)

            restored = model(image)

            restored = torch.clamp(
                restored,
                0.0,
                1.0
            )

            restored = (
                restored
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            output_file = (
                output_dir / input_file.name
            )

            np.save(
                output_file,
                restored
            )

            if (index + 1) % 100 == 0:
                print(
                    f"Processed "
                    f"{index + 1}/{len(input_files)}"
                )

    print("Inference completed.")
    print("Output directory:", output_dir)


def main():

    parser = argparse.ArgumentParser(
        description="Semiconductor image restoration inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing input NoisyLR .npy files"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Folder for restored .npy files"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained .pth checkpoint"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = load_model(
        args.checkpoint,
        device
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    restore_images(
        model,
        args.input,
        args.output,
        device
    )


if __name__ == "__main__":
    main()
