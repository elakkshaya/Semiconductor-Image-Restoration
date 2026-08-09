import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class BaselineRestoration(nn.Module):
    def __init__(self, channels=64, num_blocks=8):
        super().__init__()

        self.head = nn.Conv2d(
            1, channels, 3, padding=1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            channels, channels, 3, padding=1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            channels, 1, 3, padding=1
        )

    def forward(self, x):
        x = self.head(x)

        residual = x

        x = self.body(x)
        x = self.body_conv(x)
        x = x + residual

        x = self.upsample(x)
        x = self.tail(x)

        return x


if __name__ == "__main__":

    model = BaselineRestoration()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", parameters)

    x = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
