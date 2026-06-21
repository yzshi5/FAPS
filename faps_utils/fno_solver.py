import torch
from neuralop.models import FNO


class FNOSolver(torch.nn.Module):
    """
    Standard Fourier Neural Operator (FNO) solver wrapper.

    This module maps an input field to an output field:
        x: [batch, in_channels, height, width]
        y: [batch, out_channels, height, width]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes=(32, 32),
        hidden_channels: int = 64,
        projection_channels: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()

        self.model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

