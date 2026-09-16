import torch
import torch.nn as nn

class ESPCN(nn.Module):
    """
    ESPCN (Efficient Sub-Pixel Convolutional Network) - Shi et al., CVPR 2016
    Extracts features in low-resolution space and reconstructs high-resolution
    imagery via sub-pixel convolution (PixelShuffle).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.clamp(out, 0.0, 1.0)
