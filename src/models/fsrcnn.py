import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    """
    FSRCNN (Fast Super-Resolution Convolutional Neural Network) - Dong et al., ECCV 2016
    Uses an hourglass feature extraction, shrinking, non-linear mapping, 
    expanding, and deconvolution (sub-pixel transpose conv) structure.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, scale_factor: int = 2, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        # 1. Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )

        # 2. Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        # 3. Non-linear mapping (m layers)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ])
        self.mapping = nn.Sequential(*mapping_layers)

        # 4. Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        # 5. Deconvolution / Upsampling
        self.deconv = nn.ConvTranspose2d(
            d, out_channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)
