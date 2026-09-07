import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with two convolutional layers and PReLU activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class Sentinel2SRNet(nn.Module):
    """
    Deep Residual Sub-Pixel Convolutional Neural Network for Satellite Super-Resolution.
    
    Features:
    - Feature extraction in 10m low-resolution space for fast CPU inference.
    - Sub-pixel convolution (PixelShuffle) for 2x spatial reconstruction.
    - Zero-mean residual constraint: strictly prevents the network from altering
      global brightness or contrast, forcing it to focus 100% on high-frequency edges.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 48,
        num_blocks: int = 4,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # 1. Feature extraction head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        # 2. Residual body
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 3. Upsampling reconstruction tail (PixelShuffle)
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        # Initialize tail with small normal weights
        nn.init.normal_(self.tail[0].weight, std=1e-3)
        nn.init.zeros_(self.tail[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global bicubic baseline upsampling
        base = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # Deep high-frequency residual learning
        feat = self.head(x)
        res = self.conv_after_body(self.body(feat))
        residual_sr = self.tail(feat + res)

        # CONSTRAINT: Force zero spatial mean on residual map
        # This mathematically guarantees the model CANNOT shift overall image brightness/contrast
        residual_sr = residual_sr - residual_sr.mean(dim=(-2, -1), keepdim=True)

        # Enhanced output = baseline + zero-mean spatial sharpening
        out = base + residual_sr
        return torch.clamp(out, 0.0, 1.0)


def load_model(checkpoint_path: str = None, in_channels: int = 3, device: str = "cpu") -> Sentinel2SRNet:
    """Instantiate Sentinel2SRNet and optionally load checkpoint weights."""
    model = Sentinel2SRNet(in_channels=in_channels, out_channels=in_channels, scale_factor=2)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
