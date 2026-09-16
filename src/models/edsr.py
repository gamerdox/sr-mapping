import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual Block without Batch Normalization, with residual scaling."""
    def __init__(self, n_feats: int = 64, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    """
    Enhanced Deep Residual Networks for Single Image Super-Resolution
    Configured for satellite imagery super-resolution (Lim et al., 2017).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feats: int = 64, n_resblocks: int = 8, scale: int = 2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)

        body = [ResBlock(n_feats, res_scale=0.1) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        res = res + feat
        return torch.clamp(self.tail(res), 0.0, 1.0)
