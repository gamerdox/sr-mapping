"""
SlimDiffSR: Lightweight Remote Sensing Image Super-Resolution via Distilled Diffusion Architecture.
Based on Wang et al. (Wuhan University, 2026) - https://github.com/wwangcece/SlimDiffSR

Key Architectural Components:
1. Directional Convolution Blocks (DCB): Frequency & direction-separable kernels (1x3 + 3x1 + 3x3)
   tailored for remote sensing geometric structures (roads, building perimeters, agricultural strips).
2. Residual Channel Attention Blocks (RCAB): Squeeze-and-excitation channel attention targeting
   informative high-frequency spectral bands.
3. Residual Groups (RG) with deep and shallow skip connections.
4. Single-step sub-pixel reconstruction tail for ultra-fast feed-forward super-resolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalConvBlock(nn.Module):
    """
    Directional & Frequency-Separable Convolution (DCB).
    Captures strong directional textures in satellite imagery (roads, runways, coastlines)
    using parallel 1x3 and 3x1 convolutions combined with standard 3x3 convolution.
    """

    def __init__(self, channels: int, bias: bool = True):
        super(DirectionalConvBlock, self).__init__()
        self.conv_std = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.conv_horiz = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.conv_vert = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=bias)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fuse multi-directional feature projections
        feat = self.conv_std(x) + self.conv_horiz(x) + self.conv_vert(x)
        feat = self.act(feat)
        out = self.conv_out(feat)
        return out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention Module."""

    def __init__(self, num_features: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        reduced_features = max(num_features // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, reduced_features, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_features, num_features, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block with Directional Convolutions.
    RCAB = DCB + Channel Attention + Identity Shortcut with 0.2 residual scaling.
    """

    def __init__(self, num_features: int = 64, res_scale: float = 0.2):
        super(RCAB, self).__init__()
        self.res_scale = res_scale
        self.dcb = DirectionalConvBlock(num_features)
        self.ca = ChannelAttention(num_features, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.dcb(x)
        res = self.ca(res)
        return x + res * self.res_scale


class ResidualGroup(nn.Module):
    """Residual Group containing N RCAB blocks with a group-level shortcut."""

    def __init__(self, num_features: int = 64, num_blocks: int = 3):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(*[RCAB(num_features) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv(self.blocks(x))
        return x + res


class SlimDiffSR(nn.Module):
    """
    SlimDiffSR Student Super-Resolution Network for Remote Sensing Imagery.
    Designed for fast single-step inference with deep directional and channel attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_groups: int = 4,
        num_blocks_per_group: int = 3,
        scale: int = 2,
    ):
        super(SlimDiffSR, self).__init__()
        self.scale = scale

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # 2. Deep Feature Extraction (Residual Groups with Directional Convolutions)
        self.groups = nn.Sequential(
            *[ResidualGroup(num_features, num_blocks=num_blocks_per_group) for _ in range(num_groups)]
        )
        self.conv_after_groups = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # 3. High-Resolution Upsampling Reconstruction Tail (PixelShuffle)
        if scale == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True),
            )
        elif scale == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic baseline reference
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Deep directional feature extraction
        feat_shallow = self.conv_first(x)
        feat_deep = self.conv_after_groups(self.groups(feat_shallow))
        feat_combined = feat_shallow + feat_deep

        # Residual reconstruction
        residual = self.tail(feat_combined)

        out = base + residual
        return torch.clamp(out, 0.0, 1.0)
