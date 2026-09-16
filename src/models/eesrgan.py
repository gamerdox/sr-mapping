"""
Edge-Enhanced Super-Resolution GAN (EESRGAN) for Remote Sensing & Satellite Imagery.
Architecture based on Jakaria et al. (https://github.com/Jakaria08/EESRGAN).

Combines:
1. Deep Residual-in-Residual Dense Blocks (RRDB) for primary super-resolution.
2. Edge Enhancement Network (EESN) with dual Dense + Mask branches.
3. 2nd-order Laplacian edge gradient extraction & learned edge injection.
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    """
    Computes per-channel 2nd-order spatial Laplacian edge gradient in native PyTorch.
    Zero external dependencies, fully differentiable, and GPU-accelerated.
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device
    ).unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


class ResidualDenseBlock_5C(nn.Module):
    """5-Convolution Residual Dense Block with 0.2 residual scaling."""

    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
            m.weight.data *= 0.1
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (contains 3 RDBs)."""

    def __init__(self, nf: int = 64, gc: int = 32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Core Super-Resolution Trunk using RRDB architecture."""

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 8, gc: int = 32, scale: int = 2):
        super(RRDBNet, self).__init__()
        self.scale = scale
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling stage
        if scale == 2:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        elif scale == 4:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        if self.scale == 2:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class BeginEdgeConv(nn.Module):
    """Initial feature extraction on Laplacian edge map with 2x downsampling."""

    def __init__(self, in_nc: int = 3, out_nc: int = 64):
        super(BeginEdgeConv, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 64, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)  # Stride 2 downsamples
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(128, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        return x


class EESNRRDBNet(nn.Module):
    """Dense Sub-Branch inside the Edge Enhancement Network."""

    def __init__(self, in_nc: int = 64, out_nc: int = 256, nf: int = 64, nb: int = 3, gc: int = 32):
        super(EESNRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        out = self.lrelu(self.conv_last(fea))
        return out


class MaskConv(nn.Module):
    """Spatial Mask Attention Branch predicting edge confidence [0, 1]."""

    def __init__(self, in_nc: int = 64, out_nc: int = 256):
        super(MaskConv, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 64, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(128, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        return torch.sigmoid(x)


class FinalConv(nn.Module):
    """Upsampling and edge map reconstruction module."""

    def __init__(self, in_nc: int = 256, out_nc: int = 3):
        super(FinalConv, self).__init__()
        self.upconv1 = nn.Conv2d(in_nc, 128, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.HRconv(x)))
        return x


class EESN(nn.Module):
    """Edge Enhancement Sub-Network (Laplacian -> Dense + Mask -> Final Edge)."""

    def __init__(self, in_nc: int = 3):
        super(EESN, self).__init__()
        self.beginEdgeConv = BeginEdgeConv(in_nc=in_nc, out_nc=64)
        self.denseNet = EESNRRDBNet(in_nc=64, out_nc=256, nf=64, nb=3)
        self.maskConv = MaskConv(in_nc=64, out_nc=256)
        self.finalConv = FinalConv(in_nc=256, out_nc=in_nc)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_lap = laplacian_filter(x)
        x1 = self.beginEdgeConv(x_lap)
        x2 = self.denseNet(x1)
        x3 = self.maskConv(x1)
        # Gated fusion: continuous edge attention
        x4 = x3 * x2 + x2
        x_learned_lap = self.finalConv(x4)
        return x_learned_lap, x_lap


class EESRGAN(nn.Module):
    """
    End-to-End Edge-Enhanced Super-Resolution GAN Generator for Satellite Imagery.
    Recombines primary RRDB reconstruction with learned Laplacian edge features:
    x_sr = x_base + (x_learned_lap - x_lap)
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 8, scale: int = 2):
        super(EESRGAN, self).__init__()
        self.scale = scale
        self.netRG = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, scale=scale)
        self.netE = EESN(in_nc=out_nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = self.netRG(x)
        x_learned_lap, x_lap = self.netE(x_base)
        # Invert extracted baseline edge blur and inject sharp learned edges
        x_sr = x_base + (x_learned_lap - x_lap)
        return torch.clamp(x_sr, 0.0, 1.0)

    def forward_detailed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns intermediate tensors for specialized multi-objective loss computation."""
        x_base = self.netRG(x)
        x_learned_lap, x_lap = self.netE(x_base)
        x_sr = torch.clamp(x_base + (x_learned_lap - x_lap), 0.0, 1.0)
        return x_sr, x_base, x_learned_lap, x_lap
