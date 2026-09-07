import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sr_model import Sentinel2SRNet
from src.data.dataset import SentinelVenusDataset


class SobelEdgeLoss(nn.Module):
    """Computes L1 loss on 2D spatial image gradients (edges)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("filter_x", sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer("filter_y", sobel_y.repeat(in_channels, 1, 1, 1))
        self.channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_pred = F.conv2d(pred, self.filter_x, padding=1, groups=self.channels)
        gy_pred = F.conv2d(pred, self.filter_y, padding=1, groups=self.channels)
        gx_target = F.conv2d(target, self.filter_x, padding=1, groups=self.channels)
        gy_target = F.conv2d(target, self.filter_y, padding=1, groups=self.channels)
        return F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)


def align_radiometry(source: torch.Tensor, reference_base: torch.Tensor) -> torch.Tensor:
    """
    Align the global mean and contrast of the target (source) to match the input (reference_base).
    Preserves all high-frequency spatial structures while matching lighting.
    """
    m_src = source.mean(dim=(-2, -1), keepdim=True)
    s_src = source.std(dim=(-2, -1), keepdim=True)
    m_ref = reference_base.mean(dim=(-2, -1), keepdim=True)
    s_ref = reference_base.std(dim=(-2, -1), keepdim=True)

    aligned = (source - m_src) / (s_src + 1e-5) * s_ref + m_ref
    return torch.clamp(aligned, 0.0, 1.0)


def calibrate_and_save_weights(
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 5e-4,
    max_train_samples: int = 64,
):
    print("=" * 65)
    print("Training Sentinel2SRNet with Edge & Radiometric Constraints")
    print("=" * 65)

    dataset_path = PROJECT_ROOT / "data" / "sen2venus_sample" / "FGMANAUS"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "sr_sentinel2_2x.pt"

    dataset = SentinelVenusDataset(dataset_path)
    train_size = min(len(dataset), max_train_samples)
    train_subset = Subset(dataset, range(train_size))
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    print(f"Training on {train_size} patches with Sobel Edge & Mean-Preserving Penalties.")

    model = Sentinel2SRNet(in_channels=3, out_channels=3, scale_factor=2)
    edge_criterion = SobelEdgeLoss(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pixel_loss = 0.0
        total_edge_loss = 0.0

        for lr_patches, hr_patches in loader:
            optimizer.zero_grad()

            # 1. Bicubic base
            base = F.interpolate(lr_patches, scale_factor=2, mode="bicubic", align_corners=False)

            # 2. Radiometric alignment of target to eliminate sensor brightness shortcut
            target_aligned = align_radiometry(hr_patches, base)

            # 3. Model forward pass (with zero-mean residual constraint)
            sr_patches = model(lr_patches)

            # 4. Losses:
            # - Pixel fidelity against radiometrically matched target
            l_pixel = F.l1_loss(sr_patches, target_aligned)
            # - Edge penalty: forces sharpening of boundaries/textures
            l_edge = edge_criterion(sr_patches, target_aligned)
            # - Mean penalty: strictly penalizes any drift in global brightness
            l_mean = torch.abs(sr_patches.mean(dim=(-2, -1)) - base.mean(dim=(-2, -1))).mean()

            loss = l_pixel + 2.5 * l_edge + 5.0 * l_mean
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pixel_loss += l_pixel.item()
            total_edge_loss += l_edge.item()

        n_batches = len(loader)
        print(
            f"Epoch [{epoch}/{epochs}] - Total: {total_loss/n_batches:.4f} | "
            f"Pixel L1: {total_pixel_loss/n_batches:.4f} | "
            f"Edge L1: {total_edge_loss/n_batches:.4f}"
        )

    torch.save(model.state_dict(), checkpoint_file)
    print(f"\n[Success] Edge-focused model weights saved to:\n  {checkpoint_file}")


if __name__ == "__main__":
    calibrate_and_save_weights()
