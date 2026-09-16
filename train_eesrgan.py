"""
GPU Training Script for EESRGAN (Edge-Enhanced Super-Resolution GAN)
Trained on Stratified Multi-Biome Dataset: ESTUAMAR, FGMANAUS, and ALSACE.

Usage (on GPU machine):
    python train_eesrgan.py --epochs 100 --batch_size 16 --lr 1e-4 --device cuda
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ssim

# Project Root
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.eesrgan import EESRGAN, laplacian_filter


class MultiBiomeSatelliteDataset(Dataset):
    """
    Dataset loader for paired multi-biome Sentinel-2 (10m) and VENuS (5m) patches.
    Covers ALSACE (agricultural/urban), ESTUAMAR (coastal/estuary), FGMANAUS (rainforest).
    """

    def __init__(self, data_root: Path, augment: bool = True):
        self.data_root = Path(data_root)
        self.augment = augment
        self.lr_dir = self.data_root / "1_input_sentinel2_10m"
        self.hr_dir = self.data_root / "2_target_venus_05m"

        self.samples = []
        hr_files = sorted(list(self.hr_dir.glob("*_hr_05m.jpg")))
        for hr_path in hr_files:
            prefix = hr_path.name.split("_")[0]  # e.g., 'train' or 'test'
            patch_id = hr_path.name.split("_")[1]
            lr_path = self.lr_dir / f"{prefix}_{patch_id}_lr_10m.jpg"
            if lr_path.exists():
                self.samples.append((lr_path, hr_path))

        print(f"[Dataset] Found {len(self.samples)} paired patches in {self.data_root.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_path, hr_path = self.samples[idx]

        lr_bgr = cv2.imread(str(lr_path))
        hr_bgr = cv2.imread(str(hr_path))

        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)

        # Geometric Augmentations during training
        if self.augment:
            if np.random.rand() > 0.5:
                lr_rgb = np.fliplr(lr_rgb).copy()
                hr_rgb = np.fliplr(hr_rgb).copy()
            if np.random.rand() > 0.5:
                lr_rgb = np.flipud(lr_rgb).copy()
                hr_rgb = np.flipud(hr_rgb).copy()
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                lr_rgb = np.rot90(lr_rgb, k=rot_k).copy()
                hr_rgb = np.rot90(hr_rgb, k=rot_k).copy()

        lr_tensor = torch.from_numpy(lr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        hr_tensor = torch.from_numpy(hr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        return lr_tensor, hr_tensor


def compute_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def evaluate(model, val_loader, device):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            sr = model(lr)

            sr_np = (np.clip(sr.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            hr_np = (np.clip(hr.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

            sr_gray = cv2.cvtColor(sr_np, cv2.COLOR_RGB2GRAY)
            hr_gray = cv2.cvtColor(hr_np, cv2.COLOR_RGB2GRAY)

            p = compute_psnr_np(hr_gray, sr_gray)
            s = ssim(hr_gray, sr_gray, data_range=255)

            psnr_total += p
            ssim_total += s
            count += 1
            if count >= 100:  # Validate on 100 diverse patches for rapid validation
                break

    return psnr_total / count, ssim_total / count


def main():
    parser = argparse.ArgumentParser(description="Train EESRGAN on multi-biome Sentinel-2/VENuS dataset")
    parser.add_argument("--train_dir", type=str, default="data/sen2venus_jpegs/by_split/train_80_percent")
    parser.add_argument("--val_dir", type=str, default="data/sen2venus_jpegs/by_split/test_20_percent")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 for GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    # Device configuration
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("  EESRGAN (Edge-Enhanced Super-Resolution GAN) Multi-Biome Training")
    print("=" * 80)
    print(f"  Device:           {device}")
    if device.type == "cuda":
        print(f"  GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Learning Rate:    {args.lr}")
    print("=" * 80)

    # Output directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "eesrgan_sentinel2_2x_best.pt"
    latest_ckpt_path = ckpt_dir / "eesrgan_sentinel2_2x.pt"

    # Datasets and Loaders
    train_dataset = MultiBiomeSatelliteDataset(Path(args.train_dir), augment=True)
    val_dataset = MultiBiomeSatelliteDataset(Path(args.val_dir), augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize EESRGAN Model (8 RRDB blocks, 2x scale)
    model = EESRGAN(in_nc=3, out_nc=3, nf=64, nb=8, scale=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed Precision Scaler for high-speed GPU training
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_psnr = -1.0
    start_time = time.time()

    print("\n[Starting Training Loop...]")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_edge_loss = 0.0
        t0 = time.time()

        for step, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Detailed forward pass to get intermediate edge features
                sr, base, learned_lap, base_lap = model.forward_detailed(lr)

                # 1. Primary L1 Pixel Reconstruction Loss
                loss_pixel = F.l1_loss(sr, hr)

                # 2. Laplacian Edge Alignment Loss
                hr_lap = laplacian_filter(hr)
                loss_edge = F.l1_loss(learned_lap, hr_lap)

                # Composite Training Loss
                total_loss = loss_pixel + 0.15 * loss_edge

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_pixel.item()
            epoch_edge_loss += loss_edge.item()

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        avg_edge = epoch_edge_loss / len(train_loader)

        # Validation evaluation every 5 epochs (or first 2 epochs)
        if epoch % 5 == 0 or epoch <= 2 or epoch == args.epochs:
            val_psnr, val_ssim = evaluate(model, val_loader, device)
            is_best = val_psnr > best_val_psnr
            if is_best:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), str(best_ckpt_path))
                star_marker = "  <-- [NEW BEST SAVED!]"
            else:
                star_marker = ""

            torch.save(model.state_dict(), str(latest_ckpt_path))

            print(
                f"Epoch [{epoch:3d}/{args.epochs}] ({epoch_time:.1f}s) | "
                f"L1 Loss: {avg_loss:.4f} | Edge Loss: {avg_edge:.4f} | "
                f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}{star_marker}"
            )
        else:
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] ({epoch_time:.1f}s) | "
                f"L1 Loss: {avg_loss:.4f} | Edge Loss: {avg_edge:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}"
            )

    total_duration = (time.time() - start_time) / 60.0
    print("\n" + "=" * 80)
    print(f"  Training Finished in {total_duration:.1f} minutes!")
    print(f"  Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"  Best Checkpoint Saved: {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
