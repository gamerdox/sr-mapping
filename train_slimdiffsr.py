"""
GPU Training Script for SlimDiffSR (1-Step Distilled Super-Resolution Network)
Trained on Stratified Multi-Biome Dataset: FGMANAUS, ALSACE, and ESTUAMAR for 1 Lakh (100,000) Iterations.

Usage (on GPU machine):
    python train_slimdiffsr.py --iterations 100000 --batch_size 8 --lr 1e-4 --device cuda
    python train_slimdiffsr.py --resume checkpoints/slimdiffsr_sentinel2_2x_best.pt --start_iter 25000 --device cuda
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    def ssim(img1, img2, data_range=255):
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        f1 = img1.astype(np.float64)
        f2 = img2.astype(np.float64)
        mu1 = f1.mean()
        mu2 = f2.mean()
        s1 = np.var(f1)
        s2 = np.var(f2)
        s12 = np.mean((f1 - mu1) * (f2 - mu2))
        return float(((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2)))

# Project Root
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.slimdiffsr import SlimDiffSR


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
            prefix = hr_path.name.split("_")[0]
            patch_id = hr_path.name.split("_")[1]
            lr_path = self.lr_dir / f"{prefix}_{patch_id}_lr_10m.jpg"
            if lr_path.exists():
                self.samples.append((lr_path, hr_path))

        print(f"[Dataset] Found {len(self.samples)} paired patches in {self.data_root.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if cv2 is not None:
            lr_bgr = cv2.imread(str(lr_path))
            hr_bgr = cv2.imread(str(hr_path))
            lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
            hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        else:
            lr_rgb = np.array(Image.open(lr_path).convert("RGB"))
            hr_rgb = np.array(Image.open(hr_path).convert("RGB"))

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


def infinite_iterator(loader):
    """Infinitely cycles through the training DataLoader for iteration-based training."""
    while True:
        for batch in loader:
            yield batch


def compute_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_fft_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Computes frequency spectrum L1 loss via 2D Real Fast Fourier Transform."""
    fft1 = torch.fft.rfft2(img1, norm="backward")
    fft2 = torch.fft.rfft2(img2, norm="backward")
    return F.l1_loss(torch.abs(fft1), torch.abs(fft2))


def compute_sobel_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Computes Sobel gradient edge loss across spatial channels."""
    channels = img1.shape[1]
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    g1_x = F.conv2d(img1, gx, padding=1, groups=channels)
    g1_y = F.conv2d(img1, gy, padding=1, groups=channels)
    g2_x = F.conv2d(img2, gx, padding=1, groups=channels)
    g2_y = F.conv2d(img2, gy, padding=1, groups=channels)
    return F.l1_loss(g1_x, g2_x) + F.l1_loss(g1_y, g2_y)


def evaluate(model, val_loader, device, max_eval_samples: int = 100):
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

            if cv2 is not None:
                sr_gray = cv2.cvtColor(sr_np, cv2.COLOR_RGB2GRAY)
                hr_gray = cv2.cvtColor(hr_np, cv2.COLOR_RGB2GRAY)
            else:
                sr_gray = np.dot(sr_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                hr_gray = np.dot(hr_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

            p = compute_psnr_np(hr_gray, sr_gray)
            s = ssim(hr_gray, sr_gray, data_range=255)

            psnr_total += p
            ssim_total += s
            count += 1
            if max_eval_samples is not None and count >= max_eval_samples:
                break

    return psnr_total / count, ssim_total / count


def main():
    parser = argparse.ArgumentParser(description="Train SlimDiffSR for 1 Lakh (100,000) Iterations on Multi-Biome Dataset")
    parser.add_argument("--train_dir", type=str, default="data/sen2venus_jpegs/by_split/train_80_percent")
    parser.add_argument("--val_dir", type=str, default="data/sen2venus_jpegs/by_split/test_20_percent")
    parser.add_argument("--iterations", type=int, default=100000, help="Total iterations (default: 100,000 / 1 lakh)")
    parser.add_argument("--start_iter", type=int, default=1, help="Starting iteration number when resuming")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_interval", type=int, default=5000, help="Evaluation interval (default: every 5,000 steps)")
    parser.add_argument("--log_interval", type=int, default=500, help="Terminal logging interval (default: every 500 steps)")
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

    print("=" * 85)
    print("  SlimDiffSR: 1 Lakh (100,000) Iterations Multi-Biome GPU Training Pipeline")
    print("=" * 85)
    print(f"  Device:           {device}")
    if device.type == "cuda":
        print(f"  GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Total Steps:      {args.iterations:,} iterations (1 Lakh)")
    print(f"  Step Range:       Step {args.start_iter:,} -> {args.iterations:,}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Validation Rate:  Every {args.val_interval:,} steps on 740 test patches")
    if args.resume:
        print(f"  Resuming From:    {args.resume}")
    print("=" * 85)

    # Checkpoint paths
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "slimdiffsr_sentinel2_2x_best.pt"
    latest_ckpt_path = ckpt_dir / "slimdiffsr_sentinel2_2x_latest.pt"
    final_100k_ckpt_path = ckpt_dir / "slimdiffsr_sentinel2_2x_iter100k.pt"

    # Datasets
    train_dataset = MultiBiomeSatelliteDataset(Path(args.train_dir), augment=True)
    val_dataset = MultiBiomeSatelliteDataset(Path(args.val_dir), augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_iter = infinite_iterator(train_loader)

    # Initialize SlimDiffSR Model
    model = SlimDiffSR(scale=2).to(device)

    best_val_psnr = -1.0
    # Resume checkpoint if provided
    if args.resume:
        resume_p = Path(args.resume)
        if not resume_p.exists() and (ckpt_dir / args.resume).exists():
            resume_p = ckpt_dir / args.resume
        if resume_p.exists():
            print(f"\n[Resume] Loading pre-trained weights from {resume_p}...")
            state_dict = torch.load(str(resume_p), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("[Resume] Weights loaded successfully!")
            print("[Resume] Running initial test evaluation on test set...")
            val_psnr, val_ssim = evaluate(model, val_loader, device)
            best_val_psnr = val_psnr
            print(f"[Resume] Baseline Test Score -> PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
        else:
            print(f"[Warning] Resume checkpoint {args.resume} not found. Starting from scratch.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for pg in optimizer.param_groups:
        pg['initial_lr'] = args.lr
    last_step_idx = args.start_iter - 2 if args.start_iter > 1 else -1
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-6, last_epoch=last_step_idx)

    # Mixed Precision Scaler for high-speed GPU training
    use_amp = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_time = time.time()
    running_l1 = 0.0
    running_fft = 0.0
    running_sobel = 0.0
    interval_start_time = time.time()

    print(f"\n[Commencing 1 Lakh Iteration Training: Steps {args.start_iter:,} -> {args.iterations:,}...]")

    for step in range(args.start_iter, args.iterations + 1):
        model.train()
        lr_tensor, hr_tensor = next(train_iter)
        lr_tensor = lr_tensor.to(device)
        hr_tensor = hr_tensor.to(device)

        optimizer.zero_grad()

        try:
            autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        with autocast_ctx:
            sr_tensor = model(lr_tensor)

            # 1. Primary L1 Pixel Reconstruction Loss
            loss_l1 = F.l1_loss(sr_tensor, hr_tensor)

            # 2. Fast Fourier Transform Frequency Loss
            loss_fft = compute_fft_loss(sr_tensor, hr_tensor)

            # 3. Sobel Edge Gradient Loss
            loss_sobel = compute_sobel_loss(sr_tensor, hr_tensor)

            # Composite Multi-Objective Training Loss
            total_loss = loss_l1 + 0.10 * loss_fft + 0.05 * loss_sobel

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_l1 += loss_l1.item()
        running_fft += loss_fft.item()
        running_sobel += loss_sobel.item()

        # Terminal logging every log_interval steps
        if step % args.log_interval == 0:
            elapsed_interval = time.time() - interval_start_time
            speed = args.log_interval / elapsed_interval if elapsed_interval > 0 else 0.0
            avg_l1 = running_l1 / args.log_interval
            avg_fft = running_fft / args.log_interval
            avg_sobel = running_sobel / args.log_interval
            curr_lr = scheduler.get_last_lr()[0]

            total_elapsed = time.time() - start_time
            remaining_steps = args.iterations - step
            eta_seconds = remaining_steps / speed if speed > 0 else 0
            eta_minutes = eta_seconds / 60.0

            print(
                f"Step [{step:6d}/{args.iterations:,}] | "
                f"Speed: {speed:5.1f} it/s | "
                f"L1: {avg_l1:.4f} | FFT: {avg_fft:.4f} | Sobel: {avg_sobel:.4f} | "
                f"lr: {curr_lr:.2e} | ETA: {eta_minutes:.1f}m"
            )

            running_l1 = 0.0
            running_fft = 0.0
            running_sobel = 0.0
            interval_start_time = time.time()

        # Periodic Validation on the 740 unseen test patches every val_interval steps
        if step % args.val_interval == 0 or step == args.iterations:
            eval_samples = None if step == args.iterations else 100
            n_eval_str = f"full {len(val_dataset)}" if eval_samples is None else f"{eval_samples}"
            print(f"\n>>> Running Test Set Evaluation ({n_eval_str} patches) at Step {step:,}...")
            val_psnr, val_ssim = evaluate(model, val_loader, device, max_eval_samples=eval_samples)
            is_best = val_psnr > best_val_psnr
            if is_best:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), str(best_ckpt_path))
                star_marker = "  <-- [NEW BEST SCORE SAVED!]"
            else:
                star_marker = ""

            torch.save(model.state_dict(), str(latest_ckpt_path))

            print(
                f">>> [Validation @ Step {step:,}] Test PSNR: {val_psnr:.2f} dB | "
                f"SSIM: {val_ssim:.4f}{star_marker}\n"
            )
            interval_start_time = time.time()

    # Save final 100k iteration milestone
    torch.save(model.state_dict(), str(final_100k_ckpt_path))
    total_duration_hours = (time.time() - start_time) / 3600.0

    print("=" * 85)
    print(f"  1 Lakh Iterations Completed in {total_duration_hours:.2f} Hours!")
    print(f"  Best Validation PSNR:   {best_val_psnr:.2f} dB")
    print(f"  Best Checkpoint:        {best_ckpt_path}")
    print(f"  Final 100k Checkpoint:  {final_100k_ckpt_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
