"""
GPU Training Script for Swin2-MoSE (Swin Transformer v2 with Mixture of Spatial Experts)
Trained on Stratified Multi-Biome Dataset: FGMANAUS, ALSACE, and ESTUAMAR for 70 Epochs.
Based on Rossi et al. (IMPLab, University of Parma, 2025) - https://github.com/IMPLabUniPr/swin2-mose

Usage (on GPU machine):
    python train_swin2_mose.py --epochs 70 --batch_size 8 --lr 1e-4 --device cuda
    python train_swin2_mose.py --resume checkpoints/swin2_mose_sentinel2_2x_best.pt --start_epoch 25 --device cuda
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

from src.models.swin2_mose import Swin2MoSEModel


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
        lr_path, hr_path = self.samples[idx]

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


def compute_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ncc_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalized Cross Correlation (NCC) Loss from Swin2-MoSE paper (Rossi et al., 2025).
    NCC = (x - mean(x)) * (y - mean(y)) / (std(x) * std(y) + eps)
    Loss = 1 - mean(NCC)
    """
    b, c, h, w = y_pred.shape
    y_pred_flat = y_pred.view(b, c, -1)
    y_true_flat = y_true.view(b, c, -1)

    mean_pred = y_pred_flat.mean(dim=-1, keepdim=True)
    mean_true = y_true_flat.mean(dim=-1, keepdim=True)

    pred_centered = y_pred_flat - mean_pred
    true_centered = y_true_flat - mean_true

    cov = (pred_centered * true_centered).sum(dim=-1)
    var_pred = (pred_centered ** 2).sum(dim=-1)
    var_true = (true_centered ** 2).sum(dim=-1)

    ncc = cov / (torch.sqrt(var_pred * var_true) + eps)
    return 1.0 - torch.clamp(ncc, -1.0, 1.0).mean()


def compute_ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Differentiable 2D SSIM loss for PyTorch tensors in [0, 1].
    """
    c = img1.shape[1]
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=c) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def evaluate(model, val_loader, device, max_eval_samples: int = 100):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            sr = model(lr, return_moe_loss=False)

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
    parser = argparse.ArgumentParser(description="Train Swin2-MoSE for 70 Epochs on Multi-Biome Dataset")
    parser.add_argument("--train_dir", type=str, default="data/sen2venus_jpegs/by_split/train_80_percent")
    parser.add_argument("--val_dir", type=str, default="data/sen2venus_jpegs/by_split/test_20_percent")
    parser.add_argument("--epochs", type=int, default=70, help="Total training epochs (default: 70)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch number when resuming")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--val_interval_epochs", type=int, default=5, help="Validation interval in epochs (default: 5)")
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
    print("  Swin2-MoSE: 70 Epochs Multi-Biome GPU Training Pipeline")
    print("  Architecture: Swin Transformer v2 with Mixture of Spatial Experts (MoE-SM)")
    print("=" * 85)
    print(f"  Device:           {device}")
    if device.type == "cuda":
        print(f"  GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Total Epochs:     {args.epochs}")
    print(f"  Epoch Range:      Epoch {args.start_epoch} -> {args.epochs}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Validation Rate:  Every {args.val_interval_epochs} epochs on 740 test patches")
    if args.resume:
        print(f"  Resuming From:    {args.resume}")
    print("=" * 85)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "swin2_mose_sentinel2_2x_best.pt"
    latest_ckpt_path = ckpt_dir / "swin2_mose_sentinel2_2x_latest.pt"
    final_epoch70_ckpt_path = ckpt_dir / "swin2_mose_sentinel2_2x_epoch70.pt"

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

    # Initialize Swin2-MoSE Model
    model = Swin2MoSEModel(scale=2, in_chans=3, img_size=64).to(device)

    best_val_psnr = -1.0
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
    last_epoch_idx = args.start_epoch - 2 if args.start_epoch > 1 else -1
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=last_epoch_idx)

    use_amp = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_train_time = time.time()
    steps_per_epoch = len(train_loader)

    print(f"\n[Commencing 70-Epoch Swin2-MoSE Training: Epochs {args.start_epoch} -> {args.epochs}...]")

    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()

        running_total_loss = 0.0
        running_l1 = 0.0
        running_ncc = 0.0
        running_ssim = 0.0
        running_moe = 0.0

        for step, (lr_tensor, hr_tensor) in enumerate(train_loader, 1):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            optimizer.zero_grad()

            try:
                autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                sr_tensor, loss_moe = model(lr_tensor, return_moe_loss=True)

                # 1. L1 Pixel Reconstruction Loss
                loss_l1 = F.l1_loss(sr_tensor, hr_tensor)

                # 2. Normalized Cross-Correlation (NCC) Loss (Rossi et al., 2025)
                loss_ncc = compute_ncc_loss(sr_tensor, hr_tensor)

                # 3. Differentiable Structural Similarity (SSIM) Loss
                loss_ssim = 1.0 - compute_ssim_torch(sr_tensor, hr_tensor)

                # 4. MoE Load-Balancing Auxiliary Loss
                moe_term = loss_moe if torch.is_tensor(loss_moe) else torch.tensor(0.0, device=device)

                # Composite Swin2-MoSE Loss
                total_loss = loss_l1 + 0.10 * loss_ncc + 0.20 * loss_ssim + 0.01 * moe_term

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += total_loss.item()
            running_l1 += loss_l1.item()
            running_ncc += loss_ncc.item()
            running_ssim += loss_ssim.item()
            running_moe += moe_term.item()

            if step % 50 == 0 or step == steps_per_epoch:
                speed = step / (time.time() - epoch_start_time)
                print(
                    f"\rEpoch [{epoch:2d}/{args.epochs:2d}] Step [{step:3d}/{steps_per_epoch}] | "
                    f"Speed: {speed:4.1f} it/s | "
                    f"L1: {running_l1/step:.4f} | NCC: {running_ncc/step:.4f} | SSIM: {running_ssim/step:.4f} | "
                    f"MoE: {running_moe/step:.4f} | Total: {running_total_loss/step:.4f}",
                    end="",
                    flush=True,
                )

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        curr_lr = scheduler.get_last_lr()[0]
        print(f" | lr: {curr_lr:.2e} | Time: {epoch_duration:.1f}s")

        # Validation every val_interval_epochs or on the final epoch
        if epoch % args.val_interval_epochs == 0 or epoch == args.epochs:
            eval_samples = None if epoch == args.epochs else 100
            n_eval_str = f"full {len(val_dataset)}" if eval_samples is None else f"{eval_samples}"
            print(f"\n>>> Running Validation on Test Set ({n_eval_str} patches) at Epoch {epoch}...")
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
                f">>> [Validation @ Epoch {epoch}] Test PSNR: {val_psnr:.2f} dB | "
                f"SSIM: {val_ssim:.4f}{star_marker}\n"
            )

    # Save final 70-epoch milestone
    torch.save(model.state_dict(), str(final_epoch70_ckpt_path))
    total_duration_hours = (time.time() - start_train_time) / 3600.0

    print("=" * 85)
    print(f"  Swin2-MoSE 70 Epochs Completed in {total_duration_hours:.2f} Hours!")
    print(f"  Best Validation PSNR:  {best_val_psnr:.2f} dB")
    print(f"  Best Checkpoint:       {best_ckpt_path}")
    print(f"  Epoch 70 Checkpoint:   {final_epoch70_ckpt_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
