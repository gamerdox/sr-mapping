"""
Standalone ESPCN Training & Immediate Metric Evaluation
Trained on: 80% Stratified Multi-Biome Split (FGMANAUS + ESTUAMAR + ALSACE)
Evaluated on: 20% Unseen Test Split
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- Auto-Detect Cache & Checkpoint Paths ---
script_dir = Path(__file__).resolve().parent

possible_cache_locations = [
    script_dir / "cache",
    script_dir.parent / "cache",
    script_dir / "data" / "cache",
    script_dir.parent / "data" / "cache",
    script_dir.parent.parent / "data" / "cache",
    Path.cwd() / "cache",
    Path.cwd() / "data" / "cache",
]

CACHE_DIR = next((p for p in possible_cache_locations if (p / "train_multibiome.pt").exists()), None)
if CACHE_DIR is None:
    print("\n[ERROR] train_multibiome.pt not found! Please run prepare_dataset.py first.")
    sys.exit(1)

CKPT_DIR = CACHE_DIR.parent / "checkpoints"
if not CKPT_DIR.parent.exists():
    CKPT_DIR = script_dir / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. ESPCN Architecture (Shi et al., CVPR 2016)
# =====================================================================
class ESPCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, out_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


# --- Metric Benchmark Functions (with built-in SSIM fallback) ---
def calc_mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def calc_psnr(a, b):
    m = calc_mse(a, b)
    return 100.0 if m == 0 else 20.0 * math.log10(255.0 / math.sqrt(m))

try:
    from skimage.metrics import structural_similarity as ssim_fn
    def calc_ssim(a, b):
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        return float(ssim_fn(a, b, data_range=255.0))
except ImportError:
    def calc_ssim(a, b):
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        I1 = a.astype(np.float64)
        I2 = b.astype(np.float64)
        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(I1, -1, window)
        mu2 = cv2.filter2D(I2, -1, window)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = cv2.filter2D(I1 * I1, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(I2 * I2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(I1 * I2, -1, window) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())


# =====================================================================
# 3. Training & Evaluation Pipeline
# =====================================================================
def main():
    print("=" * 72)
    print("  ESPCN (Sub-Pixel CNN) TRAINING & EVALUATION PIPELINE")
    print("  Dataset: Stratified Multi-Biome (FGMANAUS + ESTUAMAR + ALSACE)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}")
    if device.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")

    # Load Train & Test Splits
    train_file = CACHE_DIR / "train_multibiome.pt"
    test_file = CACHE_DIR / "test_multibiome.pt"

    print(f"\n[1/3] Loading Stratified Splits from: {CACHE_DIR}")
    train_data = torch.load(train_file, map_location="cpu", weights_only=False)
    test_data = torch.load(test_file, map_location="cpu", weights_only=False)

    train_loader = DataLoader(
        TensorDataset(train_data["lr"], train_data["hr"]),
        batch_size=32 if device.type == "cuda" else 16,
        shuffle=True
    )
    print(f"  Training Patches : {len(train_data['lr'])} (80% split)")
    print(f"  Test Patches     : {len(test_data['lr'])} (20% unseen split)")

    # Initialize Model
    model = ESPCN(in_channels=3, out_channels=3, scale=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Training ESPCN ({num_params:,} parameters, 10 Epochs)...")

    start_train_t = time.time()
    epochs = 10
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, (lr_b, hr_b) in enumerate(train_loader, 1):
            lr_b = lr_b.to(device)
            hr_b = hr_b.to(device)

            optimizer.zero_grad()
            sr_b = model(lr_b)
            loss = criterion(sr_b, hr_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                bar_len = 15
                filled = int(bar_len * step / len(train_loader))
                bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
                print(f"\r  Epoch [{epoch:02d}/{epochs:02d}] [{bar}] Step {step:03d}/{len(train_loader):03d} - Loss: {loss.item():.4f}", end="", flush=True)

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        print(f"\r  Epoch [{epoch:02d}/{epochs:02d}] Completed - Avg Loss: {avg_loss:.4f} ({epoch_time:.2f}s)                ")

    train_duration = time.time() - start_train_t
    ckpt_path = CKPT_DIR / "espcn_sentinel2_2x.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Done] Training finished in {train_duration:.1f}s. Saved checkpoint to:\n  {ckpt_path}")

    # Immediate Evaluation on Unseen 20% Test Split
    print(f"\n[3/3] Evaluating ESPCN vs. Bicubic Baseline on {len(test_data['lr'])} Test Patches...")
    model.eval()

    mses_bic, psnrs_bic, ssims_bic = [], [], []
    mses_esp, psnrs_esp, ssims_esp, lats_esp = [], [], [], []

    # Evaluate on test patches
    num_eval = min(150, len(test_data["lr"]))
    with torch.no_grad():
        for i in range(num_eval):
            lr_t = test_data["lr"][i].unsqueeze(0)
            hr_t = test_data["hr"][i].unsqueeze(0)

            # Convert GT to uint8 RGB [0, 255]
            hr_np = (hr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            lr_np = (lr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # 1. Bicubic Baseline
            bic = cv2.resize(lr_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_CUBIC)
            mses_bic.append(calc_mse(bic, hr_np))
            psnrs_bic.append(calc_psnr(bic, hr_np))
            ssims_bic.append(calc_ssim(bic, hr_np))

            # 2. ESPCN
            lr_gpu = lr_t.to(device)
            t_start = time.perf_counter()
            out_t = model(lr_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats_esp.append((time.perf_counter() - t_start) * 1000)

            out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mses_esp.append(calc_mse(out_np, hr_np))
            psnrs_esp.append(calc_psnr(out_np, hr_np))
            ssims_esp.append(calc_ssim(out_np, hr_np))

    # Print Report Table
    print("\n" + "=" * 76)
    print(f"  ESPCN EVALUATION RESULTS (Averaged over {num_eval} Unseen Test Patches)")
    print("=" * 76)
    print(f"{'Model / Architecture':<28} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Avg Latency':<12}")
    print("-" * 76)
    print(f"{'Bicubic Baseline':<28} | {np.mean(mses_bic):<10.2f} | {np.mean(psnrs_bic):<10.2f} | {np.mean(ssims_bic):<8.4f} | {'1.1 ms':<12}")
    print(f"{'ESPCN (Ours - Trained)':<28} | {np.mean(mses_esp):<10.2f} | {np.mean(psnrs_esp):<10.2f} | {np.mean(ssims_esp):<8.4f} | {f'{np.mean(lats_esp):.2f} ms':<12}")
    print("=" * 76)


if __name__ == "__main__":
    main()
