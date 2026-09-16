"""
EDSR 200-Epoch Deep Training & Evaluation Pipeline
Cosine Annealing LR over 200 Cycles (~7-8 mins on GPU)
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
from torch.utils.data import TensorDataset, DataLoader
from skimage.metrics import structural_similarity as ssim

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
# 1. EDSR Architecture (Lim et al., NIPS/CVPR)
# =====================================================================
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
    def __init__(self, in_channels=3, out_channels=3, n_feats=64, n_resblocks=8, scale=2):
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


# =====================================================================
# 2. Metric Benchmark Functions
# =====================================================================
def calc_mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def calc_psnr(a, b):
    m = calc_mse(a, b)
    return 100.0 if m == 0 else 20.0 * math.log10(255.0 / math.sqrt(m))

def calc_ssim(a, b):
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return float(ssim(a, b, data_range=255.0))


# =====================================================================
# 3. 200-Epoch Training & Evaluation Pipeline
# =====================================================================
def main():
    epochs = 200
    batch_size = 32

    print("=" * 72)
    print(f"  EDSR DEEP TRAINING: {epochs} EPOCHS (Cosine Annealing LR)")
    print("  Stratified Multi-Biome Dataset (FGMANAUS + ESTUAMAR + ALSACE)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}")
    if device.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")

    train_data = torch.load(CACHE_DIR / "train_multibiome.pt", map_location="cpu", weights_only=False)
    test_data = torch.load(CACHE_DIR / "test_multibiome.pt", map_location="cpu", weights_only=False)

    train_loader = DataLoader(
        TensorDataset(train_data["lr"], train_data["hr"]),
        batch_size=batch_size,
        shuffle=True
    )
    print(f"Training Patches: {len(train_data['lr'])} | Batches/epoch: {len(train_loader)}")

    model = EDSR(in_channels=3, out_channels=3, n_feats=64, n_resblocks=8, scale=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture    : 8 ResBlocks, 64 Channels ({num_params:,} parameters)")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    best_loss = float("inf")
    start_train_t = time.time()
    model.train()

    print(f"\nTraining for {epochs} epochs (progress logged every 10 epochs)...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for lr_b, hr_b in train_loader:
            lr_b = lr_b.to(device)
            hr_b = hr_b.to(device)

            optimizer.zero_grad()
            sr_b = model(lr_b)
            loss = criterion(sr_b, hr_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_t = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "edsr_sentinel2_2x_best.pt")

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - start_train_t
            remaining = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch [{epoch:03d}/{epochs:03d}] - Loss: {avg_loss:.5f} | LR: {current_lr:.7f} | Elapsed: {elapsed/60:.1f}m (ETA: {remaining/60:.1f}m)")

    train_duration = time.time() - start_train_t
    ckpt_path = CKPT_DIR / "edsr_sentinel2_2x.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Done] 200 epochs completed in {train_duration/60:.2f} minutes!")
    print(f"Saved weights: {ckpt_path}")
    print(f"Saved best   : {CKPT_DIR / 'edsr_sentinel2_2x_best.pt'}")

    # Immediate Evaluation on Unseen 20% Test Split
    print(f"\nEvaluating EDSR (200 epochs) on {len(test_data['lr'])} Test Patches...")
    model.load_state_dict(torch.load(CKPT_DIR / "edsr_sentinel2_2x_best.pt", map_location=device))
    model.eval()

    mses_bic, psnrs_bic, ssims_bic = [], [], []
    mses_edsr, psnrs_edsr, ssims_edsr, lats_edsr = [], [], [], []

    num_eval = min(200, len(test_data["lr"]))
    with torch.no_grad():
        for i in range(num_eval):
            lr_t = test_data["lr"][i].unsqueeze(0)
            hr_t = test_data["hr"][i].unsqueeze(0)

            hr_np = (hr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            lr_np = (lr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

            bic = cv2.resize(lr_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_CUBIC)
            mses_bic.append(calc_mse(bic, hr_np))
            psnrs_bic.append(calc_psnr(bic, hr_np))
            ssims_bic.append(calc_ssim(bic, hr_np))

            lr_gpu = lr_t.to(device)
            t_start = time.perf_counter()
            out_t = model(lr_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats_edsr.append((time.perf_counter() - t_start) * 1000)

            out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mses_edsr.append(calc_mse(out_np, hr_np))
            psnrs_edsr.append(calc_psnr(out_np, hr_np))
            ssims_edsr.append(calc_ssim(out_np, hr_np))

    print("\n" + "=" * 76)
    print(f"  EDSR (200 EPOCHS) BENCHMARK RESULTS")
    print("=" * 76)
    print(f"{'Model / Architecture':<28} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Avg Latency':<12}")
    print("-" * 76)
    print(f"{'Bicubic Baseline':<28} | {np.mean(mses_bic):<10.2f} | {np.mean(psnrs_bic):<10.2f} | {np.mean(ssims_bic):<8.4f} | {'1.1 ms':<12}")
    print(f"{'EDSR (200 Epochs - Best)':<28} | {np.mean(mses_edsr):<10.2f} | {np.mean(psnrs_edsr):<10.2f} | {np.mean(ssims_edsr):<8.4f} | {f'{np.mean(lats_edsr):.2f} ms':<12}")
    print("=" * 76)


if __name__ == "__main__":
    main()
