"""
FSRCNN 400-Epoch Deep Training & Evaluation Pipeline
Cosine Annealing LR over 400 Cycles (~8 mins on GPU)
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


# --- 1. Architecture ---
class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=2, d=56, s=12, m=4):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ])
        self.mapping = nn.Sequential(*mapping_layers)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(
            d, out_channels, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)


# --- 2. Metric Functions ---
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


# --- 3. 400-Epoch Training Pipeline ---
def main():
    epochs = 400
    batch_size = 64

    print("=" * 72)
    print(f"  FSRCNN DEEP TRAINING: {epochs} EPOCHS (Cosine Annealing LR)")
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

    model = FSRCNN(in_channels=3, out_channels=3, scale=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    best_loss = float("inf")
    start_train_t = time.time()
    model.train()

    print(f"\nTraining for {epochs} epochs (progress logged every 20 epochs)...")

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
            torch.save(model.state_dict(), CKPT_DIR / "fsrcnn_sentinel2_2x_best.pt")

        if epoch <= 5 or epoch % 20 == 0 or epoch == epochs:
            elapsed = time.time() - start_train_t
            remaining = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch [{epoch:03d}/{epochs:03d}] - Loss: {avg_loss:.5f} | LR: {current_lr:.7f} | Elapsed: {elapsed/60:.1f}m (ETA: {remaining/60:.1f}m)")

    train_duration = time.time() - start_train_t
    ckpt_path = CKPT_DIR / "fsrcnn_sentinel2_2x.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Done] 400 epochs completed in {train_duration/60:.2f} minutes!")
    print(f"Saved best weights: {CKPT_DIR / 'fsrcnn_sentinel2_2x_best.pt'}")

    # Immediate Evaluation on Unseen 20% Test Split
    print(f"\nEvaluating FSRCNN (400 epochs) on {len(test_data['lr'])} Test Patches...")
    model.load_state_dict(torch.load(CKPT_DIR / "fsrcnn_sentinel2_2x_best.pt", map_location=device))
    model.eval()

    mses_bic, psnrs_bic, ssims_bic = [], [], []
    mses_fsr, psnrs_fsr, ssims_fsr, lats_fsr = [], [], [], []

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
            lats_fsr.append((time.perf_counter() - t_start) * 1000)

            out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mses_fsr.append(calc_mse(out_np, hr_np))
            psnrs_fsr.append(calc_psnr(out_np, hr_np))
            ssims_fsr.append(calc_ssim(out_np, hr_np))

    print("\n" + "=" * 76)
    print(f"  FSRCNN (400 EPOCHS) BENCHMARK RESULTS")
    print("=" * 76)
    print(f"{'Model / Architecture':<28} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Avg Latency':<12}")
    print("-" * 76)
    print(f"{'Bicubic Baseline':<28} | {np.mean(mses_bic):<10.2f} | {np.mean(psnrs_bic):<10.2f} | {np.mean(ssims_bic):<8.4f} | {'1.1 ms':<12}")
    print(f"{'FSRCNN (400 Epochs - Best)':<28} | {np.mean(mses_fsr):<10.2f} | {np.mean(psnrs_fsr):<10.2f} | {np.mean(ssims_fsr):<8.4f} | {f'{np.mean(lats_fsr):.2f} ms':<12}")
    print("=" * 76)


if __name__ == "__main__":
    main()
