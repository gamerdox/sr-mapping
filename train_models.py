"""
Multi-Model Training Pipeline for ESPCN & FSRCNN
Trained on Stratified Multi-Biome Sentinel-2 (10m) -> VENuS (5m) Pairs
(FGMANAUS + ESTUAMAR + ALSACE)
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- Path Resolution & Directory Auto-Discovery ---
script_dir = Path(__file__).resolve().parent

possible_cache_locations = [
    script_dir / "cache" / "train_multibiome.pt",
    script_dir.parent / "cache" / "train_multibiome.pt",
    script_dir / "data" / "cache" / "train_multibiome.pt",
    script_dir.parent / "data" / "cache" / "train_multibiome.pt",
    script_dir.parent.parent / "data" / "cache" / "train_multibiome.pt",
    Path.cwd() / "cache" / "train_multibiome.pt",
    Path.cwd() / "data" / "cache" / "train_multibiome.pt",
]

train_cache_file = next((p for p in possible_cache_locations if p.exists()), None)
if train_cache_file is None:
    print("\n[ERROR] train_multibiome.pt not found!")
    print("Checked locations:")
    for p in possible_cache_locations:
        print(f"  - {p}")
    print("\nPlease ensure prepare_dataset.py ran successfully before training.")
    sys.exit(1)

# Checkpoints directory
if (script_dir / "checkpoints").exists() or (script_dir.name in ["sr-mapping-main", "sr-mapping"]):
    CKPT_DIR = script_dir / "checkpoints"
elif (script_dir.parent / "checkpoints").exists() or (script_dir.parent.name in ["sr-mapping-main", "sr-mapping"]):
    CKPT_DIR = script_dir.parent / "checkpoints"
else:
    CKPT_DIR = train_cache_file.parent.parent / "checkpoints"

CKPT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. Model Definitions
# =====================================================================
class ESPCN(nn.Module):
    """ESPCN (Shi et al., CVPR 2016) - Sub-Pixel Convolutional Network"""
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


class FSRCNN(nn.Module):
    """FSRCNN (Dong et al., ECCV 2016) - Fast Super-Resolution CNN"""
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


# =====================================================================
# 2. General Training Engine
# =====================================================================
def train_single_model(model_name: str, model: nn.Module, loader: DataLoader, device: torch.device, epochs=10, lr=1e-3):
    print("\n" + "=" * 68)
    print(f"  Training {model_name} (2x Super-Resolution)")
    print("=" * 68)

    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} | Device: {device} | Batches/epoch: {len(loader)}")

    start_t = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, (lr_b, hr_b) in enumerate(loader, 1):
            lr_b = lr_b.to(device)
            hr_b = hr_b.to(device)

            optimizer.zero_grad()
            sr_b = model(lr_b)
            loss = criterion(sr_b, hr_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0 or step == len(loader):
                bar_len = 15
                filled = int(bar_len * step / len(loader))
                bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
                print(f"\r  Epoch [{epoch:02d}/{epochs:02d}] [{bar}] Step {step:03d}/{len(loader):03d} - Loss: {loss.item():.4f}", end="", flush=True)

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / len(loader)
        print(f"\r  Epoch [{epoch:02d}/{epochs:02d}] Completed - Avg Loss: {avg_loss:.4f} ({epoch_time:.2f}s)                ")

    total_time = time.time() - start_t
    print(f"[Done] {model_name} training finished in {total_time:.1f}s ({total_time/60:.2f} mins).")

    ckpt_path = CKPT_DIR / f"{model_name.lower()}_sentinel2_2x.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Success] Saved weights: {ckpt_path}")
    return ckpt_path


# =====================================================================
# 3. Main Execution
# =====================================================================
def main():
    print("=" * 68)
    print("  MULTI-MODEL TRAINING PIPELINE (ESPCN & FSRCNN)")
    print("  Stratified 80% Multi-Biome Training Split")
    print("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}")
    if device.type == "cuda":
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")

    print(f"\n[1/3] Loading Cached Multi-Biome Training Data...")
    print(f"  Source: {train_cache_file}")
    data = torch.load(train_cache_file, map_location="cpu", weights_only=False)

    train_dataset = TensorDataset(data["lr"], data["hr"])
    batch_size = 32 if device.type == "cuda" else 16
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Total Training Patches: {len(train_dataset)} (Batch Size: {batch_size})")

    # 1. Train ESPCN
    print(f"\n[2/3] Initiating ESPCN Training...")
    m_espcn = ESPCN(in_channels=3, out_channels=3, scale=2)
    train_single_model("ESPCN", m_espcn, loader, device, epochs=10, lr=1e-3)

    # 2. Train FSRCNN
    print(f"\n[3/3] Initiating FSRCNN Training...")
    m_fsrcnn = FSRCNN(in_channels=3, out_channels=3, scale=2)
    train_single_model("FSRCNN", m_fsrcnn, loader, device, epochs=10, lr=1e-3)

    print("\n" + "=" * 68)
    print("  ALL TRAININGS COMPLETED SUCCESSFULLY!")
    print(f"  Checkpoints saved to: {CKPT_DIR}")
    print("  - espcn_sentinel2_2x.pt")
    print("  - fsrcnn_sentinel2_2x.pt")
    print("=" * 68)


if __name__ == "__main__":
    main()
