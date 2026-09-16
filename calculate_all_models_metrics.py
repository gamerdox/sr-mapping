"""
Benchmark & Metric Evaluation Script for Sentinel-2 Super-Resolution Models
Calculates MSE, PSNR, and SSIM (skimage) across all 740 test patches
for all 4 trained PyTorch models vs VENuS Ground Truth ("Real").
"""

import os
import sys
import time
import math
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.sr_model import Sentinel2SRNet
from src.models.eesrgan import EESRGAN
from src.models.edsr import EDSR
from src.models.espcn import ESPCN
from src.models.fsrcnn import FSRCNN

DATA_DIR = ROOT_DIR / "data" / "sen2venus_jpegs" / "by_split" / "test_20_percent"
LR_DIR = DATA_DIR / "1_input_sentinel2_10m"
REAL_DIR = DATA_DIR / "2_target_venus_05m"
BIC_DIR = DATA_DIR / "3_bicubic_baseline_05m"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# User's exact metric functions
def psnr(img1, img2):
    mse_val = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse_val == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20.0 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

def mse(imageA, imageB):
    return float(np.mean((imageA.astype(np.float64) - imageB.astype(np.float64)) ** 2))

def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = float(ssim(imageA, imageB, data_range=255))
    p = psnr(imageA, imageB)
    return m, s, p

def main():
    print("=" * 75)
    print("  Sentinel-2 Super-Resolution Benchmark Evaluation across 740 Test Patches")
    print("=" * 75)

    device = torch.device("cpu")
    print(f"[Device] Running on {device}")

    # Model definitions and checkpoints
    models_config = {
        "Sentinel2SRNet": {
            "model_fn": lambda: Sentinel2SRNet(scale_factor=2),
            "ckpt": CHECKPOINTS_DIR / "sr_sentinel2_2x_best.pt",
            "fake_dir": DATA_DIR / "fake_sentinel2srnet",
        },
        "EESRGAN": {
            "model_fn": lambda: EESRGAN(scale=2, nb=8),
            "ckpt": CHECKPOINTS_DIR / "eesrgan_sentinel2_2x_best.pt",
            "fake_dir": DATA_DIR / "fake_eesrgan",
        },
        "EDSR": {
            "model_fn": lambda: EDSR(scale=2),
            "ckpt": CHECKPOINTS_DIR / "edsr_sentinel2_2x_best.pt",
            "fake_dir": DATA_DIR / "fake_edsr",
        },
        "ESPCN": {
            "model_fn": lambda: ESPCN(scale_factor=2),
            "ckpt": CHECKPOINTS_DIR / "espcn_sentinel2_2x_best.pt",
            "fake_dir": DATA_DIR / "fake_espcn",
        },
        "FSRCNN": {
            "model_fn": lambda: FSRCNN(scale_factor=2),
            "ckpt": CHECKPOINTS_DIR / "fsrcnn_sentinel2_2x.pt",
            "fake_dir": DATA_DIR / "fake_fsrcnn",
        },
    }

    # Load models
    loaded_models = {}
    for name, cfg in models_config.items():
        ckpt_path = cfg["ckpt"]
        if not ckpt_path.exists():
            print(f"[Warning] Checkpoint {ckpt_path.name} not found! Skipping {name}.")
            continue
        model = cfg["model_fn"]().to(device)
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        loaded_models[name] = model
        cfg["fake_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[Loaded] {name} from {ckpt_path.name}")

    # Discover test files
    real_files = sorted(list(REAL_DIR.glob("test_*_hr_05m.jpg")))
    total_samples = len(real_files)
    print(f"\n[Dataset] Found {total_samples} test patches in {REAL_DIR.name}")

    if total_samples == 0:
        print("[Error] No test patches found!")
        return

    # Tracking accumulators
    metrics_summary = {
        "Bicubic": {"mse_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "time_ms_sum": 0.0},
    }
    for name in loaded_models:
        metrics_summary[name] = {"mse_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "time_ms_sum": 0.0}

    detailed_records = []
    sample_visuals = {}

    print("\n[Running Batch Inference & Metrics Calculation...]")
    start_all = time.time()

    for idx, real_path in enumerate(real_files):
        patch_id = real_path.stem.split("_")[1] # e.g. '0000'
        lr_path = LR_DIR / f"test_{patch_id}_lr_10m.jpg"
        bic_path = BIC_DIR / f"test_{patch_id}_bicubic_05m.jpg"

        if not lr_path.exists() or not bic_path.exists():
            continue

        real_bgr = cv2.imread(str(real_path))
        lr_bgr = cv2.imread(str(lr_path))
        bic_bgr = cv2.imread(str(bic_path))

        real_gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)
        bic_gray = cv2.cvtColor(bic_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Bicubic Metrics
        m_bic, s_bic, p_bic = compare_images(real_gray, bic_gray)
        metrics_summary["Bicubic"]["mse_sum"] += m_bic
        metrics_summary["Bicubic"]["psnr_sum"] += p_bic
        metrics_summary["Bicubic"]["ssim_sum"] += s_bic

        record = {
            "patch_id": patch_id,
            "bicubic": {"mse": m_bic, "psnr": p_bic, "ssim": s_bic},
            "models": {},
        }

        # Prepare input tensor for PyTorch models
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # 2. Evaluate each trained model
        for name, model in loaded_models.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr_rgb = (np.clip(sr_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

            # Save fake image
            fake_out_path = models_config[name]["fake_dir"] / f"test_{patch_id}_fake.jpg"
            cv2.imwrite(str(fake_out_path), sr_bgr)

            # Calculate metric on Grayscale as per user specification
            sr_gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
            m_sr, s_sr, p_sr = compare_images(real_gray, sr_gray)

            metrics_summary[name]["mse_sum"] += m_sr
            metrics_summary[name]["psnr_sum"] += p_sr
            metrics_summary[name]["ssim_sum"] += s_sr
            metrics_summary[name]["time_ms_sum"] += lat_ms

            record["models"][name] = {
                "mse": m_sr,
                "psnr": p_sr,
                "ssim": s_sr,
                "latency_ms": round(lat_ms, 2),
            }

            if idx == 0:
                sample_visuals[name] = sr_rgb

        detailed_records.append(record)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            elapsed = time.time() - start_all
            pct = ((idx + 1) / total_samples) * 100
            print(f"  Processed [{idx + 1:3d}/{total_samples}] patches ({pct:.1f}%) — Elapsed: {elapsed:.1f}s")

    # Compute Averages
    n = len(detailed_records)
    results_table = []
    bic_psnr_avg = metrics_summary["Bicubic"]["psnr_sum"] / n
    bic_ssim_avg = metrics_summary["Bicubic"]["ssim_sum"] / n

    for key, val in metrics_summary.items():
        avg_mse = val["mse_sum"] / n
        avg_psnr = val["psnr_sum"] / n
        avg_ssim = val["ssim_sum"] / n
        avg_lat = val["time_ms_sum"] / n if val["time_ms_sum"] > 0 else 0.0
        gain_psnr = avg_psnr - bic_psnr_avg
        gain_ssim = avg_ssim - bic_ssim_avg

        results_table.append({
            "model": key,
            "mse": round(avg_mse, 4),
            "psnr": round(avg_psnr, 4),
            "ssim": round(avg_ssim, 4),
            "psnr_gain": round(gain_psnr, 4),
            "ssim_gain": round(gain_ssim, 4),
            "avg_latency_ms": round(avg_lat, 2),
        })

    # Display Leaderboard
    print("\n" + "=" * 88)
    print(f"{'FINAL EVALUATION LEADERBOARD (740 Unseen Test Patches)':^88}")
    print("=" * 88)
    print(f"{'Model Architecture':<22} | {'MSE':>9} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'PSNR Gain':>11} | {'SSIM Gain':>10}")
    print("-" * 88)
    for row in results_table:
        is_base = row["model"] == "Bicubic"
        psnr_gain_str = "baseline" if is_base else f"+{row['psnr_gain']:.2f} dB"
        ssim_gain_str = "baseline" if is_base else f"{row['ssim_gain']:+.4f}"
        print(f"{row['model']:<22} | {row['mse']:>9.2f} | {row['psnr']:>10.2f} | {row['ssim']:>8.4f} | {psnr_gain_str:>11} | {ssim_gain_str:>10}")
    print("=" * 88)

    # Save detailed JSON and CSV
    json_path = RESULTS_DIR / "benchmark_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": results_table, "count": n}, f, indent=2)
    print(f"\n[Saved] Summary metrics saved to {json_path}")

    csv_path = RESULTS_DIR / "benchmark_test_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["patch_id", "bicubic_psnr", "bicubic_ssim", "bicubic_mse"]
        for m in loaded_models:
            header.extend([f"{m.lower()}_psnr", f"{m.lower()}_ssim", f"{m.lower()}_mse"])
        f.write(",".join(header) + "\n")
        for rec in detailed_records:
            row = [rec["patch_id"], str(rec["bicubic"]["psnr"]), str(rec["bicubic"]["ssim"]), str(rec["bicubic"]["mse"])]
            for m in loaded_models:
                m_data = rec["models"][m]
                row.extend([str(m_data["psnr"]), str(m_data["ssim"]), str(m_data["mse"])])
            f.write(",".join(row) + "\n")
    print(f"[Saved] Per-patch detailed CSV saved to {csv_path}")

    # Generate Visual Side-by-Side Comparison Plot for Sample 0
    sample_lr = cv2.cvtColor(cv2.imread(str(LR_DIR / "test_0000_lr_10m.jpg")), cv2.COLOR_BGR2RGB)
    sample_real = cv2.cvtColor(cv2.imread(str(REAL_DIR / "test_0000_hr_05m.jpg")), cv2.COLOR_BGR2RGB)
    sample_bic = cv2.cvtColor(cv2.imread(str(BIC_DIR / "test_0000_bicubic_05m.jpg")), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    axes[0].imshow(sample_lr)
    axes[0].set_title("Input (10m LR)")
    axes[0].axis("off")

    axes[1].imshow(sample_bic)
    axes[1].set_title("Bicubic (Baseline)")
    axes[1].axis("off")

    model_names = ["FSRCNN", "ESPCN", "EDSR", "Sentinel2SRNet"]
    for i, name in enumerate(model_names):
        if name in sample_visuals:
            axes[2 + i].imshow(sample_visuals[name])
            axes[2 + i].set_title(f"{name} (SR)")
            axes[2 + i].axis("off")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "benchmark_visual_comparison.png"
    plt.savefig(str(plot_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Visual comparison figure saved to {plot_path}")
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
