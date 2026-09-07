import sys
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sr_model import load_model
from src.models.metrics import calculate_psnr, calculate_ssim
from src.data.dataset import get_demo_image_paths


def calculate_edge_sharpness(img_np: np.ndarray) -> float:
    """Computes average spatial gradient magnitude using Sobel filters."""
    # Ensure float [0, 255]
    if img_np.max() <= 1.0:
        img_np = img_np * 255.0
    gx = sobel(img_np, axis=0)
    gy = sobel(img_np, axis=1)
    mag = np.hypot(gx, gy)
    return float(round(mag.mean(), 1))


def run_inference_on_sample(
    model,
    lr_path: Path,
    bicubic_path: Path,
    target_path: Path,
    output_path: Path,
) -> dict:
    """Run super-resolution inference on a single sample, evaluate metrics, and save visual comparison."""
    lr_pil = Image.open(lr_path).convert("RGB")
    bicubic_pil = Image.open(bicubic_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")

    # Float tensor [0.0, 1.0]
    lr_np = np.array(lr_pil).astype(np.float32) / 255.0
    lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0)

    start_time = time.perf_counter()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_pil = Image.fromarray((sr_np * 255.0).astype(np.uint8))

    target_np = np.array(target_pil).astype(np.float32) / 255.0
    bicubic_np = np.array(bicubic_pil).astype(np.float32) / 255.0

    # Brightness & contrast tracking
    bic_mean = float(round(bicubic_np.mean() * 255.0, 1))
    sr_mean = float(round(sr_np.mean() * 255.0, 1))
    brightness_shift = round(sr_mean - bic_mean, 2)

    # Edge sharpness energy
    edge_bic = calculate_edge_sharpness(bicubic_np)
    edge_sr = calculate_edge_sharpness(sr_np)
    edge_hr = calculate_edge_sharpness(target_np)
    sharpness_gain = round(edge_sr - edge_bic, 1)

    # Radiometric alignment for fair spatial metric evaluation
    m_tgt = target_np.mean()
    s_tgt = target_np.std()
    target_aligned = np.clip(
        (target_np - m_tgt) / (s_tgt + 1e-5) * bicubic_np.std() + bicubic_np.mean(),
        0.0,
        1.0,
    )

    bicubic_psnr = calculate_psnr(target_aligned, bicubic_np)
    bicubic_ssim = calculate_ssim(target_aligned, bicubic_np)
    sr_psnr = calculate_psnr(target_aligned, sr_np)
    sr_ssim = calculate_ssim(target_aligned, sr_np)

    # Plot 4-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(lr_pil)
    axes[0].set_title(f"1. Sentinel-2 Input\n10m ({lr_pil.width}x{lr_pil.height})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(bicubic_pil)
    axes[1].set_title(
        f"2. Bicubic Baseline\n5m ({bicubic_pil.width}x{bicubic_pil.height})\nEdge Sharpness: {edge_bic} | PSNR: {bicubic_psnr} dB",
        fontsize=10,
    )
    axes[1].axis("off")

    axes[2].imshow(sr_pil)
    axes[2].set_title(
        f"3. Sentinel2SRNet (AI)\n5m ({sr_pil.width}x{sr_pil.height})\nEdge Sharpness: {edge_sr} ({sharpness_gain:+0.1f})\nBrightness shift: {brightness_shift:+.1f}",
        fontsize=10,
        color="darkgreen" if sharpness_gain >= 0 else "black",
        fontweight="bold",
    )
    axes[2].axis("off")

    axes[3].imshow(target_pil)
    axes[3].set_title(
        f"4. VEN\u00b5S Ground Truth\n5m Reference ({target_pil.width}x{target_pil.height})\nEdge Sharpness: {edge_hr}",
        fontsize=10,
        fontweight="bold",
    )
    axes[3].axis("off")

    plt.suptitle(
        f"Spatial Super-Resolution Evaluation (Zero-Mean Residual, Edge-Penalized Loss | {latency_ms:.1f}ms on CPU)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "latency_ms": round(latency_ms, 2),
        "bicubic_mean": bic_mean,
        "sr_mean": sr_mean,
        "brightness_shift": brightness_shift,
        "edge_bic": edge_bic,
        "edge_sr": edge_sr,
        "edge_hr": edge_hr,
        "sharpness_gain": sharpness_gain,
        "bicubic_psnr": bicubic_psnr,
        "sr_psnr": sr_psnr,
        "sr_ssim": sr_ssim,
    }


def main():
    print("=" * 80)
    print("Edge-Penalized Sentinel-2 Super-Resolution Benchmark")
    print("=" * 80)

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "sr_sentinel2_2x.pt"
    demo_dir = PROJECT_ROOT / "data" / "sen2venus_sample" / "demo_patches"
    samples = get_demo_image_paths(demo_dir)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(str(checkpoint_path))

    header = f"{'Sample':<8} | {'Brightness Shift':<17} | {'Bicubic Edge':<13} | {'Model Edge':<11} | {'Edge Gain':<10} | {'Latency':<8}"
    print(header)
    print("-" * len(header))

    for item in samples:
        s_id = item["id"]
        out_file = results_dir / f"comparison_sample_{s_id}.png"
        metrics = run_inference_on_sample(
            model,
            Path(item["lr_path"]),
            Path(item["bicubic_path"]),
            Path(item["target_path"]),
            out_file,
        )
        print(
            f"Sample {s_id:<2} | {metrics['brightness_shift']:>+15.2f}   | {metrics['edge_bic']:>11.1f}   | {metrics['edge_sr']:>9.1f}   | {metrics['sharpness_gain']:>+8.1f}   | {metrics['latency_ms']:>6.1f}ms"
        )

    print("-" * len(header))
    print(f"\n[Done] Visual comparison images saved to: {results_dir}")


if __name__ == "__main__":
    main()
