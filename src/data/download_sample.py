import os
import sys
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
import py7zr
import torch
import numpy as np
from PIL import Image

ZENODO_RECORD_URL = "https://zenodo.org/api/records/6514159/files/FGMANAUS.7z/content"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sen2venus_sample"
ARCHIVE_PATH = DEFAULT_DATA_DIR / "FGMANAUS.7z"
EXTRACT_PATH = DEFAULT_DATA_DIR / "FGMANAUS"
DEMO_DIR = DEFAULT_DATA_DIR / "demo_patches"


def download_file(url: str, dest_path: Path):
    """Download a file with a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[Info] Archive already exists at: {dest_path}")
        return

    print(f"[Downloading] Fetching sample site FGMANAUS (~62.7 MB) from Zenodo...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc="Downloading FGMANAUS.7z",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[Done] Download completed: {dest_path}")


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract .7z archive using py7zr."""
    if extract_to.exists() and any(extract_to.glob("*.pt")):
        print(f"[Info] Dataset already extracted at: {extract_to}")
        return

    print(f"[Extracting] Extracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_to)
    print(f"[Done] Extracted to {extract_to}")


def normalize_to_rgb(tensor_b2b3b4: torch.Tensor) -> np.ndarray:
    """
    Convert a 3-channel tensor (B2:Blue, B3:Green, B4:Red) to standard 8-bit RGB.
    Applies percentile clipping (2% to 98%) typical for remote-sensing reflectance.
    """
    # Sentinel-2/VENUS band order in b2b3b4b8: [B2(Blue), B3(Green), B4(Red), B8(NIR)]
    # Convert B2, B3, B4 -> R, G, B:
    # Red is index 2, Green is index 1, Blue is index 0
    rgb = tensor_b2b3b4[[2, 1, 0], :, :].cpu().numpy().astype(np.float32)

    # Reflectance values are usually in range [0, 10000] or [0, 1]
    # Percentile stretch for visual clarity
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb_norm = (rgb - p2) / (p98 - p2)
    else:
        rgb_norm = rgb / (rgb.max() + 1e-6)

    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
    return np.transpose(rgb_uint8, (1, 2, 0))  # (H, W, 3)


def export_demo_patches(extract_path: Path, demo_dir: Path, max_samples: int = 5):
    """Extract sample low-res and high-res pairs into ready-to-view PNG images."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate paired .pt files
    sen_files = sorted(extract_path.rglob("*_10m_b2b3b4b8.pt"))
    if not sen_files:
        print("[Warning] No Sentinel-2 10m files found to create demo patches.")
        return

    sample_count = 0
    for sen_file in sen_files:
        # Expected matching Venus file
        prefix = sen_file.name.replace("_10m_b2b3b4b8.pt", "")
        venus_file = sen_file.parent / f"{prefix}_05m_b2b3b4b8.pt"
        if not venus_file.exists():
            continue

        sen_data = torch.load(sen_file, map_location="cpu", weights_only=False)
        venus_data = torch.load(venus_file, map_location="cpu", weights_only=False)

        num_patches = sen_data.shape[0] if sen_data.ndim == 4 else 1
        for i in range(num_patches):
            if sample_count >= max_samples:
                break

            sen_patch = sen_data[i] if sen_data.ndim == 4 else sen_data
            venus_patch = venus_data[i] if venus_data.ndim == 4 else venus_data

            # Export LR (10m Sentinel-2) and HR (5m VENUS target)
            lr_rgb = normalize_to_rgb(sen_patch[:3])
            hr_rgb = normalize_to_rgb(venus_patch[:3])

            # Export Bicubic baseline upsampling (2x)
            lr_img = Image.fromarray(lr_rgb)
            hr_img = Image.fromarray(hr_rgb)
            bicubic_img = lr_img.resize(hr_img.size, resample=Image.BICUBIC)

            lr_img.save(demo_dir / f"sample_{sample_count}_lr_10m.png")
            hr_img.save(demo_dir / f"sample_{sample_count}_target_05m.png")
            bicubic_img.save(demo_dir / f"sample_{sample_count}_bicubic_05m.png")

            sample_count += 1

        if sample_count >= max_samples:
            break

    print(f"[Done] Exported {sample_count} sample patch sets into: {demo_dir}")
    print(f"       Each sample includes: LR (10m), Bicubic Baseline (5m), and Ground Truth VEN\u00b5S (5m).")


def main():
    print("=" * 60)
    print("SEN2VEN\u00b5S Sample Dataset Downloader")
    print("=" * 60)
    download_file(ZENODO_RECORD_URL, ARCHIVE_PATH)
    extract_archive(ARCHIVE_PATH, EXTRACT_PATH)
    export_demo_patches(EXTRACT_PATH, DEMO_DIR)
    print("\n[Success] Dataset sample ready for inference and backend testing!")


if __name__ == "__main__":
    main()
