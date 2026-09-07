from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset
import numpy as np


class SentinelVenusDataset(Dataset):
    """
    Dataset loader for paired Sentinel-2 (10m) and VENUS (5m) image patches.
    Normalizes remote sensing bands to [0.0, 1.0] with 2%-98% percentile stretch.
    """

    def __init__(self, root_dir: str | Path, rgb_only: bool = True):
        self.root_dir = Path(root_dir)
        self.rgb_only = rgb_only
        self.samples: List[Tuple[Path, Path, int]] = []

        sen_files = sorted(self.root_dir.rglob("*_10m_b2b3b4b8.pt"))
        for sen_file in sen_files:
            prefix = sen_file.name.replace("_10m_b2b3b4b8.pt", "")
            venus_file = sen_file.parent / f"{prefix}_05m_b2b3b4b8.pt"
            if venus_file.exists():
                sen_data = torch.load(sen_file, map_location="cpu", weights_only=False)
                num_patches = sen_data.shape[0] if sen_data.ndim == 4 else 1
                for i in range(num_patches):
                    self.samples.append((sen_file, venus_file, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sen_file, venus_file, patch_idx = self.samples[idx]
        sen_data = torch.load(sen_file, map_location="cpu", weights_only=False)
        venus_data = torch.load(venus_file, map_location="cpu", weights_only=False)

        lr_patch = sen_data[patch_idx] if sen_data.ndim == 4 else sen_data
        hr_patch = venus_data[patch_idx] if venus_data.ndim == 4 else venus_data

        if self.rgb_only:
            lr_np = lr_patch[[2, 1, 0], :, :].cpu().numpy().astype(np.float32)
            hr_np = hr_patch[[2, 1, 0], :, :].cpu().numpy().astype(np.float32)

            p2_l, p98_l = np.percentile(lr_np, (2, 98))
            p2_h, p98_h = np.percentile(hr_np, (2, 98))

            denom_l = p98_l - p2_l if p98_l > p2_l else lr_np.max() + 1e-5
            denom_h = p98_h - p2_h if p98_h > p2_h else hr_np.max() + 1e-5

            lr_norm = np.clip((lr_np - p2_l) / denom_l, 0.0, 1.0).astype(np.float32)
            hr_norm = np.clip((hr_np - p2_h) / denom_h, 0.0, 1.0).astype(np.float32)

            return torch.from_numpy(lr_norm).float(), torch.from_numpy(hr_norm).float()

        return lr_patch.float(), hr_patch.float()


def get_demo_image_paths(demo_dir: str | Path) -> List[dict]:
    """Returns a list of demo patch triplets (LR, Bicubic, Target)."""
    p = Path(demo_dir)
    samples = []
    for lr_file in sorted(p.glob("sample_*_lr_10m.png")):
        sample_id = lr_file.stem.split("_")[1]
        bicubic_file = p / f"sample_{sample_id}_bicubic_05m.png"
        target_file = p / f"sample_{sample_id}_target_05m.png"
        if bicubic_file.exists() and target_file.exists():
            samples.append({
                "id": int(sample_id),
                "lr_path": str(lr_file),
                "bicubic_path": str(bicubic_file),
                "target_path": str(target_file)
            })
    return samples
