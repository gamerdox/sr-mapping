from typing import Tuple, Union
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_numpy_image(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a PyTorch tensor (C, H, W) or (1, C, H, W) or NumPy array to (H, W, C) float32 [0, 1]."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze()
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = img.permute(1, 2, 0)
        img = img.numpy()

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def calculate_psnr(
    image_true: Union[np.ndarray, torch.Tensor],
    image_test: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    ref = to_numpy_image(image_true)
    test = to_numpy_image(image_test)

    # Ensure identical shape
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch for PSNR: {ref.shape} vs {test.shape}")

    val = peak_signal_noise_ratio(ref, test, data_range=1.0)
    return float(round(val, 2))


def calculate_ssim(
    image_true: Union[np.ndarray, torch.Tensor],
    image_test: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Structural Similarity Index (SSIM) in range [0, 1]."""
    ref = to_numpy_image(image_true)
    test = to_numpy_image(image_test)

    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch for SSIM: {ref.shape} vs {test.shape}")

    channel_axis = 2 if ref.ndim == 3 else None
    val = structural_similarity(ref, test, data_range=1.0, channel_axis=channel_axis)
    return float(round(val, 4))


def evaluate_pair(
    image_true: Union[np.ndarray, torch.Tensor],
    image_test: Union[np.ndarray, torch.Tensor],
) -> dict:
    """Convenience helper returning both PSNR and SSIM in a dictionary."""
    return {
        "psnr_db": calculate_psnr(image_true, image_test),
        "ssim": calculate_ssim(image_true, image_test),
    }
