# Sentinel2SRNet: Architectural Blueprint and Technical Specification

Complete architectural specification of the custom Sentinel2SRNet student model as implemented in `src/models/sr_model.py` and trained in `train_srnet.py`. Developed by Team PixelForge for Smart India Hackathon 2026 (Problem Statement SIH26142, National Technical Research Organisation - NTRO).

---

## 1. Architectural Summary

| Dimension | Value | Technical Context |
| :--- | :--- | :--- |
| **Model Role** | Distilled Student Network | Learns consensus features from three teachers (EDSR, EESRGAN, DCBCNN). |
| **Trainable Parameters** | **193,601** | Ultra-compact design, over 220 times smaller than EDSR (43M parameters). |
| **Checkpoint Size** | **765.5 KB** (`sr_sentinel2_2x_best.pt`) | Fits easily into embedded devices or low-bandwidth edge deployments. |
| **Native Scale Factor** | **2x** (10m to 5m GSD) | 4-fold increase in spatial pixel density. |
| **Cascaded Mode** | **4x** (10m to 2.5m GSD) | Two-stage cascaded inference achieving sub-4m GSD for cadastral mapping. |
| **Convolutional Layers** | **11 Conv2d Layers** | Optimal depth balancing non-linear expressiveness with fast gradient flow. |
| **Latent Feature Channels** | **48 Channels** | Empirically determined Pareto frontier, saving 43% FLOPs versus 64 channels. |
| **Activation Function** | **PReLU** (Parametric ReLU) | Prevents dead neurons across dark satellite pixels like deep water and shadows. |
| **Receptive Field** | **23 x 23 pixels** (in 10m space) | Corresponds to a physical ground footprint of 230m x 230m. |
| **Upsampling Engine** | **nn.PixelShuffle(2)** | Reorganizes channel depth into spatial pixels, avoiding checkerboard artifacts. |
| **Radiometric Guard** | **Zero-Mean Residual Layer** | Forces residual map mean to zero, mathematically preserving macroscopic brightness. |
| **Average CPU Latency** | **38.3 ms** per 256x256 patch | Interactive map panning on standard non-GPU office computers. |

---

## 2. Layer Pipeline and Tensor Flow

The diagram below details the data flow through the dual-branch network:

```text
Input Tensor: x ~ [Batch, 3, H, W] (10m Resolution)
   │
   ├──> [GLOBAL BICUBIC BRANCH]
   │    F.interpolate(x, scale=2, mode='bicubic', align_corners=False)
   │    Output Shape: [Batch, 3, 2H, 2W] ──────────────────────────────────────────┐
   │                                                                               │
   └──> [DEEP RESIDUAL STUDENT BRANCH]                                             │
        │                                                                          │
        ├── [1. FEATURE EXTRACTION HEAD]                                           │
        │    Conv2d(3 -> 48, kernel=3, padding=1) + PReLU()                        │
        │    Output Shape: [Batch, 48, H, W]                                       │
        │    │                                                                     │
        │    ├── (Identity Skip Connection) ──────────────────────────┐            │
        │    ▼                                                        │            │
        ├── [2. DEEP RESIDUAL BODY]                                   │            │
        │    ├── Residual Block 1: Conv(48,48) -> PReLU -> Conv(48,48) + Skip      │
        │    ├── Residual Block 2: Conv(48,48) -> PReLU -> Conv(48,48) + Skip      │
        │    ├── Residual Block 3: Conv(48,48) -> PReLU -> Conv(48,48) + Skip      │
        │    └── Residual Block 4: Conv(48,48) -> PReLU -> Conv(48,48) + Skip      │
        │    Output Shape: [Batch, 48, H, W]                          │            │
        │    ▼                                                        │            │
        ├── [3. MID-LEVEL FEATURE FUSION]                             │            │
        │    Conv2d(48 -> 48, kernel=3, padding=1)                   │            │
        │    Output Shape: [Batch, 48, H, W]                          │            │
        │    ▼                                                        │            │
        │    (+) Elementwise Addition ◄───────────────────────────────┘            │
        │    ▼                                                                     │
        ├── [4. SUB-PIXEL RECONSTRUCTION TAIL]                                     │
        │    Conv2d(48 -> 12, kernel=3, padding=1)                                │
        │    Tensor Shape: [Batch, 12, H, W]                                       │
        │    nn.PixelShuffle(upscale_factor=2)                                     │
        │    Tensor Shape: [Batch, 3, 2H, 2W]                                      │
        │    ▼                                                                     │
        ├── [5. ZERO-MEAN RESIDUAL CONSTRAINT]                                     │
        │    Residual = Residual - Residual.mean(dim=(-2, -1), keepdim=True)       │
        │    Output Shape: [Batch, 3, 2H, 2W]                                      │
        │    ▼                                                                     │
        └────(+) Baseline Sum ◄────────────────────────────────────────────────────┘
             ▼
        torch.clamp(Base + Residual, 0.0, 1.0)
        Final Output: [Batch, 3, 2H, 2W] (5m Super-Resolved Surface)
```

---

## 3. Mathematical Principles of Key Components

### 1. Residual Learning on Satellite Data
Standard super-resolution networks attempt to predict the full target image from scratch. This forces the model to memorize low-frequency terrain colors, consuming network capacity. Sentinel2SRNet uses residual learning:

$$\text{Output} = \text{Bicubic}(x) + \Delta I$$

The network is dedicated 100% to learning high-frequency spatial residuals ($\Delta I$), while the bicubic baseline preserves global scene context.

### 2. Analytical Zero-Mean Residual Constraint
A critical challenge in satellite imagery is radiometric drift, where super-resolution models inadvertently brighten or darken areas, shifting NDVI vegetation readings or water masks. To prevent this, the residual map is constrained at the output:

$$\Delta I_{\text{zero}} = \Delta I - \mathbb{E}_{x,y}[\Delta I]$$

By enforcing a spatial mean of zero, the network is mathematically incapable of altering the macroscopic brightness or color balance of the scene. It can only inject edge sharpness.

### 3. Parametric Rectified Linear Units (PReLU)
Standard ReLU clips all negative values to zero. In remote sensing, low-reflectance pixels (deep ocean, lakes, dense cloud shadows, narrow building alleys) produce negative activations in early convolutional layers. PReLU learns a per-channel parameter $\alpha$:

$$f(x) = \max(0, x) + \alpha \min(0, x)$$

This ensures non-zero gradient flow across low-reflectance terrain, preserving texture in water bodies and shadow zones.

---

## 4. Multi-Teacher Distillation and Training Loss

During training, Sentinel2SRNet acts as the student in a multi-teacher knowledge distillation system:
* **EDSR** provides structural supervision.
* **EESRGAN** provides edge boundary gradients.
* **DCBCNN** provides directional linear consistency.

The composite training loss is formulated as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + 2.5 \cdot \mathcal{L}_{\text{Sobel}} + 0.5 \cdot \mathcal{L}_{\text{consensus}}$$

1. **Content Fidelity ($L_1$)**: Direct absolute error between the prediction and radiometrically aligned 5m VENuS ground truth.
2. **Sobel Edge Gradient Loss**: Convolves both the prediction and ground truth with horizontal ($S_x$) and vertical ($S_y$) Sobel filters:
   $$\mathcal{L}_{\text{Sobel}} = \|S_x * \hat{y} - S_x * y\|_1 + \|S_y * \hat{y} - S_y * y\|_1$$
   This penalizes blurry transitions on roads, building perimeters, and field borders.
3. **Consensus Distillation Loss**: Distills features only on pixels where all three teachers agree within a tolerance threshold, preventing the student from absorbing synthetic or hallucinated artifacts.

---

## 5. Test-Time Augmentation (TTA) and Uncertainty Estimation

When the user activates 8x TTA in the studio, the input patch undergoes eight dihedral transformations:
* Four rotations: 0 degrees, 90 degrees, 180 degrees, 270 degrees.
* Two horizontal reflections for each rotation.

The model evaluates all eight transformed variants, inverts them to their original orientation, and calculates two products:
1. **Ensemble Mean**: The pixel-wise average, producing an image with reduced directional convolution noise and improved PSNR (+0.2 to +0.4 dB).
2. **Per-Pixel Uncertainty Map**: The pixel-wise variance across the eight predictions:
   $$\sigma^2(x, y) = \frac{1}{8} \sum_{k=1}^8 \left(I_k(x, y) - \mu(x, y)\right)^2$$
   Normalized into a confidence score, where green indicates high model certainty and red highlights ambiguous textures or sensor noise.

---

## 6. Quantitative Evaluation across 740 Test Patches

Evaluation results across 740 test patches from the SEN2VENuS dataset evaluated against authentic 5m VENuS ground truth:

| Model Architecture | Checkpoint File | MSE (Lower is better) | PSNR (dB) (Higher is better) | SSIM (Higher is better) | Delta PSNR vs Bicubic (dB) | Delta SSIM vs Bicubic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Baseline)** | *Interpolation reference* | 338.39 | 23.79 dB | 0.7628 | Baseline | Baseline |
| **DCBCNN (DCB-RCAB)** | `slimdiffsr_sentinel2_2x_best.pt` | 244.79 | 25.15 dB | 0.7769 | +1.36 dB | +0.0141 |
| **EESRGAN** | `eesrgan_sentinel2_2x_best.pt` | 246.57 | 25.05 dB | 0.7745 | +1.26 dB | +0.0117 |
| **EDSR** | `edsr_sentinel2_2x_best.pt` | 257.87 | 24.92 dB | 0.7845 | +1.13 dB | +0.0217 |
| **Sentinel2SRNet (Student)** | `sr_sentinel2_2x_best.pt` | 265.02 | 24.85 dB | **0.7898** | +1.06 dB | **+0.0270** |
| **ESPCN** | `espcn_sentinel2_2x_best.pt` | 278.95 | 24.59 dB | 0.7746 | +0.80 dB | +0.0118 |
| **FSRCNN** | `fsrcnn_sentinel2_2x.pt` | 338.20 | 23.51 dB | 0.7122 | -0.28 dB | -0.0506 |

### Architectural Takeaway
Sentinel2SRNet achieves the highest SSIM (0.7898) among all tested models, including the 43M-parameter EDSR network. This confirms that edge gradient supervision and zero-mean residual projection force the network to prioritize structural line integrity over flat pixel averaging, while running at 38.3 ms per tile on ordinary CPUs.
