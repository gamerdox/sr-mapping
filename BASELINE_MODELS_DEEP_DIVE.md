# Technical Deep-Dive: Baseline and Teacher Super-Resolution Models

This document provides a technical breakdown of the deep learning baseline and teacher architectures evaluated in our project, alongside classical bicubic interpolation, compared against our custom student network, Sentinel2SRNet.

---

## 1. Master Comparative Architectural Matrix

The table below contrasts the structural properties of each baseline and teacher architecture against our custom Sentinel2SRNet student model:

| Engineering Dimension | FSRCNN (Dong et al.) | ESPCN (Shi et al.) | EDSR (Lim et al.) | EESRGAN (Jakaria et al.) | DCBCNN (Wang et al.) | Sentinel2SRNet (Student) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Architectural Family** | Hourglass CNN | Sub-Pixel CNN | Deep ResNet | Dense RRDB GAN | Directional RCAB | Compact Residual CNN |
| **Role in Project** | Speed Baseline | Lightweight Baseline | Accuracy Teacher | Detail Teacher | Directional Teacher | Deployable Student |
| **Trainable Parameters** | 12,464 | 21,691 | 43,090,000 | 37,580,000 | 6,190,000 | 193,601 |
| **Checkpoint Size** | 38.1 KB | 84.4 KB | 37.6 MB | 37.5 MB | 6.2 MB | 765.5 KB |
| **Number of Conv Layers** | 8 | 3 | 34 (16 ResBlocks) | 48 (8 RRDB + EESN) | 28 (4 Residual Groups) | 11 (4 ResBlocks) |
| **Primary Activations** | PReLU | Tanh / ReLU | ReLU | LeakyReLU (0.2) | LeakyReLU (0.1) | PReLU |
| **Upsampling Method** | Transposed Conv | PixelShuffle | PixelShuffle | Nearest + Conv | PixelShuffle | PixelShuffle + Bicubic Skip |
| **Loss Function** | MSE (L2) | MSE (L2) | MAE (L1) | L1 + Laplacian Edge | L1 + Sobel + FFT | L1 + 2.5 Sobel + Zero-Mean |
| **CPU Latency (per tile)**| ~11.7 ms | ~7.6 ms | ~11,290 ms | ~180 ms | ~30 ms | ~38.3 ms |
| **740-Patch PSNR** | 23.51 dB | 24.59 dB | 24.92 dB | 25.05 dB | 25.15 dB | 24.85 dB |
| **740-Patch SSIM** | 0.7122 | 0.7746 | 0.7845 | 0.7745 | 0.7769 | 0.7898 |

---

## 2. Model 1: FSRCNN (Fast Super-Resolution CNN)

### Architectural Overview
FSRCNN was designed by Chao Dong et al. to eliminate the computational bottleneck of early super-resolution networks, which upsampled images prior to convolution. It processes all feature representations in compact low-resolution space through an hourglass structure, upsampling only in the final layer using transposed convolution.

### Layer Pipeline
1. **Feature Extraction**: Conv2d(3 to 56, kernel=5, padding=2) with PReLU activation.
2. **Shrinking**: Conv2d(56 to 12, kernel=1, padding=0) with PReLU activation, compressing channel depth to reduce arithmetic cost.
3. **Non-Linear Mapping**: Four cascaded Conv2d(12 to 12, kernel=3, padding=1) layers with PReLU activations.
4. **Expanding**: Conv2d(12 to 56, kernel=1, padding=0) with PReLU activation, restoring channel depth.
5. **Deconvolution**: ConvTranspose2d(56 to 3, kernel=9, stride=2, padding=4, output_padding=1).

### Remote Sensing Evaluation and Failure Modes
* **Checkerboard Artifacts**: Transposed convolution with a 9x9 kernel and stride 2 produces uneven kernel overlap. This manifests as periodic grid artifacts across agricultural fields and linear roadways.
* **Over-Smoothing**: Optimized strictly under Mean Squared Error (L2 loss), FSRCNN penalizes large pixel deviations quadratically. When faced with stochastic satellite textures like forest canopy, it outputs the statistical mean, resulting in a low SSIM score of 0.7122 and a negative PSNR gain (-0.28 dB) relative to bicubic interpolation.

---

## 3. Model 2: ESPCN (Efficient Sub-Pixel CNN)

### Architectural Overview
Introduced by Wenzhe Shi et al., ESPCN introduced the sub-pixel convolutional layer (standardized as PixelShuffle in PyTorch). It demonstrated that spatial upsampling can be achieved through regular channel rearrangement rather than transposed convolutions, avoiding checkerboard artifacts entirely.

### Layer Pipeline
1. **Low-Resolution Feature Map**: Conv2d(3 to 64, kernel=5, padding=2) with Tanh or ReLU activation.
2. **Intermediate Mapping**: Conv2d(64 to 32, kernel=3, padding=1) with Tanh or ReLU activation.
3. **Sub-Pixel Projection**: Conv2d(32 to 3 * 2^2 = 12, kernel=3, padding=1), followed by PixelShuffle(upscale_factor=2) to generate the 5m spatial output.

### Remote Sensing Evaluation and Failure Modes
* **Limited Receptive Field**: With only three convolutional layers, ESPCN has a receptive field of 9x9 pixels. It cannot capture broad spatial contexts like extended highway corridors, river bends, or large building boundaries.
* **Performance**: It achieves a PSNR of 24.59 dB (+0.80 dB over bicubic) with fast CPU execution (~7.6 ms), making it a solid lightweight baseline, but it lacks the edge sharpness necessary for cadastral mapping.

---

## 4. Model 3: EDSR (Enhanced Deep Residual Networks)

### Architectural Overview
Developed by Bee Lim et al., EDSR won the NTIRE 2017 Super-Resolution Challenge. The authors removed Batch Normalization layers from traditional residual blocks, which freed up substantial GPU memory and prevented the normalization of absolute radiometric values.

### Layer Pipeline
* **Input Projection**: Conv2d(3 to 256, kernel=3, padding=1).
* **Deep Residual Body**: 16 Residual Blocks, each containing Conv2d(256 to 256) -> ReLU -> Conv2d(256 to 256) with a residual scaling factor of 0.1 and an identity skip connection.
* **Global Residual Connection**: Elementwise addition of shallow input features with deep body features.
* **Reconstruction**: Sub-pixel convolution expanding 256 channels to the final high-resolution grid.

### Remote Sensing Evaluation and Failure Modes
* **High Structural Fidelity**: With over 43 million parameters, EDSR captures fine spatial details, achieving 24.92 dB PSNR (+1.13 dB over bicubic) and 0.7845 SSIM.
* **Severe Latency Bottleneck**: On standard CPU hardware, EDSR requires over 11 seconds per 256x256 patch, rendering it completely unusable for interactive web maps or district-level field operations without dedicated GPU clusters. In our framework, EDSR serves as the Accuracy Teacher.

---

## 5. Model 4: EESRGAN (Edge-Enhanced Super-Resolution GAN)

### Architectural Overview
Based on Jakaria et al., EESRGAN combines 8 Residual-in-Residual Dense Blocks (RRDB) with a dedicated Edge Enhancement Sub-Network (EESN). The EESN extracts second-order spatial Laplacian edge maps from intermediate representations, applies a dense mask attention mechanism, and reinjects learned edge features into the final reconstruction.

### Remote Sensing Evaluation and Failure Modes
* **Edge Acuity**: EESRGAN reconstructs sharp building footprints, field borders, and coastal margins, achieving 25.05 dB PSNR (+1.26 dB over bicubic).
* **Role in Project**: EESRGAN serves as the Detail Teacher in our framework. Its learned edge representations guide the student model, ensuring that fine spatial boundaries are preserved without hallucinating artificial features.

---

## 6. Model 5: DCBCNN (Directional Convolutional Network)

### Architectural Overview
Adapted from the SlimDiffSR directional architecture (Wang et al., Wuhan University), DCBCNN uses Directional Convolution Blocks (DCB) paired with Squeeze-and-Excitation (SE) channel attention across 4 residual groups. The DCB replaces standard 3x3 convolutions with parallel 1x3, 3x1, and 3x3 kernels to capture linear structures.

### Remote Sensing Evaluation
* **Top PSNR Performance**: Achieved the highest PSNR across all 740 test patches (25.15 dB, +1.36 dB over bicubic) and the lowest MSE (244.79). Its directional filters align naturally with satellite imagery features like roads, shorelines, and irrigation ditches.
* **Role in Project**: DCBCNN acts as the Directional Feature Expert in our multi-teacher distillation framework.

---

## 7. Model 6: Sentinel2SRNet (Student Model)

### Architectural Overview
Sentinel2SRNet was engineered as a lightweight student model (~193.6K parameters) trained via consensus knowledge distillation from EDSR, EESRGAN, and DCBCNN:
* **Feature Extraction**: Conv2d(3 to 48, kernel=3, padding=1) with PReLU activation.
* **Residual Body**: 4 cascaded residual blocks with local identity shortcuts.
* **Sub-Pixel Tail**: Conv2d(48 to 12, kernel=3, padding=1) with PixelShuffle(2).
* **Bicubic Addition**: Global skip connection adding bicubic input to the learned residual.
* **Zero-Mean Constraint**: Forces the residual map to have a spatial mean of zero, guaranteeing zero global brightness shift.

### Remote Sensing Evaluation
* **Top Structural Similarity**: Achieved the highest SSIM (0.7898, +0.0270 over bicubic), outperforming all baseline and teacher models in edge alignment.
* **Real-Time CPU Speed**: Runs in roughly 38.3 ms on standard CPU hardware, providing an optimal trade-off between structural fidelity and field deployability.

---

## 8. Quantitative Benchmark Summary (740 Unseen Test Patches)

Evaluation conducted on 740 test patches from the SEN2VENuS dataset evaluated against authentic 5m VENuS ground truth:

| Model Architecture | Checkpoint File | MSE (Lower is better) | PSNR (dB) (Higher is better) | SSIM (Higher is better) | Delta PSNR vs Bicubic (dB) | Delta SSIM vs Bicubic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Baseline)** | *Interpolation reference* | 338.39 | 23.79 dB | 0.7628 | Baseline | Baseline |
| **DCBCNN (DCB-RCAB)** | `slimdiffsr_sentinel2_2x_best.pt` | 244.79 | 25.15 dB | 0.7769 | +1.36 dB | +0.0141 |
| **EESRGAN** | `eesrgan_sentinel2_2x_best.pt` | 246.57 | 25.05 dB | 0.7745 | +1.26 dB | +0.0117 |
| **EDSR** | `edsr_sentinel2_2x_best.pt` | 257.87 | 24.92 dB | 0.7845 | +1.13 dB | +0.0217 |
| **Sentinel2SRNet (Student)** | `sr_sentinel2_2x_best.pt` | 265.02 | 24.85 dB | 0.7898 | +1.06 dB | +0.0270 |
| **ESPCN** | `espcn_sentinel2_2x_best.pt` | 278.95 | 24.59 dB | 0.7746 | +0.80 dB | +0.0118 |
| **FSRCNN** | `fsrcnn_sentinel2_2x.pt` | 338.20 | 23.51 dB | 0.7122 | -0.28 dB | -0.0506 |
