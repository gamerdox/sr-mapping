# Model Benchmark, Loss Formulations and Performance Analysis

Comprehensive empirical evaluation, loss formulations, and comparative analysis of super-resolution models on the SEN2VENuS Sentinel-2 dataset. Developed by Team PixelForge for Smart India Hackathon 2026 (Problem Statement SIH26142, National Technical Research Organisation - NTRO).

---

## 1. Mathematical Loss Formulations

Super-resolution models exhibit distinct visual and radiometric characteristics governed by their training objective functions:

| Model | Loss Type | Mathematical Formulation | Optimization Behavior and Artifact Profile |
| :--- | :--- | :--- | :--- |
| **ESPCN** | Mean Squared Error (L2) | $\mathcal{L} = \frac{1}{N} \sum (I_{\text{SR}} - I_{\text{HR}})^2$ | Squares error deviations. Tends to over-smooth high-frequency satellite textures. |
| **FSRCNN** | Mean Squared Error (L2) | $\mathcal{L} = \frac{1}{N} \sum (I_{\text{SR}} - I_{\text{HR}})^2$ | Produces blurry edges on linear features and negative PSNR gain versus bicubic baseline. |
| **EDSR** | Mean Absolute Error (L1) | $\mathcal{L} = \frac{1}{N} \sum \|I_{\text{SR}} - I_{\text{HR}}\|_1$ | Penalizes errors linearly, maintaining median color boundaries without Gaussian blurring. |
| **EESRGAN** | Composite L1 + Laplacian | $\mathcal{L} = \mathcal{L}_{1} + 0.15 \cdot \| \nabla^2 I_{\text{SR}} - \nabla^2 I_{\text{HR}} \|_1$ | Enforces sharp edge boundaries along buildings and shorelines via second-order spatial derivatives. |
| **DCBCNN** | Multi-Objective Spatial & Frequency | $\mathcal{L} = \mathcal{L}_{1} + 0.10 \cdot \mathcal{L}_{\text{FFT}} + 0.05 \cdot \mathcal{L}_{\text{Sobel}}$ | Optimizes both spatial edge gradients and Fourier frequency amplitudes for linear corridors. |
| **Sentinel2SRNet** | Physics-Informed Consensus Distillation | $\mathcal{L} = \mathcal{L}_{1} + 2.5 \cdot \mathcal{L}_{\text{Sobel}} + 0.5 \cdot \mathcal{L}_{\text{consensus}}$ | Enforces edge alignment with real road networks while distilling consensus features from teachers. |

### Mathematical Components of Sentinel2SRNet Training:

1. **Content Fidelity ($L_1$)**:
   $$\mathcal{L}_{1} = \frac{1}{H \cdot W \cdot C} \sum_{c, y, x} \left| I_{\text{SR}}(x, y, c) - I_{\text{HR}}(x, y, c) \right|$$
   Computes voxel-wise absolute differences against radiometrically aligned 5m VENuS ground truth.

2. **Sobel Edge Gradient Loss ($\mathcal{L}_{\text{Sobel}}$)**:
   $$\mathcal{L}_{\text{Sobel}} = \left\| S_x * I_{\text{SR}} - S_x * I_{\text{HR}} \right\|_1 + \left\| S_y * I_{\text{SR}} - S_y * I_{\text{HR}} \right\|_1$$
   Where $S_x$ and $S_y$ are 3x3 horizontal and vertical Sobel convolution kernels. This term forces the convolutional weights to align their high-frequency gradient magnitude with real satellite road networks and parcel borders.

3. **Consensus Knowledge Distillation Loss ($\mathcal{L}_{\text{consensus}}$)**:
   Penalizes the student model if it generates high-frequency detail on pixels where the three teachers (EDSR, EESRGAN, DCBCNN) disagree, suppressing synthetic hallucination.

4. **Zero-Mean Residual Projection Constraint**:
   $$I_{\text{residual}} \leftarrow I_{\text{residual}} - \mathbb{E}[I_{\text{residual}}]$$
   Guarantees that the learned high-frequency residual has zero net DC offset, preventing the network from shifting overall scene illumination or corrupting vegetative indices.

---

## 2. Quantitative Evaluation Across 740 Test Patches

All models were evaluated across 740 unseen test patches from the SEN2VENuS dataset covering three diverse biomes: dense Amazon rainforest (FGMANAUS), coastal wetlands and estuaries (ESTUAMAR), and European urban and agricultural plains (ALSACE).

### Master Evaluation Leaderboard

| Model Architecture | Checkpoint File | MSE (Lower is better) | PSNR (dB) (Higher is better) | SSIM (Higher is better) | Delta PSNR vs Bicubic (dB) | Delta SSIM vs Bicubic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Baseline)** | *Interpolation reference* | 338.39 | 23.79 dB | 0.7628 | Baseline | Baseline |
| **DCBCNN (DCB-RCAB)** | `slimdiffsr_sentinel2_2x_best.pt` | **244.79** | **25.15 dB** | 0.7769 | **+1.36 dB** | +0.0141 |
| **EESRGAN** | `eesrgan_sentinel2_2x_best.pt` | 246.57 | 25.05 dB | 0.7745 | +1.26 dB | +0.0117 |
| **EDSR** | `edsr_sentinel2_2x_best.pt` | 257.87 | 24.92 dB | 0.7845 | +1.13 dB | +0.0217 |
| **Sentinel2SRNet (Student)** | `sr_sentinel2_2x_best.pt` | 265.02 | 24.85 dB | **0.7898** | +1.06 dB | **+0.0270** |
| **ESPCN** | `espcn_sentinel2_2x_best.pt` | 278.95 | 24.59 dB | 0.7746 | +0.80 dB | +0.0118 |
| **FSRCNN** | `fsrcnn_sentinel2_2x.pt` | 338.20 | 23.51 dB | 0.7122 | -0.28 dB | -0.0506 |

---

## 3. In-Depth Metric Analysis

### 1. Reconstruction Fidelity (PSNR and MSE)
* **DCBCNN** achieved the highest PSNR (25.15 dB, +1.36 dB over bicubic) and lowest MSE (244.79). Its frequency-separable directional convolutions (1x3 and 3x1 kernels) align with the rectilinear structure of satellite scenes, capturing roads, runways, and agricultural drainage canals.
* **EESRGAN** achieved the second-highest PSNR (25.05 dB, +1.26 dB over bicubic) and MSE (246.57), demonstrating that Laplacian edge supervision guides spatial feature reconstruction effectively.
* **EDSR** scored 24.92 dB PSNR (+1.13 dB over bicubic), confirming the power of deep residual networks, but at substantial computational cost.

### 2. Structural Edge Preservation (SSIM)
* **Sentinel2SRNet** achieved the highest structural similarity score (0.7898, a gain of +0.0270 over bicubic), outperforming even the 43M-parameter EDSR model.
* This outcome demonstrates the advantage of combining explicit Sobel edge gradient loss with an analytical zero-mean residual constraint. By forcing the residual map to focus strictly on structural edges, Sentinel2SRNet avoids spending network capacity on flat illumination adjustments.

### 3. Computational Efficiency and Field Usability

| Model | Parameters | Checkpoint Size | CPU Latency (per tile) | Practical Deployment Tier |
| :--- | :---: | :---: | :---: | :--- |
| **FSRCNN** | 12,464 | 38.1 KB | ~11.7 ms | Interactive fast-preview mode |
| **ESPCN** | 21,691 | 84.4 KB | ~7.6 ms | Ultra-low-resource embedded preview |
| **Sentinel2SRNet (Ours)**| **193,601** | **765.5 KB** | **~38.3 ms** | **District and field office PCs (CPU)** |
| **DCBCNN** | 6,191,974 | 6.2 MB | ~30.0 ms | State and regional agency servers |
| **EESRGAN** | 37,583,211 | 37.6 MB | ~180.0 ms | Regional GPU server clusters |
| **EDSR** | 43,090,000 | 37.6 MB | ~11,290.0 ms | National headquarters supercomputing centers |

Sentinel2SRNet operates at roughly 38.3 ms on standard CPU hardware, providing approximately 70 times faster throughput than EDSR while matching or exceeding its structural sharpness on satellite imagery.
