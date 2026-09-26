# Master Project Documentation: Sentinel-2 Satellite Super-Resolution Studio

Comprehensive technical guide to the AI Satellite Super-Resolution Mapping system developed by Team PixelForge for Smart India Hackathon 2026 (Problem Statement SIH26142, National Technical Research Organisation - NTRO).

---

## Table of Contents
1. [Executive Summary and Problem Context](#1-executive-summary-and-problem-context)
2. [The Remote Sensing Challenge: Optical Limits of Sentinel-2](#2-the-remote-sensing-challenge-optical-limits-of-sentinel-2)
3. [The Core Mission: 2x Native and Cascaded 2.5m Sub-4m Super-Resolution](#3-the-core-mission-2x-native-and-cascaded-25m-sub-4m-super-resolution)
4. [Teacher-Student Knowledge Distillation Framework](#4-teacher-student-knowledge-distillation-framework)
5. [Consensus Distillation to Eliminate Feature Hallucination](#5-consensus-distillation-to-eliminate-feature-hallucination)
6. [Radiometric Safety: YCrCb Separation and Zero-Mean Residual Constraint](#6-radiometric-safety-ycrcb-separation-and-zero-mean-residual-constraint)
7. [8x Test-Time Augmentation (TTA) and Uncertainty Quantification](#7-8x-test-time-augmentation-tta-and-uncertainty-quantification)
8. [Spectral Analysis Modes: False-Color CIR and NDVI VARI](#8-spectral-analysis-modes-false-color-cir-and-ndvi-vari)
9. [Quantitative Benchmark Evaluation across 740 Test Patches](#9-quantitative-benchmark-evaluation-across-740-test-patches)
10. [Multi-Tier Deployment Across Government Echelons](#10-multi-tier-deployment-across-government-echelons)
11. [Operational Use Cases and Real-World Impact](#11-operational-use-cases-and-real-world-impact)
12. [Developer Quickstart and Local Execution](#12-developer-quickstart-and-local-execution)

---

## 1. Executive Summary and Problem Context

Every 5 days, the European Space Agency Copernicus Sentinel-2 constellation acquires open-access multispectral optical imagery covering the Earth landmass. Because the satellites operate at an orbital altitude of 786 kilometers, their highest optical resolution is limited to 10 meters per pixel ground sample distance (GSD).

At 10 meters per pixel:
* A single pixel encompasses 100 square meters on the ground, roughly the footprint of a suburban house.
* Narrow asphalt roads, rural pathways, and irrigation canals are blurred into surrounding terrain.
* Building footprints, industrial boundaries, and agricultural parcels appear blocky and indistinct.
* In emergency disaster response (floods, landslides) and defense reconnaissance, 10m imagery is insufficient to detect passable vehicle corridors or localized structural collapse.

Higher-resolution satellites (such as the CNES 5m VENuS scientific satellite or 0.5m commercial satellites) exist, but they are expensive, proprietary, or lack global continuous revisit. This project uses deep learning to transform free 10m Sentinel-2 imagery into 5m resolution (a 4-fold increase in pixel density) and cascaded 2.5m resolution (a 16-fold increase), bringing commercial-grade mapping clarity to low-resource government desktop PCs at zero acquisition cost.

---

## 2. The Remote Sensing Challenge: Optical Limits of Sentinel-2

Satellite optical sensors are constrained by physical laws:
1. **Diffraction Limits and Point Spread Function (PSF)**: Atmospheric turbulence and optical aperture diffraction blur high-frequency spatial gradients, spreading point light sources across adjacent detector pixels.
2. **Signal-to-Noise Ratio (SNR) Trade-off**: Acquiring narrower ground pixels at orbital speeds reduces photon count per detector element, increasing electronic sensor noise.
3. **Bandwidth and Storage**: Downlinking planetary imagery at sub-meter resolutions continuously exceeds civilian satellite radio frequency bandwidth.

Super-resolution mapping computationally inverts the sensor optical transfer function, reconstructing sub-pixel spatial high frequencies while preserving authentic multispectral reflectance.

---

## 3. The Core Mission: 2x Native and Cascaded 2.5m Sub-4m Super-Resolution

The project delivers a dual-stage super-resolution pipeline:
* **Stage 1: Native 2x Super-Resolution (10m to 5m GSD)**: Reconstructs authentic 5-meter ground sampling distance, matching the optical quality of the CNES/ISA VENuS satellite constellation.
* **Stage 2: Cascaded 4x Super-Resolution (10m to 2.5m GSD)**: Executes a second-pass inference on the enhanced 5m feature map, breaking the sub-4m GSD barrier necessary for cadastral mapping, individual building identification, and roadway corridor routing.

---

## 4. Teacher-Student Knowledge Distillation Framework

To bridge the gap between high-capacity research networks and low-resource district-level field offices, we employ a multi-teacher knowledge distillation strategy. Three specialized expert networks serve as teachers, transferring their learned representations to a compact student network:

```text
[Teacher 1: EDSR] ──────┐
(Accuracy Expert)        │
                         ├──> [Consensus Filtering Gate] ──> [Sentinel2SRNet Student]
[Teacher 2: EESRGAN] ────┤    (Agreement-Only Distillation)  (193.6K Params, CPU-Ready)
(Detail Expert)          │
                         │
[Teacher 3: DCBCNN] ─────┘
(Directional Expert)
```

### The Three Teacher Models:
1. **Accuracy Expert (EDSR)**: A 34-layer, 16-residual-block network with 43 million parameters. It stays strictly true to factual ground truth pixel intensities, providing a strong anchor against structural drift.
2. **Detail Expert (EESRGAN)**: An edge-enhanced generative network utilizing Residual-in-Residual Dense Blocks (RRDB) coupled with second-order Laplacian operators. It specializes in resolving sharp building eaves, boundary walls, and property lines.
3. **Directional Expert (DCBCNN)**: A directional network employing parallel 1x3, 3x1, and 3x3 convolutions with channel attention across 4 residual groups. It specializes in linear features such as roadways, shorelines, and field irrigation ditches.

### The Student Model: Sentinel2SRNet
* **Architecture**: A compact 4-residual-block network with PReLU activations, a global bicubic skip connection, and a sub-pixel convolution tail.
* **Size**: 193,601 trainable parameters and a 765.5 KB disk footprint (over 220 times smaller than EDSR).
* **Speed**: Executes in approximately 38.3 ms per 256x256 patch on standard laptop CPUs, roughly 70 times faster than EDSR on CPU.

---

## 5. Consensus Distillation to Eliminate Feature Hallucination

A primary hazard of deep super-resolution in mission-critical applications (such as defense, taxation, or disaster relief) is artificial hallucination, where a generative model invents realistic-looking roads or buildings that do not exist.

Our consensus distillation framework penalizes the student model if it generates high-frequency detail on pixels where the three teachers disagree:
* If EDSR, EESRGAN, and DCBCNN all confirm a structural edge, the student learns the high-frequency feature.
* If EESRGAN suggests an edge that EDSR and DCBCNN identify as texture noise, the consensus gate suppresses the gradient update.
* This ensures that the student model outputs conservative, reliable cartographic features without synthetic artifacts.

---

## 6. Radiometric Safety: YCrCb Separation and Zero-Mean Residual Constraint

Standard super-resolution models trained on consumer photos process RGB channels directly, causing severe radiometric distortion and color bleeding across satellite bands. Our architecture enforces two strict physics-based protections:

### 1. Luminance Separation (YCrCb Processing)
* The input 10m patch is converted to YCrCb color space.
* Deep super-resolution is applied strictly to the luminance channel (Y), where structural spatial details reside.
* The chrominance channels (Cr, Cb) are preserved and locked to the authentic baseline, preventing color drift.

### 2. Analytical Zero-Mean Residual Constraint
In Sentinel2SRNet, the residual output is constrained prior to addition with the bicubic baseline:

$$I_{\text{residual}} \leftarrow I_{\text{residual}} - \mathbb{E}_{x,y}[I_{\text{residual}}]$$

By enforcing a spatial mean of exactly zero, the network is mathematically prohibited from brightening, darkening, or tinting the scene. Macroscopic solar reflectance is preserved, ensuring downstream vegetation indices (NDVI) and water indices remain valid.

---

## 7. 8x Test-Time Augmentation (TTA) and Uncertainty Quantification

When high-precision verification is required, the user can toggle 8x TTA in the studio:
1. The input patch undergoes eight dihedral symmetry transformations (four 90-degree rotations combined with horizontal flips).
2. The network executes forward passes across all eight transformed inputs and inverts them to the original orientation.
3. **Ensemble Output**: The pixel-wise mean produces an enhanced image with reduced directional convolution noise and higher PSNR (+0.2 to +0.4 dB).
4. **Epistemic Uncertainty Heatmap**: The pixel-wise variance across the eight predictions is calculated and rendered as a spatial heatmap:
   * **Green**: High model confidence across all rotational symmetries.
   * **Red**: Ambiguous textures, shadowed terrain, or sensor noise where predictions diverge.

---

## 8. Spectral Analysis Modes: False-Color CIR and NDVI VARI

The studio provides interactive spectral proxy modes for agricultural and environmental monitoring:
* **Natural RGB**: Standard visual optical composite.
* **Color Infrared (CIR)**: Simulates false-color infrared photography by mapping near-infrared vegetation response into the red channel. Dense vegetative canopies appear in vivid crimson, distinguishing vegetation cleanly from asphalt, soil, and buildings.
* **NDVI Proxy (VARI)**: Evaluates the Visible Atmospherically Resistant Index mapped across the RdYlGn colormap to estimate canopy vigor and vegetation cover without requiring specialized multispectral software.

---

## 9. Quantitative Benchmark Evaluation across 740 Test Patches

All models were evaluated on 740 unseen test patches from the SEN2VENuS dataset, comparing model predictions against authentic 5-meter VENuS ground truth:

| Model Architecture | Checkpoint File | MSE (Lower is better) | PSNR (dB) (Higher is better) | SSIM (Higher is better) | Delta PSNR vs Bicubic (dB) | Delta SSIM vs Bicubic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Baseline)** | *Interpolation reference* | 338.39 | 23.79 dB | 0.7628 | Baseline | Baseline |
| **DCBCNN (DCB-RCAB)** | `slimdiffsr_sentinel2_2x_best.pt` | **244.79** | **25.15 dB** | 0.7769 | **+1.36 dB** | +0.0141 |
| **EESRGAN** | `eesrgan_sentinel2_2x_best.pt` | 246.57 | 25.05 dB | 0.7745 | +1.26 dB | +0.0117 |
| **EDSR** | `edsr_sentinel2_2x_best.pt` | 257.87 | 24.92 dB | 0.7845 | +1.13 dB | +0.0217 |
| **Sentinel2SRNet (Student)** | `sr_sentinel2_2x_best.pt` | 265.02 | 24.85 dB | **0.7898** | +1.06 dB | **+0.0270** |
| **ESPCN** | `espcn_sentinel2_2x_best.pt` | 278.95 | 24.59 dB | 0.7746 | +0.80 dB | +0.0118 |
| **FSRCNN** | `fsrcnn_sentinel2_2x.pt` | 338.20 | 23.51 dB | 0.7122 | -0.28 dB | -0.0506 |

### Evaluation Key Takeaways:
* **DCBCNN** achieved the highest PSNR (25.15 dB, +1.36 dB over bicubic) and lowest MSE (244.79), demonstrating that directional convolution kernels are exceptionally suited for remote sensing geometries.
* **Sentinel2SRNet** achieved the highest structural similarity score (0.7898, +0.0270 over bicubic), outperforming the 43M-parameter EDSR model. This confirms that edge gradient supervision forces the student network to preserve sharp boundary contours.
* **EDSR and EESRGAN** deliver strong accuracy and edge definition but require substantial compute, making them ideal teachers rather than real-time edge deployment models.

---

## 10. Multi-Tier Deployment Across Government Echelons

To meet the practical constraints of government field deployment:
1. **District Level (Pocket Version)**:
   * Engine: INT8-quantized Sentinel2SRNet running via ONNX Runtime or OpenVINO.
   * Hardware: Standard office desktop PCs with dual-core or quad-core CPUs and 4 GB RAM.
   * Performance: Under 40 ms per tile, fully functional offline with zero cloud dependency.
2. **State Level (Standard Version)**:
   * Engine: Full-precision student model with 8x Test-Time Augmentation.
   * Hardware: Regional workstations with quad-core or 8-core CPUs.
   * Performance: Delivers sub-pixel maps accompanied by calibrated uncertainty heatmaps.
3. **National Headquarters (Command Center Version)**:
   * Engine: Full multi-teacher ensemble combining EDSR, EESRGAN, and DCBCNN.
   * Hardware: GPU-accelerated server clusters.
   * Performance: Maximum analytical fidelity for national-scale satellite cadastral mapping.

---

## 11. Operational Use Cases and Real-World Impact

* **Pradhan Mantri Fasal Bima Yojana (PMFBY)**: Accurate separation of adjacent crop parcels, irrigation channels, and verification of localized crop lodging or drought damage.
* **National Disaster Response Force (NDRF)**: Post-disaster mapping of breached embankments, flooded transport routes, and isolated settlements within minutes of public Sentinel-2 data release.
* **Forestry and Resource Protection**: Detection of illegal clearings, unauthorized logging tracks, and forest boundary encroachment at 2.5m resolution.
* **Cadastral and Municipal Planning**: Updating local land-use maps and monitoring peri-urban sprawl without costly aerial drone surveys.

---

## 12. Developer Quickstart and Local Execution

### 1. Environment Setup
```bash
git clone https://github.com/gamerdox/sr-mapping.git
cd sr-mapping

python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launching the Application
Double-click `start.bat` on Windows or execute:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the studio at `http://127.0.0.1:8000` in your web browser.

### 3. Evaluating Benchmark Metrics
To re-evaluate models across the test set:
```bash
python calculate_all_models_metrics.py
```
Summary outputs are written to `results/benchmark_test_metrics.json` and `results/benchmark_test_metrics.csv`.

---

## License

Open-source under the [MIT License](LICENSE). Developed by Team PixelForge for Smart India Hackathon 2026.
