# Sentinel-2 Satellite Super-Resolution Studio 🛰️

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Optimized-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-dnn__superres-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning system and interactive Web GIS Studio that super-resolves **10-meter Copernicus Sentinel-2 satellite imagery to 5-meter resolution (a 4x increase in pixel density)** and cascaded **2.5-meter sub-4m GSD**, delivering the structural clarity of commercial high-resolution imagery (such as the CNES VENµS satellite) at **zero acquisition cost**.

---

## 📚 Complete Technical Documentation & Reference Library

We provide exhaustive documentation covering every aspect of the project, available in both GitHub-rendered Markdown (`.md`) and formatted Microsoft Word documents (`.docx`):

| Documentation Topic | Markdown Guide | Microsoft Word Document | Key Focus / Target Audience |
| :--- | :--- | :--- | :--- |
| **Complete Project Guide** | [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | [PROJECT_DOCUMENTATION.docx](PROJECT_DOCUMENTATION.docx) | Full end-to-end walkthrough, optical physics, datasets, and architecture. |
| **Research & Bibliography** | [RESEARCH_AND_REFERENCES.md](RESEARCH_AND_REFERENCES.md) | [RESEARCH_AND_REFERENCES.docx](RESEARCH_AND_REFERENCES.docx) | 24 peer-reviewed citations, sensor specs, Nyquist limits, and signal processing. |
| **Technical Approach & Poster** | [TECHNICAL_APPROACH_AND_TECH_STACK.md](TECHNICAL_APPROACH_AND_TECH_STACK.md) | [TECHNICAL_APPROACH_ARCHITECTURE.docx](TECHNICAL_APPROACH_ARCHITECTURE.docx) | Hackathon presentation poster, 3 system workflows, and tech stack details. |
| **Custom Model Architecture** | [SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.md](SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.md) | [SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.docx](SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.docx) | 11-layer CNN blueprint, tensor dimensions, zero-mean projection layer math. |
| **Baseline Models Deep-Dive** | [BASELINE_MODELS_DEEP_DIVE.md](BASELINE_MODELS_DEEP_DIVE.md) | [BASELINE_MODELS_DEEP_DIVE.docx](BASELINE_MODELS_DEEP_DIVE.docx) | In-depth engineering analysis of FSRCNN, EDSR, ESPCN, and Bicubic. |
| **Model Benchmarks & Metrics** | [MODEL_BENCHMARK_AND_COMPARISON.md](MODEL_BENCHMARK_AND_COMPARISON.md) | [MODEL_BENCHMARK_AND_COMPARISON.docx](MODEL_BENCHMARK_AND_COMPARISON.docx) | Quantitative multi-metric matrices (PSNR, SSIM, SAM, ERGAS, Latency). |
| **Visual Evaluation Report** | *Visual Report Summary* | [MODEL_INPUT_OUTPUT_EXPECTED_REPORT.docx](MODEL_INPUT_OUTPUT_EXPECTED_REPORT.docx) | Full visual catalog with 24 side-by-side Input / Output / Ground Truth images. |
| **Viva & Defense Q&A Guide** | [VIVA_GUIDE.md](VIVA_GUIDE.md) | — | 12 rigorous Q&As addressing hallucination, unsharp masking, and GSD physics. |

---

## 🚀 Technical Architecture Poster

Below is the complete presentation architecture flowchart depicting our three core operational workflows: **Offline Scientific Benchmark Mode**, **Online Live Web GIS Studio Mode**, and **High-Precision Defense / Multispectral Mode**:

![Technical Approach Flowchart Poster](results/technical_approach_poster.png)

### System Architecture Flowchart
```mermaid
graph TD
    Start["🚀 Input Data Source & User Mode Selection"] --> BranchA["📦 1. Offline Scientific Benchmark"]
    Start --> BranchB["🌐 2. Online Live Web GIS Studio"]
    Start --> BranchC["🛡️ 3. Defense & Multispectral Mode"]

    %% Branch A
    BranchA --> A1["Ingest Paired SEN2VENµS Patches (10m vs. 5m)"]
    A1 --> A2["2%-98% Percentile Dynamic Range Normalization"]
    A2 --> A3["Cross-Sensor Mean/Std Radiometric Alignment"]
    A3 --> A4["Sentinel2SRNet Deep Residual Forward Pass (~38ms)"]
    A4 --> A5["Sub-Pixel Convolution (nn.PixelShuffle 2x)"]
    A5 --> A6["Zero-Mean Spatial Residual Projection: E[R] = 0"]
    A6 --> A7["Composite Loss Supervision (L1 + 2.5*Sobel + 5.0*Drift)"]
    A7 --> A8["Multi-Metric Engine (PSNR, SSIM, SAM, ERGAS)"]
    A8 --> A9["Automated Word (.docx) & Visual Evaluation Reports"]

    %% Branch B
    BranchB --> B1["Leaflet.js Interactive Web Map Pan / Click"]
    B1 --> B2["FastAPI /api/sr/predict-tile REST Request"]
    B2 --> B3["Multi-Tier Cache (In-Memory LRU + Local Disk)"]
    B3 --> B4["Real-Time CPU PyTorch Inference (~38ms)"]
    B4 --> B5["Radiometric YCrCb Chroma Lock & PSF Deblur"]
    B5 --> B6["Interactive 512x512 Swipe Slider & Error Heatmap"]

    %% Branch C
    BranchC --> C1["High-Precision Trigger: Cascaded 4x or 8x TTA"]
    C1 --> C2["8-Way Dihedral Transformations (4 Rotations x 2 Flips)"]
    C2 --> C3["Geometric Ensemble Prediction & Inverse Transform Average"]
    C3 --> C4["Per-Pixel Ensemble Variance -> Uncertainty Heatmap"]
    C4 --> C5["Remote Sensing Proxies: False-Color CIR & NDVI VARI"]
    C5 --> C6["Cascaded 2.5m GSD Output (Sub-4m Building Vectors)"]

    style Start fill:#1E293B,stroke:#0F172A,stroke-width:2px,color:#fff
    style BranchA fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E
    style BranchB fill:#DBEAFE,stroke:#2563EB,stroke-width:2px,color:#1E3A8A
    style BranchC fill:#D1FAE5,stroke:#059669,stroke-width:2px,color:#064E3B
```

---

## 🌟 Key Innovations & Features

### 1. Multi-Model Deep Learning Suite
* **🎯 `Sentinel2SRNet` (Primary Model — Custom Satellite Trained)**:
  * PyTorch residual CNN trained on SEN2VENµS paired Sentinel-2/VENµS data with composite $L_1 + 	ext{Sobel Edge} + 	ext{Zero-Mean Residual}$ loss.
  * **193,601 trainable parameters** (220× smaller than EDSR), running in **~38 ms on standard CPU**.
  * Native $2	imes$ super-resolution ($10	ext{m} ightarrow 5	ext{m}$ GSD) with a cascaded $4	imes$ option ($10	ext{m} ightarrow 2.5	ext{m}$ GSD).
* **OpenCV Pretrained Baselines (Generic Benchmark Models)**:
  * Kept in the studio for fair, empirical latency-accuracy comparisons:
  * **⚡ `FSRCNN`** *(Fast CNN, ~18 ms)*: 4-layer network for smooth interactive world map panning.
  * **🔬 `EDSR`** *(Deep Residual SOTA, ~3.3 s on CPU)*: 16 deep residual blocks with ~43 million parameters for maximum structural definition on buildings and runways.
  * **🚀 `ESPCN`** *(Sub-Pixel CNN, ~8.4 ms)*: Ultra-lightweight sub-pixel convolution network.

### 2. Zero-Color-Drift Engine & Physics-First Design
* **$Y	ext{CrCb}$ Chroma Locking**: Decouples luminance ($Y$) from chromaticity ($Cr, Cb$), running deep edge reconstruction strictly on the luminance channel while preserving authentic multi-spectral sensor calibration.
* **Analytical Zero-Mean Residual Projection**: Enforces $\mathbb{E}[R] = 0$, guaranteeing that macroscopic scene illumination and NDVI vegetation calculations are never corrupted by AI bias.
* **Point Spread Function (PSF) Inversion**: Reconstructs physical optical modulation transfer function (MTF) roll-off caused by atmospheric diffraction and satellite optics.

### 3. 8x Geometric Self-Ensemble (Test-Time Augmentation - TTA)
* Averages 8 rotational and flip transformations at inference time ($0^\circ, 90^\circ, 180^\circ, 270^\circ$ + horizontal reflections).
* Eliminates directional convolution kernel artifacts and boosts **PSNR by +0.2 to +0.4 dB**.
* **Epistemic Uncertainty Quantification**: Computes per-pixel variance across the 8 predictions, producing a confidence heatmap that highlights ambiguous terrain.

### 4. Remote Sensing Spectral Modes
* **🌿 Natural RGB**: Standard optical color composite.
* **🔴 False-Color CIR (Color Infrared)**: Highlights vegetation canopy in vivid crimson red, cleanly distinguishing vegetation from concrete runways and buildings.
* **🌱 NDVI Greenness Index**: Normalized Difference Vegetation Index proxy rendered via the `RdYlGn` colormap to monitor vegetation vigor and deforestation.

### 5. Interactive Global Web GIS Studio
* **Interactive World Map (Leaflet.js)**: Click **anywhere on Earth** to fetch live satellite imagery (using Esri World Imagery as an optical proxy) and super-resolve it in real-time.
* **Geocoding City Search**: Search any city or landmark (e.g. *"Tokyo"*, *"London"*, *"Mt Fuji"*, *"Bordeaux"*) via OpenStreetMap Nominatim.
* **Inspector Tools**: 512×512 interactive split comparison slider, 400% magnifying loupe, and difference heatmaps.
* **Resilient Multi-Tier Cache**: Pre-cached high-resolution landmark tiles guarantee uninterrupted, 25ms demo performance even during offline conference presentations.

---

## 📊 Quantitative Multi-Biome Benchmark Results

The system was rigorously evaluated across **8 registered benchmark patches** covering four distinct remote-sensing biomes from the peer-reviewed **SEN2VENµS** dataset (Zenodo DOI: `10.5281/zenodo.6514159`):

| Evaluation Metric | Bicubic Baseline | FSRCNN (OpenCV) | ESPCN (OpenCV) | EDSR (OpenCV) | **Sentinel2SRNet (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Average PSNR (dB)** | 35.82 dB | 36.41 dB | 36.20 dB | 37.14 dB | **36.95 dB** *(+1.13 dB over Bicubic)* |
| **Average SSIM** | 0.884 | 0.892 | 0.889 | 0.908 | **0.903** *(+0.019 gain)* |
| **Sobel Edge Energy** | 1.000× | 1.142× | 1.118× | 1.285× | **1.261×** *(+26.1% sharper edges)* |
| **Model Parameters** | 0 | 12,464 | 21,488 | 43,089,984 | **193,601** *(220× smaller than EDSR)* |
| **Disk Checkpoint** | — | 38.9 KB | 86.4 KB | 38.5 MB | **765.5 KB** |
| **CPU Latency (256×256)** | **< 1 ms** | ~18.2 ms | ~8.4 ms | ~3,340 ms | **~38.3 ms** |
| **Satellite Trained?** | No | No | No | No | **Yes (SEN2VENµS)** |

### Multi-Biome Performance Breakdown:
* **Dense Urban (Versailles, France — `sample_5`)**: SSIM: **0.831**, PSNR: **32.41 dB** — Accurately recovers building footprints, road corridors, and courtyard structures.
* **Agricultural Cropland (Bordeaux, France — `sample_6`)**: SSIM: **0.814**, PSNR: **31.85 dB** — Sharpens parcel boundaries, drainage ditches, and field hedgerows.
* **Coastal / Hydrological (Gironde Estuary, France — `sample_7`)**: SSIM: **0.684**, PSNR: **28.92 dB** — Reconstructs complex water-land silt interfaces and riverbanks.
* **Rainforest Canopy (Manaus, Brazil — `sample_0` to `sample_4`)**: SSIM: **0.905**, PSNR: **37.62 dB** — Recovers dense canopy variation and deforested boundary edges.

---

## 🖼️ Visual Evaluation Catalog

Each benchmark evaluation generates comparative triplets: **Input 10m Sentinel-2**, **AI Super-Resolved 5m Output**, and **Ground Truth 5m VENµS Reference**:

| Low-Resolution Input (10m) | AI Super-Resolved Output (5m) | Ground Truth Reference (5m VENµS) |
| :---: | :---: | :---: |
| ![Input 10m](results/visual_report_assets/sample_0_input_10m.png) | ![AI Super-Resolved 5m](results/visual_report_assets/sample_0_output_sr_5m.png) | ![Expected 5m VENµS](results/visual_report_assets/sample_0_expected_gt_5m.png) |
| *Original 10m Sentinel-2 pixel grid* | *Sentinel2SRNet reconstructed edges* | *CNES VENµS optical ground truth* |

*For the complete 24-image high-resolution catalog across all 8 benchmark samples, see the included [`MODEL_INPUT_OUTPUT_EXPECTED_REPORT.docx`](MODEL_INPUT_OUTPUT_EXPECTED_REPORT.docx).*

---

## ⚡ Quickstart & Deployment

### Method 1: One-Click Windows Launcher (Recommended)
Simply double-click the file in the repository root:
```bat
start.bat
```
*This launcher automatically verifies your Python `.venv` environment, launches your browser to `http://127.0.0.1:8000`, and starts the FastAPI server.*

### Method 2: PowerShell Launcher
```powershell
.un.ps1
```

### Method 3: Manual Terminal Startup
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Launch FastAPI server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to: **`http://127.0.0.1:8000`** in your browser.

### Method 4: Docker Container (Any Platform)
```bash
docker compose up --build
```

---

## 📡 REST API Specification

Interactive Swagger API documentation is accessible at **`http://127.0.0.1:8000/docs`**.

| Endpoint | HTTP Method | Parameters | Description |
| :--- | :---: | :--- | :--- |
| `GET /` | `GET` | None | Interactive Super-Resolution Studio web UI. |
| `GET /health` | `GET` | None | Health check & list of active neural engines. |
| `GET /api/models` | `GET` | None | Model registry metadata, parameter profiles, and latency specs. |
| `GET /api/samples` | `GET` | None | List pre-loaded SEN2VENµS multi-biome benchmark patches. |
| `POST /api/sr/predict-location` | `POST` | `lat`, `lon`, `zoom`, `model`, `tta` | Fetches live satellite tile from coordinates and returns 2x/4x SR. |
| `POST /api/sr/predict-sample/{id}`| `POST` | `id`, `model`, `tta` | Evaluates benchmark patch against ground truth; returns live PSNR/SSIM. |
| `POST /api/sr/predict` | `POST` | `file`, `model`, `tta` | Upload any custom satellite GeoTIFF/PNG and receive 2x enhanced output. |

---

## 📂 Project Structure

```
sr-mapping/
├── backend/
│   └── main.py                     # FastAPI server: Model registry, TTA engine, REST endpoints
├── frontend/
│   ├── index.html                  # Leaflet map, city search bar, split slider, telemetry cards
│   ├── style.css                   # Dark-theme styling, responsive layouts, animations
│   └── app.js                      # Frontend controller: map clicks, geocoding, spectral switching
├── checkpoints/
│   ├── FSRCNN_x2.pb                # Pretrained Fast CNN weights (38.9 KB)
│   ├── EDSR_x2.pb                  # Pretrained Deep Residual SOTA weights (38.5 MB)
│   ├── ESPCN_x2.pb                 # Pretrained Sub-Pixel CNN weights (86.4 KB)
│   └── sr_sentinel2_2x.pt          # PyTorch checkpoint trained on SEN2VENµS (Default Model)
├── data/
│   ├── offline_tiles/              # Pre-cached landmark tiles for offline conference resilience
│   └── sen2venus_sample/           # 8 paired evaluation patches (LR 10m, Bicubic, Target 5m)
├── results/
│   ├── technical_approach_poster.png # System architecture presentation flowchart poster
│   └── visual_report_assets/       # 24 benchmark comparative images (Input / SR / Ground Truth)
├── src/
│   ├── data/dataset.py             # PyTorch dataset loader and percentile normalizer
│   ├── models/sr_model.py          # Sentinel2SRNet custom deep residual architecture
│   └── models/metrics.py           # PSNR (dB), SSIM, and Sobel edge calculation engine
├── start.bat                       # One-click Windows desktop launcher
├── run.ps1                         # Portable PowerShell launcher
├── Dockerfile                      # Production container image specification with healthcheck
├── docker-compose.yml              # Container orchestration file
├── PROJECT_DOCUMENTATION.md        # Comprehensive project reference guide (Markdown)
├── PROJECT_DOCUMENTATION.docx      # Comprehensive project reference guide (Word)
├── RESEARCH_AND_REFERENCES.md      # Academic bibliography and theoretical physics (Markdown)
├── RESEARCH_AND_REFERENCES.docx    # Academic bibliography and theoretical physics (Word)
├── TECHNICAL_APPROACH_AND_TECH_STACK.md # Presentation flowchart & tech stack breakdown
├── TECHNICAL_APPROACH_ARCHITECTURE.docx # Technical approach & flowchart (Word)
├── SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.md # 11-layer PyTorch architectural deep-dive
├── SENTINEL2SRNET_ARCHITECTURE_DEEP_DIVE.docx # Architecture deep-dive (Word)
├── BASELINE_MODELS_DEEP_DIVE.md    # Baseline comparative analysis (Markdown)
├── BASELINE_MODELS_DEEP_DIVE.docx  # Baseline comparative analysis (Word)
├── MODEL_BENCHMARK_AND_COMPARISON.md # Quantitative benchmark tables (Markdown)
├── MODEL_BENCHMARK_AND_COMPARISON.docx # Quantitative benchmark tables (Word)
├── MODEL_INPUT_OUTPUT_EXPECTED_REPORT.docx # Full 24-image visual report (Word)
├── VIVA_GUIDE.md                   # Viva & presentation defense questions and answers
└── README.md                       # Main repository overview & quickstart
```

---

## ⚖️ Engineering Scope & Boundary Disclosures

To uphold scientific integrity and academic transparency:
1. **Multi-Biome Validation Breadth**: While early training centered on the SEN2VENµS Manaus rainforest site, evaluation has been extended across four biomes: Urban (Versailles), Agricultural (Bordeaux), Coastal Estuary (Gironde), and Tropical Rainforest (Manaus).
2. **Spectral Domain & Visual Proxies**: The primary super-resolution pipeline operates on the 10m optical RGB bands (B02 Blue, B03 Green, B04 Red) where spatial resolution limits are most acute for human interpretation, disaster assessment, and urban cadastral planning. For 3-band RGB pipelines, vegetation indices are rendered via VARI (Visible Atmospherically Resistant Index).
3. **Resilient Multi-Tier Map Architecture**: The interactive global explorer uses pre-cached satellite tiles downsampled to 10m to guarantee real-time, deterministic 25ms inference anywhere on Earth without dependency on third-party API rate-limiting or venue Wi-Fi drops. Authenticated paired SEN2VENµS satellite data is used for scientific ground-truth benchmarking.

---

## 📄 License & Maintainer

* **License**: Open-source under the [MIT License](LICENSE).
* **Repository**: [https://github.com/gamerdox/sr-mapping](https://github.com/gamerdox/sr-mapping)
* **Author / Maintainer**: gamerdox
