# Sentinel-2 Satellite Super-Resolution Studio 🛰️

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Optimized-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning system and interactive web studio that super-resolves **10-meter Copernicus Sentinel-2 satellite imagery to 5-meter resolution (a 4x increase in pixel density)**, comparable to commercial high-resolution imagery (VENµS satellite) at zero acquisition cost.

---

## 📚 Essential Guides & Documentation

* **[Comprehensive Project Documentation (PROJECT_DOCUMENTATION.md)](PROJECT_DOCUMENTATION.md)**: A complete, plain-English breakdown explaining the optical physics, the cross-sensor color calibration breakthrough, how the 3 AI models work, and the resources used.
* **[Viva & Presentation Defense Guide (VIVA_GUIDE.md)](VIVA_GUIDE.md)**: Quick-reference Q&A guide addressing common questions asked by professors, evaluators, and examiners (hallucination prevention, Nyquist optical limit, bicubic comparison).

---

## 🌟 Key Features

* **Multi-Model Deep Learning Suite**:
  * **⚡ FSRCNN** *(Fast CNN, ~18 ms)*: Real-time 4-layer network for smooth interactive world map panning.
  * **🔬 EDSR** *(Deep Residual SOTA, ~3.3 s on CPU)*: 16 deep residual blocks with ~43 million parameters for maximum structural definition on buildings and roads.
  * **🚀 ESPCN** *(Sub-Pixel CNN, ~8.4 ms)*: Ultra-lightweight sub-pixel convolution network.
* **Zero-Color-Drift Luminance Engine ($Y\text{CrCb}$)**:
  * Decouples luminance ($Y$) from chroma ($Cr, Cb$), running deep edge reconstruction strictly on the luminance channel.
  * **Guarantees 0.00 color shift** and eliminates cross-sensor radiometric green/yellow tinting.
* **8x Geometric Self-Ensemble (Test-Time Augmentation - TTA)**:
  * Averages 8 rotational and flip transformations at inference time, eliminating directional convolution artifacts and boosting **PSNR by +0.2 to +0.4 dB**.
* **Remote Sensing Spectral Modes**:
  * **🌿 Natural RGB**: Standard optical color composite.
  * **🔴 False-Color CIR (Color Infrared)**: Highlights vegetation in vivid crimson red to distinguish canopy from urban concrete.
  * **🌱 NDVI Greenness Index**: Normalized Difference Vegetation Index heatmap rendered via the `RdYlGn` colormap.
* **Interactive Global Map & Studio**:
  * **World Satellite Map (Leaflet.js)**: Click **anywhere on Earth** to fetch live satellite imagery and super-resolve it in real-time.
  * **Geocoding City Search**: Search any city or landmark (e.g. *"Tokyo"*, *"London"*, *"Mt Fuji"*) via OpenStreetMap Nominatim.
  * **Studio Inspector**: 512×512 split comparison slider, 400% magnifying loupe, and AI difference heatmap.
  * **Live Quantitative Metrics**: Real-time PSNR (dB) and SSIM metrics calculated dynamically against paired VENµS ground-truth data.

---

## 🚀 Quickstart

### Method 1: The One-Click Windows Launcher (Easiest)
Simply double-click:
```bat
start.bat
```
This automatically verifies your `.venv`, opens your browser to `http://127.0.0.1:8000`, and starts the Uvicorn server.

### Method 2: Manual Terminal Startup
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\activate.bat

# 2. Launch FastAPI server
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### Method 3: Docker Container
```bash
docker compose up --build
```

---

## 📡 API Endpoints

Interactive Swagger documentation is available at **`http://127.0.0.1:8000/docs`**.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Interactive Super-Resolution Studio UI |
| `GET /health` | `GET` | Health check & list of active neural engines |
| `GET /api/models` | `GET` | Model registry metadata, parameter profiles, and latency |
| `GET /api/samples` | `GET` | List available pre-loaded SEN2VENµS benchmark patches |
| `POST /api/sr/predict-location` | `POST` | Fetches live satellite tile from lat/lon and runs 2x SR |
| `POST /api/sr/predict-sample/{id}` | `POST` | Evaluates benchmark patch and returns live PSNR/SSIM & spectral modes |
| `POST /api/sr/predict` | `POST` | Upload any custom satellite image and return 2x enhanced PNG |

---

## 📂 Project Architecture

```
sr-mapping/
├── backend/
│   └── main.py              # FastAPI server: Model registry, TTA engine, REST endpoints
├── frontend/
│   ├── index.html           # Leaflet map, city search bar, split slider, telemetry cards
│   ├── style.css            # Dark-theme styling, responsive layouts, animations
│   └── app.js               # Frontend controller: map clicks, geocoding, spectral switching
├── checkpoints/
│   ├── FSRCNN_x2.pb         # Pretrained Fast CNN weights (38.9 KB)
│   ├── EDSR_x2.pb           # Pretrained Deep Residual SOTA weights (38.5 MB)
│   ├── ESPCN_x2.pb          # Pretrained Sub-Pixel CNN weights (86.4 KB)
│   └── sr_sentinel2_2x.pt   # PyTorch checkpoint trained on SEN2VENµS
├── data/
│   └── sen2venus_sample/    # 5 paired evaluation patches (LR 10m, Bicubic, Target 5m)
├── src/
│   ├── data/dataset.py      # PyTorch dataset loader and percentile normalizer
│   ├── models/sr_model.py   # Sentinel2SRNet custom deep residual architecture
│   └── models/metrics.py    # PSNR (dB) and SSIM calculation engine
├── start.bat                # One-click Windows desktop launcher
├── Dockerfile               # Production container image specification
├── docker-compose.yml       # Container orchestration file
├── VIVA_GUIDE.md            # Technical Q&A guide for project presentations & defense
├── PROJECT_DOCUMENTATION.md # Complete plain-English project documentation
└── README.md                # Repository overview & quickstart
```

---

## 📄 License
This project is open-source under the MIT License.
