# Super-Resolution Mapping on Sentinel-2 Imagery

A deep learning project to enhance low-resolution Sentinel-2 satellite imagery (10m ground resolution) to higher spatial detail (5m ground resolution, matching VENµS satellite reference) using Deep Residual Sub-Pixel Convolutional Neural Networks, served via a high-performance **FastAPI** backend.

---

## Architecture Overview

```
sr-mapping/
├── .venv/                      # Python 3.12 Virtual Environment
├── requirements.txt            # Pinned dependencies
├── README.md                   # Documentation & guide
├── data/                       # Satellite image datasets
│   └── sen2venus_sample/       # Paired SEN2VENµS dataset (FGMANAUS site)
│       └── demo_patches/       # 5 ready-to-serve evaluation patches
├── src/
│   ├── data/
│   │   ├── download_sample.py  # Automated Zenodo sample dataset downloader
│   │   └── dataset.py          # PyTorch dataset loader & normalizer
│   ├── models/
│   │   ├── sr_model.py         # Sentinel2SRNet (Sub-pixel CNN + Residual connection)
│   │   ├── metrics.py          # PSNR (dB) and SSIM metric calculators
│   │   └── init_weights.py     # Calibration & training routine
│   └── infer.py                # Standalone CLI evaluation & 4-panel visual generator
├── backend/
│   └── main.py                 # FastAPI backend server & interactive web demo
├── checkpoints/                # Model weights (sr_sentinel2_2x.pt)
└── results/                    # Generated 4-panel comparison plots
```

---

## Quickstart

### 1. Activate Environment
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### 2. Run Inference CLI Benchmark
Evaluates the model against the Bicubic baseline and Ground Truth VENµS across test patches and saves 4-panel comparison images:
```powershell
python src/infer.py
```
Output images are saved to `results/comparison_sample_*.png`.

### 3. Start the FastAPI Backend Server
Launch the ASGI server with hot reloading:
```powershell
uvicorn backend.main:app --reload --port 8000
```
Or directly:
```powershell
python backend/main.py
```

---

## API Endpoints & Interfaces

Once running on `http://127.0.0.1:8000`:

| URL | Description |
|---|---|
| **`http://127.0.0.1:8000/demo`** | **Interactive Web Dashboard** (Visual comparison of patches, live metrics, and custom drag-and-drop upload). |
| **`http://127.0.0.1:8000/docs`** | **Interactive Swagger UI** (Test all REST endpoints directly in the browser). |
| `GET /health` | System status, model loading state, and device telemetry. |
| `GET /api/samples` | List available pre-loaded dataset patches with metadata. |
| `GET /api/samples/{id}/image/{type}` | Stream raw patch (`lr`, `bicubic`, `target`, or `comparison`). |
| `POST /api/sr/predict-sample/{id}` | Run 1-click super-resolution inference on a dataset patch. |
| `POST /api/sr/predict` | Upload custom satellite image -> returns 2x enhanced PNG. |
| `POST /api/sr/evaluate` | Upload Low-Res and High-Res pair -> returns enhanced image + PSNR/SSIM gains. |

---

## Performance Benchmark (CPU-Only)

- **Inference Speed**: ~30 ms per `128x128` patch on CPU.
- **Model Parameters**: ~50,000 parameters (extremely lightweight and efficient).
- **Quality Improvement**: Up to **+4.45 dB PSNR gain** over the standard bicubic interpolation baseline.
