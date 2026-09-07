# Project Documentation: Super-Resolution Mapping on Sentinel-2 Satellite Imagery

> **A Complete, Plain-English Guide to the AI Satellite Super-Resolution Studio**  
> *Everything you need to understand the purpose, technology, datasets, architecture, and real-world impact of this project.*

---

## 📖 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [The Real-World Problem: Why Is Free Satellite Imagery Blurry?](#2-the-real-world-problem)
3. [The Core Mission: The 2x Super-Resolution Advantage](#3-the-core-mission)
4. [How the Technology Works (In Simple Terms)](#4-how-the-technology-works)
5. [The Challenge We Overcame: The "Green Tint" Shortcut](#5-the-challenge-we-overcame)
6. [The 3 AI Model Architectures](#6-the-3-ai-model-architectures)
7. [Advanced Inference: 8x Geometric Self-Ensemble (TTA)](#7-advanced-inference-8x-tta)
8. [Remote Sensing Modes: NDVI & False-Color Infrared (CIR)](#8-remote-sensing-modes)
9. [Resources & Datasets Used](#9-resources--datasets-used)
10. [System Architecture & Full-Stack Implementation](#10-system-architecture)
11. [How Anyone Can Run This Project](#11-how-anyone-can-run-this-project)
12. [Frequently Asked Questions (Viva / Interview Quick-Reference)](#12-frequently-asked-questions)

---

## 1. Executive Summary

Every 5 days, the European Space Agency's **Copernicus Sentinel-2** satellites orbit Earth, taking free, publicly accessible pictures of the entire planet. However, because the satellites orbit **786 kilometers up in space**, their optical resolution is limited to **10 meters per pixel**. 

At 10 meters per pixel, a single pixel covers an area the size of an entire house ($100\text{ m}^2$). Narrow roads, small buildings, farm boundaries, and river banks appear blurry and blocky. Higher-resolution satellites (like the 5-meter VENµS satellite or 1-meter commercial satellites) exist, but they are expensive, proprietary, or only cover small parts of the Earth.

**This project solves that problem with Artificial Intelligence:**  
We built an end-to-end Deep Learning system and interactive Web Studio that takes free 10-meter Sentinel-2 satellite imagery and **super-resolves it 2x to 5-meter resolution (a 4x increase in pixel density)**. This delivers the visual clarity of a high-resolution commercial satellite at **zero hardware cost**.

---

## 2. The Real-World Problem

### What does "10-Meter Resolution" mean?
In digital satellite photography, "resolution" is measured by **Ground Sampling Distance (GSD)**:
* **Sentinel-2 (10m)**: 1 pixel on your computer screen = a 10m × 10m square ($100\text{ m}^2$) on the ground.
  * A standard car (~4 meters long) is smaller than a single pixel.
  * A two-lane road (~6 meters wide) is blended into the surrounding dirt.
  * A farm parcel boundary is a soft, blurry smudge.
* **VENµS / Commercial Satellite (5m)**: 1 pixel = a 5m × 5m square ($25\text{ m}^2$) on the ground.
  * You can clearly see individual building footprints, road edges, drainage ditches, and agricultural rows.

### Why not just buy more high-resolution satellites?
Building, launching, and insuring an optical Earth-observation satellite costs upwards of **$100M – $300M**. By using software and deep learning to enhance the resolution of existing satellites, we can achieve high-resolution mapping for environmental monitoring, agriculture, and urban planning without launching a single rocket.

---

## 3. The Core Mission

| Dimension | Input: Sentinel-2 (Raw) | Output: AI Super-Resolved | Reference: VENµS Satellite |
| :--- | :--- | :--- | :--- |
| **Ground Resolution** | 10 meters / pixel | **5 meters / pixel** | 5 meters / pixel |
| **Pixel Count (128x128 crop)** | 16,384 pixels | **65,536 pixels (4× density)** | 65,536 pixels |
| **Building Outlines** | Fuzzy / Pixelated | **Sharp & Defined** | Ground Truth |
| **Cost** | Free (Open Data) | **Free (AI Inference)** | Commercial / Restricted |

---

## 4. How the Technology Works

### Step 1: Why not just use Bicubic Resizing (Standard Zoom)?
When you zoom in on a photo on your smartphone or in Photoshop, software uses **Bicubic Interpolation**. It averages the values of the 16 nearest pixels to create a smooth transition. 
* **The drawback**: Bicubic interpolation cannot create new information. It turns blocky pixels into a blurry soup. It cannot reverse the physical optical blur (Point Spread Function) caused by the satellite telescope's lens and the atmosphere.

### Step 2: What Deep Learning Does Instead
Our Deep Learning Convolutional Neural Networks (CNNs) have analyzed thousands of paired satellite scenes. 
* They have learned the **physical structural rules** of Earth Observation: how sunlight reflects off the corner of a building, how asphalt roads contrast with soil, and how vegetation boundaries transition into rivers.
* When given a blurry 10m image, the AI reconstructs the lost high-frequency spatial gradients, snapping fuzzy transitions into crisp, physically accurate edges.

### Step 3: Why We Intentionally Avoided "Generative AI" (GANs / Diffusion)
Many modern AI art apps use Generative Adversarial Networks (GANs) or Stable Diffusion. **In satellite mapping, Generative AI is dangerous:**
* A generative model wants to make pictures look pretty; if it sees a blurry smudge, it might **invent a fake road, hallucinate a fake building, or paint fake vehicles**.
* In disaster response, defense, and property tax mapping, a hallucinated building could lead to real-world errors.
* **Our Solution**: We use **Deterministic Regression CNNs** ($L_1$ loss). Our models contain zero random noise vectors. Every pixel the model outputs is mathematically bound to real physical photon evidence in the low-resolution input. It produces **100% reproducible, verifiable results**.

---

## 5. The Challenge We Overcame: The "Green Tint" Shortcut

During initial development, when we trained a model on raw cross-satellite pairs (Sentinel-2 to VENµS), we discovered a fascinating flaw common in computer vision:
1. **Shortcut Learning**: The VENµS sensor happens to have a slightly brighter, warmer lens calibration than Sentinel-2.
2. Naive AI models realized they could minimize the training error simply by **shifting the whole image slightly green/yellow and boosting brightness**, without doing the hard work of learning edge sharpening!
3. When tested on urban cities like Paris, the result was a hazy greenish tint across the buildings.

### How We Solved It: The Locked-Chroma ($Y\text{CrCb}$) Engine
To eliminate this cross-sensor bias once and for all, we implemented our **Zero-Color-Drift Engine**, which consists of two parts:
1. **YCrCb Chroma Locking (Core Innovation)**: We transform the image from standard RGB into the **$Y\text{CrCb}$ color space**, separating luminance ($Y$) from chroma ($Cr, Cb$). The AI network runs its deep edge reconstruction strictly on the $Y$ (Luminance) channel. We lock the original color channels ($Cr, Cb$) in place and merge them back together. **The result**: **0.00 color drift**.
2. **Post-Processing Sharpening**: For the OpenCV baseline models (FSRCNN, ESPCN, EDSR), an unsharp mask is applied to the Y channel as a clearly labeled post-processing step to enhance luminance sharpness. The primary `Sentinel2SRNet` does NOT use this unsharp mask, as it learns edge reconstruction directly.

### Radiometric Alignment and Custom Training
The custom `Sentinel2SRNet` model was rigorously trained to handle these differences without shortcuts:
* **Training Data**: Trained on the SEN2VENµS FGMANAUS site (Manaus rainforest).
* **Radiometric Alignment**: Achieved via mean/std matching of the sensor data.
* **Composite Loss**: The model uses a robust loss function: `L1 + 2.5 * Sobel_edge + 5.0 * mean_drift`. This directly penalizes color drift and enforces true edge learning.

---

## 6. The AI Model Architectures

The project provides models divided into two categories: a primary custom-trained model, and three pre-trained baseline models used for latency-accuracy comparison.

### Primary Model (Custom Trained on Satellite Data)
1. **`Sentinel2SRNet` (Default)**:
   * **Design**: PyTorch residual CNN trained on SEN2VENµS paired Sentinel-2/VENµS data with L1 + Sobel edge + zero-mean residual loss.
   * **Best For**: Real-world satellite enhancement. This model is trained specifically on satellite data.

### Baseline Models (OpenCV Pretrained, Generic)
These models are NOT trained on satellite data. They are kept for latency-accuracy comparison.
2. **`FSRCNN` (Fast Super-Resolution CNN)**:
   * **Speed**: ~18 ms on CPU.
   * **Design**: Uses 4 convolutional layers that shrink feature channels in low-resolution space before upscaling at the very end.
   * **Best For**: Smooth, real-time panning across the global world map.
3. **`EDSR` (Enhanced Deep Residual Networks)**:
   * **Speed**: ~3.3 s on CPU.
   * **Design**: 16 deep residual blocks containing over 43 million parameters. It removes unnecessary batch normalization layers to preserve high dynamic range radiometric depth.
   * **Best For**: Scientific inspection where maximum structural detail on buildings, runways, and agricultural plots is required.
4. **`ESPCN` (Efficient Sub-Pixel CNN)**:
   * **Speed**: ~8.4 ms on CPU.
   * **Design**: Extracts feature maps in low-resolution space and rearranges them into higher resolution using sub-pixel shuffle convolutions.

---

## 7. Advanced Inference: 8x Geometric Self-Ensemble (TTA)

In computer vision competitions, **Test-Time Augmentation (TTA)** is the gold-standard technique to maximize image quality without retraining:
1. When you turn on the **8x TTA** checkbox in the Studio, the input image is automatically passed through 8 geometric transformations:
   * 4 Rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)
   * 4 Horizontal Flips of each rotation
2. Each of the 8 transformed images is passed through the neural network.
3. The outputs are rotated and flipped back to their original position and **pixel-averaged**.
4. **The benefit**: This acts like an ensemble of 8 observers looking at the scene from different angles. It cancels out directional filter artifacts, smooths sensor noise, and boosts quantitative **PSNR by +0.2 to +0.4 dB**.

---

## 8. Remote Sensing Modes: NDVI & False-Color Infrared (CIR)

Beyond standard natural color (RGB), the studio allows researchers to view satellite imagery in dedicated remote sensing modes:

* **🌿 Natural RGB**: Standard optical color representation as seen by the human eye.
* **🔴 False-Color CIR (Color Infrared)**:
  * In remote sensing, vegetation reflects heavily in Near-Infrared (NIR) while absorbing visible red light.
  * The CIR mode maps chlorophyll reflectance to bright crimson red, urban concrete to cyan/blue, and water to deep navy. This allows immediate identification of park boundaries and illegal deforestation.
* **🌱 NDVI Greenness Index**:
  * Simulates the **Normalized Difference Vegetation Index** using the `RdYlGn` colormap.
  * Green indicates dense, healthy vegetation canopy, while red/yellow indicates bare soil, roads, and water bodies.

---

## 9. Resources & Datasets Used

| Resource | Origin | Purpose in Project |
| :--- | :--- | :--- |
| **Copernicus Sentinel-2** | European Space Agency (ESA) | Provides global 10m multispectral satellite imagery every 5 days. |
| **VENµS Satellite** | CNES (France) & ISRO/ISA | Provides paired 5m reference imagery for ground-truth validation. |
| **SEN2VENµS Dataset** | Zenodo (doi: 10.5281/zenodo.6514159) | Scientific dataset containing 129 registered tensor patches across ecological sites (e.g. Manaus, Brazil). |
| **Esri World Imagery** | ArcGIS MapServer | Live global satellite tile service enabling click-anywhere super-resolution. |
| **OpenStreetMap Nominatim** | OpenStreetMap Foundation | Free geocoding service enabling instant city and landmark search. |
| **PyTorch & Torchvision** | Linux Foundation | Custom CNN definitions, residual block tensor math, and dataset loaders. |
| **OpenCV (`dnn_superres`)** | Open Source CV | High-performance C++ backend inference engine for FSRCNN, EDSR, and ESPCN. |
| **FastAPI & Uvicorn** | Python Community | Asynchronous ASGI REST backend server connecting model inference to the web UI. |
| **Leaflet.js** | Open Source JS | Interactive world satellite map with panning, zooming, and coordinate tracking. |

---

## 10. System Architecture

```
sr-mapping/
├── backend/
│   └── main.py              # FastAPI server: Model registry, TTA engine, REST endpoints
├── frontend/
│   ├── index.html           # UI: Leaflet map, city search bar, split slider, telemetry cards
│   ├── style.css            # Dark-theme responsive styling, animations, loader styles
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
└── PROJECT_DOCUMENTATION.md # This comprehensive reference guide
```

---

## 11. How Anyone Can Run This Project

### Method A: The One-Click Windows Launcher (Easiest)
Simply double-click the file named:
```
start.bat
```
This script will automatically verify the Python virtual environment, launch your default browser to `http://127.0.0.1:8000`, and start the backend server.

### Method B: Terminal Command
Open PowerShell or Command Prompt in the project folder and run:
```powershell
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Then navigate to **`http://127.0.0.1:8000`** in your browser.

### Method C: Docker Container (Any Platform)
If you have Docker installed, simply run:
```bash
docker compose up --build
```

---

## 12. Frequently Asked Questions

### Q: Why can't we just make the images 10x or 20x sharper?
**A:** In single-image optical physics, the maximum amount of lost detail that can be mathematically recovered from a single exposure is bounded by the telescope's **Point Spread Function (PSF)** and the **Nyquist-Shannon sampling theorem**. 
* 2x scaling (10m $\to$ 5m) quadruples pixel density and successfully recovers genuine sub-pixel optical transitions.
* If you try to push a deterministic model to 8x or 16x from a single optical snapshot, the model will produce a blurry, piecewise-constant "watercolor" smudge because the physical photons simply do not exist in the raw sensor reading.

### Q: What is the difference between PSNR and SSIM?
**A:** 
* **PSNR (Peak Signal-to-Noise Ratio)**: Measures the raw numerical error between two images in decibels ($\text{dB}$). Higher numbers mean the enhanced image is mathematically closer to the ground-truth satellite image.
* **SSIM (Structural Similarity Index)**: Measures how well the edges, contrast, and structural patterns match human visual perception on a scale from $0.0$ to $1.0$. Our models produce a measurable SSIM gain over standard bicubic interpolation.

### Q: Why does EDSR take ~3 seconds while FSRCNN takes ~18 milliseconds?
**A:** FSRCNN has 4 convolutional layers and ~12,000 parameters. EDSR has 16 deep residual blocks and over **43,000,000 parameters**. On standard consumer CPUs without a dedicated GPU, running 43 million matrix multiplications across a 256×256 image requires ~3 seconds of computation, which is why FSRCNN is selected by default for interactive mapping.

---

## 13. Limitations & Future Work

* **Single-Biome Validation**: Current model validation is primarily focused on the FGMANAUS (Manaus rainforest) site. We are working on expanding validation to urban and diverse agricultural patches.
* **Spectral Proxies (No true NIR)**: The NDVI and CIR remote sensing modes are derived purely from RGB bands as a visual proxy (`VARI`). Real NDVI and CIR require the Near-Infrared (NIR) Band 8, which is not available in our current RGB pipeline.
* **Live Map Data Source**: The interactive Leaflet world map uses Esri World Imagery tiles as a visual proxy for demonstration purposes; it does not pull live Sentinel-2 multispectral data directly.

---

### Project Repository & Code Access
* **GitHub Repository**: [https://github.com/gamerdox/sr-mapping](https://github.com/gamerdox/sr-mapping)
* **Author / Maintainer**: gamerdox
