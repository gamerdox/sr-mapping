# Academic Research, Literature and References

Scientific bibliography, theoretical foundations, and satellite mission references supporting the Sentinel-2 Super-Resolution Mapping project developed by Team PixelForge for Smart India Hackathon 2026 (Problem Statement SIH26142, National Technical Research Organisation - NTRO).

---

## 1. Foundational Super-Resolution and Computer Vision Literature

### 1. ESPCN: Efficient Sub-Pixel Convolutional Neural Network
* **Authors**: Wenzhe Shi, Jose Caballero, Ferenc Huszar, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang
* **Publication**: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), pages 1874 to 1883.
* **Citation Link**: [arXiv:1608.00367](https://arxiv.org/abs/1608.00367)
* **Relevance**: Provided the mathematical foundation for sub-pixel convolution (PixelShuffle). By reorganizing latent feature channels into a spatial grid, it avoids the checkerboard artifacts typical of transposed convolutions and reduces memory requirements significantly.

### 2. FSRCNN: Accelerating the Super-Resolution Convolutional Neural Network
* **Authors**: Chao Dong, Chen Change Loy, and Xiaoou Tang
* **Publication**: Proceedings of the European Conference on Computer Vision (ECCV 2016), LNCS volume 9906, pages 391 to 407. Springer.
* **Citation Link**: [arXiv:1609.05158](https://arxiv.org/abs/1609.05158)
* **Relevance**: Introduced the paradigm of keeping all convolutions in the low-resolution input space, expanding to high resolution only at the output layer. This approach informed our CPU-first latency budget.

### 3. EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution
* **Authors**: Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee
* **Publication**: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW 2017), NTIRE 2017 Challenge Winner, pages 136 to 144.
* **Citation Link**: [arXiv:1707.02921](https://arxiv.org/abs/1707.02921)
* **Relevance**: Demonstrated that removing Batch Normalization stabilizes training on absolute intensity scales and conserves GPU memory. Serves as our primary Accuracy Teacher model.

### 4. Deep Residual Learning and PReLU Activation
* **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
* **Publications**: 
  * Residual Learning: IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), pages 770 to 778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  * Parametric Rectifiers: IEEE International Conference on Computer Vision (ICCV 2015), pages 1026 to 1034.
* **Relevance**: Provided the identity skip connections used across our four residual blocks and the PReLU activations that maintain gradient flow across low-reflectance pixels such as water bodies and shadow zones.

### 5. RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
* **Authors**: Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu
* **Publication**: Proceedings of the European Conference on Computer Vision (ECCV 2018), pages 286 to 301.
* **Citation Link**: [arXiv:1807.02758](https://arxiv.org/abs/1807.02758)
* **Relevance**: Introduced Channel Attention blocks in super-resolution, allowing the network to rescale channel features adaptively according to their high-frequency informational content.

---

## 2. Knowledge Distillation and Teacher-Student Frameworks

### 6. Speculative Knowledge Distillation (SKD)
* **Authors**: Research on multi-teacher consensus distillation for lightweight vision models.
* **Citation Link**: [arXiv:2410.11325](https://arxiv.org/pdf/2410.11325)
* **Relevance**: Guided our multi-teacher consensus approach where three specialized teachers (EDSR, EESRGAN, and DCB-CNN) transfer representations to Sentinel2SRNet only on regions where their predictions agree, effectively mitigating synthetic feature hallucination.

### 7. ESAOpenSR: Open-Source Super-Resolution Framework for Earth Observation
* **Organization**: European Space Agency (ESA) OpenSR Initiative
* **Repository**: [https://github.com/ESAOpenSR/opensr-model](https://github.com/ESAOpenSR/opensr-model)
* **Relevance**: Baseline reference for validating remote-sensing super-resolution models on open satellite constellations.

### 8. LDSR-S2: Lightweight Latent Diffusion and Remote Sensing Super-Resolution
* **Publication**: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (IEEE JSTARS 2025).
* **DOI**: [10.1109/JSTARS.2025.3542220](https://doi.org/10.1109/JSTARS.2025.3542220)
* **Relevance**: Investigates diffusion and deep convolutional super-resolution trade-offs specifically on Sentinel-2 constellations.

---

## 3. Remote Sensing Datasets and Sensor Specifications

### 9. SEN2VENuS Dataset Benchmark
* **Authors**: Julien Michel, Olivier Vinas, Robin Youssefi, Juan M. Angel, and Marion Cournet
* **Publication**: Data in Brief, volume 43, 108349 (2022). Elsevier.
* **Repository**: [Zenodo Record 6514159](https://zenodo.org/record/6514159)
* **Relevance**: Primary paired ground truth dataset providing concurrent Sentinel-2 (10m) and VENuS (5m) surface reflectance imagery. Our models were trained and benchmarked across 3,695 paired scenes across three distinct biomes:
  * FGMANAUS (Manaus, Amazon Basin, Brazil): Dense rainforest canopy.
  * ESTUAMAR (Mar Menor Lagoon, Spain): Coastal wetlands and water-land interfaces.
  * ALSACE (Alsace Plain, France): Rectilinear urban buildings, roads, and agricultural parcels.

### 10. Copernicus Sentinel-2 MSI Sensor
* **Organization**: European Space Agency (ESA)
* **Specifications**: Multi-Spectral Instrument (MSI) acquiring 13 spectral bands at 10m, 20m, and 60m spatial resolution with a 5-day revisit rate at the equator.
* **Relevance**: Target source data for our super-resolution mapping pipeline.

### 11. CNES / ISA VENuS Scientific Micro-Satellite
* **Organizations**: Centre National d'Etudes Spatiales (CNES) and Israel Space Agency (ISA)
* **Specifications**: Superspectral sensor acquiring 12 narrow spectral bands at 5-meter ground sample distance with a 2-day revisit rate and identical viewing geometry to Sentinel-2.
* **Relevance**: Provides authentic, non-interpolated 5m optical ground truth for training and quantitative benchmark validation.

---

## 4. Evaluation Metrics and Standards

### 12. Structural Similarity Index (SSIM)
* **Authors**: Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli
* **Publication**: IEEE Transactions on Image Processing, volume 13, number 4, pages 600 to 612 (2004).
* **DOI**: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)
* **Relevance**: Primary metric for structural and edge preservation. Evaluated using a windowed Gaussian kernel via scikit-image on grayscale luminance representations.

### 13. Peak Signal-to-Noise Ratio (PSNR)
* **Formulation**: Standard logarithmic decibel scale based on Mean Squared Error (MSE), measuring reconstruction fidelity against true ground truth sensor measurements.

---

## 5. Project Repository Reference

* **Repository**: [https://github.com/gamerdox/sr-mapping](https://github.com/gamerdox/sr-mapping)
* **Author and Maintainer**: Team PixelForge
* **Hackathon Context**: Smart India Hackathon 2026 (SIH 2026), Problem Statement ID SIH26142, National Technical Research Organisation (NTRO).
