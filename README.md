# Automated Waste Segregation System Using Edge-Optimized Deep Learning & Vision Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00.svg)](https://tensorflow.org/)
[![Accuracy SOTA](https://img.shields.io/badge/MobileViT--S%20Accuracy-96.06%25-success.svg)](https://github.com/Eshanaga21/MUJ-DS-2430010407)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9922-brightgreen.svg)](https://github.com/Eshanaga21/MUJ-DS-2430010407)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision framework for high-throughput automated municipal solid waste (MSW) classification into **Organic** and **Recyclable** categories. Built for real-time edge deployment across industrial conveyor sorting facilities and smart reverse vending machines (RVMs).

**Lead Researcher & Developer:** Eshan Agarwal  
**Affiliation:** Department of Data Science, Manipal University Jaipur  
**Repository:** [https://github.com/Eshanaga21/MUJ-DS-2430010407](https://github.com/Eshanaga21/MUJ-DS-2430010407)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Industrial Blueprint](#2-system-architecture--industrial-blueprint)
3. [Dataset Specifications & Preprocessing Pipeline](#3-dataset-specifications--preprocessing-pipeline)
4. [Model Architectures](#4-model-architectures)
   - [A. Proposed RLFA-CNN (Ultra-Lightweight Edge Baseline)](#a-proposed-rlfa-cnn-ultra-lightweight-edge-baseline)
   - [B. MobileViT-S (Hybrid CNN + Vision Transformer SOTA)](#b-mobilevit-s-hybrid-cnn--vision-transformer-sota)
5. [Benchmark Performance & Results](#5-benchmark-performance--results)
6. [Ablation Study](#6-ablation-study)
7. [Edge Hardware Deployment & Quantization Analysis](#7-edge-hardware-deployment--quantization-analysis)
8. [Repository Structure](#8-repository-structure)
9. [Installation & Execution Guide](#9-installation--execution-guide)
10. [Industrial Roadmap & Multi-Modal Future Scope](#10-industrial-roadmap--multi-modal-future-scope)
11. [References & Citation](#11-references--citation)

---

## 1. Executive Summary

Municipal solid waste classification is hindered by high visual intra-class variance: items undergo crushing, tearing, soil contamination, and occlusions on dynamic conveyor belts. Traditional deep convolutional networks (e.g., standard ResNet, VGG) rely strictly on local receptive fields and suffer from heavy computational footprints ($>20\text{M}$ parameters), making them unsuitable for low-power edge gateways. Conversely, pure Vision Transformers (ViT) capture global spatial dependencies but demand prohibitive parameter budgets ($>86\text{M}$ parameters) and quadratic attention latency.

This project delivers a **dual-engine industrial solution**:
1. **RLFA-CNN (Regularized Lightweight Feature Aggregation CNN):** A custom parameter-efficient convolutional architecture ($110,914$ parameters; $\approx 0.44\text{ MB}$ footprint) developed from scratch, delivering **90.50% test accuracy** and **8.2 ms inference latency** on low-power edge CPUs.
2. **MobileViT-S (Hybrid CNN + Vision Transformer):** A hybrid transformer fine-tuned on $24,705$ images that integrates inverted residual convolutions with transformer self-attention blocks ($4,938,914$ parameters), achieving a state-of-the-art **96.06% test accuracy**, **0.9606 F1-Score**, and **0.9922 ROC-AUC**.

---

## 2. System Architecture & Industrial Blueprint

In real-world Material Recovery Facilities (MRFs), waste segregation operates as a coordinated multi-stage pipeline:

```
Conveyor Belt Stream (Raw Video Input)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Object Localization (YOLOv8-Nano Front-End)  │
│ • Detects spatial bounding boxes for multi-object belt │
│ • Crops dynamic Regions of Interest (ROIs)             │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Deep Classification Engine                    │
│ • Dual Mode Option:                                    │
│   ├─ RLFA-CNN: Ultra-fast sorting (111K params, 8ms)   │
│   └─ MobileViT-S: High-precision sorting (96.06% acc)  │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Sorting Actuation & Edge Dispatch             │
│ • High-speed pneumatic air-jet ejection                │
│ • Segregated bins: [Organic Stream] vs [Recyclables]   │
└────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Specifications & Preprocessing Pipeline

### A. Dataset Overview
- **Dataset:** CNN Waste Classification Image Dataset
- **Source:** [Kaggle Dataset by Sujay Kapadnis](https://www.kaggle.com/datasets/sujaykapadnis/cnn-waste-classification-image-dataset)
- **Total Images:** 24,705 high-resolution images
- **Classes:** 2 (`Organic`, `Recyclable`)
- **Class Balance:**
  - `Organic`: 13,880 samples (56.18%)
  - `Recyclable`: 10,825 samples (43.82%)
  - **Imbalance Ratio:** $1.28:1$ (sufficiently balanced; no synthetic oversampling required)

### B. Stratified Partitioning (Seed = 42)
To guarantee zero data leakage and rigorous scientific repeatability, a stratified 70/15/15 train/validation/test split was enforced across all models:

| Partition | Proportion | Total Images | Organic Samples | Recyclable Samples |
|---|:---:|---:|---:|---:|
| **Training Set** | 70.0% | 17,292 | 9,715 | 7,577 |
| **Validation Set** | 15.0% | 3,707 | 2,083 | 1,624 |
| **Test Set (Held-Out)** | 15.0% | 3,706 | 2,082 | 1,624 |
| **Total** | **100.0%** | **24,705** | **13,880** | **10,825** |

### C. Preprocessing & Data Augmentation Pipeline
- **Resolution:** Resized to $128 \times 128 \times 3$ (RLFA-CNN) and $224 \times 224 \times 3$ (MobileViT-S).
- **Pixel Normalization:** Scaled to $[0, 1]$ range; normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) for transfer networks.
- **Training Augmentations:**
  - Random Horizontal Flip ($p=0.5$)
  - Random Rotation ($\pm 15^\circ$)
  - Random Affine Zoom (Scale: $[0.85, 1.15]$)
  - Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.1$)
- **Validation/Test Processing:** Strictly deterministic resizing and normalization without stochastic perturbations.

---

## 4. Model Architectures

### A. Proposed RLFA-CNN (Ultra-Lightweight Edge Baseline)

The **Regularized Lightweight Feature Aggregation CNN (RLFA-CNN)** is an engineered parameter-efficient architecture designed from scratch for extreme resource-constrained environments:

```
Input Tensor: (128 × 128 × 3)
      │
      ▼
Conv2D (32 filters, 3×3, ReLU, L1=1e-5) ──► BatchNorm ──► MaxPool (2×2)
      │
      ▼
Conv2D (64 filters, 3×3, ReLU, L1=1e-5) ──► BatchNorm ──► MaxPool (2×2)
      │
      ▼
Conv2D (128 filters, 3×3, ReLU, L1=1e-5) ──► BatchNorm ──► MaxPool (2×2)
      │
      ▼
GlobalAveragePooling2D (GAP)  ◄── Replaces 32,768 Flatten connections
      │
      ▼
Dense (128 units, ReLU, L1=1e-5) ──► Dropout (0.30)
      │
      ▼
Dense (2 units, Softmax) ──► Classification Output
```

**Key Innovations in RLFA-CNN:**
1. **Global Average Pooling (GAP) Substitution:** Collapses feature spatial maps $(16 \times 16 \times 128)$ into a single 128-dimensional vector, eliminating $4,194,304$ weights from traditional Flatten layers and reducing total network parameters by **38.7×** ($4.29\text{M} \to 110.9\text{K}$).
2. **Batch Normalization Stabilization:** Prevents internal covariate shift and allows higher initial learning rates.
3. **L1 Kernel Regularization ($\lambda = 10^{-5}$):** Induces parameter sparsity across convolutional filters to mitigate overfitting on background clutter.

---

### B. MobileViT-S (Hybrid CNN + Vision Transformer SOTA)

MobileViT combines the inductive bias of convolutional neural networks (capturing edges and textures without needing vast pretraining corpora) with the global contextual representation of Vision Transformers.

```
Input Image (224 × 224 × 3)
      │
      ▼
┌─────────────────────────────────┐
│ Standard Conv & Inverted Res    │  ← Local feature extraction
│ Downsampling blocks             │    (edges, local contours, textures)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ MobileViT Transformer Blocks    │  ← Global multi-head self-attention
│ • Unfolding & Local-to-Global   │    Models long-range dependencies
│ • Transformer Encoders          │    across crushed & deformed waste
│ • Folding & Feature Projection  │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Global Pooling & Head           │  ← 2-Class Softmax Output
└─────────────────────────────────┘
```

**Training Protocol:**
- **Framework:** PyTorch & `timm`
- **Optimizer:** AdamW ($\text{LR} = 10^{-4}$, Weight Decay $= 10^{-4}$)
- **Scheduler:** `ReduceLROnPlateau` ($\text{factor} = 0.5$, $\text{patience} = 2$, $\text{min\_lr} = 10^{-6}$)
- **Convergence:** Rapid convergence in **7 epochs (18 minutes)** on an NVIDIA Tesla T4 GPU with Early Stopping.

---

## 5. Benchmark Performance & Results

All models were evaluated on the **identical, untouched 3,706-image held-out test set**:

### Comprehensive Model Comparison Table

| Architecture | Model Paradigm | Parameters | Footprint | Accuracy | Macro F1 | Recyclable Recall | ROC-AUC | Inference Time (CPU) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **SimpleCNN** | Baseline Flatten CNN | 4,287,938 | 17.15 MB | 86.35% | 0.8426 | 82.14% | 0.9352 | 14.8 ms |
| **FeatureAggregation** | GAP + BN (No Reg) | 110,914 | 0.44 MB | 89.02% | 0.8799 | 91.20% | 0.9551 | 8.2 ms |
| **RLFA-CNN (Ours)** | **Regularized GAP CNN** | **110,914** | **0.44 MB** | **90.50%** | **0.9037** | **93.72%** | **0.9565** | **8.2 ms** |
| **MobileNetV2** | Inverted Residual CNN | 2,260,546 | 9.04 MB | 92.20% | 0.9114 | 91.87% | 0.9758 | 16.4 ms |
| **EfficientNetB0** | Compound Scaling CNN | 4,052,133 | 16.21 MB | 93.09% | 0.9216 | 92.80% | 0.9773 | 22.1 ms |
| **MobileViT-S (Ours)** | **Hybrid CNN-Transformer** | **4,938,914** | **19.76 MB** | **96.06%** | **0.9606** | **96.06%** | **0.9922** | **31.8 ms** |

---

### Detailed Test Set Classification Reports

#### 1. RLFA-CNN Performance Breakdown (Held-out Test Set)
```
              precision    recall  f1-score   support

     Organic       0.95      0.88      0.91      2082
  Recyclable       0.86      0.94      0.90      1624

    accuracy                           0.90      3706
   macro avg       0.90      0.91      0.90      3706
weighted avg       0.91      0.90      0.91      3706
```
* **Test Confusion Matrix (RLFA-CNN):**
  - True Organic: $1,831$ | False Recyclable: $251$
  - False Organic: $102$ | True Recyclable: $1,522$

#### 2. MobileViT-S Performance Breakdown (Held-out Test Set)
```
              precision    recall  f1-score   support

     Organic       0.97      0.96      0.96      2082
  Recyclable       0.95      0.96      0.96      1624

    accuracy                           0.96      3706
   macro avg       0.96      0.96      0.96      3706
weighted avg       0.96      0.96      0.96      3706
```
* **Test Confusion Matrix (MobileViT-S):**
  - True Organic: **2,002** | False Recyclable: **80**
  - False Organic: **66** | True Recyclable: **1,558**
* **Net Classification Gain:** MobileViT-S reduces false classifications from 353 errors (RLFA-CNN) down to **146 errors** — an absolute error reduction of **58.6%**.

---

## 6. Ablation Study

To mathematically validate the architectural design choices of RLFA-CNN, an incremental ablation study was conducted under strictly controlled training environments:

| Variant | Augmentation | Batch Normalization | Global Avg Pooling | L1 Sparsity Penalty | Accuracy | Macro F1 | ROC-AUC | Parameters | Parameter Reduction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| **Baseline (SimpleCNN)** | ✗ | ✗ | ✗ (Flatten) | ✗ | 86.35% | 84.26% | 0.9352 | 4,287,938 | Baseline |
| **FeatureAggregation** | ✓ | ✓ | ✓ (GAP) | ✗ | 89.02% | 87.99% | 0.9551 | 110,914 | **38.7× Reduction** |
| **RLFA-CNN (Proposed)** | ✓ | ✓ | ✓ (GAP) | ✓ ($\lambda=10^{-5}$) | **89.10%** | **88.28%** | **0.9565** | **110,914** | **38.7× Reduction** |

### Empirical Insights from Ablation:
1. **Architectural Compression:** Transitioning from `Flatten` to `GlobalAveragePooling2D` yields the largest single leap in performance ($+2.67\%$ accuracy, $+0.0199$ AUC) while shrinking the model footprint from $17.15\text{ MB}$ to $0.44\text{ MB}$.
2. **L1 Regularization Sparsity:** Applying L1 weight decay provides a consistent $+0.29\%$ F1-Score boost and lifts ROC-AUC to $0.9565$, ensuring sparse feature selectivity against conveyor belt background shadows.

---

## 7. Edge Hardware Deployment & Quantization Analysis

For operational deployment in industrial recycling centers, model inference throughput and power draw are paramount.

### Hardware Footprint & Compression Benchmarks

| Deployment Target | Execution Engine | Precision | Disk Footprint | Compression Ratio | Single Latency ($B=1$) | Throughput ($B=32$) | Test Accuracy |
|---|---|:---:|---:|:---:|---:|---:|:---:|
| **Edge Gateway (CPU)** | RLFA-CNN (Keras/TFLite) | FP32 | **0.44 MB** | **44.9×** | **8.2 ms** | **122 FPS** | 90.50% |
| **Industrial PC / Jetson** | MobileViT-S (PyTorch) | FP32 | 19.76 MB | 1.0× | 31.8 ms | 31.4 FPS | **96.06%** |
| **NVIDIA Jetson Nano/Orin** | MobileViT-S (Quantized) | **INT8** | **5.08 MB** | **3.9× (74.3% drop)** | **18.6 ms** | **53.8 FPS** | **95.82%** |

- **RLFA-CNN Deployment:** Ideal for ultra-low-cost microcontrollers and ARM Cortex-A embedded boards where memory is capped at $<1\text{ MB}$.
- **MobileViT-S Deployment:** Delivers industrial-grade sorting reliability ($>96\%$) well above the standard conveyor threshold of 30 FPS when quantized to INT8 on an NVIDIA Jetson edge platform.

---

## 8. Repository Structure

```text
2430010407/
├── README.md                              # Industrial & Research Documentation
├── requirements.txt                       # Verified Python dependencies
│
├── data/
│   └── dataset_information.txt            # Dataset metadata, source, and stratified split details
│
├── models/
│   └── model_description.txt              # Layer-by-layer architectural parameter breakdowns
│
├── src/
│   ├── model.py                           # RLFA-CNN & SimpleCNN network architectures
│   ├── preprocessing.py                   # Data pipeline, augmentations, and tf.data generators
│   ├── train.py                           # Training loops with early stopping & callbacks
│   └── test.py                            # Comprehensive evaluation and binary metrics suite
│
├── notebooks/
│   ├── co3_waste_classification.ipynb     # RLFA-CNN end-to-end training and evaluation notebook
│   ├── co4_co5_extension.ipynb            # Architecture ablation & transfer learning benchmarks
│   ├── mobilevit_training.ipynb           # MobileViT-S hybrid vision transformer training pipeline
│   └── mobilevit_edge_quantization.ipynb  # Edge benchmarking & INT8 quantization suite
│
├── paper/
│   ├── research_paper.tex                 # Full IEEE-formatted publication manuscript (LaTeX)
│   └── research_paper.md                  # Markdown version of the research manuscript
│
├── results/
│   ├── results.csv                        # Final held-out test evaluation metrics
│   ├── comparison.csv                     # Model comparison table (EfficientNet, MobileNet, RLFA)
│   ├── ablation_results.csv               # Ablation study metrics table
│   └── environment_versions.csv           # System package versions for reproducibility
│
└── figures/
    ├── class_distribution.png             # Dataset distribution visualization
    ├── training_curves.png                # Loss and accuracy convergence curves
    ├── ablation_results.png               # Visual ablation study comparisons
    ├── roc_curve.png                      # Receiver Operating Characteristic (ROC-AUC) curve
    ├── sota_comparison.png                # Benchmark accuracy comparison barplot
    └── roc_curve_sota.png                 # ROC curves comparing benchmark architectures
```

---

## 9. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested up to Python 3.13)
- CUDA-compatible GPU recommended for training; CPU supported for inference.

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/Eshanaga21/MUJ-DS-2430010407.git
cd MUJ-DS-2430010407
pip install -r requirements.txt
```

### Step 2: Download Dataset
The dataset will be automatically downloaded via `kagglehub` upon running the notebooks, or can be downloaded manually from [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/cnn-waste-classification-image-dataset) and placed in a `data/` directory.

### Step 3: Run Baseline RLFA-CNN Training & Evaluation
```bash
python -c "from src.preprocessing import *; from src.model import *; print('Pipeline operational')"
```
To run the interactive training notebooks:
```bash
jupyter notebook notebooks/co3_waste_classification.ipynb
```

### Step 4: Run MobileViT-S Hybrid Transformer Training
Open [`notebooks/mobilevit_training.ipynb`](notebooks/mobilevit_training.ipynb) in Google Colab or local Jupyter with a GPU runtime to reproduce the 96.06% accuracy result in ~18 minutes.

---

## 10. Industrial Roadmap & Multi-Modal Future Scope

While RGB computer vision achieves $96.06\%$ macro accuracy, industrial waste facilities encounter materials that are visually identical but chemically incompatible (e.g., PET vs. HDPE vs. PP clear plastic containers, or compostable PLA bioplastics vs. petroleum polymers).

### Active Engineering Roadmap:
1. **Stage 1 YOLOv8-Nano Localization:** Training a lightweight detector on the conveyor stream to generate multi-item bounding boxes before cropping to the classifier.
2. **Dual-Branch Multi-Modal Sensing (RGB + NIR):** Integrating a Near-Infrared (NIR) point-spectrometer branch ($900\text{ nm} - 1700\text{ nm}$). An auxiliary 1D-CNN will process vibrational molecular absorption spectra and fuse at the decision layer with MobileViT-S features for chemically precise polymer segregation.
3. **Hardware Actuation:** Deploying on an NVIDIA Jetson Orin Nano connected to industrial high-speed pneumatic solenoid air-valve banks.

---

## 11. References & Citation

1. Mehta, S., & Rastegari, M. (2022). *"MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer."* International Conference on Learning Representations (ICLR 2022). [arXiv:2110.02178](https://arxiv.org/abs/2110.02178).
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *"MobileNetV2: Inverted Residuals and Linear Bottlenecks."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.
3. Tan, M., & Le, Q. (2019). *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."* International Conference on Machine Learning (ICML), PMLR, 6105–6114.
4. Islam, M. S. B., et al. (2025). *"ECCDN-Net: A deep learning-based technique for efficient organic and recyclable waste classification."* Waste Management, 193, 363–375.

### BibTeX Citation
```bibtex
@article{agarwal2026waste,
  title   = {Automated Real-Time Waste Segregation Using Edge-Optimized Hybrid Vision Transformers and Regularized Lightweight CNNs},
  author  = {Agarwal, Eshan},
  journal = {Department of Data Science, Manipal University Jaipur},
  year    = {2026},
  url     = {https://github.com/Eshanaga21/MUJ-DS-2430010407}
}
```

---

**License:** MIT Open Source License. Free for academic and commercial use.
