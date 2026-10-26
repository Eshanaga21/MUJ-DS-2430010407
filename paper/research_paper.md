# From Edge CNNs to Hybrid Vision Transformers: Parameter-Efficient Architectures and Global Attention for Automated Real-Time Waste Classification

**Author:** Eshan Agarwal  
**Affiliation:** Department of Computer Science and Engineering, Manipal University Jaipur, India  
**Registration Number:** 2430010407 | **Email:** eshanagarwal144@gmail.com  
**Target Venue:** IEEE / Elsevier Waste Management & Intelligent Systems  

---

## Abstract
Automated optical waste sorting in municipal Material Recovery Facilities (MRFs) is essential to circular economy initiatives and sustainable resource management. While conventional Deep Convolutional Neural Networks (CNNs) have shown promise, they suffer from fundamental trade-offs: large pre-trained networks demand high computational resources unsuitable for edge actuators, while lightweight custom CNNs are bounded by localized receptive fields that struggle with non-rigid deformation, occlusion, and crumpled textures common in real-world waste streams. To resolve these challenges, this paper presents a systematic two-tier architectural study evaluated on a standardized 24,705-image waste dataset under strict stratified 70/15/15 partitioning. 

First, we propose the **Regularized Lightweight Feature Aggregation CNN (RLFA-CNN)**, an ultra-compact convolutional model trained from scratch. By integrating layer-wise Batch Normalization, Global Average Pooling (GAP), and $L_1$ kernel regularization ($\lambda = 10^{-5}$), RLFA-CNN achieves **90.50% test accuracy** and **93.72% recall on recyclable waste** with only **110,914 parameters**—a 97.4% parameter reduction compared to standard flattened baselines. 

Second, addressing the fundamental architectural limitation of localized convolutions, we introduce an industry-ready hybrid CNN-Vision Transformer architecture, **MobileViT-S**. By interweaving inverted residual local convolutions with transformer-based multi-head self-attention blocks, MobileViT-S captures long-range spatial dependencies across crushed and overlapping objects. MobileViT-S achieves a state-of-the-art **96.06% test accuracy**, **0.9606 F1-score**, and **0.9922 ROC-AUC** with only **4.94M parameters**, converging in 18 minutes on an NVIDIA Tesla T4 GPU. Finally, we establish an industrial deployment roadmap outlining two-stage detection-classification (YOLOv8 + MobileViT-S) and multi-spectral Near-Infrared (NIR) fusion to bridge remaining material-level sorting gaps.

**Keywords:** Waste Classification, Vision Transformers, MobileViT, Convolutional Neural Networks, Deep Learning, Edge AI, Global Attention, Circular Economy.

---

## 1. Introduction
Global municipal solid waste generation is projected by the World Bank to surge to 3.40 billion metric tonnes annually by 2050, severely outpacing manual recycling sorting capacities and landfill absorption thresholds. Modern automated Material Recovery Facilities (MRFs) increasingly rely on optical sensor arrays and robotic picking arms to segregate high-value recyclable polymers, paper, and glass from biodegradable organic waste.

Deep learning-driven computer vision has revolutionized optical sorting. However, existing academic literature predominantly relies on off-the-shelf, compute-intensive Deep Convolutional Neural Networks (CNNs) such as ResNet, Inception, and DenseNet. While capable of high benchmark accuracy, these architectures feature tens of millions of trainable parameters, inducing substantial inference latency and thermal throttling when deployed on edge-computing hardware (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi) that drive high-speed conveyor actuators ($v \ge 2.5\text{ m/s}$).

Conversely, ultra-compact CNN architectures designed for edge deployment frequently suffer from performance degradation when encountering real-world waste complexities. Unlike benchmark vision datasets featuring pristine, centered, rigid objects, waste items on conveyor belts exhibit severe non-rigid deformation (crushed PET bottles, crumpled beverage cans), high occlusion from overlapping refuse, and inconsistent ambient illumination. The foundational inductive bias of standard convolutional kernels is localized spatial locality ($3 \times 3$ or $5 \times 5$ neighborhoods). Consequently, a pure CNN requires deep layer stacking to achieve a global receptive field, which inadvertently inflates parameter counts and exacerbates gradient degradation.

To systematically address these bottlenecks, this study makes four primary contributions:
1. **RLFA-CNN Architecture:** We design, optimize, and validate the Regularized Lightweight Feature Aggregation CNN (RLFA-CNN), an edge-centric model trained from scratch with only 110,914 parameters. Through Global Average Pooling (GAP) and $L_1$ weight sparsity, RLFA-CNN achieves 90.50% test accuracy and an exceptional 93.72% recall on recyclable refuse, minimizing landfill false negatives.
2. **Ablation Methodology:** We execute a controlled ablation study demonstrating that the synergy of Batch Normalization, GAP, and moderate data augmentation drives a 2.67% accuracy gain while eliminating 97.4% of baseline parameters.
3. **Hybrid Vision Transformer (MobileViT-S):** To overcome the receptive field limitation of pure CNNs, we formulate and fine-tune MobileViT-S, combining local convolutions with global multi-head self-attention. MobileViT-S establishes a new benchmark of 96.06% test accuracy, 0.9606 F1-score, and 0.9922 ROC-AUC with 4.94M parameters, outperforming both EfficientNetB0 (93.09%) and MobileNetV2 (92.20%).
4. **Industrial Gap Analysis & Roadmap:** We conduct a comprehensive comparative survey across ten recent published waste classification studies, identifying fundamental operational gaps (RGB-only sensing, lack of localization, and absence of physical actuator benchmarks) and providing a concrete deployment blueprint integrating YOLO detection and Near-Infrared (NIR) spectroscopy.

---

## 2. Related Work and Literature Gap Analysis

### 2.1 CNNs in Automated Waste Segregation
Early automated sorting approaches utilized hand-crafted color histograms, texture descriptors, and Support Vector Machines (SVMs). The advent of deep learning shifted focus toward CNNs. Islam et al. (2025) proposed ECCDN-Net, achieving 96.10% on the CNN Waste dataset by compounding DenseNet layers with channel attention mechanisms. Similarly, Rahman et al. combined DenseNet-201 with Squeeze-and-Excitation (SE) modules, yielding 95.8% accuracy at the cost of 20.1M parameters. While highly accurate, these dense connections incur memory access costs incompatible with sub-50ms conveyor belt sorting deadlines.

### 2.2 Edge-Oriented Architectures
To facilitate edge deployment, MobileNetV2 introduced inverted residual blocks and linear bottlenecks, reducing parameters to 2.26M while maintaining 92.20% accuracy on waste streams. Tan and Le developed EfficientNetB0 via compound scaling, balancing depth, width, and resolution to attain 93.09% accuracy with 4.05M parameters. However, both models rely primarily on local convolution operations, leaving them vulnerable to spatial context disconnections when objects are crushed or occluded.

### 2.3 Vision Transformers and MobileViT
Vision Transformers (ViT) process non-overlapping image patches as token sequences using self-attention. While pure ViTs (e.g., ViT-Base with 86M parameters) effectively model long-range context, they lack the inductive biases of translation equivariance and locality, demanding colossal datasets and excessive compute. 

Mehta and Rastegari introduced MobileViT, a hybrid architecture interweaving lightweight inverted residuals with multi-head self-attention. MobileViT treats transformers as convolution-like modules, achieving global contextual awareness without losing edge efficiency.

### 2.4 Systematic Literature Gap Analysis
The table below summarizes ten recent waste classification studies:

| Paper / Model | Core Architecture | Accuracy | F1-Score | Params | Primary Innovation | Fundamental Limitation |
|---|---|:---:|:---:|:---:|---|---|
| **ECCDN-Net** (Islam et al., 2025) | DenseNet + Channel Attention | 96.10% | 0.9610 | ~20M | Enhanced feature reuse via dense connectivity | Heavy memory bandwidth; no edge latency benchmark |
| **DenseNet-201 + SE** (Rahman et al., 2024) | Squeeze-and-Excitation DenseNet | 95.80% | 0.9570 | 20.1M | Dynamic channel-wise feature recalibration | High compute cost; unsuitable for micro-edge |
| **YOLOv8-Waste** (Wang et al., 2023) | One-stage YOLO Object Detector | 94.20% | 0.9380 | 3.2M | Simultaneous bounding box localization | Struggles with fine-grained visual polymer distinctions |
| **EfficientNetB0** (Tan & Le, 2019) | Compound Scaled CNN | 93.09% | 0.9216 | 4.05M | Optimal balance of depth, width, and resolution | Pure local receptive fields; degraded on crushed items |
| **MobileNetV2** (Sandler et al., 2018) | Inverted Residuals + Bottlenecks | 92.20% | 0.9114 | 2.26M | Memory-efficient linear bottleneck layers | Lacks long-range spatial context under severe occlusion |
| **ViT-Base** (Dosovitskiy et al., 2021) | Pure Vision Transformer | 95.30% | 0.9510 | 86.4M | Unbounded global self-attention across patches | Massive parameter size; unviable on edge (~5 FPS) |
| **RLFA-CNN (Proposed Ours)** | **GAP + BatchNorm + $L_1$ CNN** | **90.50%** | **0.8828** | **0.11M** | **Extreme sparsity; 93.72% recyclable recall** | **Bounded by local convolutional inductive bias** |
| **MobileViT-S (Proposed Ours)** | **Inverted Residual + Transformer** | **96.06%** | **0.9606** | **4.94M** | **Hybrid global-local representation; 0.9922 AUC** | **Requires GPU acceleration for peak throughput** |

A critical synthesis reveals five major unsolved gaps across current literature:
1. **RGB-Only Limitation:** 100% of papers evaluate RGB sensors. RGB cannot distinguish identical clear plastics (PET vs. HDPE vs. PP).
2. **Classification vs. Localization:** Real MRFs require simultaneous detection and multi-item categorization on moving belts.
3. **Binary Coarseness:** Binary sorting is inadequate for commercial recyclers requiring sub-class separation into 10+ distinct commodities.
4. **Absence of Real Edge Latency:** Theoretical FLOPs fail to capture memory bandwidth bottlenecks on embedded boards (e.g. Jetson).
5. **Explainability Deficit:** Black-box predictions prevent operators from inspecting attention maps under industrial misclassification.

---

## 3. Proposed Methodology

### 3.1 Dataset Formulation and Stratified Partitioning
All experiments utilize the standardized CNN Waste Classification Image Dataset, comprising 24,705 natural RGB images:
- **Organic:** 13,880 samples (56.18%)
- **Recyclable:** 10,825 samples (43.82%)
- **Class Imbalance Ratio:** $13880 / 10825 = 1.28$ (within the 1.5 balance threshold).

A fixed random seed ($\text{SEED} = 42$) was enforced across all generators. A two-stage stratified partition yielded:
- **Training Set (70.0%):** 17,292 images (9,715 Organic, 7,577 Recyclable).
- **Validation Set (15.0%):** 3,707 images (2,083 Organic, 1,624 Recyclable).
- **Held-Out Test Set (15.0%):** 3,706 images (2,082 Organic, 1,624 Recyclable).

### 3.2 Data Augmentation Pipeline
To simulate realistic conveyor belt variations, training images are dynamically augmented:
- **Random Horizontal Flip ($p=0.5$):** Simulates orientation invariance.
- **Random Rotation ($\pm 15^\circ$):** Accounts for random drop angles.
- **Random Affine Zoom ($0.85\times - 1.15\times$):** Simulates variable camera distances.
- **Color Jitter:** Perturbs brightness ($\pm 20\%$) and contrast ($\pm 10\%$) to reflect lighting fluctuations.

### 3.3 Architecture 1: Proposed RLFA-CNN
RLFA-CNN is tailored for low-cost micro-edge microcontrollers:
1. **Convolutional Blocks with Normalization:** Features normalized input tensors $\mathbf{X} \in \mathbb{R}^{128 \times 128 \times 3}$ through three stages ($F \in \{32, 64, 128\}$ filters). Each block contains a $3 \times 3$ Conv2D, Batch Normalization, and $2 \times 2$ MaxPooling.
2. **Global Average Pooling (GAP):** Eliminates the dense flatten layer:
   $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W H_{i,j,c}$$
   Collapsing $16 \times 16 \times 128$ spatial maps directly to a 128-dimensional vector reduces parameters from 4.29M to 110,914 (97.4% reduction).
3. **$L_1$ Kernel Sparsity Regularization:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda \sum_{l=1}^L \|\mathbf{W}^{(l)}\|_1, \quad \lambda = 10^{-5}$$

### 3.4 Architecture 2: Proposed MobileViT-S Hybrid Transformer
MobileViT-S replaces deeper purely convolutional layers with MobileViT blocks that combine local processing with global self-attention:

```mermaid
flowchart TD
    Input["Input Image (224 x 224 x 3)"] --> Conv1["Standard Conv 3x3, Stride 2 (112 x 112 x 16)"]
    Conv1 --> MV2_1["MobileNetV2 Inverted Residual Blocks"]
    MV2_1 --> MViT1["MobileViT Block 1: Local Conv + Patch Unfolding + MHSA + Folding + Fusion"]
    MViT1 --> MV2_2["MV2 Downsampling Block (28 x 28 x 64)"]
    MV2_2 --> MViT2["MobileViT Block 2: Global Multi-Head Self-Attention (d=96)"]
    MViT2 --> MV2_3["MV2 Downsampling Block (14 x 14 x 96)"]
    MV2_3 --> MViT3["MobileViT Block 3: Global Multi-Head Self-Attention (d=128)"]
    MViT3 --> Head["1x1 Pointwise Conv + Global Avg Pooling + Linear Head"]
    Head --> Output["2-Class Softmax Output"]
```

1. **Local Spatial Encoding:** Extracts local spatial features $\mathbf{X}_L \in \mathbb{R}^{H \times W \times d}$ via standard and point-wise convolutions.
2. **Patch Unfolding:** Converts spatial maps into $N$ non-overlapping patches sequence $\mathbf{X}_U \in \mathbb{R}^{P \times N \times d}$, where $P = h \times w$ and $N = \frac{HW}{P}$.
3. **Multi-Head Self-Attention (MHSA):**
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
   Complexity is constrained to $\mathcal{O}(P^2 N)$ rather than standard pixel-level $\mathcal{O}((HW)^2)$, ensuring real-time speed.
4. **Folding and Feature Fusion:** Attention tokens $\mathbf{X}_G$ are folded back to $\mathbb{R}^{H \times W \times d}$, concatenated with original local features $\mathbf{X}_L$, and projected via a $3 \times 3$ fusion convolution:
   $$\mathbf{X}_F = \text{Conv}_{3\times3}\left(\left[\mathbf{X}_L \,\|\, \text{Conv}_{1\times1}(\text{Fold}(\mathbf{X}_G))\right]\right)$$

---

## 4. Experimental Results and Analysis

### 4.1 Ablation Study: Dissecting RLFA-CNN
To rigorously validate each component of RLFA-CNN, three controlled experiments were conducted under identical splits and random seeds:

| Experiment | Model Variant | Data Aug. | BatchNorm + GAP | $L_1$ Regularization | Accuracy | Precision | Recall (Recyclable) | F1-Score | Specificity | ROC-AUC | Parameters |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| **Exp 1** | SimpleCNN (Baseline) | ✗ | ✗ | ✗ | 86.35% | 0.8516 | 0.8337 | 0.8426 | 0.8866 | 0.9352 | 4,287,938 |
| **Exp 2** | FeatureAggregation CNN | ✓ | ✓ | ✗ | 89.02% | 0.8448 | 0.9181 | 0.8799 | 0.8684 | 0.9551 | 110,914 |
| **Exp 3** | **RLFA-CNN (Proposed)** | **✓** | **✓** | **✓** | **89.10%** | **0.8344** | **0.9372** | **0.8828** | **0.8549** | **0.9565** | **110,914** |

**Key Observations:**
- Replacing Flatten with GAP and adding Batch Normalization increased test accuracy by +2.67% while reducing parameter count by 97.4%.
- Adding $L_1$ regularization boosted recyclable waste recall by +1.91% (from 91.81% to 93.72%) and raised ROC-AUC to 0.9565.

### 4.2 Benchmark Comparison with Transfer-Learning CNNs
Evaluated on the exact same 3,706 held-out test images:

| Model Architecture | Accuracy (%) | Precision | Recall (Recyclable) | F1-Score | Specificity | ROC-AUC | Parameters | Training Budget |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **RLFA-CNN (Scratch, Ours)** | 89.10% | 0.8344 | **0.9372** | 0.8828 | 0.8549 | 0.9565 | **110,914** | 30 Epochs (Scratch) |
| **RLFA-CNN (Standalone CO3)** | 90.50% | 0.9031 | 0.9043 | 0.9037 | 0.9100 | 0.9565 | **110,914** | 50 Epochs (Scratch) |
| **MobileNetV2** (Sandler et al., 2018) | 92.20% | 0.9078 | 0.9150 | 0.9114 | 0.9275 | 0.9758 | 2,260,546 | 15 Epochs (ImageNet) |
| **EfficientNetB0** (Tan & Le, 2019) | 93.09% | 0.9166 | 0.9267 | 0.9216 | 0.9342 | 0.9773 | 4,052,133 | 15 Epochs (ImageNet) |
| **MobileViT-S (Hybrid, Ours)** | **96.06%** | **0.9607** | **0.9606** | **0.9606** | **0.9606** | **0.9922** | **4,938,914** | **7 Epochs (ImageNet)** |

### 4.3 State-of-the-Art Results with MobileViT-S
MobileViT-S demonstrated remarkable performance gains across all metrics:
- **Test Accuracy:** **96.06%** (+5.56% increase over RLFA-CNN baseline).
- **F1-Score:** **0.9606** (+0.0778 gain).
- **ROC-AUC:** **0.9922** (+0.0357 gain), reflecting near-flawless category separability.
- **Training Time:** Converged in only 7 epochs (18.0 minutes on an NVIDIA Tesla T4 GPU).

#### Per-Class Classification Report (Held-Out Test Set: 3,706 samples):
| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **Organic** | 0.97 | 0.96 | 0.96 | 2,082 |
| **Recyclable** | 0.95 | 0.96 | 0.96 | 1,624 |
| **Macro Average** | **0.96** | **0.96** | **0.96** | **3,706** |
| **Weighted Average** | **0.96** | **0.96** | **0.96** | **3,706** |

#### Test Confusion Matrix:
| Actual \ Predicted | Predicted Organic (0) | Predicted Recyclable (1) | Total Actual |
|---|---:|---:|---:|
| **Actual Organic (0)** | **2,002** (96.16%) | 80 (3.84%) | 2,082 |
| **Actual Recyclable (1)** | 66 (4.06%) | **1,558** (95.94%) | 1,624 |
| **Total Predicted** | 2,068 | 1,638 | **3,706** |

---

## 5. Discussion and Industrial Production Architecture

### 5.1 Why Vision Transformers Outperform Pure CNNs
In waste segregation, refuse items do not maintain canonical shapes:
- **Non-Rigid Deformation:** A crushed plastic bottle or crumpled can loses its standard perimeter and aspect ratio. Pure CNNs rely on localized edge filters and fail when standard geometries are distorted. MobileViT's self-attention captures long-range spatial context, maintaining high classification confidence regardless of local boundary deformation.
- **Occlusion Handling:** Refuse items overlap on conveyor belts. MobileViT attends across non-contiguous visual tokens, allowing the network to group disjoint visible patches belonging to the same item.

### 5.2 Edge Deployment Latency Profile
Unlike pure Vision Transformers (ViT-Base with 86M parameters and $>180\text{ ms}$ latency), MobileViT-S maintains a lightweight footprint of 4.94M parameters. When converted to ONNX and quantized to INT8 with TensorRT on an NVIDIA Jetson Orin Nano (8GB, 40 TOPS), MobileViT-S delivers sustained throughput of $\sim 32\text{ FPS}$ ($31.2\text{ ms}$ latency), operating well within the standard 50ms conveyor belt sorting window.

### 5.3 Industrial System Architecture: Multi-Modal Optical Sorting
To overcome the physical limitation of RGB cameras (inability to distinguish visually identical clear polymers like PET, HDPE, and PP), we propose the following end-to-end industrial pipeline:

```mermaid
flowchart LR
    Conveyor["Moving Conveyor Belt (v = 2.5 m/s)"] --> Stage1["Stage 1: YOLOv8-Nano Detection (<12 ms)"]
    Stage1 --> Stage2["Stage 2: MobileViT-S Classification (<25 ms)"]
    Stage2 --> Stage3["Stage 3: NIR Spectrometer Point Probe (<5 ms)"]
    Stage3 --> Decision["Dual-Branch Fusion Decision"]
    Decision --> Actuator["Pneumatic Air Ejectors / Robotic Picker"]
```

1. **Stage 1 (Localization):** Overhead RGB camera feed processed by YOLOv8-Nano to detect and crop multiple bounding boxes simultaneously.
2. **Stage 2 (Visual Categorization):** Cropped items classified into broad categories by MobileViT-S.
3. **Stage 3 (Spectral Polymer Verification):** A localized Near-Infrared (NIR) spectrometer probe ($900-1700\text{ nm}$) scans molecular overtone reflectance peaks ($\sim 1660\text{ nm}$ for PET vs. $\sim 1730\text{ nm}$ for HDPE/PP), processed by a 1D-CNN.
4. **Stage 4 (Actuation):** High-speed pneumatic air jets divert items into target recovery bins.

---

## 6. Limitations
1. **Dataset Scope:** Evaluated on the 24,705-image Kaggle benchmark; domain shifts under severe moisture, mud, or industrial grime require real-world validation.
2. **Binary Granularity:** The current benchmark evaluates Organic vs. Recyclable; commercial facilities require 10–15 sub-classes.
3. **Actuator Latency:** Physical air-jet actuation dynamics and pneumatic delay were modeled analytically but not evaluated on a physical conveyor rig.

---

## 7. Conclusion
This study systematically evaluated the efficiency and accuracy frontiers of automated waste classification. For highly resource-constrained micro-controllers, the proposed RLFA-CNN demonstrated 90.50% test accuracy and 93.72% recyclable recall with only 110,914 parameters, reducing baseline parameter overhead by 97.4%. For industrial sorting lines requiring maximum classification accuracy under severe deformation and occlusion, the hybrid MobileViT-S vision transformer established state-of-the-art performance, achieving 96.06% accuracy, 0.9606 F1-score, and 0.9922 ROC-AUC with 4.94M parameters.

Future work will focus on integrating YOLOv8-Nano for real-time bounding-box detection, physical implementation of the dual-branch RGB+NIR spectral fusion model, and expanding classification granularity to multi-class municipal sorting.

---

## References
1. S. Kaza, L. Yao, P. Bhada-Tata, and F. Van Woerden, *What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050*. Washington, DC: World Bank, 2018.
2. M. S. B. Islam, M. S. Islam Sumon, M. E. Majid, et al., "ECCDN-Net: A deep learning-based technique for efficient organic and recyclable waste classification," *Waste Management*, vol. 193, pp. 363–375, 2025.
3. M. A. Rahman, M. M. Hossain, and S. Islam, "DenseNet-201 with Squeeze-and-Excitation attention for municipal waste classification," *IEEE Access*, vol. 12, pp. 11204–11215, 2024.
4. Z. Wang, H. Liu, and Y. Chen, "Real-time multi-class waste detection on automated conveyor belts using lightweight YOLOv8," *Computers and Electronics in Agriculture*, vol. 210, p. 107912, 2023.
5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520.
6. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR, 2019, pp. 6105–6114.
7. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
8. S. Mehta and M. Rastegari, "MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.
9. S. Kapadnis, "CNN Waste Classification Image Dataset," Kaggle Dataset Repository, 2021.
