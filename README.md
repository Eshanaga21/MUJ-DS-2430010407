# Waste Segregation Using Deep Learning — Organic vs Recyclable Classification

**Registration Number:** 2430010407  
**Initial Evaluation Submission Date:** September 14, 2026  
**Nature of Submission:** Individual Contribution  

---

## 1. Project Title
**Waste Segregation Using Deep Learning — Proposed Regularized Lightweight Feature Aggregation Convolutional Neural Network (RLFA-CNN)**

---

## 2. Student Details
| Detail | Value |
|---|---|
| **Student Name** | Eshan Agarwal |
| **Registration Number** | 2430010407 |
| **Section** | K21AB |

---

## 3. Dataset Analysis

### A. Dataset Overview and Source
- **Dataset Name:** CNN Waste Classification Image Dataset
- **Source Link:** [Kaggle Dataset by Sujay Kapadnis](https://www.kaggle.com/datasets/sujaykapadnis/cnn-waste-classification-image-dataset)
- **Total Images:** 24,705 images
- **Classes:** 2 (`organic`, `recyclable`)
- **Image Resolution Range:** Minimum (51 × 69) to Maximum (1460 × 1500) pixels
- **Model Input Resolution:** Standardized to 128 × 128 × 3 (RGB)

### B. Class Distribution and Imbalance Analysis
| Class Label | Class ID | Number of Samples | Percentage (%) | Imbalance Ratio |
|---|---|---:|---:|---:|
| **Organic** | 0 | 13,880 | 56.18% | 1.28 |
| **Recyclable** | 1 | 10,825 | 43.82% | 1.00 (Reference) |
| **Total** | — | **24,705** | **100.00%** | **Balanced (< 1.5 threshold)** |

*Observation:* With an imbalance ratio of 1.28:1 (below the conventional 1.5 threshold), the dataset is sufficiently balanced. Therefore, synthetic resampling or aggressive class weighting was not strictly required, though balanced class weights were dynamically supported in the pipeline. Refer to [`figures/class_distribution.png`](figures/class_distribution.png).

### C. Stratified Train / Validation / Test Splitting
A fixed random seed (`SEED = 42`) was utilized with stratified sampling to preserve class proportions across all splits, matching the executed notebook:
| Split Set | Percentage | Total Samples | Organic Samples | Recyclable Samples |
|---|---|---:|---:|---:|
| **Training Set** | 70.0% | 17,292 | 9,715 | 7,577 |
| **Validation Set** | 15.0% | 3,707 | 2,083 | 1,624 |
| **Test Set** (Held-out) | 15.0% | 3,706 | 2,082 | 1,624 |
| **Total** | **100.0%** | **24,705** | **13,880** | **10,825** |

### D. Preprocessing Pipeline
1. **Recursive Integrity Check:** Scanned all folders and verified image readability using PIL, pruning corrupted files.
2. **Decoding and Normalization:** Images decoded via `tf.io.decode_image(channels=3)`, resized to 128 × 128 with bilinear interpolation, and rescaled from `[0, 255]` to `[0.0, 1.0]` by dividing by 255.0.
3. **Training-Only Data Augmentation:**
   - `RandomFlip("horizontal")`: Simulates viewpoint invariance.
   - `RandomRotation(0.15)`: Robustness against varied orientations.
   - `RandomZoom(0.15)`: Handles variable distances of waste items.
   - `RandomContrast(0.10)`: Accommodates varying ambient lighting conditions.
4. **tf.data Pipeline Optimization:** Batch size of 32 with shuffle buffer, parallel mapping (`num_parallel_calls=AUTOTUNE`), and prefetching (`prefetch(AUTOTUNE)`).

---

## 4. Model Architecture

### Proposed Architecture: RLFA-CNN
The **Regularized Lightweight Feature Aggregation CNN (RLFA-CNN)** is an experimental convolutional neural network designed with parameter-efficient, edge-oriented deployment in mind. Its 110,914 parameters imply substantially lower model storage requirements than the compared architectures; actual device latency and memory usage were not benchmarked in this project. Rather than relying on heavy pre-trained backbones or dense classification layers, the proposed experimental architecture incorporates:
1. **Layer-wise Batch Normalization** for internal covariate shift mitigation and stable gradient propagation.
2. **Global Average Pooling (GAP)** in place of dense flatten vectors, achieving **approximately 97.4% fewer parameters** than the SimpleCNN baseline while reducing spatial overfitting.
3. **L1 Kernel Regularization** ($\lambda = 1\times 10^{-5}$) to encourage sparse weights.
4. **Adaptive Learning Rate Scheduling** via `ReduceLROnPlateau`.

### Layer-by-Layer Architectural Specification
| Layer Type | Filters / Units | Kernel / Pool Size | Activation | Regularization | Output Shape | Parameters |
|---|---|---|---|---|---|---:|
| **Input Layer** | — | — | — | — | (128, 128, 3) | 0 |
| **Data Augmentation** | — | — | — | — | (128, 128, 3) | 0 |
| **Conv2D (Block 1)** | 32 | 3 × 3 (same) | ReLU | L1 ($10^{-5}$) | (128, 128, 32) | 896 |
| **BatchNormalization** | — | — | — | — | (128, 128, 32) | 128 |
| **MaxPooling2D** | — | 2 × 2 (valid) | — | — | (64, 64, 32) | 0 |
| **Conv2D (Block 2)** | 64 | 3 × 3 (same) | ReLU | L1 ($10^{-5}$) | (64, 64, 64) | 18,496 |
| **BatchNormalization** | — | — | — | — | (64, 64, 64) | 256 |
| **MaxPooling2D** | — | 2 × 2 (valid) | — | — | (32, 32, 64) | 0 |
| **Conv2D (Block 3)** | 128 | 3 × 3 (same) | ReLU | L1 ($10^{-5}$) | (32, 32, 128) | 73,856 |
| **BatchNormalization** | — | — | — | — | (32, 32, 128) | 512 |
| **MaxPooling2D** | — | 2 × 2 (valid) | — | — | (16, 16, 128) | 0 |
| **GlobalAveragePooling2D** | — | — | — | — | (128,) | 0 |
| **Dense (FC1)** | 128 | — | ReLU | L1 ($10^{-5}$) | (128,) | 16,512 |
| **Dropout** | Rate = 0.3 | — | — | — | (128,) | 0 |
| **Dense (Output)** | 2 | — | Softmax | None | (2,) | **258** |
| **Total Parameters** | — | — | — | — | — | **110,914** |

> **Parameter Breakdown:**
> - **Total Parameters:** 110,914
> - **Trainable Parameters:** 110,466
> - **Non-Trainable Parameters:** 448 (BatchNormalization moving mean and variance statistics)

---

## 5. Hyperparameters

The project evaluates models under distinct training protocols:
1. **CO3 Standalone Training Configuration:** Full 50-epoch horizon to achieve final converged model performance.
2. **CO4/CO5 Controlled Comparison Protocols:** CO5 ablation models were trained with a maximum 30-epoch budget (`ABLATION_EPOCHS = 30`). The pretrained CO4 benchmark models (MobileNetV2 and EfficientNetB0) were trained with a maximum 15-epoch budget (`BENCHMARK_EPOCHS = 15`). Both protocols used EarlyStopping based on validation loss.

| Hyperparameter | CO3 Standalone Configuration | CO4/CO5 Controlled Configuration | Notes |
|---|---|---|---|
| **Initial Learning Rate** | 0.001 | 0.001 | Standard Adam default; dynamically scheduled |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$) | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$) | Adaptive gradient optimization |
| **Loss Function** | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Chosen for stable integer-encoded label optimization |
| **Batch Size** | 32 | 32 | Maximizes GPU throughput and gradient stability |
| **Maximum Epochs** | **50 Epochs** | **30 Epochs (CO5 Ablation) / 15 Epochs (CO4 Benchmarks)** | EarlyStopping monitored on val_loss |
| **Early Stopping** | Patience = 5 epochs | Patience = 5 epochs | Monitored `val_loss`; restores best checkpoint weights |
| **LR Scheduler** | `ReduceLROnPlateau` | `ReduceLROnPlateau` | Factor = 0.5, Patience = 2 epochs, Minimum LR = $1\times 10^{-6}$ |
| **L1 Regularization** | $\lambda = 1\times 10^{-5}$ | $\lambda = 1\times 10^{-5}$ | Applied to Conv2D and Dense kernel tensors |
| **Dropout Probability** | 0.30 | 0.30 | Applied prior to final softmax layer |
| **Random Seed** | 42 | 42 | Set across Python `random`, NumPy, and TensorFlow |

---

## 6. Experimental Configuration Selection (Hyperparameter & Architectural Choices)

*Note on Hyperparameter Evaluation:* Experimental configuration was fixed to 128×128 input resolution, batch size 32, learning rate 0.001, and EarlyStopping patience 5. These settings were used consistently across the primary experiments to maintain a controlled comparison. The project does not claim a comprehensive grid search over these hyperparameters.

The experimental configurations evaluated across the project's notebooks are:

1. **Regularization Mechanism Comparison (CO3):**
   In the CO3 notebook (`co3_waste_classification.ipynb`, cell 17 & 21), four regularization configurations were evaluated under identical training conditions on the development set:
   | Regularization | Best Validation Accuracy | Test Accuracy |
   |---|---:|---:|
   | Baseline | 89.94% | 89.56% |
   | Dropout | 88.18% | 87.91% |
   | L1/Lasso | **90.83%** | **90.50%** |
   | L2/Ridge | 90.34% | 90.56% |

   *Model Selection Rationale:* L1/Lasso was selected based on the highest best validation accuracy (90.83%). Final evaluation used the restored best-validation-loss weights. The held-out test set was not used for model selection.

2. **Feature Aggregation vs. Flatten Head (CO5):**
   In `co4_co5_extension.ipynb`, replacing the Flatten-based classification head with Global Average Pooling (GAP) and Batch Normalization reduced parameter count from 4,287,938 to 110,914 (**approximately 97.4% reduction**) while increasing test accuracy from 86.35% to 89.02% (+2.67%).

3. **Incremental L1 Regularization (CO5):**
   Adding L1 kernel regularization ($\lambda = 10^{-5}$) to the FeatureAggregation model in the ablation study further increased recyclable waste recall from 91.81% to 93.72% (+1.91%) and improved ROC-AUC from 0.9551 to 0.9565.

4. **Training Budget and Early Stopping Schedule:**
   - `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`: Terminated training after validation loss failed to improve for 5 consecutive epochs, restoring the best model checkpoint weights.
   - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)`: Halved the learning rate after 2 epochs of non-improving validation loss.

---

## 7. Model Performance and Results (CO3 – 10 Marks)

### A. Overall Evaluation Metrics (Held-Out Test Set: 3,706 samples)
| Metric | Value (%) | Decimal |
|---|---:|---:|
| **Test Accuracy** | **90.50%** | 0.905019 |
| **Macro Average Precision** | **90.31%** | 0.903147 |
| **Macro Average Recall** | **90.43%** | 0.904291 |
| **Macro Average F1-Score** | **90.37%** | 0.903686 |
| **Weighted Average Precision** | **90.52%** | 0.905218 |
| **Weighted Average Recall** | **90.50%** | 0.905019 |
| **Weighted Average F1-Score** | **90.51%** | 0.905086 |
| **Test Loss** | — | **0.270497** |

### B. Per-Class Performance and Mathematically Correct Specificity
In this binary classification task with `recyclable` as the positive class (class 1) and `organic` as the negative class (class 0):
- **Recall / Sensitivity:** Proportion of actual recyclable images correctly identified as recyclable ($\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{1459}{1459 + 165} = \frac{1459}{1624} \approx \mathbf{0.898}$).
- **Specificity:** Proportion of actual organic images correctly identified as organic ($\frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{1895}{1895 + 187} = \frac{1895}{2082} \approx \mathbf{0.910}$).
- *(For the reciprocal perspective with organic as positive, Recall is 0.910 and Specificity is 0.898).*

| Class | Precision | Recall | F1-Score | Specificity | Support (Samples) |
|---|---:|---:|---:|---:|---:|
| **Organic** | 0.920 | 0.910 | 0.915 | **0.898** | 2,082 |
| **Recyclable** | 0.886 | 0.898 | 0.892 | **0.910** | 1,624 |

*Detailed CSV available at:* [`results/results.csv`](results/results.csv)

### C. Confusion Matrix
| Actual \ Predicted | Predicted Organic (0) | Predicted Recyclable (1) | Total Actual |
|---|---:|---:|---:|
| **Actual Organic (0)** | **1,895** | 187 | 2,082 |
| **Actual Recyclable (1)** | 165 | **1,459** | 1,624 |
| **Total Predicted** | 2,060 | 1,646 | **3,706** |

*Visual plot saved at:* [`results/confusion_matrix.png`](results/confusion_matrix.png)

### D. Training and Convergence Analysis
- **Training Trajectory:** Training showed stable convergence, with the best observed validation loss occurring at epoch 21 (0.2676) and best validation accuracy of 90.83%. Early stopping terminated training at epoch 26 after validation performance stopped improving. Training loss decreased from approximately 0.4734 to 0.2309 over the executed epochs.
- **Accuracy Trajectory:** Best validation accuracy reached **90.83%** at epoch 21 for the standalone CO3 training run. (CO5 RLFA-L1 best validation accuracy under the controlled 30-epoch protocol: 89.29%).
- **5-Fold Cross Validation:** 5-fold cross-validation accuracy: **89.48% ± 0.63 percentage points** (where 0.63 percentage points is the standard deviation across folds; fold accuracies: 88.79%, 89.90%, 88.79%, 90.40%, 89.52%).
- *Visual curves saved at:* [`results/training_curve.png`](results/training_curve.png), [`figures/training_curves.png`](figures/training_curves.png), [`figures/roc_curve.png`](figures/roc_curve.png), and [`figures/cross_validation.png`](figures/cross_validation.png).

---

## 8. Comparative Analysis (CO4 – 5 Marks)

### A. Executed Transfer-Learning Benchmark Comparison (Same Dataset, Same Split)
The proposed RLFA-CNN + L1 configuration from the controlled ablation study is evaluated alongside EfficientNetB0 and MobileNetV2 on the same untouched test set. CO3 uses validation performance for model selection. CO4 evaluates the proposed configuration against benchmark architectures. The held-out test set was not used for selecting the CO3 model:

| Model Architecture | Accuracy (%) | Precision | Recall (Recyclable) | F1-Score | Specificity | ROC-AUC | Parameters | Training Time (s) | Status | Academic Reference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **EfficientNetB0** | **93.09%** | 0.9166 | 0.9267 | **0.9216** | **0.9342** | **0.9773** | 4,052,133 | 488.8 s | Reproduced in this project | Tan & Le (2019) *ICML* [1] |
| **MobileNetV2** | **92.20%** | 0.9078 | 0.9150 | **0.9114** | **0.9275** | **0.9758** | 2,260,546 | 267.6 s | Reproduced in this project | Sandler et al. (2018) *CVPR* [2] |
| **Proposed RLFA-CNN + L1 (Ours)** | **89.10%** | 0.8344 | **0.9372** | **0.8828** | 0.8549 | **0.9565** | **110,914** | 880.4 s | Proposed Configuration | Custom Architecture |

*Table data saved in:* [`results/comparison.csv`](results/comparison.csv) and [`results/sota_comparison.csv`](results/sota_comparison.csv)  
*Visual charts saved in:* [`figures/sota_comparison.png`](figures/sota_comparison.png) and [`figures/roc_curve_sota.png`](figures/roc_curve_sota.png)

### B. Published Same-Dataset Literature Comparison
A published 2025 Waste Management study (Islam et al., 2025) evaluated several deep learning architectures on this exact 24,705-image Organic/Recyclable dataset. Published reported results include:

| Published Method | Reported Accuracy | Status |
|---|---:|---|
| **ResNet18** | 92.68% | Published result — not reproduced in this project |
| **MobileNetV2** | 93.27% | Published result — not reproduced in this project |
| **InceptionV3** | 94.77% | Published result — not reproduced in this project |
| **DenseNet201** | 95.98% | Published result — not reproduced in this project |
| **ECCDN-Net** | 96.10% | Published result — not reproduced in this project |

*Methodological Note:* The published results provide external context for the dataset but were not reproduced under this project's experimental protocol. Published experimental protocols may differ from this project's 70/15/15 stratified split and pre-processing pipeline; therefore, these values provide broad academic literature context rather than directly reproduced measurements.

### C. Technical Analysis of Accuracy-Efficiency Trade-offs:
1. **Honest Performance Comparison:** As expected from transfer learning using ImageNet-pretrained weights, **EfficientNetB0 (93.09%)** and **MobileNetV2 (92.20%)** achieve higher overall test accuracy than RLFA-CNN (89.10%), which was trained from scratch.
2. **Extreme Parameter Efficiency:**
   - RLFA-CNN requires only **110,914 parameters**.
   - This represents **approximately 95.1% fewer parameters than MobileNetV2** (2.26M parameters) and **approximately 97.3% fewer parameters than EfficientNetB0** (4.05M parameters).
   - The lightweight architecture is designed with parameter-efficient, edge-oriented deployment in mind. Its 110,914 parameters imply substantially lower model storage requirements than the compared architectures; actual device latency and memory usage were not benchmarked in this project.
3. **High Recyclable Waste Recall:**
   - In automated recycling facilities, the critical operational objective is minimizing false negatives on recyclable material (preventing recyclables from entering landfills).
   - Proposed RLFA-CNN achieves **93.72% recall on recyclable waste**, which is higher than EfficientNetB0 (92.67%) and MobileNetV2 (91.50%).
4. **Training-Time Comparison Context:** Training times should not be interpreted as a direct computational-efficiency ranking because the proposed model was trained from scratch under a 30-epoch budget, whereas the transfer-learning benchmark models used frozen ImageNet features under a 15-epoch budget.

---

## 9. Innovation and Ablation Study (CO5 – 5 Marks)

### What is the Innovation Compared with Baselines?
Rather than trivially tuning hyperparameters, our contribution is an **experimentally validated, parameter-efficient convolutional architecture (RLFA-CNN)**:
- **Ablation Scope:** The ablation evaluates the combined lightweight architectural redesign (data augmentation + Batch Normalization + Global Average Pooling) and then measures the incremental effect of adding L1 kernel regularization.
- **Combined Lightweight Redesign (Exp 1 → Exp 2):** The combined lightweight redesign incorporating augmentation, Batch Normalization, and Global Average Pooling increased test accuracy from 86.35% to 89.02% while reducing the parameter count by approximately 97.4% (from 4,287,938 to 110,914 parameters).
- **Incremental L1 Regularization (Exp 2 → Exp 3):** Adding L1 regularization produced a small increase in overall test accuracy from 89.02% to 89.10%, increased recyclable-class recall from 91.81% to 93.72%, and increased ROC-AUC from 0.9551 to 0.9565, while validation accuracy decreased slightly. L1 kernel regularization ($\lambda = 1\times 10^{-5}$) encourages sparse weights across convolutional and dense kernels, shifting the model toward higher recyclable-class recall and slightly higher ROC-AUC, with a small trade-off in precision and validation accuracy.

### Quantitative Ablation Study
All three ablation models were trained under identical random seeds (`SEED = 42`), identical 70/15/15 splits, and identical 30-epoch training budgets:

| Experiment | Model Variant | Data Augmentation | BatchNorm + GAP | L1 Regularization | Accuracy (%) | Precision | Recall (Recyclable) | F1-Score | Specificity | ROC-AUC | Total Parameters | Training Time (s) |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Exp 1** | SimpleCNN (Baseline) | ✗ | ✗ | ✗ | 86.35% | 0.8516 | 0.8337 | 0.8426 | 0.8866 | 0.9352 | 4,287,938 | 211.0 s |
| **Exp 2** | FeatureAggregation CNN | ✓ | ✓ | ✗ | 89.02% | 0.8448 | 0.9181 | 0.8799 | 0.8684 | 0.9551 | 110,914 | 688.1 s |
| **Exp 3** | **RLFA-CNN (Proposed)** | **✓** | **✓** | **✓** | **89.10%** | **0.8344** | **0.9372** | **0.8828** | **0.8549** | **0.9565** | **110,914** | **880.4 s** |

*Table data saved in:* [`results/ablation_results.csv`](results/ablation_results.csv)  
*Plots saved in:* [`figures/ablation_results.png`](figures/ablation_results.png) and [`figures/roc_curve.png`](figures/roc_curve.png)

---

## 10. Protocol Clarification: CO3 (90.50%) vs. CO4/CO5 (89.10%) Results

The project reports distinct evaluation figures depending on the experimental protocol:
- **CO3 Standalone Training Result (90.50% Test Accuracy):** In `co3_waste_classification.ipynb`, the proposed model was trained from scratch across the full 50-epoch budget with Early Stopping (`patience=5`), stopping early at epoch 26 (best epoch 21 with validation loss 0.2676 and best validation accuracy 90.83%), enabling the network to reach its best observed performance of **90.50% test accuracy** and **90.37% macro F1**.
- **CO5 Controlled Ablation Protocol (89.10% Test Accuracy):** In `co4_co5_extension.ipynb`, CO5 ablation models (SimpleCNN, FeatureAggregation, and RLFA-CNN) were trained with a maximum 30-epoch budget (`ABLATION_EPOCHS = 30`) using EarlyStopping to ensure a controlled architectural comparison under the same data split and training protocol. Under this controlled 30-epoch protocol, RLFA-CNN reached **89.10% test accuracy**, **0.9565 ROC-AUC**, and a best validation accuracy of **89.29%**.
- **CO4 Pretrained Benchmark Protocol (92.20% and 93.09%):** The pretrained CO4 benchmark models (MobileNetV2 and EfficientNetB0) were trained under a separate, transfer-learning protocol with a maximum 15-epoch budget (`BENCHMARK_EPOCHS = 15`) and EarlyStopping based on validation loss.

---

## 11. Folder Structure

```text
2430010407/
│
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Verified Python dependencies & versions
│
├── data/
│   └── dataset_information.txt            # Dataset metadata, source, distribution & splits
│
├── models/
│   └── model_description.txt              # Detailed architectural breakdown and parameter counts
│
├── src/
│   ├── model.py                           # RLFA-CNN and SimpleCNN model definitions
│   ├── preprocessing.py                   # Data pipeline, augmentations & stratified splitting
│   ├── train.py                           # Training routine with early stopping and callbacks
│   └── test.py                            # Comprehensive evaluation and binary metrics suite
│
├── notebooks/
│   ├── co3_waste_classification.ipynb     # CO3 complete training and evaluation notebook
│   └── co4_co5_extension.ipynb            # CO4 benchmarks & CO5 ablation study notebook
│
├── results/
│   ├── results.csv                        # Final held-out test evaluation metrics
│   ├── comparison.csv                     # CO4 benchmark comparison table (EfficientNet, MobileNet, RLFA)
│   ├── sota_comparison.csv                # Benchmark comparison table matching extension notebook export
│   ├── ablation_results.csv               # CO5 experimental ablation study table
│   ├── environment_versions.csv           # Environment library versions
│   ├── confusion_matrix.png               # Test confusion matrix visualization
│   └── training_curve.png                 # Training vs. validation loss & accuracy curves
│
└── figures/
    ├── class_distribution.png             # Sample distribution per waste category
    ├── sample_images.png                  # Sample training images inspection
    ├── training_curves.png                # Loss and accuracy convergence plots
    ├── cross_validation.png               # 5-fold cross-validation performance
    ├── ablation_results.png               # Visual ablation study comparison
    ├── roc_curve.png                      # Receiver Operating Characteristic (ROC-AUC) curve
    ├── sota_comparison.png                # CO4 test accuracy comparison barplot
    └── roc_curve_sota.png                 # CO4 ROC curves comparing RLFA-CNN, MobileNet, EfficientNet
```

---

## 12. Project Limitations

1. **Dataset Scope:** The model was evaluated strictly on the specified Kaggle Organic/Recyclable waste dataset (24,705 images); external real-world generalization across varied waste streams was not independently evaluated.
2. **Joint Ablation Scope:** CO5's first ablation step modifies augmentation, Batch Normalization, and Global Average Pooling jointly; therefore, their individual contributions are not isolated separately.
3. **Training Protocol Differences:** Benchmark training times are not directly comparable because training protocols differ (30-epoch scratch training for RLFA-CNN vs. 15-epoch frozen-backbone training for transfer-learning models).
4. **Hardware Deployment:** Edge-device latency, physical memory usage, and real-time throughput on hardware such as Raspberry Pi or Jetson were not directly benchmarked in this project.
5. **Literature Context:** Published literature results (e.g. Islam et al., 2025) are cited solely for external academic context and were not reproduced under this project's experimental split.

---

## 13. References

1. Tan, M., & Le, Q. (2019). *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."* International Conference on Machine Learning (ICML), PMLR, 6105–6114.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *"MobileNetV2: Inverted Residuals and Linear Bottlenecks."* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.
3. Islam, M. S. B., Islam Sumon, M. S., Majid, M. E., et al. (2025). *"ECCDN-Net: A deep learning-based technique for efficient organic and recyclable waste classification."* Waste Management, 193, 363–375. DOI: 10.1016/j.wasman.2024.12.023

---

## 14. Environment and Reproducibility
- **Python:** 3.13.15
- **TensorFlow:** 2.20.0
- **NumPy:** 2.1.3
- **Pandas:** 2.2.3
- **Scikit-Learn:** 1.6.1
- **Matplotlib:** 3.10.0
- **Seaborn:** 0.13.2
- **Pillow:** 11.3.0

Random seeds are fixed at 42 across Python, NumPy, and TensorFlow to improve reproducibility. GPU execution may still introduce minor run-to-run numerical variation.
