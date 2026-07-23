# RFInsight: Deep Learning Framework for Wireless Signal Classification Using Spectrogram Images

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-yellow)

**RFInsight** is a deep learning-based framework that reframes wireless RF signal classification as a computer vision problem. By converting raw RF signals into spectrogram images, RFInsight leverages state-of-the-art CNN, Transformer, and hybrid architectures to automatically learn discriminative time-frequency patterns — eliminating the need for manual, handcrafted feature engineering traditionally used in RF signal analysis.

This repository documents an end-to-end research pipeline: from spectrogram-based data preprocessing to model benchmarking, class-imbalance handling, and explainable AI, developed as part of an M.Tech Data Science and Machine Learning capstone project.

---

## 1. Overview

Wireless RF signal classification is the task of identifying the type or modulation scheme of a radio-frequency signal from its captured waveform. RFInsight approaches this problem through the lens of computer vision:

- **RF Signal Classification**: Instead of analyzing raw I/Q (in-phase/quadrature) time-series data directly, RF signals are transformed into visual representations that encode both time and frequency information.
- **Spectrogram-Based Image Representation**: Each RF signal is converted into a spectrogram — a 2D time-frequency image where intensity represents signal power at a given time and frequency. This transforms a 1D signal processing problem into a 2D image classification problem.
- **Deep Learning Approach**: Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are applied directly on spectrogram images to automatically learn hierarchical and contextual features that distinguish between signal classes.
- **Difference from Traditional Handcrafted Feature Methods**: Classical RF classification methods rely on manually engineered features (e.g., cyclic spectral analysis, higher-order statistics, wavelet coefficients) combined with traditional classifiers like SVM or Random Forest. These methods are labor-intensive, domain-specific, and often fail to generalize across varying SNR conditions or unseen signal types. RFInsight's deep learning pipeline learns features directly from data, enabling better generalization, scalability, and adaptability to noisy real-world RF environments.

---

## 2. Problem Statement

The radio frequency spectrum is increasingly congested due to the proliferation of wireless devices, IoT systems, and communication standards. Reliable and automated identification of RF signals is critical but challenging due to:

- Overlapping frequency bands and signal interference
- Variable signal-to-noise ratio (SNR) conditions
- Similarity between modulation schemes at low SNR
- Scarcity of labeled real-world RF data
- Need for real-time, low-latency inference in operational settings

Robust RF signal classification directly supports several real-world applications:

- **Spectrum Monitoring** — Tracking spectrum occupancy and usage patterns across frequency bands
- **Cognitive Radio Systems** — Enabling dynamic spectrum access by identifying unused or underutilized channels
- **Signal Intelligence (SIGINT)** — Detecting and categorizing signals of interest in defense and security contexts
- **Interference Detection** — Identifying sources of unwanted or malicious interference in communication networks
- **Wireless Network Management** — Supporting network optimization, anomaly detection, and quality-of-service improvements

---

## 3. Dataset

- **Source**: Kaggle RF Signal Image Classification Dataset
- **Classes**: 21 distinct wireless signal classes (e.g., various modulation schemes and signal types)
- **Data Format**: Spectrogram images generated from raw RF signal captures
- **Representation**: Time-frequency representation, where each image encodes signal energy distribution across time and frequency axes, allowing deep learning models to learn spatial patterns corresponding to unique signal characteristics

---

## 4. Project Objectives

- Develop an accurate and generalizable **RF signal classification** pipeline using spectrogram images
- Perform a comparative study between **CNN and Transformer-based architectures** for RF spectrogram classification
- Apply **transfer learning** using pretrained backbones to improve performance and reduce training time
- Address **class imbalance** across the 21 signal categories using specialized loss functions and sampling strategies
- Incorporate **Explainable AI (XAI)** techniques to interpret model predictions and visualize discriminative spectrogram regions

---

## 5. Methodology

The project follows a structured deep learning workflow:

```
RF Spectrogram Images
        ↓
Data Preprocessing
        ↓
Augmentation
        ↓
Deep Learning Models
        ↓
Evaluation
        ↓
Explainability
```

**Pipeline Description:**

1. **Data Preprocessing** — Resizing, normalization, and formatting of spectrogram images for model compatibility
2. **Augmentation** — Applying transformations (flips, rotations, cropping, noise injection) to improve model robustness and reduce overfitting
3. **Deep Learning Models** — Training and benchmarking multiple architectures (CNNs, Transformers, hybrid models) on the preprocessed dataset
4. **Evaluation** — Assessing model performance using standard classification metrics
5. **Explainability** — Applying Grad-CAM to interpret and visualize model decision-making

---

## 6. Model Architectures

| Model | Architecture Idea | Advantages | Why Useful for RF Classification |
|---|---|---|---|
| **ResNet50** | Deep CNN with residual (skip) connections | Avoids vanishing gradients; strong, well-tested baseline | Captures hierarchical spatial patterns like frequency bands and energy bursts |
| **EfficientNet** | Compound scaling of depth, width, and resolution | High accuracy with fewer parameters and lower compute | Suitable for edge/resource-constrained spectrum monitoring |
| **Vision Transformer (ViT)** | Splits image into patches, applies self-attention instead of convolutions | Captures long-range, global dependencies | Models non-local time-frequency patterns in spectrograms |
| **CNN-ViT Hybrid** | CNN layers for local features + transformer blocks for global context | Combines local texture sensitivity with global structure awareness | Captures both fine-grained bursts and global spectrogram structure |
| **LDAM + Supervised Contrastive Learning** | Margin-based loss for class imbalance + contrastive embedding learning | Improves minority-class recall and embedding separability | Handles imbalance across the 21 signal classes and yields more robust features |

---

## 7. Explainable AI (XAI)

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to interpret model predictions:

- **Model Attention Visualization**: Generates heatmaps highlighting which regions of the input spectrogram most influenced the model's prediction
- **Important Spectrogram Regions**: Identifies specific time-frequency regions (e.g., particular frequency bands or burst patterns) that the model relies on for classification
- **Prediction Interpretability**: Enhances trust and transparency in model decisions, which is critical for applications in security, defense, and regulatory spectrum monitoring where "black-box" predictions are insufficient

---

## 8. Evaluation Metrics

Model performance is assessed using the following standard classification metrics:

- **Accuracy** — Overall correctness of predictions across all classes
- **Precision** — Proportion of correctly predicted positive instances among all predicted positives
- **Recall** — Proportion of correctly predicted positive instances among all actual positives
- **F1 Score** — Harmonic mean of precision and recall, useful for imbalanced class distributions
- **Confusion Matrix** — Detailed breakdown of predictions vs. actual labels across all 21 signal classes, useful for identifying class-specific misclassification patterns

---

## 9. Technologies Used

**Programming Language**
- Python

**Deep Learning**
- PyTorch
- Torchvision
- TIMM (PyTorch Image Models)

**Data Handling**
- NumPy
- Pandas

**Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn

**Explainability**
- Grad-CAM

---

## 10. Key Features

- 🛰️ **Vision-Based RF Analysis** — Treats RF signal classification as an image classification problem using spectrograms
- 🧠 **Multi-Architecture Benchmarking** — Compares CNN, Transformer, and hybrid models on a common dataset
- 🔁 **Transfer Learning** — Leverages pretrained backbones for faster convergence and improved accuracy
- ⚖️ **Class Imbalance Handling** — Implements LDAM loss and supervised contrastive learning for robust minority-class performance
- 🔍 **Explainable Predictions** — Uses Grad-CAM to visualize and interpret model decisions
- 📊 **Comprehensive Evaluation** — Reports accuracy, precision, recall, F1-score, and confusion matrices for thorough model assessment
- 📡 **Real-World Relevance** — Directly applicable to spectrum monitoring, cognitive radio, and signal intelligence domains
- 🧩 **Modular Pipeline** — Clean, extensible workflow from preprocessing to explainability

---

## 11. Repository Structure

```
RFInsight/
│
├── RFInsight_Wireless_Signal_Classification.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

---

## 12. Applications

- **Wireless Communication** — Improving signal identification in modern communication systems
- **Cognitive Radio** — Enabling dynamic and intelligent spectrum access decisions
- **Spectrum Monitoring** — Automated tracking of spectrum usage and occupancy
- **RF Security** — Detecting unauthorized, spoofed, or malicious RF transmissions
- **Telecommunications Research** — Supporting academic and industrial research into next-generation wireless systems

---

## 13. Future Improvements

- ⏱️ **Real-Time RF Classification** — Optimizing inference pipelines for live signal stream processing
- 📈 **Deployment Dashboard** — Building an interactive dashboard for monitoring classification results in real time
- 🔧 **Edge AI Optimization** — Compressing and quantizing models for deployment on edge/embedded RF hardware
- 🧬 **Self-Supervised Learning** — Exploring self-supervised pretraining to reduce dependency on labeled RF data

---

## 14. Author

**Nidhi Dhameliya**
M.Tech Data Science and Machine Learning
Rashtriya Raksha University

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
