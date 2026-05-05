# 🌍 EuroSat Land Use Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![ONNX](https://img.shields.io/badge/ONNX-Supported-lightgrey)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-94.54%25-brightgreen)

A comprehensive Deep Learning project for classifying satellite images into 10 different land-use categories. This repository explores custom CNN architectures, state-of-the-art models (Inception, ResNet, SENet, SqueezeNet), hyperparameter tuning, and cross-framework model deployment using ONNX.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Preprocessing & Pipeline](#-preprocessing--pipeline)
- [Model Architectures](#-model-architectures)
- [Results & Evaluation](#-results--evaluation)
- [Model Deployment (ONNX)](#-model-deployment-onnx)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)

---

## 🔍 Project Overview
The goal of this project is to accurately classify satellite images from the **EuroSat** dataset. By designing from-scratch Baseline CNNs, tuning their hyperparameters, and implementing modern architectures like **Inception**, **ResNet**, **SENet**, and **SqueezeNet**, this project demonstrates a deep understanding of CNN design patterns. The final, most optimized model is then exported and evaluated in a different deep learning framework (PyTorch) using **ONNX** to simulate a real-world production deployment.

---

## 🛰️ Dataset
We use the **[EuroSat Dataset](https://github.com/phelber/eurosat)** (downloaded via Kaggle), which contains:
- **Total Images:** 27,000
- **Format:** RGB Images, $64 \times 64$ pixels
- **Classes:** 10 distinct land-use categories (e.g., Forest, Highway, Residential, etc.)

**Data Split (Stratified to maintain class balance):**
- **Training Set:** 70% (18,900 images)
- **Validation Set:** 15% (4,050 images)
- **Test Set:** 15% (4,050 images)

---

## ⚙️ Preprocessing & Pipeline
To ensure robust training and prevent data-loading bottlenecks:
- **Normalization:** Pixel values scaled by $1/255$ to the $[0, 1]$ range.
- **Randomization:** Images were shuffled randomly to eliminate biases.
- **Data Pipeline:** Utilized `tf.data.Dataset` with `.prefetch()` to optimize GPU utilization during training.

---

## 🧠 Model Architectures
Several architectures were implemented and compared:

1. **Baseline CNN:** Custom 3-layer Convolutional network (32, 64, 128 filters) with $3 \times 3$ kernels, MaxPooling, BatchNorm, Dropout (0.25), L2 Regularization ($1e-4$), and He Initialization.
2. **Tuned CNN:** Extensively tuned using 10 different hyperparameter combinations to find the optimal setup.
3. **Inception-Based Model:** Custom implementation featuring a Stem, Inception-A/C blocks, and Reduction-A block.
4. **ResNet:** Residual architecture utilizing skip connections to improve gradient flow.
5. **SENet:** Incorporation of Squeeze-and-Excitation blocks to adaptively recalibrate channel-wise feature responses.
6. **SqueezeNet:** A lightweight architecture designed to minimize parameters while maintaining accuracy.

---

## 📊 Results & Evaluation

Despite SqueezeNet having around 700,000 parameters, the **Inception-based architecture** proved to be the most efficient and accurate, achieving the best performance with only **~300,000 parameters**. This highlights that architectural design and feature extraction mechanisms are often more crucial than sheer parameter count.

### 🏆 Model Comparison

| Model | Parameters | Training Acc | Validation Acc | Test Acc / Notes |
|-------|------------|--------------|----------------|------------------|
| **Baseline CNN** | - | - | - | 94.02% |
| **Tuned CNN** | - | - | 89.75% | 94.30% |
| **ResNet** | - | 99.99% | 94.40% | - |
| **⭐ Inception-based**| **~300k** | **99.98%** | **94.54%** | **Best Performing Model** |

> *Note: SENet and SqueezeNet full metric tables are available in the project report.*

### 📈 Training Curves & Confusion Matrices
*(Replace the placeholder links below with your actual image paths once you push to GitHub)*

**Inception Model - Confusion Matrix:**
<p align="center">
  <img src="path_to_your_images/inception_confusion_matrix.png" alt="Inception Confusion Matrix" width="400"/>
</p>

**Inception Model - Accuracy & Loss:**
<p align="center">
  <img src="path_to_your_images/inception_loss_acc.png" alt="Inception Training Plots" width="600"/>
</p>

*The diagonal dominance in our confusion matrices indicates strong class separation with very minimal inter-class confusion.*

---

## 🚀 Model Deployment (ONNX)
To demonstrate production-readiness, the best model (**Inception-based**) was exported to the **ONNX** format.
- **Trained in:** TensorFlow / Keras
- **Exported to:** `.onnx`
- **Inference in:** PyTorch
- **Deployment Test Accuracy:** **94.05%**

This proves the model's portability across different Deep Learning frameworks without losing performance.

---

## 📁 Repository Structure
```text
📦 EuroSat-Classification
 ┣ 📂 data/               # Instructions/scripts for downloading the dataset
 ┣ 📂 models/             # Saved models (.h5) and ONNX files
 ┣ 📂 notebooks/          # Jupyter notebooks for training and EDA
 ┣ 📂 src/                # Python scripts (architectures, utils, train, eval)
 ┣ 📂 images/             # Plots, Confusion Matrices, and diagram images
 ┣ 📜 requirements.txt    # Dependencies
 ┗ 📜 README.md           # Project documentation

---

## 💻 How to Run

1. **Clone the repository:**
   
```bash
   git clone https://github.com/your-username/EuroSat-Classification.git
   cd EuroSat-Classification

