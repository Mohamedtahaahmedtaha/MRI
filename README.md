# NeuroSynth AI: Brain Tumor Detection & MRI Synthesis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

**NeuroSynth AI** is a state-of-the-art deep learning framework designed to assist radiologists by automating the detection of brain tumors and synthesizing contrast-enhanced MRI scans (T1c) from standard T1 scans, reducing the need for potentially harmful Gadolinium contrast agents.

---

## Key Features

* **Multi-Modal Analysis:** Processes 4 MRI modalities (T1, T2, FLAIR, T1c) for robust diagnosis.
* **Smart Triage Pipeline:** A cost-effective 2-stage workflow:
    1.  **Detection:** A CNN classifier filters scans. If "Healthy", the process ends.
    2.  **Synthesis:** If "Tumor" is detected, a GAN model generates the missing T1c modality.
* **Generative Adversarial Network (GAN):** Implements a **Pix2Pix** architecture with U-Net generator and PatchGAN discriminator.
* **Production Ready:** Deployed using **FastAPI** for real-time inference.

---

## System Architecture

The pipeline consists of two core modules:

### 1. The Classifier (ResNet-based)
* **Input:** 4-Channel MRI Tensor.
* **Function:** Binary Classification (Tumor / No Tumor).
* **Goal:** Optimize resource usage by running the heavy GAN model only on positive cases.

### 2. The Generator (Conditional GAN)
* **Architecture:** Pix2Pix (U-Net with Skip Connections).
* **Loss Function:** Adversarial Loss (BCE) + L1 Pixel-wise Loss.
* **Objective:** Learns the mapping $G: T1 \rightarrow T1c$ to visualize tumor boundaries without contrast injection.

---

## Results & Visualization

### Medical Image Synthesis (T1 to T1c)
*Comparison between Original T1, Generated T1c, and Ground Truth T1c.*


### Performance Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **SSIM** | **0.89** | Structural Similarity Index (Higher is better) |
| **PSNR** | **28.4 dB** | Peak Signal-to-Noise Ratio |
| **Detection F1** | **94%** | Tumor Classification Accuracy |

---

## Tech Stack

* **Core:** Python, PyTorch, NumPy.
* **Computer Vision:** PIL, Scikit-Image.
* **Deployment:** FastAPI, Uvicorn.
* **Infrastructure:** Support for CUDA (NVIDIA GPUs).

---

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohamedTahaAhmedTaha/MRI.git
   cd MRI
