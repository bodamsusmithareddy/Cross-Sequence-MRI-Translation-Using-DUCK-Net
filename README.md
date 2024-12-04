# Cross-Sequence MRI Translation Using Optimized DUCK-Net for BraTS2020 dataset

## Project Overview
This project implements DUCK-Net, an optimized neural network architecture for cross-sequence MRI translation. The model is designed to synthesize missing MRI modalities, enhancing diagnostic capabilities in medical imaging.

## Features
- **Dynamic Tensor Alignment:** Ensures tensor dimensions match during concatenation.
- **Efficient Memory Usage with PyTorch:** Optimized for memory efficiency during training and inference.
- **Data Augmentation:** Incorporates flipping and rotations for improved generalization.
- **Dynamic Learning Rate Scheduler:** Adjusts learning rates based on validation loss trends.
- **Efficient Upsampling:** Utilizes transposed convolutions for smooth feature scaling.
- **Bias Field Correction:** Mitigates scanner-induced intensity inconsistencies.
- **Skull Stripping:** Focuses on relevant anatomical features by removing non-brain tissue.
- **Normalization:** Applies Z-score normalization for consistent intensity values.
- **Batch Normalization, ReLU Activation, and Max Pooling:** Ensures feature stability and reduces spatial dimensions.
- **Attention Fusion:** Enhances focus on relevant regions in the decoder stage.
- **Weight Initialization:** Employs Xavier initialization for better convergence.
- **Overfitting Prevention:** Includes dropout regularization.
- **Adam Optimizer with Weight Decay:** Ensures efficient convergence and reduces overfitting.
- **Loss Function (MSE):** Minimizes pixel-wise intensity differences effectively.

---

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Scikit-image
- Nibabel

### Hardware Requirements
- CPU (if GPU unavailable)
- Recommended: GPU with CUDA support for faster training.

---

## Dataset
- **BraTS 2020 Dataset**
  - Download from [Kaggle](https://www.kaggle.com/) or the official BraTS competition site.
  - Place the dataset in your system or environments local directory.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DUCK-Net
---
2. To install the libraries you can run the command:
   ```
   pip install -r requirements.txt
   ```
---
3. Set up project directories:
   ```
   mkdir -p /home/sbodam/NNProject/saved_models/BraTS2020
   mkdir -p /home/sbodam/NNProject/visualizations
   ```
---
## How to Run
### Training and Evaluation
1. Run the script:
    ```
    python ducknet_optimized.py
    ```
---
