# Cross-Sequence MRI Translation Using Optimized DUCK-Net for BraTS2020 dataset

Authors: [Navya Kesani], [Susmitha Reddy Bodam], [Jahnavi Bellapukonda], [Jalli Sadharma Kireeti Dev Sai] & [Revanth Chowdary CH]


## Project Overview
This project implements DUCK-Net, a state-of-the-art neural network architecture optimized for cross-sequence MRI translation. DUCK-Net is designed to synthesize missing MRI modalities, enhancing diagnostic capabilities in medical imaging and improving patient outcomes.

---

## Features
- **Dynamic Tensor Alignment:** Ensures consistent tensor dimensions for seamless feature concatenation.
- **Efficient Memory Management:** Optimized for memory efficiency during training and inference with PyTorch.
- **Data Augmentation:** Incorporates random flipping and rotations for enhanced generalization.
- **Dynamic Learning Rate Scheduler:** Adjusts learning rates dynamically based on validation loss trends.
- **Efficient Upsampling:** Transposed convolutions for smooth and precise feature scaling.
- **Bias Field Correction:** Mitigates scanner-induced intensity inconsistencies using N4ITK correction.
- **Skull Stripping:** Removes non-brain tissue to focus on relevant anatomical features.
- **Normalization:** Applies Z-score normalization for consistent intensity values across images.
- **Batch Normalization and ReLU Activation:** Ensures stable feature extraction and reduces spatial dimensions.
- **Attention Mechanisms:** Enhances focus on relevant regions in the decoding stage for high-fidelity synthesis.
- **Weight Initialization:** Utilizes Xavier initialization for improved convergence.
- **Overfitting Prevention:** Implements dropout regularization during training.
- **Adam Optimizer with Weight Decay:** Ensures efficient convergence while reducing overfitting.
- **Loss Function (MSE):** Minimizes pixel-wise intensity differences for accurate image reconstruction.

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
- SimpleITK
- Nilearn

### Hardware Requirements
- CPU (if GPU unavailable)
- Recommended: GPU with CUDA support for faster training and inference.

---

## Dataset
- **BraTS 2020 Dataset**
  - Download from [Kaggle](https://www.kaggle.com/) or the official BraTS competition site.
  - Ensure the dataset structure is preserved during extraction.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DUCK-Net
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```


---

## How to Run

### Training and Evaluation
1. Run the main training script:
   ```bash
   python ducknet_optimized.py
   ```

2. The script will automatically train DUCK-Net for the following modality translations:
   - T1 → T2
   - T2 → FLAIR
   - FLAIR → T1

3. Outputs include:
   - Model Checkpoints: Saved in `/saved_models/BraTS2020`.
   - Visualizations: Saved in `/visualizations`.

---

### Visualizations
- **Loss Curves:** Training and validation loss trends.
- **MRI Outputs:** Input MR, Synthetic MR, Ground Truth, and Difference Maps for each modality translation.

---

## Configuration
You can modify the following parameters in `ducknet_optimized.py`:
- **Epochs:**  500.
- **Batch Size:**  16.
- **Learning Rate:**  1e-4.
- **Scheduler Patience:**  5 epochs.

---

## DUCK-Net Architecture
### Key Components
- **Encoder:** Extracts hierarchical spatial features through convolutional layers and max pooling.
- **Decoder:** Reconstructs the target modality using transposed convolutions and attention fusion.
- **Attention Mechanisms:** Focus on clinically relevant regions for high-fidelity reconstruction.

---

## Results

### Modality Translation Results for T1 → T2 and T2 → FLAIR

| Metric/Model      | Optimized DUCK-Net | DUCK-Net | U-Net | CycleGAN | Pix2Pix |
|--------------------|--------------------|----------|-------|----------|---------|
| **T1 → T2 Results**              |                |          |       |          |         |
| Dice Coefficient   | 0.95              | 0.92     | 0.89  | 0.87     | 0.88    |
| IoU                | 0.89              | 0.86     | 0.83  | 0.81     | 0.82    |
| SSIM               | 0.98              | 0.96     | 0.94  | 0.91     | 0.92    |
| PSNR (dB)          | 36.5              | 34.8     | 32.0  | 30.5     | 31.2    |
| MSE                | 0.0035            | 0.0041   | 0.0060| 0.0085   | 0.0076  |
| MAE                | 0.030             | 0.033    | 0.042 | 0.048    | 0.045   |
| Accuracy (%)       | 96.5              | 94.5     | 92.3  | 90.1     | 91.2    |
| **T2 → FLAIR Results**           |                |          |       |          |         |
| Dice Coefficient   | 0.94              | 0.91     | 0.87  | 0.85     | 0.86    |
| IoU                | 0.88              | 0.84     | 0.81  | 0.78     | 0.80    |
| SSIM               | 0.97              | 0.95     | 0.92  | 0.90     | 0.91    |
| PSNR (dB)          | 35.8              | 34.5     | 31.8  | 30.1     | 30.9    |
| MSE                | 0.0039            | 0.0044   | 0.0063| 0.0087   | 0.0078  |
| MAE                | 0.032             | 0.035    | 0.044 | 0.049    | 0.046   |
| Accuracy (%)       | 96.2              | 94.2     | 91.8  | 89.7     | 90.5    |

---

### Known Issues
- Ensure the dataset is correctly structured with valid paths.
- GPU support is strongly recommended for faster execution.

---

## Future Work
1. **Integration with Additional Datasets:** Training DUCK-Net with diverse datasets such as BraTS 2021 or ADNI.
2. **GAN-Based Architectures:** Adding GAN loss functions to enhance perceptual quality.
3. **Advanced Data Augmentation:** Exploring advanced augmentation techniques such as elastic transformations.
4. **Real-Time Deployment:** Developing lightweight versions for real-time clinical use on edge devices.

---
