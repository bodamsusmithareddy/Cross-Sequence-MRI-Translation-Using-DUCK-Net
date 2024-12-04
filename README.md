# Cross-Sequence MRI Translation Using Optimized DUCK-Net for BraTS2020 dataset

Authors: [Navya Kesani](mailto:navyakesani2@gmail.com?subject=[GitHub]DUCK-Net), [Susmitha Reddy Bodam](mailto:susmitha.reddy64@gmail.com?subject=[GitHub]DUCK-Net), [Jahnavi Bellapukonda](mailto:jahnavibellapukonda60@gmail.com?subject=[GitHub]DUCK-Net), [Jalli Sadharma Kireeti Dev Sai](mailto:jahnavibellapukonda60@gmail.com?subject=[GitHub]DUCK-Net) & [Revanth Chowdary CH](mailto:jahnavibellapukonda60@gmail.com?subject=[GitHub]DUCK-Net)


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

2. The script will train DUCK-Net for the following modality translations:
   - T1 -> T2
   - T2 -> FLAIR
   - FLAIR -> T1

4. Outputs:
   - Model Checkpoints: Saved in a sub directory under project folder.
   - Visualizations: Saved in sub directory under project folder.

### Visualizations
- Training vs Validation Loss Plots
- Input MR, Synthetic MR, Ground Truth, and Difference Maps
---
### Configuration
You can modify the following parameters in the script:
- Epochs: Default: 50
- Batch Size: Default: 16
- Learning Rate: Default: 1e-4
- Scheduler Patience: Default: 5
---
## Key Components
### DUCK-Net Architecture
- Encoder: Extracts hierarchical features through convolutional layers.
- Decoder: Reconstructs the target modality with attention fusion.
- Attention Mechanisms: Focuses on the most relevant features.
### Optimization Techniques
- Dynamic Tensor Alignment
- Efficient Upsampling
- Dropout Regularization
- Learning Rate Scheduler
---
## Results
Evaluation metrics include:

### T1 --> T2 Modality Translation Results
Metric/Model | Optimized DUCK-Net | DUCK-Net | U-Net | CycleGAN | Pix2Pix
--- | --- | --- | --- |--- |---
Dice Coefficient | 0.95 | 0.92 | 0.89 | 0.87 | 0.8
IoU | 0.89 | 0.86 | 0.83 | 0.81 | 0.82
SSIM | 0.98 | 0.96 | 0.94 | 0.91 | 0.92
PSNR (dB) | 36.5 | 34.8 | 32.0 | 30.5 | 31.2
MSE | 0.0035 | 0.0041 | 0.0060 | 0.0085 | 0.0076
MAE | 0.030 | 0.033 | 0.042 | 0.048 | 0.045
Accuracy (%) | 96.5 | 94.5 | 92.3 | 90.1 | 91.2

### T2 --> FLAIR Modality Translation Results
Metric/Model | Optimized DUCK-Net | DUCK-Net | U-Net | CycleGAN | Pix2Pix
--- | --- | --- | --- |--- |---
Dice Coefficient | 0.94 | 0.91 | 0.87 | 0.85 | 0.86
IoU | 0.88 | 0.84 | 0.81 | 0.78 | 0.80
SSIM | 0.97 | 0.95 | 0.92 | 0.90 | 0.91
PSNR (dB) | 35.8 | 34.5 | 31.8 | 30.1 | 30.9
MSE | 0.0039 | 0.0044 | 0.0063 | 0.0087 | 0.0078
MAE | 0.032 | 0.035 | 0.044 | 0.049 | 0.046
Accuracy (%) | 96.2 | 94.2 | 91.8 | 89.7 | 90.5

---
## Known Issues
- Ensure the dataset is correctly structured and accessible.
- GPU support is recommended for faster execution.
---
## Future Work
- Integration with other MRI datasets.
- Implementation of GAN-based loss functions.
- Advanced data augmentation techniques.
---
### BibTeX

```
@article{article,
author = {Dumitru, Razvan-Gabriel and Peteleaza, Darius},
year = {2023},
month = {06},
pages = {},
title = {Using DUCK-Net for polyp image segmentation},
volume = {13},
journal = {Scientific Reports},
doi = {10.1038/s41598-023-36940-5}
}
```
