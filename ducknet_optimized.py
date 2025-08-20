import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score
import SimpleITK as sitk
from nilearn.masking import compute_epi_mask

# Paths for data, model saving, and visualizations
data_dir = '/home/sbodam/NNProject/MICCAI_BraTS2020_TrainingData'
model_save_path = '/home/sbodam/NNProject/saved_models/BraTS2020'
visualization_save_path = '/home/sbodam/NNProject/visualizations'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(visualization_save_path, exist_ok=True)

# Device setup for training (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Function to normalize image intensity
def normalize_intensity(image):
    """
    Normalize the intensity values of an image to [0, 1].
    Args:
        image (ndarray): Input image array.
    Returns:
        ndarray: Normalized image.
    """
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(image)

# Function to resample an image to the target shape
def resample_image(image, target_shape=(240, 240)):
    """
    Resample an image to the specified shape using bilinear interpolation.
    Args:
        image (ndarray): Input image.
        target_shape (tuple): Target shape for the resampled image.
    Returns:
        ndarray: Resampled image.
    """
    factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
    return zoom(image, factors, order=1)

# Function to perform bias field correction
def bias_field_correction(image):
    """
    Correct intensity inhomogeneity in an image using N4ITK.
    Args:
        image (ndarray): Input image array.
    Returns:
        ndarray: Corrected image.
    """
    itk_image = sitk.GetImageFromArray(image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(itk_image)
    return sitk.GetArrayFromImage(corrected_image)

# Function for skull stripping
def skull_strip(image):
    """
    Remove non-brain tissues from an MRI image using a simple threshold.
    Args:
        image (ndarray): Input MRI image.
    Returns:
        ndarray: Skull-stripped image.
    """
    threshold = 0.1 * np.max(image)  # Adjust threshold for masking
    mask = (image > threshold).astype(np.float32)
    return image * mask

# Function to preprocess an image
def preprocess_image(image, target_shape=(240, 240)):
    """
    Apply preprocessing steps including skull stripping, bias field correction,
    normalization, and resampling.
    Args:
        image (ndarray): Input image array.
        target_shape (tuple): Target shape for resampling.
    Returns:
        ndarray: Preprocessed image.
    """
    stripped_image = skull_strip(image)
    corrected_image = bias_field_correction(stripped_image)
    normalized_image = normalize_intensity(corrected_image)
    resampled_image = resample_image(normalized_image, target_shape)
    return resampled_image

# Function to load and preprocess dataset
def load_data(data_dir, source_suffix, target_suffix, slice_idx=77, target_shape=(240, 240)):
    """
    Load, preprocess, and split data into training, validation, and test sets.
    Args:
        data_dir (str): Path to the dataset directory.
        source_suffix (str): Suffix for source images.
        target_suffix (str): Suffix for target images.
        slice_idx (int): Slice index to extract from 3D MRI volumes.
        target_shape (tuple): Target shape for preprocessing.
    Returns:
        tuple: Split datasets (training, validation, and test).
    """
    source_images, target_images = [], []
    for patient_folder in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        try:
            source_path = os.path.join(patient_path, f"{patient_folder}{source_suffix}")
            target_path = os.path.join(patient_path, f"{patient_folder}{target_suffix}")
            source_image = nib.load(source_path).get_fdata()[:, :, slice_idx]
            target_image = nib.load(target_path).get_fdata()[:, :, slice_idx]
            source_image = preprocess_image(source_image, target_shape)
            target_image = preprocess_image(target_image, target_shape)
            source_images.append(np.expand_dims(source_image, -1))
            target_images.append(np.expand_dims(target_image, -1))
        except FileNotFoundError as e:
            print(f"Missing file: {e}")
            continue
    source_images = np.array(source_images, dtype="float32")
    target_images = np.array(target_images, dtype="float32")
    x_train_val, x_test, y_train_val, y_test = train_test_split(source_images, target_images, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.20, random_state=42)
    print(f"Loaded {len(x_train)} training samples, {len(x_val)} validation samples, and {len(x_test)} test samples.")
    return x_train, x_val, x_test, y_train, y_val, y_test

# Custom Dataset class for DataLoader
class BraTSDataset(Dataset):
    """
    Custom Dataset for loading BraTS MRI data.
    """
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        source_image = self.source_data[idx]
        target_image = self.target_data[idx]
        return torch.tensor(source_image).permute(2, 0, 1), torch.tensor(target_image).permute(2, 0, 1)

# Define DUCK-Net model
class DUCKNet3D(nn.Module):
    """
    DUCK-Net model for MRI translation with 3D convolutions.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(DUCKNet3D, self).__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.dec3 = self._conv_block(256 + 128, 128)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.dec1 = nn.Conv3d(64, out_channels, kernel_size=1)
        self.att3 = nn.Conv3d(128, 128, kernel_size=1)
        self.att2 = nn.Conv3d(64, 64, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        """
        Define a convolutional block with BatchNorm and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of DUCK-Net.
        """
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        d3 = self.upsample3(e3)
        d3 = torch.cat([d3, self._align_dimensions(self.att3(e2), d3)], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upsample2(d3)
        d2 = torch.cat([d2, self._align_dimensions(self.att2(e1), d2)], dim=1)
        d2 = self.dec2(d2)
        d1 = self.dec1(d2)
        return d1

    def _align_dimensions(self, tensor_a, tensor_b):
        """
        Align dimensions of two tensors for concatenation.
        """
        _, _, h_a, w_a, d_a = tensor_a.size()
        _, _, h_b, w_b, d_b = tensor_b.size()
        if h_a != h_b or w_a != w_b or d_a != d_b:
            tensor_a = F.interpolate(tensor_a, size=(h_b, w_b, d_b), mode='trilinear', align_corners=False)
        return tensor_a

# Visualization function
def visualize_translation(source, output, target, save_path, title="Translation Results"):
    """
    Visualize the translation results and save the plots.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(source.squeeze(), cmap="gray")
    axes[0].set_title("Input MR")
    axes[1].imshow(output.squeeze(), cmap="gray")
    axes[1].set_title("Synthetic MR")
    axes[2].imshow(target.squeeze(), cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[3].imshow(np.abs(target.squeeze() - output.squeeze()), cmap="hot")
    axes[3].set_title("Difference Map")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization: {save_path}")

# Define modality pairs for training and evaluation
modality_pairs = {
    'T1 -> T2': ('_t1.nii', '_t2.nii'),
    'T2 -> FLAIR': ('_t2.nii', '_flair.nii'),
    'FLAIR -> T1': ('_flair.nii', '_t1.nii'),
}

# Training loop for each modality pair
for translation, (source_suffix, target_suffix) in modality_pairs.items():
    print(f"Training for {translation}")
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir, source_suffix, target_suffix)
    train_dataset = BraTSDataset(x_train, y_train)
    val_dataset = BraTSDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = DUCKNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(500):
        model.train()
        train_loss = 0
        for source_images, target_images in train_loader:
            source_images, target_images = source_images.to(device), target_images.to(device)
            optimizer.zero_grad()
            outputs = model(source_images.unsqueeze(1))
            outputs_resized = F.interpolate(outputs, size=target_images.shape[-3:], mode='trilinear', align_corners=False)
            loss = criterion(outputs_resized, target_images.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for source_images, target_images in val_loader:
                source_images, target_images = source_images.to(device), target_images.to(device)
                outputs = model(source_images.unsqueeze(1))
                outputs_resized = F.interpolate(outputs, size=target_images.shape[-3:], mode='trilinear', align_corners=False)
                loss = criterion(outputs_resized, target_images.unsqueeze(1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/50, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(visualization_save_path, f"{translation}_epoch_{epoch+1}.png")
            visualize_translation(
                x_val[0], outputs_resized[0].squeeze().detach().cpu().numpy(),
                y_val[0], save_path, title=f"{translation} Epoch {epoch+1}"
            )

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {translation}")
    plt.legend()
    loss_plot_path = os.path.join(visualization_save_path, f"{translation}_loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot: {loss_plot_path}")
