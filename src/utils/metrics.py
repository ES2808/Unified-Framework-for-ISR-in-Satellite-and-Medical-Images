import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img1 (ndarray): Image 1 (0-255).
        img2 (ndarray): Image 2 (0-255).
    Returns:
        float: PSNR value.
    """
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index).
    Args:
        img1 (ndarray): Image 1 (0-255).
        img2 (ndarray): Image 2 (0-255).
    Returns:
        float: SSIM value.
    """
    # Convert to Y channel (luminance) for SSIM calculation as per standard practice in SR
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        
    return ssim(img1, img2, data_range=255)
