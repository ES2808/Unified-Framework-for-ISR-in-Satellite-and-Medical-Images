import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class UnifiedDataset(Dataset):
    """
    Dataset class for Unified Super-Resolution.
    Returns (LR, HR, domain) tuples.
    """
    def __init__(self, root_dir, domain, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory (containing 'hr' and 'lr' subfolders).
            domain (str): 'medical' or 'satellite'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        
        # Lowercase hr/lr based on previous exploration
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lr_dir = os.path.join(root_dir, 'lr')
        
        # Support multiple extensions including .ppm and .tif
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm')
        
        # Only select files that start with 'resized_' which are the processed HR images
        self.image_files = [f for f in os.listdir(self.hr_dir) if f.lower().endswith(valid_extensions) and f.startswith('resized_')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_filename = self.image_files[idx]
        # Remove 'resized_' prefix to get the LR filename
        lr_filename = hr_filename.replace('resized_', '')
        
        hr_path = os.path.join(self.hr_dir, hr_filename)
        lr_path = os.path.join(self.lr_dir, lr_filename) 

        hr_image = cv2.imread(hr_path)
        lr_image = cv2.imread(lr_path)
        
        if hr_image is None:
            raise FileNotFoundError(f"HR image not found: {hr_path}")
        if lr_image is None:
             raise FileNotFoundError(f"LR image not found: {lr_path}")

        # Convert BGR to RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to Tensor
        hr_image = torch.from_numpy(hr_image.transpose((2, 0, 1))).float() / 255.0
        lr_image = torch.from_numpy(lr_image.transpose((2, 0, 1))).float() / 255.0

        # Data Augmentation (Random Flip/Rotate)
        if self.transform:
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                lr_image = torch.flip(lr_image, [2])
                hr_image = torch.flip(hr_image, [2])
            
            # Random Vertical Flip
            if np.random.random() > 0.5:
                lr_image = torch.flip(lr_image, [1])
                hr_image = torch.flip(hr_image, [1])
                
            # Random Rotation (90 degrees)
            if np.random.random() > 0.5:
                lr_image = torch.rot90(lr_image, 1, [1, 2])
                hr_image = torch.rot90(hr_image, 1, [1, 2])

        return lr_image, hr_image, self.domain
