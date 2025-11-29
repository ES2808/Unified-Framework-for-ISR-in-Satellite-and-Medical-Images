import os
import cv2
import numpy as np
import rasterio

def preprocess_dataset(hr_folder, lr_folder, scale=4, method='bicubic', size=(128, 128)):
    """
    Downsamples and resizes HR images to create LR pairs for training.
    Args:
        hr_folder (str): Path to high-resolution images.
        lr_folder (str): Path to save low-resolution images.
        scale (int): Downscaling factor.
        method (str): Interpolation method ('bicubic' or 'lanczos').
        size (tuple): Size to resize HR images before downscaling.
    """
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    interp_map = {
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    for filename in os.listdir(hr_folder):
        if filename.startswith('resized_'): continue # Skip already processed files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm')):
            img = cv2.imread(os.path.join(hr_folder, filename))
            if img is None:
                continue  # Skip unreadable files

            # Resize HR to fixed size
            hr_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            h, w = hr_img.shape[:2]
            lr_img = cv2.resize(hr_img, (w // scale, h // scale), interpolation=interp_map[method])

            # Save processed images (avoid overwriting original HR images)
            cv2.imwrite(os.path.join(hr_folder, f"resized_{filename}"), hr_img)
            cv2.imwrite(os.path.join(lr_folder, filename), lr_img)

def preprocess_stare_dataset(hr_folder, lr_folder, scale=4, method='bicubic', size=(128, 128)):
    """Preprocess STARE dataset for SR training."""
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)
    interp_map = {'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
    for filename in os.listdir(hr_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(hr_folder, filename))
            hr_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            h, w = hr_img.shape[:2]
            lr_img = cv2.resize(hr_img, (w // scale, h // scale), interpolation=interp_map[method])
            cv2.imwrite(os.path.join(hr_folder, filename), hr_img)
            cv2.imwrite(os.path.join(lr_folder, filename), lr_img)

def preprocess_spacenet2_dataset(input_folder, hr_output, lr_output, scale=4, patch_size=(128, 128)):
    """Preprocess SpaceNet-2 multispectral TIFFs for SR training."""
    os.makedirs(hr_output, exist_ok=True)
    os.makedirs(lr_output, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if fname.endswith('.tif') and 'MUL' in fname:
                path = os.path.join(root, fname)
                with rasterio.open(path) as src:
                    img = src.read([1, 2, 3])  # RGB from multispectral
                    img = np.transpose(img, (1, 2, 0))
                    img = cv2.normalize(img.astype('float32'), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hr_img = cv2.resize(img, patch_size, interpolation=cv2.INTER_AREA)
                    lr_img = cv2.resize(hr_img, (patch_size[0] // scale, patch_size[1] // scale), interpolation=cv2.INTER_CUBIC)
                    base_name = os.path.splitext(fname)[0]
                    cv2.imwrite(os.path.join(hr_output, f"{base_name}_HR.png"), hr_img)
                    cv2.imwrite(os.path.join(lr_output, f"{base_name}_LR.png"), lr_img)
