import os
import shutil
import random
import sys
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.preprocessing import preprocess_dataset

def update_medical_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    medical_train_hr = os.path.join(base_dir, 'data', 'medical', 'train', 'hr')
    medical_val_hr = os.path.join(base_dir, 'data', 'medical', 'val', 'hr')
    
    medical_train_lr = os.path.join(base_dir, 'data', 'medical', 'train', 'lr')
    medical_val_lr = os.path.join(base_dir, 'data', 'medical', 'val', 'lr')

    # 1. Identify New Files
    # New files are those that are NOT 'resized_' and don't have a corresponding LR yet (or we just process all non-resized)
    # Actually, since we modified preprocessing to skip 'resized_', we just need to look for non-resized files.
    
    all_files = [f for f in os.listdir(medical_train_hr) if not f.startswith('resized_') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
    
    print(f"Found {len(all_files)} raw images in train/hr.")
    
    if not all_files:
        print("No new raw images found.")
        return

    # 2. Split New Data (20% to Val)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.2)
    val_files = all_files[:split_idx]
    
    print(f"Moving {len(val_files)} new images to validation...")
    for f in val_files:
        shutil.move(os.path.join(medical_train_hr, f), os.path.join(medical_val_hr, f))
        
    # 3. Process Datasets
    print("Processing Medical Train...")
    preprocess_dataset(medical_train_hr, medical_train_lr, scale=4)
    
    print("Processing Medical Val...")
    preprocess_dataset(medical_val_hr, medical_val_lr, scale=4)
    
    print("Update Complete.")

if __name__ == "__main__":
    update_medical_data()
