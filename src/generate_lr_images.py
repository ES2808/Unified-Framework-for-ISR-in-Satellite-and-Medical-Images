import os
import shutil
import random
from utils.preprocessing import preprocess_dataset

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'satelitte')
    
    train_hr = os.path.join(data_dir, 'train', 'hr')
    train_lr = os.path.join(data_dir, 'train', 'lr')
    
    val_hr = os.path.join(data_dir, 'val', 'hr')
    val_lr = os.path.join(data_dir, 'val', 'lr')
    
    # Process Training Data
    print(f"Processing Training Data...")
    if os.path.exists(train_hr):
        preprocess_dataset(train_hr, train_lr, scale=4, method='bicubic', size=(128, 128))
    else:
        print(f"Warning: Training HR directory not found at {train_hr}")

    # Process Validation Data
    print(f"Processing Validation Data...")
    if os.path.exists(val_hr):
        preprocess_dataset(val_hr, val_lr, scale=4, method='bicubic', size=(128, 128))
    else:
        print(f"Warning: Validation HR directory not found at {val_hr}")

    # Process Medical Data
    print(f"Processing Medical Data...")
    medical_data_dir = os.path.join(base_dir, 'data', 'medical')
    medical_train_hr = os.path.join(medical_data_dir, 'train', 'hr')
    medical_train_lr = os.path.join(medical_data_dir, 'train', 'lr')
    medical_val_hr = os.path.join(medical_data_dir, 'val', 'hr')
    medical_val_lr = os.path.join(medical_data_dir, 'val', 'lr')

    # Create directories if they don't exist
    os.makedirs(medical_val_hr, exist_ok=True)

    # Check if validation set is empty and split if necessary
    if os.path.exists(medical_train_hr):
        val_files = os.listdir(medical_val_hr)
        if not val_files:
            print("Splitting medical data into Train/Val (80/20)...")
            all_files = [f for f in os.listdir(medical_train_hr) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm'))]
            random.shuffle(all_files)
            split_idx = int(len(all_files) * 0.2)
            val_files_to_move = all_files[:split_idx]
            
            for f in val_files_to_move:
                shutil.move(os.path.join(medical_train_hr, f), os.path.join(medical_val_hr, f))
            print(f"Moved {len(val_files_to_move)} files to validation set.")
        
        print("Processing Medical Train...")
        preprocess_dataset(medical_train_hr, medical_train_lr, scale=4, method='bicubic', size=(128, 128))
        
        print("Processing Medical Val...")
        preprocess_dataset(medical_val_hr, medical_val_lr, scale=4, method='bicubic', size=(128, 128))

    else:
        print(f"Warning: Medical HR directory not found at {medical_train_hr}")


    print("Done.")

if __name__ == "__main__":
    main()
