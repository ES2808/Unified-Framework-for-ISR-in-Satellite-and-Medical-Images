import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.unified_model import UnifiedSRModel
from src.utils.dataset import UnifiedDataset
from src.models.loss import VGGLoss, CharbonnierLoss

def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    model = UnifiedSRModel(upscale=opt.scale).to(device)
    
    # Loss and Optimizer
    criterion_sr = CharbonnierLoss().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_perceptual = VGGLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
    
    # Datasets
    print("Initializing Datasets...")
    # UnifiedDataset expects the folder containing 'hr' and 'lr' subfolders
    # Kaggle notebook structure: {medical_data}/train/hr, {medical_data}/train/lr
    
    medical_train_path = os.path.join(opt.medical_data, 'train')
    satellite_train_path = os.path.join(opt.satellite_data, 'train')
    
    medical_train = UnifiedDataset(medical_train_path, 'medical')
    satellite_train = UnifiedDataset(satellite_train_path, 'satellite')
    
    # Dataloaders
    medical_loader = DataLoader(medical_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    satellite_loader = DataLoader(satellite_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    
    print(f"Medical Train: {len(medical_train)} images")
    print(f"Satellite Train: {len(satellite_train)} images")
    
    print("Starting Training...")
    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total_batches = 0
        
        # Train on Medical Data (Class 0)
        for i, (lr, hr, domain) in enumerate(medical_loader):
            lr, hr = lr.to(device), hr.to(device)
            target_cls = torch.zeros(lr.size(0), dtype=torch.long).to(device) # 0 for medical
            
            optimizer.zero_grad()
            sr, domain_logits = model(lr, 'medical') 
            
            loss_sr = criterion_sr(sr, hr)
            loss_cls = criterion_cls(domain_logits, target_cls)
            loss_perc = criterion_perceptual(sr, hr)
            
            # Weighted Loss
            loss = loss_sr + 0.1 * loss_cls + 0.05 * loss_perc
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy
            pred_cls = torch.argmax(domain_logits, dim=1)
            epoch_acc += (pred_cls == target_cls).float().mean().item()
            total_batches += 1
            
        # Train on Satellite Data (Class 1)
        for i, (lr, hr, domain) in enumerate(satellite_loader):
            lr, hr = lr.to(device), hr.to(device)
            target_cls = torch.ones(lr.size(0), dtype=torch.long).to(device) # 1 for satellite
            
            optimizer.zero_grad()
            sr, domain_logits = model(lr, 'satellite')
            
            loss_sr = criterion_sr(sr, hr)
            loss_cls = criterion_cls(domain_logits, target_cls)
            loss_perc = criterion_perceptual(sr, hr)
            
            loss = loss_sr + 0.1 * loss_cls + 0.05 * loss_perc
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy
            pred_cls = torch.argmax(domain_logits, dim=1)
            epoch_acc += (pred_cls == target_cls).float().mean().item()
            total_batches += 1
            
        avg_loss = epoch_loss / total_batches
        avg_acc = epoch_acc / total_batches
        print(f"Epoch [{epoch+1}/{opt.epochs}] Loss: {avg_loss:.4f} | Domain Acc: {avg_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            if not os.path.exists(opt.save_dir):
                os.makedirs(opt.save_dir)
            torch.save(model.state_dict(), os.path.join(opt.save_dir, f"model_epoch_{epoch+1}.pth"))

    # Save Final Model
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    torch.save(model.state_dict(), os.path.join(opt.save_dir, "model_final.pth"))
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--medical_data', type=str, required=True, help='Path to medical data root')
    parser.add_argument('--satellite_data', type=str, required=True, help='Path to satellite data root')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    opt = parser.parse_args()
    
    train(opt)
