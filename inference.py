import argparse
import os
import cv2
import torch
import numpy as np
from src.models.unified_model import UnifiedSRModel
from src.utils.metrics import calculate_psnr, calculate_ssim
import math

def forward_chop(model, x, domain, scale=4, shave=10, min_size=10000):
    """
    Process image in patches to avoid OOM.
    """
    n_GPUs = 1 # Force 1 for CPU/Single GPU
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    
    # If image is small enough, process directly
    if w * h <= min_size:
        return model(x, domain)
    
    # Split
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]
    ]
    
    sr_list = []
    logits_list = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        sr_batch, logits_batch = forward_chop(model, lr_batch, domain, scale, shave, min_size)
        sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        logits_list.extend(logits_batch.chunk(n_GPUs, dim=0))
        
    # Merge
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
    
    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
    # Average logits
    logits = torch.cat(logits_list, dim=0).mean(dim=0, keepdim=True)
    
    return output, logits

def forward_x8(model, x, domain):
    # Self-ensemble (x8)
    # 0: original
    # 1: flip_h
    # 2: flip_v
    # 3: transpose
    # 4: rotate 90
    # 5: rotate 180
    # 6: rotate 270
    # 7: rotate 90 + flip_h (etc.)
    # Simplified: 4 rotations * 2 flips
    
    sr_list = []
    for rotate in range(4):
        for flip in range(2):
            # Augment
            x_aug = x.clone()
            if rotate > 0:
                x_aug = torch.rot90(x_aug, k=rotate, dims=(2, 3))
            if flip == 1:
                x_aug = torch.flip(x_aug, dims=(3,))
            
            # Inference
            with torch.no_grad():
                out, _ = model(x_aug, domain)
            
            # De-augment
            if flip == 1:
                out = torch.flip(out, dims=(3,))
            if rotate > 0:
                out = torch.rot90(out, k=-rotate, dims=(2, 3))
            
            sr_list.append(out)
            
    return torch.stack(sr_list).mean(dim=0), None # Return mean and dummy logits


def inference(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = UnifiedSRModel(upscale=opt.scale).to(device)
    if os.path.exists(opt.model_path):
        model.load_state_dict(torch.load(opt.model_path, map_location=device))
        print(f"Model loaded from {opt.model_path}")
    else:
        print(f"Error: Model not found at {opt.model_path}")
        return

    model.eval()
    
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # Handle Input
    if os.path.isdir(opt.input_path):
        files = [os.path.join(opt.input_path, f) for f in os.listdir(opt.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.ppm'))]
    else:
        files = [opt.input_path]
        
    print(f"Processing {len(files)} images...")
    
    avg_psnr = 0
    avg_ssim = 0
    count = 0
    
    with torch.no_grad():
        for file_path in files:
            basename = os.path.basename(file_path)
            
            # Read LR Image
            lr_img = cv2.imread(file_path)
            if lr_img is None:
                print(f"Failed to read {file_path}")
                continue
                
            # Preprocess
            lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            lr_tensor = torch.from_numpy(lr_img_rgb.transpose((2, 0, 1))).float() / 255.0
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            
            # Inference (Pass domain=None to let model predict)
            # If user specified domain, we can still pass it to force it, or just let model decide.
            # Let's respect user input if given, otherwise None.
            
            if opt.self_ensemble:
                # For ensemble, we need a fixed domain. If not provided, predict once first.
                if opt.domain is None:
                     _, logits = model(lr_tensor, None)
                     pred_idx = torch.argmax(logits, dim=1).item()
                     ens_domain = 'medical' if pred_idx == 0 else 'satellite'
                else:
                     ens_domain = opt.domain
                
                sr_tensor, _ = forward_x8(model, lr_tensor, ens_domain)
                domain_logits = torch.tensor([[1.0, 0.0]]) if ens_domain == 'medical' else torch.tensor([[0.0, 1.0]]) # Dummy
            else:
                sr_tensor, domain_logits = model(lr_tensor, opt.domain)
            
            # Determine used domain for logging
            if opt.domain:
                used_domain = opt.domain
            else:
                pred_idx = torch.argmax(domain_logits, dim=1).item()
                used_domain = 'medical' if pred_idx == 0 else 'satellite'
            
            # Postprocess
            sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
            sr_img = np.clip(sr_img, 0, 1) * 255.0
            sr_img = sr_img.round().astype(np.uint8)
            sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            
            # Save
            save_path = os.path.join(opt.output_dir, f"sr_{basename}")
            cv2.imwrite(save_path, sr_img_bgr)
            
            # Calculate Metrics (if HR provided)
            if opt.hr_path:
                if os.path.isdir(opt.hr_path):
                    # Try to find corresponding HR image
                    # Assuming HR filename is 'resized_' + basename or just basename
                    hr_file_path = os.path.join(opt.hr_path, f"resized_{basename}")
                    if not os.path.exists(hr_file_path):
                         hr_file_path = os.path.join(opt.hr_path, basename)
                else:
                    hr_file_path = opt.hr_path
                    
                if os.path.exists(hr_file_path):
                    hr_img = cv2.imread(hr_file_path)
                    if hr_img is not None:
                        # Ensure dimensions match (crop if needed, though SR should match HR size if scale is correct)
                        h, w = hr_img.shape[:2]
                        sr_img_bgr = cv2.resize(sr_img_bgr, (w, h)) # Resize SR to match HR just in case of slight mismatch
                        
                        psnr = calculate_psnr(hr_img, sr_img_bgr)
                        ssim = calculate_ssim(hr_img, sr_img_bgr)
                        
                        avg_psnr += psnr
                        avg_ssim += ssim
                        count += 1
                        print(f"{basename} ({used_domain}): PSNR={psnr:.2f}, SSIM={ssim:.4f}")
                    else:
                        print(f"Warning: Could not read HR image {hr_file_path}")
                else:
                     print(f"Warning: HR image not found for {basename}")

    if count > 0:
        print(f"\nAverage PSNR: {avg_psnr / count:.2f}")
        print(f"Average SSIM: {avg_ssim / count:.4f}")
    
    print("Inference Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_path', type=str, required=True, help='Path to LR image or directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--domain', type=str, default=None, choices=['medical', 'satellite'], help='Force domain (optional, otherwise auto-detected by model)')
    parser.add_argument('--hr_path', type=str, help='Path to HR image or directory for metrics')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble (x8) for better quality')
    opt = parser.parse_args()
    
    inference(opt)
