import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.unified_model import UnifiedSRModel

def test_model_instantiation():
    print("Testing Model Instantiation...")
    model = UnifiedSRModel(upscale=4)
    print("Model created successfully.")
    
    # Dummy Input (B, C, H, W)
    lr_medical = torch.randn(1, 3, 32, 32)
    lr_satellite = torch.randn(1, 3, 32, 32)
    
    # Forward Medical
    print("Testing Medical Forward Pass...")
    sr_medical = model(lr_medical, 'medical')
    print(f"Medical Output Shape: {sr_medical.shape}")
    assert sr_medical.shape == (1, 3, 128, 128), f"Expected (1, 3, 128, 128), got {sr_medical.shape}"
    
    # Forward Satellite
    print("Testing Satellite Forward Pass...")
    sr_satellite = model(lr_satellite, 'satellite')
    print(f"Satellite Output Shape: {sr_satellite.shape}")
    assert sr_satellite.shape == (1, 3, 128, 128), f"Expected (1, 3, 128, 128), got {sr_satellite.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_model_instantiation()
