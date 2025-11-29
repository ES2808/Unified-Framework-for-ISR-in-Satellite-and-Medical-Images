import torch
import torch.nn as nn
from .backbone import SharedRRDBBackbone
from .heads import MedicalTransformerHead, SatelliteCNNHead

class UnifiedSRModel(nn.Module):
    """
    Unified Super-Resolution Model with shared backbone and domain-specific heads.
    Args:
        in_nc (int): Number of input channels.
        nf (int): Number of feature channels.
        nb (int): Number of RRDB blocks in the backbone.
        upscale (int): Upscaling factor.
        out_nc (int): Number of output channels.
    """
    def __init__(self, in_nc=3, nf=64, nb=5, upscale=4, out_nc=3):
        super().__init__()
        self.backbone = SharedRRDBBackbone(in_nc, nf, nb)
        self.medical_head = MedicalTransformerHead(nf, out_nc, upscale)
        self.satellite_head = SatelliteCNNHead(nf, out_nc, upscale)
        
        # Domain Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf, 2) # 2 classes: 0=medical, 1=satellite
        )

    def forward(self, x, domain=None):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor.
            domain (str, optional): 'medical' or 'satellite'. If None, uses classifier prediction.
        Returns:
            Tensor: Super-resolved output.
            Tensor: Domain logits (for training).
        """
        features = self.backbone(x)
        
        # Predict Domain
        domain_logits = self.classifier(features)
        
        if domain is None:
            # Inference Mode: Use predicted domain
            pred_idx = torch.argmax(domain_logits, dim=1).item()
            domain = 'medical' if pred_idx == 0 else 'satellite'
        
        # Route to correct head
        if domain == 'medical':
            out = self.medical_head(features)
        elif domain == 'satellite':
            out = self.satellite_head(features)
        else:
            raise ValueError("Domain must be 'medical' or 'satellite'")
            
        return out, domain_logits
