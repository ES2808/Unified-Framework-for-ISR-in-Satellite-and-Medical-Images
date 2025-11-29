import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 17, 26, 35]):
        """
        Perceptual Loss using VGG19 features.
        layer_ids corresponds to relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        """
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg
        self.layer_ids = layer_ids
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for VGG
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x: Generated Image, y: Target Image
        # Assume input is [0, 1]
        
        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += nn.functional.l1_loss(x, y)
            if i >= max(self.layer_ids):
                break
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
