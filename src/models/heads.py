import torch
import torch.nn as nn
from .blocks import SelfAttentionBlock, MultiScaleResidualBlock

class MedicalTransformerHead(nn.Module):
    """
    Upsampling head for medical images using deeper self-attention blocks.
    """
    def __init__(self, nf=64, out_nc=3, upscale=4, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttentionBlock(nf) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * upscale ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(nf, out_nc, 3, 1, 1)
        )

    def forward(self, x):
        res = x
        for block in self.blocks:
            x = block(x)
        x = self.conv_after_body(x)
        x = x + res # Residual connection
        return self.upsample(x)

class SatelliteCNNHead(nn.Module):
    """
    Upsampling head for satellite images using multi-scale residual block.
    """
    def __init__(self, nf=64, out_nc=3, upscale=4):
        super().__init__()
        self.resblock = MultiScaleResidualBlock(nf)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * upscale ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(nf, out_nc, 3, 1, 1)
        )

    def forward(self, x):
        x = self.resblock(x)
        return self.upsample(x)
