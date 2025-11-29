import torch
import torch.nn as nn
from .blocks import RRDB

class SharedRRDBBackbone(nn.Module):
    """
    Shared feature extraction backbone using RRDB blocks.
    """
    def __init__(self, in_nc=3, nf=64, nb=5):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.rrdb_trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        return fea + trunk
