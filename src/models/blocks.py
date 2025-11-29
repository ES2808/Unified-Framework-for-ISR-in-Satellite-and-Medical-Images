import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Residual Dense Block (RDB) ---------- #

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block as used in ESRGAN/RRDB.
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + 0.2 * x5

# ---------- RRDB Block (Backbone) ---------- #

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))

# ---------- Transformer-based Texture Attention Head (Medical) ---------- #

class SelfAttentionBlock(nn.Module):
    """
    Simple self-attention block for spatial attention.
    """
    def __init__(self, nf):
        super().__init__()
        self.query = nn.Conv2d(nf, nf // 8, 1)
        self.key = nn.Conv2d(nf, nf // 8, 1)
        self.value = nn.Conv2d(nf, nf, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        proj_key = self.key(x).view(B, -1, H * W)                       # B x C' x HW
        energy = torch.bmm(proj_query, proj_key)                        # B x HW x HW
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)                   # B x C x HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # B x C x HW
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# ---------- Multi-scale Residual CNN Head (Satellite) ---------- #

class MultiScaleResidualBlock(nn.Module):
    """
    Multi-scale residual block using different kernel sizes.
    """
    def __init__(self, nf):
        super().__init__()
        self.conv3x3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv5x5 = nn.Conv2d(nf, nf, 5, 1, 2)
        self.conv1x1 = nn.Conv2d(nf, nf, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.lrelu(self.conv3x3(x))
        out2 = self.lrelu(self.conv5x5(x))
        out3 = self.lrelu(self.conv1x1(x))
        out = out1 + out2 + out3
        return x + out
