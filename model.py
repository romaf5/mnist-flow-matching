"""
Flow Matching U-Net.

Lightweight U-Net that predicts the vector field u_t(x, t) for transporting
samples from N(0, I) to the data distribution. Supports class-conditional
generation via learned embeddings. Works with 28x28 (MNIST) and 32x32 (CIFAR-10).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning via adaptive scale+shift."""
    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class UNet(nn.Module):
    """
    U-Net for image generation.
    Supports 28x28 (MNIST/Fashion-MNIST) and 32x32 (CIFAR-10).
    Predicts the vector field u_t(x, t).
    """
    def __init__(self, in_channels=1, base_channels=64, time_dim=128, num_classes=10):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)  # +1 for unconditional

        c = base_channels
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, c, 3, padding=1)
        self.res1 = ResBlock(c, time_dim)
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)       # /2

        self.res2 = ResBlock(c, time_dim)
        self.down2 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)   # /4

        # Bottleneck
        self.res_mid1 = ResBlock(c * 2, time_dim)
        self.res_mid2 = ResBlock(c * 2, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.res3 = ResBlock(c * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.res4 = ResBlock(c * 2, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
            nn.Conv2d(c * 2, in_channels, 1),
        )

    def forward(self, x, t, class_label=None):
        t_emb = self.time_embed(t)
        if class_label is not None:
            t_emb = t_emb + self.class_embed(class_label)

        h1 = self.enc1(x)
        h1 = self.res1(h1, t_emb)

        h2 = self.down1(h1)
        h2 = self.res2(h2, t_emb)

        h3 = self.down2(h2)
        h3 = self.res_mid1(h3, t_emb)
        h3 = self.res_mid2(h3, t_emb)

        h = self.up2(h3)
        h = torch.cat([h, h2], dim=1)
        h = self.res3(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.res4(h, t_emb)

        return self.out(h)


class _ClassifierBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class Classifier(nn.Module):
    """ResNet-style classifier. Adapts to input channels and spatial size."""
    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            _ClassifierBlock(64, 64),
            _ClassifierBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            _ClassifierBlock(64, 128, stride=2),
            _ClassifierBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            _ClassifierBlock(128, 256, stride=2),
            _ClassifierBlock(256, 256),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)
