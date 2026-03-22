"""
Flow Matching U-Net for MNIST.

Lightweight U-Net that predicts the vector field u_t(x, t) for transporting
samples from N(0, I) to the data distribution. Supports class-conditional
generation via learned embeddings.
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
    U-Net for 28x28 grayscale images.
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
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)       # 28->14

        self.res2 = ResBlock(c, time_dim)
        self.down2 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)   # 14->7

        # Bottleneck
        self.res_mid1 = ResBlock(c * 2, time_dim)
        self.res_mid2 = ResBlock(c * 2, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)  # 7->14
        self.res3 = ResBlock(c * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)  # 14->28
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


class MNISTClassifier(nn.Module):
    """Simple CNN classifier for MNIST."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28->14
        x = self.pool(F.relu(self.conv2(x)))   # 14->7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
