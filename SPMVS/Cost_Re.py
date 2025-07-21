import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TriDirectionalScanner ---
class TriDirectionalScanner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_x = nn.Linear(dim, dim // 3)
        self.proj_y = nn.Linear(dim, dim // 3)
        self.proj_z = nn.Linear(dim, dim // 3)
        self.weights_gen = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_seq_x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)  # [B*H*W, D, C]
        x_seq_y = x.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)  # [B*D*W, H, C]
        x_seq_z = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)  # [B*D*H, W, C]

        weights = self.weights_gen(x).view(B, 3, 1, 1, 1)  # shape: (B, 3, 1, 1, 1)
        w_x, w_y, w_z = weights[:, 0:1].reshape(-1), weights[:, 1:2].reshape(-1), weights[:, 2:3].reshape(-1)

        out_x = self.proj_x(x_seq_x).mean(1)
        out_y = self.proj_y(x_seq_y).mean(1)
        out_z = self.proj_z(x_seq_z).mean(1)

        # Padding to same size for addition
        min_len = min(out_x.size(0), out_y.size(0), out_z.size(0))
        out = (w_x[:min_len].unsqueeze(1) * out_x[:min_len] +
               w_y[:min_len].unsqueeze(1) * out_y[:min_len] +
               w_z[:min_len].unsqueeze(1) * out_z[:min_len])
        return out

# --- DeformablePositionEmbed ---
class DeformablePositionEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.offset_net = nn.Sequential(
            nn.Conv3d(dim, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim // 4, 3, 3, padding=1)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device
        offsets = self.offset_net(x).permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

        # Create base grid
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x_lin = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x_lin, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, D, H, W, 3)

        sampling_grid = grid + offsets  # (B, D, H, W, 3)
        sampled = F.grid_sample(x, sampling_grid, align_corners=False, mode='bilinear', padding_mode='border')
        return sampled

# --- Dummy Mamba (placeholder for mamba_ssm.Mamba) ---
class DummyMamba(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model)
        )

    def forward(self, x):
        return self.linear(x)

# --- MambaBottleneck ---
class MambaBottleneck(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scanner = TriDirectionalScanner(in_dim)
        self.mamba = DummyMamba(d_model=in_dim, d_state=64, d_conv=4, expand=2)
        self.deform_pe = DeformablePositionEmbed(in_dim)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim, in_dim // 4, 1),
            nn.GELU(),
            nn.Conv3d(in_dim // 4, in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_seq = self.scanner(x)  # [N, C]
        pos_emb = self.deform_pe(x).view(B * C * D * H * W // C, C)
        seq = x_seq + pos_emb
        global_feat = self.mamba(seq).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        att = self.channel_att(global_feat)
        return global_feat * att

# --- MambaCostRegNet ---
class MambaCostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.conv1 = nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1)
        self.conv3 = nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1)

        self.bottleneck = MambaBottleneck(base_channels * 4)

        self.conv7 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 3,
                                        stride=2, padding=1, output_padding=1)
        self.conv9 = nn.ConvTranspose3d(base_channels * 2, base_channels, 3,
                                        stride=2, padding=1, output_padding=1)
        self.prob = nn.Conv3d(base_channels, 1, 3, padding=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x_b = self.bottleneck(x4)

        x7 = self.conv7(x_b)
        x9 = self.conv9(x7)
        out = self.prob(x9)
        return out

# Run a test
model = MambaCostRegNet(in_channels=4, base_channels=8)
x = torch.randn(1, 4, 16, 32, 32)  # (B, C, D, H, W)
y = model(x)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
print("Output shape:", y.shape)
