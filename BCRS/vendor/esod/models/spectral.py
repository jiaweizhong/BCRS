"""BCRS Phase 2 Dual-Evidence Spectral Fusion Module

Provides lightweight Depthwise Multi-Kernel High-Pass Filters (Laplacian/Sobel)
and Gated Evidence Fusion for low-contrast, non-rigid tiny targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiKernelSpectralFilter(nn.Module):
    """Multi-kernel depthwise spectral filter utilizing Laplacian and Sobel kernels

    Extracts high-frequency spatial edge and texture saliency maps.
    """
    def __init__(self, in_channels, kernels=(3, 5)):
        super().__init__()
        self.in_channels = in_channels
        self.kernels = kernels

        # 3x3 Laplacian Kernel
        laplacian_3x3 = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]
        ], dtype=torch.float32)

        # 3x3 Sobel Horizontal & Vertical
        sobel_x_3x3 = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32)

        sobel_y_3x3 = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], dtype=torch.float32)

        # 5x5 Laplacian Kernel
        laplacian_5x5 = torch.tensor([
            [0., 0., -1., 0., 0.],
            [0., -1., -2., -1., 0.],
            [-1., -2., 16., -2., -1.],
            [0., -1., -2., -1., 0.],
            [0., 0., -1., 0., 0.]
        ], dtype=torch.float32)

        # Depthwise conv 3x3 (Laplacian + Sobel-X + Sobel-Y)
        weight_3x3 = torch.stack([laplacian_3x3, sobel_x_3x3, sobel_y_3x3], dim=0) # shape (3, 3, 3)
        weight_3x3 = weight_3x3.repeat(in_channels, 1, 1).unsqueeze(1) # shape (in_channels*3, 1, 3, 3)

        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 3,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.conv3x3.weight = nn.Parameter(weight_3x3, requires_grad=True)

        # Depthwise conv 5x5 (Laplacian 5x5)
        weight_5x5 = laplacian_5x5.unsqueeze(0).repeat(in_channels, 1, 1).unsqueeze(1) # shape (in_channels, 1, 5, 5)
        self.conv5x5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False
        )
        self.conv5x5.weight = nn.Parameter(weight_5x5, requires_grad=True)

        # Channel reduction & normalization
        out_channels_total = in_channels * 4
        self.norm = nn.BatchNorm2d(out_channels_total)
        self.act = nn.SiLU()

    def forward(self, x):
        feat_3x3 = self.conv3x3(x) # shape (B, C*3, H, W)
        feat_5x5 = self.conv5x5(x) # shape (B, C, H, W)
        concat_feat = torch.cat([feat_3x3, feat_5x5], dim=1) # shape (B, C*4, H, W)
        out = self.act(self.norm(concat_feat))
        return out


class SpectralBranch(nn.Module):
    """Spectral Evidence Branch for patch priority scoring

    Processes multi-kernel spectral features to produce spectral priority logits.
    """
    def __init__(self, in_channels, out_channels=1, kernels=(3, 5)):
        super().__init__()
        self.filter = MultiKernelSpectralFilter(in_channels, kernels=kernels)
        spectral_dim = in_channels * 4
        self.proj = nn.Sequential(
            nn.Conv2d(spectral_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        spec_feat = self.filter(x)
        logits = self.proj(spec_feat)
        return logits, spec_feat


class GatedEvidenceFusion(nn.Module):
    """Gated Evidence Fusion for Semantic and Spectral Evidence

    Computes dynamic element-wise gate g = Sigmoid(Conv([F_semantic, F_spectral]))
    Fused Logits = g * P_semantic + (1 - g) * P_spectral
    """
    def __init__(self, semantic_channels, spectral_channels):
        super().__init__()
        total_channels = semantic_channels + spectral_channels
        self.gate_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels // 2),
            nn.SiLU(),
            nn.Conv2d(total_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, p_semantic, p_spectral, f_semantic, f_spectral):
        concat_feat = torch.cat([f_semantic, f_spectral], dim=1)
        gate = self.gate_conv(concat_feat) # shape (B, 1, H, W)
        p_fused = gate * p_semantic + (1.0 - gate) * p_spectral
        return p_fused, gate
