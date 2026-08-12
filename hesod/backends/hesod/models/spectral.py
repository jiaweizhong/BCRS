"""HESOD dual-evidence selector: spectral/saliency branch and evidence fusion.

Implements the frequency/local-saliency evidence branch and fusion heads
described in HESOD-Proposal.md section 5.3 (Spectral/Saliency Branch, Evidence
Fusion and Priority Head). The spectral branch does not run FFT per patch;
it uses fixed depthwise Laplacian/Sobel high-pass filters over the shared
selector feature map, matching section 5.8's "no second backbone, no
per-patch FFT" constraint.
"""

import torch
import torch.nn as nn


class MultiKernelSpectralFilter(nn.Module):
    """Trainable depthwise filters initialized as Laplacian/Sobel high-pass kernels.

    Extracts high-frequency edge/texture saliency from a shared feature map
    without a second backbone.  The high-pass kernels are initializations,
    not frozen constants: their parameters and the downstream 1x1 stem/head
    are optimized jointly.  A truly fixed-filter arm remains a separate
    ablation in HESOD-Proposal.md section 7.9.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        laplacian_3x3 = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        sobel_x_3x3 = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y_3x3 = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )
        laplacian_5x5 = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, -2.0, -1.0, 0.0],
                [-1.0, -2.0, 16.0, -2.0, -1.0],
                [0.0, -1.0, -2.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        weight_3x3 = torch.stack([laplacian_3x3, sobel_x_3x3, sobel_y_3x3], dim=0)
        weight_3x3 = weight_3x3.repeat(in_channels, 1, 1).unsqueeze(1)  # (C*3, 1, 3, 3)
        self.conv3x3 = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.conv3x3.weight = nn.Parameter(weight_3x3, requires_grad=True)

        weight_5x5 = laplacian_5x5.unsqueeze(0).repeat(in_channels, 1, 1).unsqueeze(1)  # (C, 1, 5, 5)
        self.conv5x5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False
        )
        self.conv5x5.weight = nn.Parameter(weight_5x5, requires_grad=True)

        self.norm = nn.BatchNorm2d(in_channels * 4)
        self.act = nn.SiLU()

    def forward(self, x):
        feat_3x3 = self.conv3x3(x)  # (B, C*3, H, W)
        feat_5x5 = self.conv5x5(x)  # (B, C, H, W)
        return self.act(self.norm(torch.cat([feat_3x3, feat_5x5], dim=1)))  # (B, C*4, H, W)


class SpectralBranch(nn.Module):
    """Full-channel spectral evidence branch: filter -> 1x1 stem -> 1x1 head."""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.filter = MultiKernelSpectralFilter(in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        spec_feat = self.stem(self.filter(x))
        return self.head(spec_feat), spec_feat


class ChannelPooledSpectralFilter(nn.Module):
    """Channel-pooled trainable spectral filter initialized from Sobel/Laplacian.

    Trades some spectral resolution for a large FLOPs reduction versus
    MultiKernelSpectralFilter (filters run on 2 pooled channels instead of
    `in_channels`), per HESOD-Proposal.md SS7.9's low-overhead implementation
    ablation.
    """

    def __init__(self):
        super().__init__()
        laplacian_3x3 = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        sobel_x_3x3 = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y_3x3 = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )
        weight_3x3 = torch.stack([laplacian_3x3, sobel_x_3x3, sobel_y_3x3], dim=0)
        weight_3x3 = weight_3x3.repeat(2, 1, 1).unsqueeze(1)  # (6, 1, 3, 3)

        self.conv = nn.Conv2d(2, 6, kernel_size=3, padding=1, groups=2, bias=False)
        self.conv.weight = nn.Parameter(weight_3x3, requires_grad=True)
        self.norm = nn.BatchNorm2d(6)
        self.act = nn.SiLU()

    def forward(self, x):
        pooled = torch.cat([x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        return self.act(self.norm(self.conv(pooled)))  # (B, 6, H, W)


class ChannelPooledSpectralBranch(nn.Module):
    """Channel-pooled spectral evidence branch, same interface as SpectralBranch."""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.filter = ChannelPooledSpectralFilter()
        self.stem = nn.Sequential(
            nn.Conv2d(6, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        spec_feat = self.stem(self.filter(x))
        return self.head(spec_feat), spec_feat


class GatedEvidenceFusion(nn.Module):
    """Input-dependent gate over semantic vs. spectral priority (HESOD-Proposal.md SS5.3.C).

    g = sigmoid(Conv([f_semantic, f_spectral])); fused = g * p_semantic + (1-g) * p_spectral.
    """

    def __init__(self, semantic_channels, spectral_channels):
        super().__init__()
        total = semantic_channels + spectral_channels
        self.gate_conv = nn.Sequential(
            nn.Conv2d(total, total // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(total // 2),
            nn.SiLU(),
            nn.Conv2d(total // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, p_semantic, p_spectral, f_semantic, f_spectral):
        gate = self.gate_conv(torch.cat([f_semantic, f_spectral], dim=1))  # (B, 1, H, W)
        return gate * p_semantic + (1.0 - gate) * p_spectral, gate
