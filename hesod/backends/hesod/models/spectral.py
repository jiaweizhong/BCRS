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
            in_channels,
            in_channels * 3,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.conv3x3.weight = nn.Parameter(weight_3x3, requires_grad=True)

        weight_5x5 = (
            laplacian_5x5.unsqueeze(0).repeat(in_channels, 1, 1).unsqueeze(1)
        )  # (C, 1, 5, 5)
        self.conv5x5 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.conv5x5.weight = nn.Parameter(weight_5x5, requires_grad=True)

        self.norm = nn.BatchNorm2d(in_channels * 4)
        self.act = nn.SiLU()

    def forward(self, x):
        feat_3x3 = self.conv3x3(x)  # (B, C*3, H, W)
        feat_5x5 = self.conv5x5(x)  # (B, C, H, W)
        return self.act(
            self.norm(torch.cat([feat_3x3, feat_5x5], dim=1))
        )  # (B, C*4, H, W)


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
        pooled = torch.cat(
            [x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1
        )
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


class ReliabilitySpectralBranch(nn.Module):
    """Channel-pooled spectral branch producing a signed logit AND a separate
    learned confidence score c_i^spec, for the reliability-aware residual gate
    (BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md SS5.3C, F5 in
    HESOD-Agri-Experiment-Plan.md SS6.2).

    Kept as its own class rather than adding a confidence output to
    ChannelPooledSpectralBranch: existing callers (ConcatEvidenceSegmenter,
    GatedEvidenceFusion arms) unpack a 2-tuple and would break on a 3-tuple.
    The logit head is unchanged from ChannelPooledSpectralBranch -- it is
    already an unconstrained Conv2d output, so it can already go negative
    (needed so the gate can suppress texture background, not just rescue
    low-objectness targets) without any change.
    """

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.filter = ChannelPooledSpectralFilter()
        self.stem = nn.Sequential(
            nn.Conv2d(6, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.confidence_head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        spec_feat = self.stem(self.filter(x))
        logit = self.head(spec_feat)
        confidence = torch.sigmoid(self.confidence_head(spec_feat))
        return logit, spec_feat, confidence


class TextureRiskHead(nn.Module):
    """Estimates per-location texture-contamination risk t_i^bg in [0,1] from
    the shared shallow feature map (HESOD-Agri-Proposal.md SS4.2) -- a
    lightweight 1x1-conv head, not a second backbone, so its cost must still
    be included in any reported selector overhead (SS4.2.2's requirement).
    """

    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 4)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class ReliabilityGateMLP(nn.Module):
    """Gate a_i = sigmoid(MLP([q_i, h_i, c_i^spec, t_i^bg])), implemented as
    1x1 convs since evidence is per-spatial-location, not global.

    Matches HESOD-Agri-Proposal.md SS4.2's gate formula except the budget
    embedding e_B is omitted: this project has no budget-conditioning
    mechanism yet (BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md
    SS5.5 is unimplemented), so the gate sees 4 inputs, not 5.
    """

    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, evidence):
        return self.mlp(evidence)


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
        gate = self.gate_conv(
            torch.cat([f_semantic, f_spectral], dim=1)
        )  # (B, 1, H, W)
        return gate * p_semantic + (1.0 - gate) * p_spectral, gate
