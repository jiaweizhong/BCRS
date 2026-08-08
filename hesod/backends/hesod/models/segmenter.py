"""HESOD selector heads: the baseline objectness Segmenter and its
semantic/spectral evidence variants (HESOD-Proposal.md section 5.3 and 7.3.1's
locked five-arm roster, plus the channel-pooled low-overhead pair from
section 7.9). Each class shares the same contract: given per-level features
`x` (a list, one tensor per input level -- one level for ESOD's single-stem
ObjSeeker slot, or many for a multi-level FPN head), return a list of the
same length with one `nc`-channel logit map per level.

Kept separate from models/yolo.py (which only wires these into the Model
forward/parse pipeline) the same way models/spectral.py splits out the
building blocks these classes compose.
"""

import torch
import torch.nn as nn

from models.spectral import ChannelPooledSpectralBranch, GatedEvidenceFusion, SpectralBranch


class Segmenter(nn.Module):
    """Baseline: pure semantic objectness head (E1.0 upstream baseline / E2.1
    semantic coverage-supervised -- the two arms share this architecture and
    differ only in training loss, not module structure)."""

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)  # output conv

    def forward(self, x):
        return [self.m[i](x[i]) for i in range(len(x))]


class SpectralOnlySegmenter(nn.Module):
    """E2.5: spectral-only priority (Evidence Ablation, HESOD-Proposal.md SS7.4).

    No semantic head contributes to the routing logit -- isolates whether
    spectral/local-saliency evidence alone can find real tiny targets that
    objectness misses (H1/H2).
    """

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.spectral_branches = nn.ModuleList(SpectralBranch(x, nc) for x in ch)

    def forward(self, x):
        return [self.spectral_branches[i](x[i])[0] for i in range(len(x))]


class DualEvidenceSegmenter(nn.Module):
    """E2.4: semantic + spectral gated fusion (Fusion Ablation, HESOD-Proposal.md SS5.3.C)."""

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(SpectralBranch(x, nc) for x in ch)
        self.gated_fusions = nn.ModuleList(GatedEvidenceFusion(x, x) for x in ch)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, f_spectral = self.spectral_branches[i](x[i])
            p_fused, _ = self.gated_fusions[i](p_semantic, p_spectral, x[i], f_spectral)
            res.append(p_fused)
        return res


class ConcatEvidenceSegmenter(nn.Module):
    """E2.3: semantic + spectral concatenation (Fusion Ablation, HESOD-Proposal.md SS7.4).

    Contrasted against DualEvidenceSegmenter to isolate the effect of the
    fusion mechanism itself (input-dependent gate vs. a plain learned 1x1
    combiner) holding the evidence branches fixed.
    """

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(SpectralBranch(x, nc) for x in ch)
        self.concat_convs = nn.ModuleList(nn.Conv2d(nc * 2, nc, 1) for _ in ch)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, _ = self.spectral_branches[i](x[i])
            res.append(self.concat_convs[i](torch.cat([p_semantic, p_spectral], dim=1)))
        return res


class ChannelPooledSpectralOnlySegmenter(nn.Module):
    """SpectralOnlySegmenter with the low-overhead channel-pooled filter (SS7.9)."""

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.spectral_branches = nn.ModuleList(ChannelPooledSpectralBranch(x, nc) for x in ch)

    def forward(self, x):
        return [self.spectral_branches[i](x[i])[0] for i in range(len(x))]


class ChannelPooledDualEvidenceSegmenter(nn.Module):
    """DualEvidenceSegmenter with the low-overhead channel-pooled spectral filter (SS7.9)."""

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(ChannelPooledSpectralBranch(x, nc) for x in ch)
        self.gated_fusions = nn.ModuleList(GatedEvidenceFusion(x, x) for x in ch)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, f_spectral = self.spectral_branches[i](x[i])
            p_fused, _ = self.gated_fusions[i](p_semantic, p_spectral, x[i], f_spectral)
            res.append(p_fused)
        return res


class ChannelPooledConcatEvidenceSegmenter(nn.Module):
    """E2.9: channel-pooled spectral + concat (locked five-arm roster, HESOD-Proposal.md SS7.3.1).

    ConcatEvidenceSegmenter with the low-overhead channel-pooled spectral
    filter (SS7.9) -- tests whether the lightweight spectral implementation
    keeps its complementary value once its own compute is reduced.
    """

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(ChannelPooledSpectralBranch(x, nc) for x in ch)
        self.concat_convs = nn.ModuleList(nn.Conv2d(nc * 2, nc, 1) for _ in ch)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, _ = self.spectral_branches[i](x[i])
            res.append(self.concat_convs[i](torch.cat([p_semantic, p_spectral], dim=1)))
        return res
