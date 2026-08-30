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

import math

import torch
import torch.nn as nn

from models.spectral import (
    ChannelPooledSpectralBranch,
    GatedEvidenceFusion,
    ReliabilityGateMLP,
    ReliabilitySpectralBranch,
    SpectralBranch,
    TextureRiskHead,
)


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
        self.spectral_branches = nn.ModuleList(
            ChannelPooledSpectralBranch(x, nc) for x in ch
        )

    def forward(self, x):
        return [self.spectral_branches[i](x[i])[0] for i in range(len(x))]


class ChannelPooledDualEvidenceSegmenter(nn.Module):
    """DualEvidenceSegmenter with the low-overhead channel-pooled spectral filter (SS7.9)."""

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(
            ChannelPooledSpectralBranch(x, nc) for x in ch
        )
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

    HESOD-Agri context (HESOD-Agri-Experiment-Plan.md SS6.2): this is F1, a
    required capacity-matched control, not the proposed HESOD-Agri method --
    see ReliabilityGateEvidenceSegmenter (F5) below for that.
    """

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(
            ChannelPooledSpectralBranch(x, nc) for x in ch
        )
        self.concat_convs = nn.ModuleList(nn.Conv2d(nc * 2, nc, 1) for _ in ch)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, _ = self.spectral_branches[i](x[i])
            res.append(self.concat_convs[i](torch.cat([p_semantic, p_spectral], dim=1)))
        return res


class ChannelPooledMaxEvidenceSegmenter(nn.Module):
    """Elementwise-max fusion of semantic + channel-pooled spectral evidence.

    Contrasted against ChannelPooledConcatEvidenceSegmenter (arm 5, the
    UAVDT roster's Concat-only) to isolate the fusion RULE itself. Concat's
    learned 1x1 combiner has no floor: it can, and on UAVDT apparently does,
    converge to a decision boundary worse than either evidence branch alone
    (HESOD-Experiment-Plan.md SS3.4 point 2 -- a --hm-threshold/--top-k
    inference-time sweep ruled out candidate volume as the cause, and a
    --pos-weight retrain is testing the coverage-loss asymmetry). This
    module has no learned combination step at all: `torch.max(p_semantic,
    p_spectral)` in logit space guarantees the fused score at every location
    is at least as high as whichever single branch is more confident there
    -- concat-only's central failure mode (a fused score *below* both
    inputs) is structurally impossible here. Downside: no way to learn a
    genuinely synergistic combination beyond "best of either branch" -- if
    concat's problem turns out to be under-training rather than an
    unbounded-worse-than-either-input pathology, this won't help and may
    cap out no better than spectral-only.
    """

    def __init__(self, nc=10, ch=()):
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(
            ChannelPooledSpectralBranch(x, nc) for x in ch
        )

    def forward(self, x):
        res = []
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, _ = self.spectral_branches[i](x[i])
            res.append(torch.max(p_semantic, p_spectral))
        return res


class ReliabilityGateEvidenceSegmenter(nn.Module):
    """F5: reliability-aware residual gate (BCRS-Budget-Constrained-Recall-Safe-
    Selector-Proposal.md SS5.3C; HESOD-Agri-Proposal.md SS4.2;
    HESOD-Agri-Experiment-Plan.md SS6.2/SS6.2.1 -- the proposed HESOD-Agri
    selector, R2/R3 once validated against the F0-F6 ladder).

    Differs from both existing fusion arms on purpose:
    - vs. ConcatEvidenceSegmenter (F1): fusion is an explicit residual gate,
      not a learned 1x1 combiner over concatenated logits -- at a_i=0 this
      degenerates EXACTLY to the semantic-only baseline, so spectral evidence
      can only ever perturb, never replace, the semantic prior.
    - vs. GatedEvidenceFusion (F2, used by *DualEvidenceSegmenter): the gate
      is conditioned on interpretable per-location scalars -- semantic
      uncertainty h_i, the spectral branch's own confidence c_i^spec, and a
      dedicated texture-risk t_i^bg -- not on the raw semantic/spectral
      feature maps. That is what makes this "reliability-aware" rather than
      just "another learned gate": F2 can learn to depend on spectral
      unconditionally, this cannot without also raising t_i^bg or c_i^spec
      in the wrong direction.

    u_i = logit(q_i) + alpha * a_i * z_i^spec, evaluated in logit space
    (HESOD-Agri-Proposal.md SS4.2). No budget embedding e_B -- see
    ReliabilityGateMLP's docstring. alpha is a fixed (not learned)
    constructor arg: HESOD-Agri-Experiment-Plan.md SS6.2.1 treats it as an
    unvalidated hyperparameter requiring a sweep, not something to fold into
    end-to-end optimization before that sweep happens.

    Only the F5 fusion mechanism is implemented here -- F6's rescue-ranking
    and conditional-gate-regularization losses (HESOD-Agri-Proposal.md
    SS4.2.2) are a separate, not-yet-implemented step in loss.py, deferred
    until F5 itself is validated against F1 (SS6.2's promotion gate).
    """

    def __init__(self, nc=1, ch=(), alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.spectral_branches = nn.ModuleList(
            ReliabilitySpectralBranch(x, nc) for x in ch
        )
        self.texture_risk_heads = nn.ModuleList(TextureRiskHead(x) for x in ch)
        self.gates = nn.ModuleList(ReliabilityGateMLP() for _ in ch)

    def forward(self, x, return_extras=False):
        """return_extras=False (default) is unchanged from the original
        contract: a list of fused priority maps, one per level. F6's
        rescue-ranking/conditional-gate-regularization losses (HESOD-Agri-
        Proposal.md SS4.2.2) need the intermediates that used to be computed
        and discarded here (q/h/a/z_spec/c_spectral/t_bg) -- return_extras=True
        additionally returns a per-level list of dicts holding them, without
        changing what's in `res` or how any existing caller unpacks it.
        """
        res = []
        extras = [] if return_extras else None
        for i in range(len(x)):
            p_semantic = self.m[i](x[i])
            p_spectral, _, c_spectral = self.spectral_branches[i](x[i])
            t_bg = self.texture_risk_heads[i](x[i])

            q = torch.sigmoid(p_semantic)
            h = -(
                q * torch.log(q + 1e-7) + (1 - q) * torch.log(1 - q + 1e-7)
            ) / math.log(2)
            a = self.gates[i](torch.cat([q, h, c_spectral, t_bg], dim=1))

            res.append(p_semantic + self.alpha * a * p_spectral)
            if return_extras:
                extras.append(
                    {
                        "q": q,
                        "h": h,
                        "a": a,
                        "z_spec": p_spectral,
                        "c_spectral": c_spectral,
                        "t_bg": t_bg,
                    }
                )
        return (res, extras) if return_extras else res
