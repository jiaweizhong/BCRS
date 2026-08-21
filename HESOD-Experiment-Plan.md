# HESOD Experiment Plan

**Canonical status: 2026-08-12.** This file is the execution contract, not an
investigation log. Superseded configs, failed-run details, and patch history
belong in Git history or `ESOD-Baseline-Patches.md`.

## 1. Fixed protocols

| Dataset | Data | Classes | Model | Hyp | Input | Evaluation split |
|---|---|---:|---|---|---:|---|
| VisDrone | `/root/autodl-tmp/VisDrone_v2.yaml` | 10 | `visdrone_yolov5m.yaml` | `hyp.visdrone.yaml` | 1536 | val: 548 images, 38,759 GT |
| TinyPerson | protocol-specific audited YAML | 1 | `tinyperson_yolov5m.yaml` | `hyp.tinyperson.yaml` | 2048 | official test: 786 images, no dense images |
| UAVDT | `/root/autodl-tmp/UAVDT_v3.yaml` | 3 | `uavdt_yolov5m.yaml` | `hyp.uavdt.yaml` | 1280 | test: car/truck/bus |

Common training protocol: YOLOv5m, 50 epochs, SGD, global batch 8, cosine
scheduling, and weight decay 0.0005. TinyPerson's paper-text profile uses
`lr0=0.01`, `pixl=0.4`, and focal:dice 20:1. Alternate TinyPerson
hyperparameter profiles are no longer retained. No alternate UAVDT class
protocol is valid.

Metric contract:

- `mAP@.5 = mean(ap[:, 0])`; `mAP@.5:.95 = mean(ap over IoU 0.50:0.95)`.
  The evaluator mapping is correct; there is no AP/AP50 swap.
- VisDrone and UAVDT use native COCO-style AP. TinyPerson headline metrics use
  the official APt50 (area 1-400 px²) and APs50 (area 400-1024 px²) evaluator.
- `audit_buckets.py` is class-aware, confidence-ordered, one-to-one recall at
  one confidence/IoU setting. It is final detector recall, neither AP nor
  selector BPR.
- Paper BPRbox is `intersection(GT, any patch) / area(GT) > 0.5` (strict `>`,
  not IoU). Paper BPRctr is the fraction of GT-center cells in the `M >= 0.5`,
  3x3-local-maxima collection. `dump_selected_patches.py` records both the
  exact routing configuration and those maxima; `audit_selector_coverage.py`
  rejects incomplete/mismatched image sets before reporting either metric.
- Patch-center coverage is a separate routing diagnostic and must never be
  reported as BPRctr. The released `--hm-metric` path is a tolerant pyramid
  proxy; use the artifact audit for paper-defined BPRctr.
- Exact K is a HESOD/BCRS-only construct (`HESOD-Proposal.md` SS5.6 Action
  Space A: fixed-cost patch top-k, chosen specifically because ESOD's own
  fixed threshold gives an input-dependent, unpredictable patch count and
  BCRS wants a guaranteed compute budget). Confirmed absent from the paper's
  own released code: `grep -n "top.k\|top_k" hesod/backends/esod/test.py
  hesod/backends/esod/models/yolo.py` returns zero hits, and
  `hesod/backends/esod/test.py --sparse-head` calls
  `model.model[-1].set_sparse()` standalone, with no companion top-K flag.
  Efficiency/method comparisons BETWEEN HESOD ARMS use exact K + SparseHead +
  measured end-to-end latency/GFLOPs, so every arm gets the same fixed patch
  budget. For accuracy AND efficiency reproduction AGAINST THE PAPER'S OWN
  reported numbers (AP/AP50 and GFLOPs/FPS alike), fixed-threshold routing is
  the paper-matching choice -- `hesod/backends/esod/models/yolo.py`'s
  `get_indices(..., thresh=0.3)` applies the same hardcoded second threshold
  inside each fixed-threshold-selected patch with no override, exactly
  matching upstream. HESOD adds a `sparse_all_selected` flag (absent from
  `hesod/backends/esod/`) specifically
  to stop that same 0.3 threshold from re-filtering positions inside an
  exact-K-selected patch, which would otherwise silently break exact K's own
  "select K patches, evaluate each fully" semantics; `test.py` enables it only
  when `--top-k` and `--sparse-head` are passed together. So `--sparse-head`
  alone (no `--top-k`) is the paper-comparable SparseHead test; `--sparse-head
  --top-k N` measures HESOD's own corrected exact-K mode, a different setup.
  See the Routing contract below for the full Algorithm 1/2 description.
- Box-regression comparisons additionally report AP75, APs, and size-bin
  detector recall at IoU 0.50 and 0.75. An AP50-only gain is insufficient.

The paper text and public code define distinct label controls. These masks are
offline pseudo-label preprocessing (`masks/<split>/<stem>.npy`), not changes to
the YOLO bbox labels. TinyPerson experiments use the paper-text protocol:
focal:dice 20:1 (`--selector-loss paper`) plus the literal RGB-SAM Eq. 4 mask
(`--tinyperson-mask-mode paper-hybrid`), with no extra union term. Gaussian is
available only as an explicit preprocessing diagnostic, not as an experiment
configuration.

Every prepared TinyPerson dataset carries `tinyperson_protocol.json`, including
annotation hashes and mask mode. Hybrid preprocessing fails if SAM is missing;
it never silently falls back to Gaussian.

Routing contract:

- ObjSeeker predicts the objectness mask. AdaSlicer performs slicing. Released
  inference uses the GPU-parallel, suboptimal Algorithm 2; Algorithm 1 is the
  iterative greedy reference and is not the default inference path.
- The paper's `0.5` is a fixed foreground/local-maxima cutoff, not a learned or
  initial value. VisDrone/TinyPerson configs use 0.5; released UAVDT uses 0.3.
- Paper lowercase `k` means a `k x k` candidate grid (`k=8` gives at most 64
  coarse cells). HESOD `TOP_K` means how many of those candidates are retained.
  Smaller `TOP_K` reduces downstream neck/head work roughly linearly, but the
  dense stem and ObjSeeker cost remain, so end-to-end latency is not `1/K`.
- Current HESOD Top-K ranks the 8x8 coarse cells by maximum response and emits
  the selected fixed cells directly; it does not run Algorithm 2's offset
  adjustment/overlap removal. This is an explicit method delta, not a baseline
  reproduction claim. A budgeted-Algorithm-2 variant is a separate ablation.

## 2. Accepted evidence

| Dataset / setting | Paper | Audited | Residual |
|---|---:|---:|---:|
| VisDrone Gaussian AP / AP50 | 35.7 / 59.5 | 34.9 / 58.6 | -0.8 / -0.9 pp |
| VisDrone released SAM hybrid AP / AP50 | 36.0 / 59.7 | 34.6 / 58.4 | -1.4 / -1.3 pp |
| TinyPerson APt50 / APs50 | 61.3 / 74.4 | pending audited rerun | -- |
| UAVDT nc=3 AP / AP50 (frozen-tree `run_baseline.sh`) | 22.5 / 40.7 | 20.1 / 37.0 | -2.4 / -3.7 pp |
| UAVDT nc=3 AP / AP50 (`UAVDT_fresh`, active tree, 2026-08-20/21) | 22.5 / 40.7 | 17.1 / 34.4 | -5.4 / -6.3 pp |

Interpretation is intentionally narrow:

- VisDrone's former large gap came mainly from the wrong conversion; the
  audited released-code reproduction is now within 1.0 pp.
- TinyPerson's historical 55.26/71.23 result is not accepted as a current
  reproduction: its retained training log identifies ratio 16 while the
  current source is ratio 8, and no matching official-evaluation log remains.
- UAVDT nc=3 is paper-comparable; the released nc=1 collapse is another task.
- UAVDT's fresh reconversion (`UAVDT_fresh`, re-extracted from the raw
  `M_attr.zip`/`UAV-benchmark-M.zip`/`UAV-benchmark-MOTD_v1.0.zip` this
  session) widened the residual to the paper rather than confirming the
  frozen-tree number (-5.4/-6.3pp vs. -2.4/-3.7pp). Ruled out as the cause:
  the `image_id` disambiguation bug found and fixed in `test.py` (UAVDT
  reuses frame numbers like `img000001.jpg` across every video, colliding
  under plain `path.stem`) -- that only affected the *external* audit tools
  (`audit_buckets.py`/`vt_diagnose.py`), not `test.py`'s own internal AP/AP50,
  which is computed directly from in-memory per-batch tensors and never
  re-matches by filename, in either run. Leading candidates, neither
  confirmed: a real data-provenance difference between `UAVDT_v3` (origin
  unknown) and `UAVDT_fresh`, or single-seed noise (though a 3pp+ shift on a
  373,997-box test set is large for noise alone). Open item, not yet
  resolved.
- TinyPerson uses only the retained official paper-text two-arm runner. Hardware
  topology (one GPU versus the paper's two V100s at global batch 8) remains a
  documented reproduction variable.

Retained bundles under `results/`:

| Bundle | Status |
|---|---|
| `visdrone_yolov5m_baseline` | Valid Gaussian protocol; retrained 2026-08-18 on a re-provisioned GPU host (full train/test/measure/audit logs retained, see `results/visdrone/visdrone_yolov5m_baseline/`) -- 34.9/58.6, within 0.1-0.2pp of the prior Audited value and the smallest residual-to-paper of all four rows in this table |
| `visdrone_yolov5m_sam_masks` | Valid released-code SAM hybrid control |
| `tinyperson_yolov5m_baseline` | Legacy evidence only, not a canonical reproduction. The retained training log records code commit `cbbbc86` with ratio 16, the current source uses ratio 8, and the claimed 55.26/71.23 score has no matching retained `official_eval` log. Do not reuse this checkpoint as R0 for the corrected protocol. |
| `uavdt_yolov5m_baseline` | Valid official-source, three-class artifacts (frozen-tree `run_baseline.sh` provenance) |
| `uavdt_yolov5m_baseline` (`UAVDT_fresh`) | Valid, active-tree reproduction on independently re-extracted raw data -- see SS7 for the full comparison against `uavdt_yolov5m_channel_pooled_concat_sabl_isphead` |

Empty `buckets.json` is never evidence. Accept compute data only from a
populated `--task measure` artifact or aggregate GFLOPs/FPS in its log.

## 3. Locked VisDrone matrix

All arms share `VisDrone_v2`, initialization, schedule, input, batch, seed
policy, exact K, SparseHead, evaluation, and measurement path.

### 3.1 Selector roster

| Arm | Selector | Selector loss | Question |
|---|---|---|---|
| E1.0 | Semantic baseline | released weighted BCE | Released-code reproduction |
| E1.P | Semantic baseline | focal:dice 20:1 | Paper/code loss drift |
| E2.1 | Semantic baseline | coverage | Coverage supervision only |
| E2.5 | Spectral only | coverage | Spectral evidence only |
| E2.4 | Semantic + spectral gate | coverage | Gated fusion |
| E2.3 | Semantic + spectral concat | coverage | Full concat fusion |
| E2.9 | Channel-pooled spectral + semantic concat | coverage | Low-overhead fusion |

### 3.2 Box-regression factorial

SABL is an orthogonal detector-head ablation, not part of the BCRS selector
claim. Its first-pass evidence is the loss-only VisDrone result in
`reference/new/SSABNet.pdf` (+2.2 AP, +2.2 APs, +6.3 AP75). Do not blend SABL
with WIoU, Inner-MPDIoU, or size weighting in this factorial.

| Arm | Selector | Selector loss | Box loss | Role |
|---|---|---|---|---|
| R0 | Semantic baseline | released weighted BCE | upstream CIoU | Existing E1.0 control |
| R1 | Semantic baseline | released weighted BCE | exact SABL | General loss-only effect |
| R2 | Channel-pooled spectral + semantic concat | coverage | upstream CIoU | Existing E2.9 control |
| R3 | Channel-pooled spectral + semantic concat | coverage | exact SABL | SABL x HESOD interaction |

Exact SABL contract:

- `s = sqrt(w_gt * h_gt)` in network-input pixels; never use raw detection-grid
  units with the paper constants.
- `mu(s) = exp(-(s / 32)^6)` and normalized Wasserstein constant `C = 12`.
- SABL changes only the regression contribution to `lbox`. Anchor assignment,
  classification, selector supervision, and the original CIoU-derived
  objectness quality target remain unchanged.
- `--box-loss size_weighted` is disabled. Model graph, inference, parameters,
  GFLOPs, threshold route, exact Top-K route, and SparseHead are unchanged.
- R1/R3 must retrain from the same pretrained initialization and seed as their
  controls. A CIoU-trained `best.pt` cannot establish a SABL result.
- The literal `kappa=32, beta=6, C=12` arm is tested first. Resolution-scaled
  constants are a later sensitivity study, not part of R1/R3.

The main effects are `R1-R0` and `R3-R2`; the selector-loss interaction is
`(R3-R2) - (R1-R0)`. Existing E1.0/E2.9 artifacts may serve as R0/R2 only when
all data, initialization, seed, schedule, routing, and evaluator fields match.

Implementation record (2026-08-12):

- `hesod/backends/hesod/utils/loss.py` implements SSABNet Equations (10)-(15),
  converts grid-space boxes to input-pixel units only for Wasserstein distance
  and `s`, and retains the vendor CIoU value for objectness quality.
- `hesod/backends/hesod/train.py` exposes `--box-loss sabl`; the default remains
  `upstream`, and no inference or checkpoint-schema path is changed.
- `scripts/esod_baseline/run_visdrone_roster.sh` keeps SABL opt-in through
  `INCLUDE_SABL=1` and adds only the R1/R3 treatment checkpoints.
- `tests/test_sabl_loss.py` locks the exact-match, zero-overlap gradient,
  input-pixel scaling, large-object CIoU limit, objectness isolation, CLI, and
  runner contracts.

## 4. Execution and acceptance

First reproduce the three baselines:

```bash
cd /root/BCRS
RUN_ROOT=/root/esod_baseline_runs \
  bash scripts/esod_baseline/run_baseline.sh all 0
```

To run the released-code hybrid preprocessing on all three datasets (requires
the SAM dependency/checkpoint), use the same runner with an explicit protocol:

```bash
cd /root/BCRS
MASK_MODE=released-hybrid RUN_ROOT=/root/esod_hybrid_runs \
  bash scripts/esod_baseline/run_baseline.sh all 0
```

Then run the paper-loss control and locked roster:

```bash
cd /root/BCRS
CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT=/root/hesod_roster_runs \
REUSE_CHECKPOINTS=0 \
INCLUDE_PAPER=1 \
TOP_K=32 \
SPARSE_HEAD=1 \
  bash scripts/esod_baseline/run_visdrone_roster.sh 0
```

SABL execution is gated by `tests/test_sabl_loss.py`, which verifies finite,
non-zero zero-overlap gradients, input-pixel scaling, large-object convergence
toward CIoU, and unchanged objectness targets. The gate is implemented; run it
before launching an experiment:

```bash
cd /root/BCRS
pytest -q tests/test_sabl_loss.py
```

To add R1/R3 after the control roster exists:

```bash
cd /root/BCRS
CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT=/root/hesod_roster_runs \
REUSE_CHECKPOINTS=1 \
INCLUDE_PAPER=0 \
INCLUDE_SABL=1 \
TOP_K=32 \
SPARSE_HEAD=1 \
  bash scripts/esod_baseline/run_visdrone_roster.sh 0
```

Then:

1. Run one matched VisDrone seed for R1 and R3; reuse valid R0/R2 controls.
2. Advance an arm if AP improves by at least 0.5 pp or AP75 by at least 1.0 pp,
   without degrading AP50. AP50-only improvement does not advance.
3. Repeat the advanced control/treatment pairs with at least three seeds.
4. Transfer SABL to TinyPerson, then UAVDT nc=3, only after the VisDrone
   multi-seed result is positive. Keep the paper constants fixed for this
   transfer; any resolution-normalized variant receives a new arm name.

A run is accepted only if image/GT counts, class ranges, prediction IDs,
finite metrics, mask completeness, exact-route patch artifact, BPRbox/BPRctr
audit, and populated measurement artifacts pass. Regression arms must also
contain AP75/APs and IoU-0.75 recall diagnostics. Audit mismatch is fatal unless
`SKIP_AUDIT=1` is explicitly chosen. Evaluator-only fixes permit checkpoint
re-evaluation; a checkpoint trained with a removed data/class/loss protocol is
not reusable.

After single-run validation, repeat E1.0, E1.P, the best clean HESOD arm, and
any advanced SABL control/treatment pair with at least three seeds. Report
mean/std for AP, AP50, AP75, APs where defined, BPRbox, BPRctr, occupancy,
GFLOPs, FPS/latency, and recall buckets at IoU 0.50/0.75. Transfer only an
identified winner to TinyPerson, then UAVDT nc=3; do not tune jointly on all
datasets.

## 5. Guardrails and next decisions

Do not restore these invalid/confounded protocols: generic VisDrone
conversion, TinyPerson VOC-finetune hyp, UAVDT nc=1, hidden-threshold Top-K
sweeps, old concat/pooled-concat bundles, aggressive tiny-box weighting,
ad-hoc IoU/VOC evaluation, or ratio-16 runners. Their only retained conclusion
is that they cannot support a paper-comparable or causal claim.

Code invariants:

- Plain ESOD and the HESOD baseline graph remain checkpoint-compatible:
  35,842,600 parameters and the same 581-entry state schema.
- All evaluators retain trailing zero-patch images; TinyPerson must report
  exactly 786 images and 13,687 labels.
- Exact Top-K plus SparseHead has no second heatmap threshold.
- Prediction JSON keeps raw zero-indexed classes; the auditor rejects unknown
  or ambiguous image IDs.
- Native epoch validation uses the paper's strict `BPRbox > 0.5` boundary; the
  final runner audit independently recomputes BPRbox/BPRctr from an exact-route
  artifact and final detector recall from one-to-one matched predictions.
- Spectral kernels are trainable Sobel/Laplacian initializations, not fixed
  filters.
- SABL is training-only and may not alter checkpoint schema or inference. Its
  regression score must not replace the existing CIoU objectness target.

Decision order:

1. Quantify the paper-loss drift before altering the detector.
2. Judge HESOD on AP-versus-latency Pareto performance, not selector recall;
   add area/budget regularization or a rank-aware Top-K surrogate if needed.
3. Once the clean selector comparison is valid, run the R0-R3 SABL factorial
   before trying combined or hand-tuned box losses.
4. If SABL is neutral, target localization/calibration next with anchor
   re-clustering, a higher-resolution detection level, or QFL/Varifocal-style
   ranking; do not revive aggressive tiny-box weighting.
5. Test fixed versus trainable spectral filters only after the clean
   spectral-only arm demonstrates value.

No deleted exploratory result meets the bar for a publishable claim.

## 6. SeaDronesSeeV2 reference literature (reported-only, non-reproduction)

No YOLOv5m-based, common-protocol baseline exists for SeaDronesSeeV2 within
this project or in the original SeaDronesSeeV2 paper (its own Table 4
baselines are Faster R-CNN/CenterNet/EfficientDet, non-YOLO, and its class
taxonomy predates the V2 revision our `CompressedVersion` download uses --
confirmed 2026-08-xx: category_id 1-5 = swimmer/boat/jetski/
life_saving_appliances/buoy, category_id 0 'ignored' never used in any
annotation). The numbers below are drawn from four papers in
`reference/HighResolution/` (reviewed for resolution-protocol guidance;
`run_seadronesseev2.sh`'s img-size=1536 choice cites this section). Per the
reported-only discipline already established in
`HESOD-Lightweight-Detector-Review-and-Roadmap.md` SS8: these are
context/orientation only, never to be mixed into a common-protocol sortable
column with our own `results/seadronesseev2/*` runs. Confounds that break
direct comparability, all four papers:

- **Backbone/scale mismatch**: EUAVDet.pdf and SeaLSOD-YOLO.pdf compare
  against YOLOv5-nano/YOLOv5s; "Maritime Small Object Detection.pdf" uses
  YOLOv5s. Our own runs use YOLOv5m.
- **Resolution mismatch**: see SS6.1; none match our chosen 1536.
- **Metric-definition mismatch**: all four report standard COCO-style
  mAP/AP50/AP75 (area-based small/medium/large bins per COCO convention). Our
  own `audit_buckets.py`/`vt_diagnose.py` report plain recall over a
  different, codebase-specific pixel-side-length bucket scheme (Very Tiny
  <16x16, Tiny 16-32, Small 32-96, Medium/Large >96) -- not the same
  statistic.
- **Class-taxonomy alignment unverified**: whether each paper's
  SeaDronesSee(V2) instance uses the same 5-class V2 taxonomy as our
  `CompressedVersion` download has not been independently confirmed
  per-paper.
- **FPS hardware mismatch**: EUAVDet.pdf measures FPS on Jetson Nano/Orin
  (embedded boards); SeaLSOD-YOLO.pdf's FPS hardware is unstated in the
  extracted table; our own FPS is measured on an RTX 5090 desktop GPU. None of
  these FPS numbers are comparable to each other or to ours -- same class of
  issue as the V100-vs-RTX5090 TinyPerson/VisDrone FPS gap already discussed
  in SS1.

### 6.1 Resolution protocol (why SeaDronesSeeV2 runs at img-size=1536)

| Paper | Protocol | Rationale (as stated) |
|---|---|---|
| EUAVDet.pdf | 640x640, P3/P4/P5 at 80/40/20 | Standard YOLOv5 default, no dataset-specific tuning stated |
| SeaLSOD-YOLO.pdf | 640x640 | "determin[es] the resolution of training samples" -- no further justification given |
| Maritime Small Object Detection.pdf | 640x640 SAHI tiles (altitude-aware dynamic tiling effectively reaches ~1280p via 2x2 tiling) | Fixed-tile SAHI baseline; their own altitude-aware ScaledV1/V2 vary effective resolution by flight altitude |
| YOLOv7-sea.pdf | ~2400 side length (large, no fixed grid stated) | "For small targets, the larger the input image scale, the better the detection performance" -- explicit resolution-accuracy argument, the opposite framing from the other three |

Three of four papers converge on 640 (matched to plain YOLOv5 defaults, no
dataset-specific reasoning given); YOLOv7-sea.pdf is the outlier, arguing
explicitly for much higher resolution specifically because these targets are
small. img-size=1536 for our own runs was chosen as a deliberate middle
ground -- citing VisDrone's own in-project precedent (same 8x8-patch-grid
architecture) rather than either literature extreme, since 640 is the same
order of magnitude as Pest24 (where SS11.2.9-equivalent routing already lost
to dense) and would risk re-testing nothing new. Not yet validated against a
2400 run.

### 6.2 Reported accuracy/efficiency (reported-only, not a reproduction target)

**EUAVDet.pdf Table 5** (SeaDronesSeeV2 validation, YOLOv5/7/8/10-nano/tiny
family, 640) -- only the YOLOv5-lineage row is reproduced here; the full table
also has YOLOv7-tiny/YOLOv8-n/YOLOv10-n + their own EUAVDet variants and
AP_S/AP_M/AP_L breakdowns:

| Method | Params (M) | GFLOPs | AP50 | AP75 | mAP | FPS (Jetson Nano) | FPS (Jetson Orin) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5-n | 1.77 | 4.2 | 70.2 | 36.0 | 38.4 | 24.5 | 92.0 |
| EUAVDet-nv5 (their method) | 1.03 | 4.0 | 74.6 | 39.7 | 40.9 | 24.8 | 94.5 |

**SeaLSOD-YOLO.pdf Tables 3+4** (SeaDroneSee validation, 640) -- only the
plain-YOLOv5s row is reproduced here as the closest architecture-family
anchor; the full table also has Faster R-CNN/SSD/YOLOv3/v6/v8/v9/v10/v11/v12/
RT-DETR/three other recent small-object YOLO variants:

| Method | P | R | mAP@.5 | mAP@.5:.95 | Params (MB) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 75.1 | 61.5 | 66.7 | 40.2 | 17.6 | 24.1 | 344 |
| SeaLSOD-YOLO (their method) | 81.9 | 73.5 | 77.0 | 44.9 | 30.8 | 33.9 | 256 |

**Maritime Small Object Detection.pdf Table III** (SeaDronesSee, YOLOv5s):

| Setup | mAP | mAP50 | mAP50-small | mAP50-medium | mAP50-large | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Original image (640px, no tiling) | 0.16 | 0.35 | 0.34 | 0.88 | 1.00 | 7.00 |
| Tiled (static SAHI) | 0.34 | 0.62 | 0.42 | 0.75 | 0.91 | 0.60 |
| ScaledV1 (altitude-aware, their method) | 0.25 | 0.48 | 0.45 | 0.68 | 0.97 | 2.60 |
| ScaledV2 (altitude-aware, their method) | 0.27 | 0.52 | 0.47 | 0.73 | 0.97 | 1.87 |
| Full-resolution, no size restriction (side-mention only, not in their main table, no FPS reported) | -- | -- | 0.21 | -- | -- | not reported |

**YOLOv7-sea.pdf Table 1** (SeaDronesSee test-set challenge, ~2400; no
GFLOPs/FPS reported anywhere in this paper):

| Method | AP | AP50 | AP75 | AR1 | AR10 |
|---|---:|---:|---:|---:|---:|
| Baseline | 41.81 | 72.33 | 41.25 | 36.33 | 48.91 |
| YOLOv7 | 53.61 | 83.12 | 56.72 | 43.79 | 60.41 |
| Ours (final, their method) | 59.00 | 90.72 | 64.15 | 46.41 | 67.98 |

Matches the paper's own headline claim (3rd place on the SeaDronesSee
benchmark-server leaderboard, 59% mAP; they cite the then-current top
leaderboard entry at 36% mAP for context). A separate ablation table
(their Table 2, ADD-one-module-at-a-time on an internal validation set) also
exists in this paper but had an internally inconsistent-looking AP50 value at
one intermediate row during extraction (AP50 dropping from 87.8 to 45.2 then
recovering to 92.3 while AP moved smoothly) -- not reproduced here; verify
against the source PDF directly if that specific ablation is needed.

### 6.3 Own R0 result and decision to drop the dataset (2026-08-20)

R0 (`seadronesseev2_yolov5m.yaml`, img-size=1536, 50 epochs) trained and
evaluated cleanly -- no data issues, no pipeline bugs, a legitimate result:

| Size bin | GT Count | Recalled | Recall Rate |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 183 | 151 | 82.51% |
| Tiny (16x16-32x32) | 2784 | 2607 | 93.64% |
| Small (32x32-96x96) | 4795 | 4637 | 96.70% |
| Medium/Large (>96x96) | 1868 | 1827 | 97.81% |
| **Total** | **9630** | **9222** | **95.76%** |

mAP@.5=0.894, mAP@.5:.95=0.533, GFLOPs=76.4, FPS=134.2 (`test.py --task
measure`, batch=1, RTX 5090).

Very Tiny objects are only **183/9630 ~ 1.9%** of this val split's GT --
matching the pre-run diagnostic (SS6, above) that already flagged val
specifically at ~1.9% Very Tiny (train ~5.5%), far below VisDrone's ~31% or
TinyPerson's ~53%. R0 already recalls 82.5%+ even in the hardest bin, and
total recall is 95.76% -- there is essentially no accuracy headroom left:
even a hypothetical 100% Very Tiny recall would only move total recall from
95.76% to ~96.6% (183 more correct detections out of 9630), since that bin
is such a small share of the data. This directly confirms the pre-run
concern raised before any training started -- the dataset's actual object-
size composition doesn't match the "sparse, genuinely tiny targets" regime
HESOD's routing thesis needs to show an accuracy advantage, unlike VisDrone/
TinyPerson.

**Decision: dropped from further HESOD-arm investment.** The queued
concat+SABL+ISPPHead arm was not run. Same class of finding as Pest24's A0
result (SS11.2.9 in `HESOD-Agri-Experiment-Plan.md`) -- a legitimate,
informative negative result about where the routing thesis does and doesn't
have room to operate, not a data or pipeline failure. Retained here rather
than silently dropped so the reasoning stays traceable.

## 7. UAVDT: first concat-vs-R0 comparison, and an `image_id` fix (2026-08-20/21)

No concat/SABL/lightweight-head arm had ever been run for UAVDT before this
session (only R0 existed). `UAVDT_fresh` (re-extracted from the raw
`M_attr.zip`/`UAV-benchmark-M.zip`/`UAV-benchmark-MOTD_v1.0.zip` on the active
`hesod/backends/hesod` tree, not the frozen-tree `run_baseline.sh` pipeline)
ran both R0 and `concat+SABL+ISPPHead` (plain concat+SABL skipped, same
skip-the-middle-arm reasoning as SeaDronesSeeV2/fresh-TinyPerson).

### 7.1 `image_id` collision bug found and fixed

UAVDT reuses frame numbers (`img000001.jpg`, `img000002.jpg`, ...) identically
across every video sequence, with no split-level subdirectory separating
train from test videos in `UAV-benchmark-M/`. `test.py`'s `image_id`
assignment (`int(path.stem) if path.stem.isnumeric() else path.stem`, used by
TinyPerson/VisDrone/SeaDronesSeeV2 without issue) collided across videos for
UAVDT specifically -- predictions from frame 1 of *any* of the ~20 test
videos all wrote `image_id="img000001"` into `best_predictions.json`,
indistinguishable to any external tool that re-associates predictions with
images by that field. Fixed in `test.py` (both the `save_json` and
`selected_regions` assignments) to use `path.parent.stem + '_' + path.stem`
for UAVDT specifically -- the same disambiguation pattern the `--save-txt`
label-filename code already used elsewhere in the same file, just not
applied to `image_id` itself. A matching symlink farm
(`images/test/`+`labels/test/`, built from `split/test.txt` with the same
`{video}_{stem}` naming) was created so `audit_buckets.py`/`vt_diagnose.py`
can resolve images by this same key.

**Scope of the bug, precisely: this never affected `test.py`'s own internal
AP/AP50** (computed directly from in-memory per-batch tensors during the
dataloader loop, never re-matched by filename) -- only the external audit
tools, which previously either crashed outright or silently reported 0%
recall (predictions and GT never matched at all under the old plain-stem
symlink layout).

### 7.2 R0 vs concat+SABL+ISPPHead (`UAVDT_fresh`, both size-bin audits confirmed consistent)

| | R0 | Concat+SABL+ISPPHead | Delta |
|---|---:|---:|---:|
| mAP@.5 | 0.344 | 0.371 | +2.7pp |
| mAP@.5:.95 | 0.171 | 0.195 | +2.4pp |
| Total recall | 84.67% | 90.17% | +5.5pp |
| Very Tiny recall (<16x16) | 83.63-83.64% | 84.18-84.21% | +0.55pp |
| car / truck / bus recall | 84.87 / 84.41 / 75.20 | 90.56 / 84.88 / 76.28 | +5.7 / +0.5 / +1.1pp |
| Occupy | 0.245 | 0.48 | ~2x |

audit_buckets.py and vt_diagnose.py's independently-computed Very Tiny
recall figures agree closely for both arms (83.63/83.64 and 84.18/84.21),
a useful cross-check that the fixed `image_id` pipeline is now self-
consistent.

**Clear, consistent win for concat+SABL+ISPPHead** across mAP, total
recall, and Very Tiny recall -- not just a tradeoff on one axis. This is
the first-ever arm-vs-R0 comparison run for UAVDT in this project.

**Caveat, not yet isolated: Occupy nearly doubled (0.245 -> 0.48).** The
concat selector routes to roughly twice as much image area as R0's
semantic-only selector -- some of this accuracy gain is plausibly "seeing
more of the image," not purely smarter patch selection. Same class of
open question as Pest24's R2 result (HESOD-Agri-Experiment-Plan.md
SS11.2.2's "smarter selection vs. selecting more area" caveat). A
matched-Occupy or matched-GFLOPs comparison would be needed to isolate the
selection-quality effect from the budget-increase effect; not run here.

### 7.3 R0-vs-paper gap widened, not confirmed (see SS2)

`UAVDT_fresh` R0's residual to the paper (-5.4/-6.3pp AP/AP50) is larger
than the previously-accepted frozen-tree number's residual (-2.4/-3.7pp),
not a confirmation of it. The `image_id` fix in SS7.1 does not explain this
gap (it never touched `test.py`'s own AP computation). Leading candidates
-- a real data-provenance difference between `UAVDT_v3` and `UAVDT_fresh`,
or single-seed noise -- are both unconfirmed. Open item.

## 8. SeaPerson (aka TinyPersonV2): 6-arm roster, in progress (2026-08-20/21)

A further-validation dataset requested after the TinyPerson protocol rework
(SS1/SS2) -- same schema lineage as TinyPerson (JSON image/annotation keys
verified identical, including `ignore`/`uncertain`/`in_dense_image`), much
denser (5711 train images, 262,063 annotations, ~45.9 boxes/image vs
TinyPerson's own 794/~much sparser), and ships a genuine official 3-way
split (`train`/`valid`/`test`; confirmed `train_ids | valid_ids ==
trainvalid_ids` exactly and zero train/test id overlap) -- unlike TinyPerson,
which has to manufacture its own random valid slice.

**Protocol decisions:**
- Whole-image annotations only (`release/rgb_{train,valid,test}.json`), not
  the pre-tiled `release/corner/*.json` sliding-window variants -- same
  reasoning as TinyPerson's own excluded `sw640_sh512` variant: ESOD's method
  self-routes full images, it does not want pre-cropped tiles.
- img-size=2048, matching TinyPerson's own choice, justified by SeaPerson's
  own resolution distribution (92.9% of train images at 1920x1080).
- No erase-preprocessed image variant ships with this release (only raw
  `imgs_rgb.zip`), unlike TinyPerson, which was just reworked specifically
  *toward* the erase-preprocessed protocol. Rather than fall back to
  label-level-only `ignore`/`uncertain` filtering on unmodified pixels (the
  approach TinyPerson's own protocol was reworked away from), `prepare_seaperson()`
  erases those box regions directly into the image (mid-gray fill, same
  convention `prepare_visdrone()` already uses for its `ignored`/`others`
  classes), writing an `_erased` copy only when erasing actually occurred.
- Labels/masks are written in-place per source video subfolder (same
  convention as `prepare_uavdt()`/`prepare_tinyperson()`), which leaves no
  flat directory for `audit_buckets.py`/`vt_diagnose.py`'s `--labels`/
  `--images` convention -- the same gap UAVDT hit. Fixed proactively this
  time with a proper committed reorganizer
  (`scripts/esod_baseline/reorganize_seaperson.py`, modeled on the
  already-existing `reorganize_uavdt.py`) rather than a scratchpad symlink
  patch, producing a self-contained `seaperson_v2/{images,labels,masks}/{split}/`
  tree that `data/seaperson.yaml` points at directly.

**Roster (6 arms, confirmed distinct by the user):**

| Arm | Selector | Loss | Box |
|---|---|---|---|
| R0 | `Segmenter` | upstream | CIoU |
| semantic-only | `Segmenter` (same arch as R0) | coverage | CIoU |
| spectral-only | `SpectralOnlySegmenter` | coverage | CIoU |
| concat-only | `ChannelPooledConcatEvidenceSegmenter` | coverage | CIoU |
| concat+SABL | `ChannelPooledConcatEvidenceSegmenter` | coverage | SABL |
| concat+SABL+ISPPHead | `ChannelPooledConcatEvidenceSegmenter` + `ISPPHead` | coverage | SABL |

R0 and semantic-only share the same model config (`seaperson_yolov5m.yaml`)
and differ only in `--selector-loss`, matching VisDrone's own E1.0/E2.1 pair
(`run_visdrone_roster.sh`) -- deliberately isolating the loss-function effect
from the selector-architecture effect. concat-only and concat+SABL likewise
share one config (`seaperson_yolov5m_channel_pooled_concat.yaml`), differing
only in `--box-loss`.

**Status (2026-08-21): roster running. R0 and semantic-only complete;
spectral-only, concat-only, concat+SABL, concat+SABL+ISPPHead in progress.**

### 8.1 `vt_diagnose.py` mixed-image-format bug found and fixed

`vt_diagnose.py --images-dir` matched each label file to its image via a
single hardcoded `--image-ext` (default `.jpg`). SeaPerson mixes formats --
most images are `.jpg`, but the `rgb1000/` subfolder ships `.bmp` -- so every
`.bmp` image failed to match and silently fell back to an assumed native size
of 800x600 for size-bucket classification. Confirmed systematic, not random:
exactly 1000/5752 test label files hit the fallback on both R0 and
semantic-only. Real actual resolution for these images is far from 800x600
(SeaPerson is 92.9% 1920x1080), so every fallback image's GT boxes were
bucketed against the wrong denominator.

**Impact, precisely:** this only ever corrupted `vt_diagnose.py`'s own
standalone Very Tiny/Tiny/etc. bucket counts -- `audit_buckets.py` was never
affected (it already tries multiple image suffixes) and its numbers below are
unchanged by the fix. Before the fix, `vt_diagnose.py` reported Very Tiny
recall of 52.79% (R0) / 53.12% (semantic-only) against 74.03%/75.49% from
`audit_buckets.py` on the same predictions -- a stark disagreement that was
the actual tell something was wrong, matching the cross-tool-agreement check
already established for UAVDT (SS7.2).

**Fix:** `vt_diagnose.py` now tries every common image suffix (`.jpg`,
`.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`), `--image-ext` first, before
falling back to `--native-w`/`--native-h`. Re-run against the existing
`best_predictions.json` for both completed arms confirmed agreement with
`audit_buckets.py` to within 0.02pp (Table below) -- exact Very Tiny GT count
also now matches exactly (82417 both tools, was 113672 vt_diagnose vs 82417
audit_buckets before the fix, since the wrong denominator also pulled boxes
into/out of the Very Tiny bucket incorrectly, not just misclassifying which
image they belonged to).

### 8.2 Full per-arm results (test split unless noted; updated as each arm completes)

**Size-bucket recall** (`audit_buckets.py`, test split, 300375 total GT
targets: 82417 Very Tiny / 174816 Tiny / 42987 Small / 155 Medium-Large):

| Arm | Very Tiny (<16x16) | Tiny (16-32) | Small (32-96) | Medium/Large (>96) | Total |
|---|---:|---:|---:|---:|---:|
| R0 | 74.03% (61014/82417) | 86.78% (151707/174816) | 94.74% (40725/42987) | 79.35% (123/155) | 84.42% (253569/300375) |
| semantic-only | 75.49% (62217/82417) | 91.57% (160087/174816) | 95.71% (41141/42987) | 81.94% (127/155) | 87.75% (263572/300375) |
| spectral-only | pending (OOM on first attempt, SS8.3) | | | | |
| concat-only | pending | | | | |
| concat+SABL | pending | | | | |
| concat+SABL+ISPPHead | pending | | | | |

**Accuracy + efficiency** (mAP/P/R/BPR/Occupy from `test.py --task test`;
GFLOPs/FPS/latency from `test.py --task measure`, valid split, batch=1):

| Arm | mAP@.5 | mAP@.5:.95 | P | R | BPR | Occupy (test) | GFLOPs | FPS | Latency (ms, infer/NMS/total) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.750 | 0.320 | 0.827 | 0.696 | 0.947 | 0.228 | 202.4 | 88.9 | 11.2/1.2/12.4 |
| semantic-only | 0.769 | 0.325 | 0.825 | 0.708 | 0.991 | 0.363 | 266.8 | 75.4 | 13.3/1.2/14.4 |
| spectral-only | pending | | | | | | | | |
| concat-only | pending | | | | | | | | |
| concat+SABL | pending | | | | | | | | |
| concat+SABL+ISPPHead | pending | | | | | | | | |

**Very Tiny miss-reason breakdown** (`vt_diagnose.py`, post mixed-format fix,
SS8.1; percentages are of that arm's own missed-Very-Tiny count):

| Arm | Missed Very Tiny GT | `right_class_low_iou` (localization failure) | `no_nearby_prediction` (selector-dropped) | `matched_but_stolen_by_other_gt` |
|---|---:|---:|---:|---:|
| R0 | 21388 | 74.3% | 25.6% | 0.1% |
| semantic-only | 20181 | 79.7% | 20.1% | 0.1% |
| spectral-only | pending | | | |
| concat-only | pending | | | |
| concat+SABL | pending | | | |
| concat+SABL+ISPPHead | pending | | | |

**Interpretation so far** (2 of 6 arms; will be revisited once the roster
completes): semantic-only (same `Segmenter` architecture as R0, coverage
loss instead of upstream weighted BCE) improves every accuracy metric,
isolating the loss-function effect cleanly since the selector architecture is
unchanged. As with UAVDT's concat comparison (SS7.2), Occupy increased
alongside accuracy -- the same "smarter selection vs. selecting more area"
caveat applies here and is not yet isolated. More notably, for both arms
localization failure dominates missed Very Tiny objects, not selector drops
-- coverage supervision is doing exactly what SS3.3/H4 predict: it reduces
selector-caused misses (25.6%->20.1% of a smaller total) and shifts the
remaining bottleneck further toward detection-head localization precision at
very tiny scale. This suggests headroom in this roster is more likely in
box-regression quality (SABL, evaluated in a later arm) than in further
selector tuning alone -- a hypothesis the remaining arms will test directly.

### 8.3 spectral-only: CUDA OOM, unresolved as of this writing

`SpectralOnlySegmenter` (E2.5-style, full-channel `SpectralBranch` --
`MultiKernelSpectralFilter` runs its depthwise 3x3/5x5 filters on the full
P3 channel width, unlike the channel-pooled variant the concat arms use)
OOM'd at epoch 0/batch 0 on the same batch=8 that trained R0 and
semantic-only cleanly: "31.30 GiB memory in use" out of a 31.36 GiB RTX 5090,
failing to allocate 48 MiB more. AMP is already on by default
(`half_precision = not opt.disable_half`, no `--disable-half` passed), so
this isn't an unused lever. A retry at `--batch-size 4` reproduced the same
OOM; `--batch-size 2` is queued next, not yet confirmed. If this arm needs a
smaller batch than every other arm to complete, that is a real, documented
protocol deviation for this one arm specifically (its own row in SS8.2's
tables should note the batch size actually used), not a silent
inconsistency -- matches the same-class caveat already applied to R1/R3 in
SS3.2 when a treatment arm needs a setting its control didn't.
