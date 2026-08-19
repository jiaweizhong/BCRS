# HESOD Experiment Plan

**Canonical status: 2026-08-12.** This file is the execution contract, not an
investigation log. Superseded configs, failed-run details, and patch history
belong in Git history or `ESOD-Baseline-Patches.md`.

## 1. Fixed protocols

| Dataset | Data | Classes | Model | Hyp | Input | Evaluation split |
|---|---|---:|---|---|---:|---|
| VisDrone | `/root/autodl-tmp/VisDrone_v2.yaml` | 10 | `visdrone_yolov5m.yaml` | `hyp.visdrone.yaml` | 1536 | val: 548 images, 38,759 GT |
| TinyPerson | `/root/autodl-tmp/TinyPerson_v1.yaml` | 1 | `tinyperson_yolov5m.yaml` | `hyp.tinyperson.yaml` | 2048 | val: 786 images, 13,687 GT |
| UAVDT | `/root/autodl-tmp/UAVDT_v3.yaml` | 3 | `uavdt_yolov5m.yaml` | `hyp.uavdt.yaml` | 1280 | test: car/truck/bus |

Common training protocol: YOLOv5m, 50 epochs, SGD, global batch 8, cosine
scheduling, and weight decay 0.0005. TinyPerson alone uses the canonical
dataset-specific `lr0=0.005`, `pixl=0.4` profile. No alternate TinyPerson hyp
or UAVDT class protocol is valid.

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
  own released code: `grep -n "top.k\|top_k" esod/test.py esod/models/yolo.py`
  returns zero hits, and `esod/test.py --sparse-head` calls
  `model.model[-1].set_sparse()` standalone, with no companion top-K flag.
  Efficiency/method comparisons BETWEEN HESOD ARMS use exact K + SparseHead +
  measured end-to-end latency/GFLOPs, so every arm gets the same fixed patch
  budget. For accuracy AND efficiency reproduction AGAINST THE PAPER'S OWN
  reported numbers (AP/AP50 and GFLOPs/FPS alike), fixed-threshold routing is
  the paper-matching choice -- `esod/models/yolo.py`'s `get_indices(...,
  thresh=0.3)` applies the same hardcoded second threshold inside each
  fixed-threshold-selected patch with no override, exactly matching upstream.
  HESOD adds a `sparse_all_selected` flag (absent from `esod/`) specifically
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
the YOLO bbox labels, and the protocol applies to VisDrone, UAVDT, and
TinyPerson:

1. **Released-code baseline:** weighted BCE selector loss plus explicitly
   generated Gaussian masks (`--mask-mode gaussian`).
2. **Paper-text loss control:** focal:dice 20:1 (`--selector-loss paper`).

A literal paper-mask control (RGB SAM plus Eq. 4, without the public code's
extra `SAM*0.5` union) remains pending. Do not label a released-code run as a
bit-exact paper implementation.

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
| TinyPerson APt50 / APs50 | 61.3 / 74.4 | 55.26 / 71.23 | -6.04 / -3.17 pp |
| UAVDT nc=3 AP / AP50 | 22.5 / 40.7 | 20.1 / 37.0 | -2.4 / -3.7 pp |

Interpretation is intentionally narrow:

- VisDrone's former large gap came mainly from the wrong conversion; the
  audited released-code reproduction is now within 1.0 pp.
- TinyPerson improved with its dataset hyp but retains the largest APt50 gap.
- UAVDT nc=3 is paper-comparable; the released nc=1 collapse is another task.
- The next shared causal test is focal+dice versus weighted BCE, followed only
  if necessary by the isolated paper-mask control. Single-GPU BatchNorm versus
  the paper's two-GPU/global-batch-8 setup is a secondary unproven variable.

Retained bundles under `results/`:

| Bundle | Status |
|---|---|
| `visdrone_yolov5m_baseline` | Valid Gaussian protocol; retrained 2026-08-18 on a re-provisioned GPU host (full train/test/measure/audit logs retained, see `results/visdrone/visdrone_yolov5m_baseline/`) -- 34.9/58.6, within 0.1-0.2pp of the prior Audited value and the smallest residual-to-paper of all four rows in this table |
| `visdrone_yolov5m_sam_masks` | Valid released-code SAM hybrid control |
| `tinyperson_yolov5m_baseline` | Valid canonical-hyp artifacts; fixed-evaluator (`scripts/esod_baseline/tinyperson_eval/eval_tinyperson_official.py`, official APt50/APs50 protocol) applied 2026-08-18 to an independently retrained checkpoint on a re-provisioned GPU host -- 55.26/71.23, within 0.2pp of the prior Audited value (opposite-signed on APt50 vs APs50, consistent with single-seed training noise). This is a different checkpoint from the one that produced 55.46/71.04, not a re-eval of it, so it does not yet isolate evaluator-pipeline stability from run-to-run training variance; a true reproduction check would re-run this evaluator against the original checkpoint's predictions if still available |
| `uavdt_yolov5m_baseline` | Valid official-source, three-class artifacts |

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
