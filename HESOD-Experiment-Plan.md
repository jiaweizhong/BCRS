# HESOD Experiment Plan

**Canonical status: 2026-08-23.** This file is the execution contract, not an
investigation log. Superseded configs, failed-run details, and patch history
belong in Git history or `ESOD-Baseline-Patches.md`.

## 1. Fixed protocols

| Dataset | Data | Classes | Model | Hyp | Input | Evaluation split |
|---|---|---:|---|---|---:|---|
| VisDrone | `/root/autodl-tmp/VisDrone_v2.yaml` | 10 | `visdrone_yolov5m.yaml` | `hyp.visdrone.yaml` | 1536 | val: 548 images, 38,759 GT |
| TinyPerson | protocol-specific audited YAML | 1 | `tinyperson_yolov5m.yaml` | `hyp.tinyperson.yaml` | 2048 | official test: 786 images, no dense images |
| UAVDT | `/root/autodl-tmp/UAVDT_v3.yaml` | 3 | `uavdt_yolov5m.yaml` | `hyp.uavdt.yaml` | 1280 | test: car/truck/bus |
| SeaPerson (TinyPersonV2) | `/root/autodl-tmp/seaperson.yaml` (`seaperson_v2/`, via `reorganize_seaperson.py`) | 1 | arm-specific, see SS8 | `hyp.seaperson.yaml` | 2048 | official test: 5752 images |

SeaDronesSeeV2 was evaluated and dropped (SS6.3) -- not a live protocol.

Common training protocol: YOLOv5m, 50 epochs, SGD, global batch 8, cosine
scheduling, and weight decay 0.0005. TinyPerson's paper-text profile uses
`lr0=0.01`, `pixl=0.4`, and focal:dice 20:1. Alternate TinyPerson
hyperparameter profiles are no longer retained. No alternate UAVDT class
protocol is valid. SeaPerson's `spectral-only` arm is the one documented
exception to global batch 8 (SS8, OOM at batch 8 and 4; trained at batch 2).

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
  `hesod/backends/esod/`) specifically to stop that same 0.3 threshold from
  re-filtering positions inside an exact-K-selected patch, which would
  otherwise silently break exact K's own "select K patches, evaluate each
  fully" semantics; `test.py` enables it only when `--top-k` and
  `--sparse-head` are passed together. So `--sparse-head` alone (no
  `--top-k`) is the paper-comparable SparseHead test; `--sparse-head
  --top-k N` measures HESOD's own corrected exact-K mode, a different setup.
  See the Routing contract below for the full Algorithm 1/2 description.
- Box-regression comparisons additionally report AP75, APs, and size-bin
  detector recall at IoU 0.50 and 0.75. An AP50-only gain is insufficient.
- `test.py`'s `image_id` field (used only by external tools that
  re-associate predictions with images by filename -- never by `test.py`'s
  own internal AP, which is computed from in-memory per-batch tensors) is
  `path.stem` for every dataset except UAVDT, which reuses frame numbers
  (`img000001.jpg`, ...) identically across every video and needs
  `path.parent.stem + '_' + path.stem` to stay unique; `--save-txt`
  filenames already used this disambiguation, `image_id` now matches it.
- `vt_diagnose.py --images-dir` tries every common image suffix (`.jpg`,
  `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`), `--image-ext` first, before
  falling back to `--native-w`/`--native-h` for a genuinely unmatched file --
  needed for SeaPerson's mixed `.jpg`/`.bmp` images (the `rgb1000/`
  subfolder ships `.bmp`). `audit_buckets.py` was never affected by either
  gap above; it already handles both cases.

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
| UAVDT nc=3 AP / AP50 (`UAVDT_fresh`, active tree) | 22.5 / 40.7 | 17.1 / 34.4 | -5.4 / -6.3 pp |

Interpretation is intentionally narrow:

- VisDrone's former large gap came mainly from the wrong conversion; the
  audited released-code reproduction is now within 1.0 pp.
- TinyPerson's historical 55.26/71.23 result is not accepted as a current
  reproduction: its retained training log identifies ratio 16 while the
  current source is ratio 8, and no matching official-evaluation log remains.
- UAVDT nc=3 is paper-comparable; the released nc=1 collapse is another task.
- UAVDT's fresh reconversion (`UAVDT_fresh`, re-extracted from raw source
  archives) widened the residual to the paper rather than confirming the
  frozen-tree number (-5.4/-6.3pp vs. -2.4/-3.7pp). The `image_id` fix (SS1)
  is ruled out as the cause -- it never touched `test.py`'s own AP
  computation, only external audit tools. Leading candidates, neither
  confirmed: a real data-provenance difference between `UAVDT_v3` and
  `UAVDT_fresh`, or single-seed noise (a 3pp+ shift on a 373,997-box test
  set is large for noise alone). Open item.
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

E2.1/E2.5/E2.4/E2.3 have never actually been trained/retained for VisDrone in
isolation (SS9.2) -- only E1.0 and the fully-bundled E2.9+SABL exist under
`results/visdrone/`.

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

Implementation:

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

## 6. SeaDronesSeeV2: evaluated and dropped

No YOLOv5m-based, common-protocol baseline exists for SeaDronesSeeV2 within
this project or in the original SeaDronesSeeV2 paper (its own Table 4
baselines are Faster R-CNN/CenterNet/EfficientDet, non-YOLO, and its class
taxonomy predates the V2 revision our `CompressedVersion` download uses --
confirmed: category_id 1-5 = swimmer/boat/jetski/life_saving_appliances/buoy,
category_id 0 'ignored' never used in any annotation). SS6.1/6.2 below are
drawn from four papers in `reference/HighResolution/`, reported-only context
per the discipline in `HESOD-Lightweight-Detector-Review-and-Roadmap.md` SS8
-- never mixed into a common-protocol sortable column with our own
`results/seadronesseev2/*` run. Confounds that break direct comparability,
all four papers:

- **Backbone/scale mismatch**: EUAVDet.pdf and SeaLSOD-YOLO.pdf compare
  against YOLOv5-nano/YOLOv5s; "Maritime Small Object Detection.pdf" uses
  YOLOv5s. Our own run used YOLOv5m.
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
  issue as the V100-vs-RTX5090 TinyPerson/VisDrone FPS gap in SS1.

### 6.1 Resolution protocol (why SeaDronesSeeV2 ran at img-size=1536)

| Paper | Protocol | Rationale (as stated) |
|---|---|---|
| EUAVDet.pdf | 640x640, P3/P4/P5 at 80/40/20 | Standard YOLOv5 default, no dataset-specific tuning stated |
| SeaLSOD-YOLO.pdf | 640x640 | "determin[es] the resolution of training samples" -- no further justification given |
| Maritime Small Object Detection.pdf | 640x640 SAHI tiles (altitude-aware dynamic tiling effectively reaches ~1280p via 2x2 tiling) | Fixed-tile SAHI baseline; their own altitude-aware ScaledV1/V2 vary effective resolution by flight altitude |
| YOLOv7-sea.pdf | ~2400 side length (large, no fixed grid stated) | "For small targets, the larger the input image scale, the better the detection performance" -- explicit resolution-accuracy argument, the opposite framing from the other three |

Three of four papers converge on 640 (matched to plain YOLOv5 defaults, no
dataset-specific reasoning given); YOLOv7-sea.pdf is the outlier, arguing
explicitly for much higher resolution specifically because these targets are
small. img-size=1536 was chosen as a deliberate middle ground -- citing
VisDrone's own in-project precedent (same 8x8-patch-grid architecture)
rather than either literature extreme, since 640 is the same order of
magnitude as Pest24 (where routing already lost to dense) and would risk
re-testing nothing new.

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
(their Table 2) also exists but had an internally inconsistent-looking AP50
value at one intermediate row during extraction -- not reproduced here;
verify against the source PDF directly if that specific ablation is needed.

### 6.3 Own R0 result and decision to drop the dataset

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
is such a small share of the data. The dataset's actual object-size
composition doesn't match the "sparse, genuinely tiny targets" regime
HESOD's routing thesis needs to show an accuracy advantage, unlike VisDrone/
TinyPerson.

**Decision: dropped from further HESOD-arm investment.** The queued
concat+SABL+ISPPHead arm was not run. Same class of finding as Pest24's A0
result (`HESOD-Agri-Experiment-Plan.md` SS11.2.9) -- a legitimate,
informative negative result about where the routing thesis does and doesn't
have room to operate, not a data or pipeline failure. Retained here rather
than silently dropped so the reasoning stays traceable.

## 7. UAVDT: R0 vs concat+SABL+ISPPHead

`UAVDT_fresh` (re-extracted from raw source archives on the active
`hesod/backends/hesod` tree, not the frozen-tree `run_baseline.sh` pipeline)
is the first concat/SABL/lightweight-head comparison ever run for UAVDT in
this project -- previously only R0 existed. It ran R0 and
`concat+SABL+ISPPHead` (plain concat+SABL was skipped at the time; SS9.2
revisits that gap now that SeaPerson's own concat-only result turned out to
be non-obvious). **Params below is known-incorrect, same bug as SS8.1's
now-fixed table:** computed from an unfused `Model(cfg).parameters()`
rather than the actual deployed (post-`.fuse()`) checkpoint `test.py
--task measure` reports -- SS8.1's audit (2026-08-23) found this
understates the real fusion savings by a small, architecture-dependent
amount (~22.8K params on SeaPerson's configs; UAVDT's own offset has not
been measured). Left uncorrected pending a fresh `--task measure` re-run
for these two UAVDT checkpoints (not yet done; low priority, same as
SS8.1's own artifact-consistency item, but this one is a known-wrong value,
not just an unrefreshed artifact).

| | R0 | Concat+SABL+ISPPHead | Delta |
|---|---:|---:|---:|
| mAP@.5 | 0.344 | 0.371 | +2.7pp |
| mAP@.5:.95 | 0.171 | 0.195 | +2.4pp |
| Total recall | 84.67% | 90.17% | +5.5pp |
| Very Tiny recall (<16x16) | 83.63-83.64% | 84.18-84.21% | +0.55pp |
| car / truck / bus recall | 84.87 / 84.41 / 75.20 | 90.56 / 84.88 / 76.28 | +5.7 / +0.5 / +1.1pp |
| Occupy | 0.245 | 0.48 | ~2x |
| Params (M) | 35.87 | 26.01 | -27.5% |

`audit_buckets.py` and `vt_diagnose.py`'s independently-computed Very Tiny
recall figures agree closely for both arms (83.63/83.64 and 84.18/84.21), a
useful cross-check that the `image_id` disambiguation (SS1) leaves the
pipeline self-consistent.

**Clear, consistent win for concat+SABL+ISPPHead** across mAP, total
recall, and Very Tiny recall -- not just a tradeoff on one axis. It also
has 27.5% fewer parameters than R0 (26.01M vs 35.87M) despite the added
concat selector branch, confirming ISPPHead's own efficiency motivation
(`HESOD-Lightweight-Detector-Review-and-Roadmap.md` SS5.2) holds at the
whole-model level here too, not just in the head-only bake-off it was
originally measured in.

**Caveat, not yet isolated: Occupy nearly doubled (0.245 -> 0.48).** The
concat selector routes to roughly twice as much image area as R0's
semantic-only selector -- some of this accuracy gain is plausibly "seeing
more of the image," not purely smarter patch selection. Same class of
open question as Pest24's R2 result (`HESOD-Agri-Experiment-Plan.md`
SS11.2.2's "smarter selection vs. selecting more area" caveat). A
matched-Occupy or matched-GFLOPs comparison would be needed to isolate the
selection-quality effect from the budget-increase effect; not run here.

`UAVDT_fresh` R0's residual to the paper (-5.4/-6.3pp AP/AP50, SS2) is
larger than the previously-accepted frozen-tree number's residual
(-2.4/-3.7pp), not a confirmation of it -- see SS2 for the still-open
provenance-vs-noise question.

## 8. SeaPerson (aka TinyPersonV2): 7-arm roster

A further-validation dataset requested after the TinyPerson protocol rework
-- same schema lineage as TinyPerson (JSON image/annotation keys verified
identical, including `ignore`/`uncertain`/`in_dense_image`), much denser
(5711 train images, 262,063 annotations, ~45.9 boxes/image vs TinyPerson's
own 794/~much sparser), and ships a genuine official 3-way split
(`train`/`valid`/`test`; confirmed `train_ids | valid_ids ==
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
  `imgs_rgb.zip`), unlike TinyPerson, which was reworked specifically
  *toward* the erase-preprocessed protocol. Rather than fall back to
  label-level-only `ignore`/`uncertain` filtering on unmodified pixels (the
  approach TinyPerson's own protocol was reworked away from),
  `prepare_seaperson()` erases those box regions directly into the image
  (mid-gray fill, same convention `prepare_visdrone()` already uses for its
  `ignored`/`others` classes), writing an `_erased` copy only when erasing
  actually occurred.
- Labels/masks are written in-place per source video subfolder (same
  convention as `prepare_uavdt()`/`prepare_tinyperson()`), which leaves no
  flat directory for `audit_buckets.py`/`vt_diagnose.py`'s `--labels`/
  `--images` convention -- the same gap UAVDT hit. Fixed proactively this
  time with a proper committed reorganizer
  (`scripts/esod_baseline/reorganize_seaperson.py`, modeled on the
  already-existing `reorganize_uavdt.py`) rather than a scratchpad symlink
  patch, producing a self-contained `seaperson_v2/{images,labels,masks}/{split}/`
  tree that `data/seaperson.yaml` points at directly.

**Roster (8 arms. First 6 confirmed distinct by the user, gated-fusion
added after concat-only's per-bucket trade-off below motivated adding
VisDrone's own long-defined but never-isolated E2.4 arm; concat+ISPPHead
added 2026-08-23 after concat+SABL showed no clean accuracy win over
concat-only -- isolates ISPPHead's own efficiency gain from SABL's
mixed/inconclusive effect, see the interpretation below):**

| Arm | Selector | Loss | Box |
|---|---|---|---|
| R0 | `Segmenter` | upstream | CIoU |
| semantic-only | `Segmenter` (same arch as R0) | coverage | CIoU |
| spectral-only | `SpectralOnlySegmenter` | coverage | CIoU |
| concat-only | `ChannelPooledConcatEvidenceSegmenter` | coverage | CIoU |
| gated-fusion | `ChannelPooledDualEvidenceSegmenter` | coverage | CIoU |
| concat+SABL | `ChannelPooledConcatEvidenceSegmenter` | coverage | SABL |
| concat+SABL+ISPPHead | `ChannelPooledConcatEvidenceSegmenter` + `ISPPHead` | coverage | SABL |
| concat+ISPPHead | `ChannelPooledConcatEvidenceSegmenter` + `ISPPHead` | coverage | CIoU |

R0 and semantic-only share the same model config (`seaperson_yolov5m.yaml`)
and differ only in `--selector-loss`, matching VisDrone's own E1.0/E2.1 pair
-- deliberately isolating the loss-function effect from the
selector-architecture effect. concat-only and concat+SABL likewise share one
config (`seaperson_yolov5m_channel_pooled_concat.yaml`), differing only in
`--box-loss`. gated-fusion shares concat-only's evidence branches (same
channel-pooled spectral filter, same semantic head) via
`seaperson_yolov5m_channel_pooled_dual_evidence.yaml`, differing only in the
fusion op (`GatedEvidenceFusion`'s learned sigmoid gate vs. concat-only's
fixed 1x1 conv over concatenated logits) -- isolates the fusion mechanism
itself, holding evidence sources fixed, directly testing whether a gate can
recover concat-only's Tiny/Small/Medium-Large losses without giving up its
Very Tiny gain.

`spectral-only` (full-channel `SpectralOnlySegmenter`, unlike the
channel-pooled variant the concat/gated arms use) OOM'd at the shared
batch=8 and at batch=4; it trained successfully only at batch=2, a
documented protocol deviation for this one arm (SS1). Given its result
below already lands essentially on par with semantic-only despite the
handicap, the smaller batch is not an obvious confound in its favor.

**Status: first 7 arms complete (2026-08-23), all results downloaded and
independently re-verified against the raw logs (`results/seaperson_raw/`)
-- see the Accuracy + efficiency table's note for the one correction that
came out of that audit (Params methodology). 8th arm (concat+ISPPHead)
added 2026-08-23, queued/in progress.**

### 8.1 Results (test split unless noted; updated as each arm completes)

**Size-bucket recall** (`audit_buckets.py`, test split, 300375 total GT
targets: 82417 Very Tiny / 174816 Tiny / 42987 Small / 155 Medium-Large):

| Arm | Very Tiny (<16x16) | Tiny (16-32) | Small (32-96) | Medium/Large (>96) | Total |
|---|---:|---:|---:|---:|---:|
| R0 | 74.03% (61014/82417) | 86.78% (151707/174816) | 94.74% (40725/42987) | 79.35% (123/155) | 84.42% (253569/300375) |
| semantic-only | 75.49% (62217/82417) | 91.57% (160087/174816) | 95.71% (41141/42987) | 81.94% (127/155) | 87.75% (263572/300375) |
| spectral-only | 75.23% (62006/82417) | 91.69% (160286/174816) | 95.89% (41220/42987) | 86.45% (134/155) | 87.77% (263646/300375) |
| concat-only | 76.00% (62637/82417) | 91.50% (159962/174816) | 95.68% (41131/42987) | 80.00% (124/155) | 87.84% (263854/300375) |
| gated-fusion | 75.49% (62219/82417) | 90.79% (158711/174816) | 95.50% (41052/42987) | 86.45% (134/155) | 87.26% (262116/300375) |
| concat+SABL | 76.67% (63193/82417) | 91.49% (159935/174816) | 94.77% (40740/42987) | 80.65% (125/155) | 87.89% (263993/300375) |
| concat+SABL+ISPPHead | 77.19% (63619/82417) | 90.70% (158562/174816) | 94.89% (40792/42987) | 83.23% (129/155) | 87.59% (263102/300375) |
| concat+ISPPHead | pending | | | | |

**Accuracy + efficiency** (mAP/P/R/BPR/Occupy from `test.py --task test`;
Params/GFLOPs/FPS/latency from `test.py --task measure`, valid split,
batch=1; all values below independently re-verified 2026-08-23 against the
downloaded raw `*_test.log`/`*_measure.log`/`buckets.json` artifacts, see
`results/seaperson_raw/`). **Params correction (2026-08-23):** this table
previously listed an architecture-only Params estimate for the first 6 arms
(`sum(p.numel() for p in Model(cfg).parameters())`, computed before their
`--task measure` was re-run) that did not match the real measured value --
`test.py` loads every checkpoint through `attempt_load()`, which calls
`.fuse()` (`models/yolo.py` `Model.fuse()`, folding each Conv2d+BatchNorm2d
pair into one Conv2d, the "Fusing layers..." line in every log), so the
real deployed parameter count is systematically lower (by ~22.8K here) than
counting an unfused `Model(cfg)` naively. Every arm below now uses the
`buckets.json`/measure-log value directly, matching how a paper reports
deployed-model parameters, not the training-time graph. GFLOPs matched
exactly between the two computations (architecture-only, unaffected by
fusion's parameter-count change); FPS/latency below are also refreshed to
this same 2026-08-23 remeasurement pass for all 6 (consistent with each
other and with concat+SABL+ISPPHead, which was already a single live
measurement) rather than mixing an older single R0 measurement in:

| Arm | mAP@.5 | mAP@.5:.95 | P | R | BPR | Occupy (test) | Params (M) | GFLOPs | FPS | Latency (ms, infer/NMS/total) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.750 | 0.320 | 0.827 | 0.696 | 0.947 | 0.228 | 35.78 | 202.4 | 85.7 | 11.7/1.2/12.9 |
| semantic-only | 0.769 | 0.325 | 0.825 | 0.708 | 0.991 | 0.363 | 35.78 | 266.8 | 73.5 | 13.6/1.2/14.8 |
| spectral-only | 0.770 | 0.327 | 0.829 | 0.705 | 0.992 | 0.335 | 35.94 | 267.4 | 71.7 | 14.0/1.2/15.2 |
| concat-only | 0.772 | 0.326 | 0.825 | 0.707 | 0.986 | 0.374 | 35.79 | 281.2 | 69.8 | 14.3/1.3/15.6 |
| gated-fusion | 0.765 | 0.324 | 0.824 | 0.704 | 0.990 | 0.330 | 35.86 | 261.5 | 74.3 | 13.5/1.2/14.7 |
| concat+SABL | 0.771 | 0.323 | 0.820 | 0.713 | 0.990 | 0.348 | 35.79 | 263.7 | 73.4 | 13.6/1.3/14.9 |
| concat+SABL+ISPPHead | 0.769 | 0.323 | 0.815 | 0.716 | 0.990 | 0.340 | 25.92 | 209.1 | 77.1 | 13.0/1.2/14.2 |
| concat+ISPPHead | pending | | | | | | | | | |

**Very Tiny miss-reason breakdown** (`vt_diagnose.py`; percentages are of
that arm's own missed-Very-Tiny count):

| Arm | Missed Very Tiny GT | `right_class_low_iou` (localization failure) | `no_nearby_prediction` (selector-dropped) | `matched_but_stolen_by_other_gt` |
|---|---:|---:|---:|---:|
| R0 | 21388 | 74.3% | 25.6% | 0.1% |
| semantic-only | 20181 | 79.7% | 20.1% | 0.1% |
| spectral-only | 20395 | 79.7% | 20.1% | 0.2% |
| concat-only | 19760 | 80.6% | 19.2% | 0.2% |
| gated-fusion | 20181 | 79.5% | 20.4% | 0.1% |
| concat+SABL | 19215 | 79.6% | 20.3% | 0.1% |
| concat+SABL+ISPPHead | 18780 | 79.8% | 20.1% | 0.1% |
| concat+ISPPHead | pending | | | |

**Interpretation (7 of 8 arms; concat+ISPPHead queued):**
semantic-only (same `Segmenter` architecture as R0, coverage loss instead
of upstream weighted BCE) improves every accuracy metric over R0, isolating
the loss-function effect cleanly since the selector architecture is
unchanged. As with UAVDT's concat comparison (SS7), Occupy increased
alongside accuracy -- the same "smarter selection vs. selecting more area"
caveat applies here and is not yet isolated. Across all six coverage-loss
arms, localization failure dominates missed Very Tiny objects, not selector
drops -- coverage supervision reduces selector-caused misses (R0's 25.6% ->
19-20% for every coverage-loss arm, of a smaller total each time) and
shifts the remaining bottleneck further toward detection-head localization
precision at very tiny scale. This suggests headroom in this roster is more
likely in box-regression quality (SABL) than in further selector tuning
alone -- confirmed only partially: concat+SABL's own localization-failure
share barely moved (see its interpretation below), so this remains an open
question rather than a confirmed explanation.

**spectral-only is essentially on par with semantic-only** (total recall
87.77% vs 87.75%, Very Tiny recall ~75.2% vs ~75.5%, mAP@.5 0.770 vs 0.769)
despite using no semantic evidence at all, and despite training at 1/4 the
batch size -- spectral evidence alone, with the coverage loss, recovers
about as many real tiny targets as the semantic-only baseline architecture
does. This is a genuine H2 data point (spectral evidence can substitute
for, not just supplement, semantic evidence for tiny-target selection).
Medium/Large recall is notably higher for spectral-only (86.45% vs
79.35%/81.94%), but N=155 GT is small enough (a 7-11 detection swing moves
the percentage several points) that this should not be over-interpreted
without a multi-seed check.

**concat-only did not clearly beat either single-evidence arm on aggregate,
and per-bucket it is a genuine trade-off, not a uniform improvement.**
Total recall (87.84%) is only +0.07pp over spectral-only (87.77%) and
+0.09pp over semantic-only (87.75%); mAP@.5 (0.772) is +0.002/+0.003 over
the same two arms. Breaking the aggregate down by bucket (recalled counts):
concat wins Very Tiny by a real margin (62637 vs 62006 spectral-only vs
62217 semantic-only, +420 to +631 objects), but *loses* to **both**
single-evidence arms on Tiny (159962, worst of the three), Small (41131,
worst), and Medium/Large (124, worst). The small aggregate win exists only
because Very Tiny's gain outweighs the losses in every larger bucket
combined. Read plainly: concat is not "semantic + spectral, strictly
better" -- it is a redistribution of selector sensitivity toward the very
smallest objects at a measurable cost everywhere else, and the net is only
barely positive. This is an order of magnitude smaller than the
R0-to-single-evidence jump (+3.3-3.4pp total recall from adding coverage
loss alone, architecture unchanged) -- essentially all of this roster's
gain so far comes from switching to the coverage loss, not from fusing
evidence. It is also not free: Occupy 0.374/0.488, higher than
spectral-only's 0.335/0.420 and close to semantic-only's 0.363/0.457. This
is a genuine, structured data point for the "dual-evidence fusion" framing
-- a real Very-Tiny-specific effect, not noise (both audit tools agree to
within 0.02pp), but not the unqualified win a fusion-improves-everything
narrative would need. Matches the falsifiable spirit of H2/H3 in
`HESOD-Proposal.md` SS6. concat+SABL and concat+SABL+ISPPHead may still
separately justify themselves on box-regression and head-efficiency grounds
respectively, but neither should be framed as validating fusion-over-single-
evidence unless their own per-bucket deltas over concat-only (not over R0)
show it.

**gated-fusion refutes the hypothesis that motivated it: the learned gate
does not recover concat-only's Tiny/Small/Medium-Large losses, and gives up
part of its Very Tiny gain too.** Same evidence branches as concat-only
(SS8, `ChannelPooledDualEvidenceSegmenter` vs. `ChannelPooledConcatEvidenceSegmenter`),
only the fusion op differs -- a learned per-location sigmoid gate instead
of a fixed 1x1 conv over concatenated logits. Every accuracy metric is
worse than concat-only: total recall 87.26% vs 87.84% (-0.58pp), Very Tiny
75.49% vs 76.00% (-0.51pp), Tiny 90.79% vs 91.50% (-0.71pp), mAP@.5 0.765
vs 0.772. gated-fusion is in fact the worst of all five coverage-loss arms
on total recall and mAP@.5, beating only R0's plain upstream-loss baseline
-- worse than semantic-only and spectral-only alone, not just worse than
concat-only. Occupy (test) dropped further still (0.330, lowest of any
coverage-loss arm) alongside GFLOPs (261.5, also lowest) and the highest
FPS (74.3) -- the gate learned to suppress evidence and route to less area
than the fixed concatenation does, and that reduction cost accuracy across
almost every bucket rather than sharpening selection. The one bucket where
it wins is Medium/Large (86.45% vs 80.00%), but N=155 GT makes that a
10-detection swing, not a reliable signal. Read plainly: for this task, a
fixed, unconditional fusion of both evidence sources beats a learned gate
that gets to suppress one -- the gate's extra flexibility did not pay for
itself. This is a genuine negative result for the fusion-mechanism
ablation, not a data or pipeline issue (both audit tools agree to within
0.02pp, same as every other arm).

**concat+SABL over concat-only (its matched control, same selector/evidence,
only the box loss changed) is a small, mixed result, not a clean win.**
Very Tiny recall improves (+0.67pp, 62637->63193) and total recall ticks up
(+0.05pp, 263854->263993), but Small recall drops (-0.91pp, 41131->40740)
and mAP@.5/mAP@.5:.95 are flat-to-slightly-down (0.772->0.771,
0.326->0.323) -- box-loss precision gains at the smallest scale did not
translate into an aggregate accuracy improvement. The one clear, consistent
change is efficiency: Occupy (test) dropped from 0.374 to 0.348 and
measured GFLOPs from 281.2 to 263.7 (FPS 69.8->73.4), even though only the
regression loss changed and the selector's own supervision
(`--selector-loss coverage`) is identical between the two arms -- a real,
if indirect, training-dynamics interaction between the box and selector
losses, not something to over-read as SABL "learning to select better."
Very Tiny miss-reason mix is essentially unchanged (localization failure
80.6%->79.6%, selector-dropped 19.2%->20.3%, on 545 fewer total misses),
so SABL is not resolving the localization-failure bottleneck the SS8.1
interpretation above flagged as the roster's main remaining headroom.

**concat+SABL+ISPPHead over concat+SABL (its matched control -- same
selector, same evidence, same box loss, only `YOLOv6Head` -> `ISPPHead`)
cleanly isolates the head swap, and confirms it: a large parameter/compute
saving at essentially no accuracy cost.** Params drops 27.6% (35.79M ->
25.92M) and measured GFLOPs 20.7% (263.7 -> 209.1), with FPS up modestly
(73.4 -> 77.1). Accuracy is a wash, not a win or a loss: total recall dips
slightly (-0.30pp, 87.89% -> 87.59%), mAP@.5 is flat-to-slightly-down
(0.771 -> 0.769), but Very Tiny recall actually improves (+0.52pp, 76.67%
-> 77.19%) and the Very Tiny miss-reason mix stays effectively unchanged
(localization failure 79.6% -> 79.8%, on 435 fewer total misses) -- so, as
with concat+SABL itself, the lighter head does not resolve the
localization-failure bottleneck either; it just gets to the same accuracy
for meaningfully less compute. This is the cleanest isolation of the
ISPPHead effect in the whole project (UAVDT's SS7 comparison bundled the
head swap with the selector and loss changes at once; SS9.1's still-open
gap is about isolating it against plain R0, not against this SABL+concat
combo) -- and it matches the 27.5% UAVDT reduction (SS7) closely, a useful
cross-dataset consistency check for ISPPHead's efficiency claim.

**Against R0 -- the roster's actual start-to-end comparison -- the full
recipe is not simply "better and faster."** concat+SABL+ISPPHead beats R0
on every accuracy axis (total recall 87.59% vs 84.42%, +3.17pp; mAP@.5
0.769 vs 0.750) with 27.6% fewer parameters (25.92M vs 35.78M), but
measured GFLOPs is slightly *higher* (209.1 vs 202.4) and FPS is *lower*
(77.1 vs 85.7, -10.0%) -- because Occupy is much larger throughout this
roster than R0's (0.34 here vs R0's 0.228), the concat selector routes
substantially more image area through the shared backbone/neck, and that
cost outweighs what the lighter head saves. Fewer parameters did not
translate into a net latency win here; the accuracy gain came partly from
processing more of the image, the same "smarter selection vs. selecting
more area" caveat flagged for UAVDT (SS7) and never isolated in this
roster either. Report both numbers together -- neither "the final method
has 28% fewer parameters" nor "the final method is faster than R0" is true
in isolation without the other.

## 9. Known gaps: missing head-swap controls and competitor baselines

Audited directly (grep over every model cfg, `results/`, and the codebase),
not from memory, in response to a direct question about whether the paper's
claimed comparisons actually exist. Recorded here as placeholders so the gap
stays visible and tracked rather than silently assumed-covered; none of
these are built or scheduled yet -- SeaPerson's roster (SS8) is the current
priority.

### 9.1 "ESOD + ISPPHead" head-only control -- missing for every dataset

Every dataset's "final method" arm (`*_channel_pooled_concat_sabl_isphead`)
is currently compared only against plain R0, which bundles three
simultaneous changes at once: selector architecture (`Segmenter` ->
`ChannelPooledConcatEvidenceSegmenter`), loss (`upstream` -> `coverage` +
`sabl`), and head (`YOLOv6Head` -> `ISPPHead`). There is no arm anywhere
that swaps only the head, holding R0's selector and loss fixed, so none of
the reported "+ISPPHead" gains can currently be attributed to the head swap
specifically rather than the selector/loss changes bundled alongside it.
Confirmed via `grep -rl ISPPHead hesod/backends/hesod/models/cfg/esod/*.yaml`
-- exactly 3 configs use it (SeaPerson, TinyPerson, UAVDT; none for
VisDrone), and every one pairs it with
`ChannelPooledConcatEvidenceSegmenter`, never plain `Segmenter`.

**Placeholder arm, per dataset:** `{dataset}_yolov5m_baseline_isphead`
-- R0's exact config (plain `Segmenter`, `--selector-loss upstream
--box-loss upstream`) with only `Detect, [nc, anchors, 'ISPPHead']` swapped
in for the head. Cheap to build once scheduled: no new code, `ISPPHead`
already exists and is wired into `parse_model`'s dispatch; one new `.yaml`
per dataset plus one `run_arm()` line. Not yet built for any dataset.

### 9.2 "BCRS selector + original head" -- incomplete outside SeaPerson

SeaPerson's `concat-only` (SS8.1, `ChannelPooledConcatEvidenceSegmenter`,
coverage loss, CIoU, default `YOLOv6Head`) is the only complete instance of
this control across all four datasets:

- **VisDrone**: defined as R2/E2.9 in SS3.1/3.2, but never actually trained
  -- `results/visdrone/` has only `baseline`, `channel_pooled_concat_sabl`
  (already bundled with SABL), and `reliability_gate_sabl`. No plain
  concat-only (no SABL) result exists to retrieve; this needs a real
  training run against the already-existing
  `visdrone_yolov5m_channel_pooled_concat.yaml` config, not new code.
- **UAVDT**: never run in isolation -- SS7 only ever trained R0 and the
  fully-bundled `concat_sabl_isphead`. Revisit that skip now that
  SeaPerson's own concat-only result turned out to be non-obvious (SS8.1's
  per-bucket trade-off) rather than assuming UAVDT would show the same
  pattern.
- **TinyPerson**: only concat**+SABL** exists (the H1a/H1b head comparison
  in `HESOD-Lightweight-Detector-Review-and-Roadmap.md` SS5.2 used it as the
  baseline row); a plain concat-only (no SABL) arm was never isolated.

### 9.3 QueryDet, Faster R-CNN, RetinaNet

Zero training or evaluation of any of these three had happened in this
project under our own protocol as of the previous pass through this
section. Every `grep` hit for these names resolved to either the
*pristine, vendored* TinyPerson benchmark toolkit
(`hesod/backends/esod/evaluation/tiny_benchmark/`, an unmodified Detectron1-
era `maskrcnn_benchmark` codebase shipped with the original ESOD release,
never adapted or run by us) or GPViT's own reference configs -- not
something integrated with our datasets, splits, or eval tooling.
`HESOD-Proposal.md` SS7.3 lists all three as required baselines
conceptually, but they are not equally cheap to add:

- **Faster R-CNN and RetinaNet** ship as ready-to-use COCO-pretrained
  detectors in `torchvision.models.detection` -- no custom framework
  integration needed. Infra now exists under `hesod/backends/baseline/`
  (new 2026-08-22, UAVDT + SeaPerson only, VisDrone/TinyPerson deferred but
  not blocked -- see that directory's README.md for full protocol notes:
  COCO-pretrained fine-tune matching this project's own convention,
  torchvision resize geometry instead of YOLOv5 letterboxing, pycocotools
  COCOeval as a separate code path from this doc's `ap_per_class()` mAP
  column, `fvcore`-based GFLOPs). `scripts/esod_baseline/audit_buckets.py`
  and `vt_diagnose.py` run against it completely unmodified, so its
  recall-bucket numbers are directly comparable to every arm in SS7/SS8.
  **Pending its first run** (GPU busy finishing SeaPerson's
  `concat+SABL+ISPPHead`, SS8) -- no results yet, placeholder rows only:

  | Arm | Very Tiny recall | Total recall | mAP@.5 | mAP@.5:.95 | GFLOPs | FPS |
  |---|---:|---:|---:|---:|---:|---:|
  | uavdt_fasterrcnn | pending | | | | | |
  | uavdt_retinanet | pending | | | | | |
  | seaperson_fasterrcnn | pending | | | | | |
  | seaperson_retinanet | pending | | | | | |

- **QueryDet** has no off-the-shelf package -- it needs a real
  from-scratch implementation (its sparse/selective inference mechanism is
  the whole point of comparing against it, not something a pretrained
  torchvision model can stand in for). Getting it onto common footing (same
  data conversion, same evaluator, same hardware) is a multi-day
  integration effort, a different class of work than everything else in
  this document. Not scheduled; still requires its own explicit separate
  scoping decision before starting.
