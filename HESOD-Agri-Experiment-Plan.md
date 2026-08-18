# HESOD-Agri Experiment Plan

> **Companion proposal:** [HESOD-Agri-Proposal.md](HESOD-Agri-Proposal.md)  
> **Method source:** [HESOD-Agri-Proposal.md](HESOD-Agri-Proposal.md) SS4.2 (reliability-aware residual gate)
> **Datasets in scope:** AgriPest, Pest24, GWHD 2021 only  
> **Primary question:** Does a reliability-aware semantic–spectral gate improve the detection accuracy–compute Pareto frontier, and does it do so *specifically* by conditioning spectral evidence on semantic uncertainty rather than treating it as a co-equal signal (plain concat)?  
> **Status (2026-08-18):** All six core Pest24 arms (R0–R3, F5, Gate+SABL) plus both pre-registered F6 attempts are complete and audited (SS11.2.1–SS11.2.8). **The F6 gate line is closed**: attempt 2 (retry) set new project-best AP/AP50/total-recall/TFR but still fell 1.34pp short of F5's Very Tiny recall, failing the one condition (SS11.2.7/SS11.2.8) the mechanism was pre-registered to satisfy -- per the rule, no further retries. **Pest24's final fusion-ladder answer is R3 (concat+SABL)**: best Very Tiny recall (66.86%) of any arm tried, second-best AP@.5:.95 (0.349, behind F6 attempt 2's 0.352). **Separately, and more consequentially: A0 (dense, no routing at all, SS11.2.9) beats every routed/gated arm on AP/AP50/total-recall/Very-Tiny-recall AND measured FPS, confirmed at matched 640 resolution against concat+SABL** -- repeated, same-resolution evidence that spatial routing is not earning its computational keep on Pest24 specifically (low native resolution + extreme density), not a single anomalous measurement. Pest24 experimentation is effectively paused pending a decision to move to AgriPest or back to VisDrone/UAVDT/TinyPerson (where the same A0-style dense-vs-routed check has not yet been run). AgriPest and GWHD 2021 have not started. No result is claim-bearing until it meets the three-seed rule in SS7.

## 1. Non-negotiable protocol

### 1.1 Scope

- Do not add UAV, biomedical, or generic COCO results to this plan.
- Do not merge the three datasets or their class spaces.
- Train a separate detector/checkpoint for each dataset and arm.
- Use official/published splits. Any diagnostic subset is reported in addition to—not instead of—the official test/validation set.
- Freeze the test split until model, threshold, and budget decisions have been made on training/validation data.

### 1.2 Metric names

- `AP` = COCO-style mean AP at IoU 0.50:0.05:0.95.
- `AP50` = AP at IoU 0.50. It must never be written into an `AP`/`mAP` field without the suffix.
- `AP75`, `AP_small`, per-class AP, recall@0.50, and recall@0.75 are separate fields.
- Custom recall uses class-aware, one-to-one prediction/ground-truth matching.
- The report always includes evaluated image, ground-truth, prediction, class, and empty-image counts.

### 1.3 Routing metrics

For selected region mask `M` and ground-truth box `G`:

- `BPR_box = mean(area(G ∩ M) / area(G) > 0.5)`;
- `BPR_ctr = mean(center(G) ∈ M)`;
- `occupancy = selected routing cells / 64` for the 8×8 router;
- `oracle_BPR@K` uses an explicit, deterministic oracle selection algorithm and the same coverage definition;
- report the K distribution and zero-patch frequency, not just the mean.

`BPR_box` is not IoU. The union with the selected region must not appear in its denominator.

### 1.3.1 Gate-specific metrics (F2–F6 arms only)

- `TFR(B) = |selected regions ∩ textured-background regions| / max(1, |selected regions|)` at a fixed budget `B` — Pest24's specific admission criterion (HESOD-Agri-Proposal.md SS2/SS9); a texture-hard-negative region set `B_tex` must be defined per dataset before this is computed, not post-hoc;
- gate activation `a_i` distribution, reported separately for tiny-object regions and `B_tex` regions — not a single global mean;
- conditional spectral lift: F5/F6's recall or coverage minus F1's, computed **within** each objectness x texture-risk bucket (SS1.3.2), not as a global average that can hide a mechanism failure behind an aggregate win;
- priority-coverage rank correlation: Spearman correlation between $u_i$ and the GT-coverage target, per dataset.

### 1.3.2 Difficulty buckets for gate analysis

In addition to size/density/objectness buckets used elsewhere in this plan, gate-specific analysis (SS1.3.1, SS6.2) requires a **2D cross-tab**: $q_i$ quantile (low/mid/high) x texture-risk quantile (low/mid/high), 3x3, reporting tiny recall, background false-routing rate, and mean gate activation in each cell — not two separate 1D means, which can average away exactly the conditional behavior the gate is meant to exhibit.

### 1.4 Systems metrics

- end-to-end latency: batch 1 and one declared throughput batch;
- router and detector latency where separable;
- peak VRAM, processed local pixel area, parameters, and reliable MACs/FLOPs;
- GPU/CPU, software version, precision, image resolution, warm-up count, timed iterations, synchronization, and dataloader inclusion.

All accuracy–compute comparisons use the same timing contract and native information policy.

## 2. Dataset preparation and admission

### 2.1 Shared preflight audit

Before any training, generate `dataset_audit.json` and a short Markdown report containing:

1. source URL/version, archive checksum, licence/redistribution note, and split source;
2. image count by split, class count/names, boxes/class, empty images, corrupt images;
3. invalid, zero-area, out-of-bounds, clipped, and duplicate boxes;
4. native width/height and box width/height/area distributions;
5. objects/image and class imbalance distributions;
6. native-resolution COCO size strata (`small < 32^2`, `medium < 96^2`, `large ≥ 96^2`);
7. 8×8 positive-cell occupancy and oracle BPR@K = 8, 16, 32;
8. conversion manifest mapping each output image/label to its raw source and original annotation ID.

Required gates:

- class IDs are contiguous and match the data YAML;
- all retained boxes are finite, positive-area, and in bounds;
- official image IDs and split identity survive conversion;
- empty images are retained unless the official protocol explicitly excludes them;
- a visual sample covers every class plus dense, sparse, tiny, boundary, and empty cases.

### 2.2 AgriPest

Expected characteristics from the source paper: 49,700 field images, 264,700 instances, 14 pest species, four crops, and annotated scene-condition analyses.

Preparation requirements:

- record the exact official split files and category mapping;
- preserve condition metadata for dense/sparse, illumination variation, and background clutter when available;
- quantify whether boxes are already clipped or contain ignored/difficult flags;
- generate deterministic box-derived routing targets after geometric augmentation, never before it;
- report native and training-resolution box size distributions.

Dataset-specific result table:

- overall AP/AP50/AP75/AP_small;
- per-class AP and recall;
- dense vs sparse;
- normal vs challenging illumination/background, only where faithfully reproducible;
- detection count error per image as a secondary monitoring metric.

### 2.3 Pest24

Expected characteristics from the source literature: roughly 25,000 high-resolution light-trap images, 24 pest classes, dense and adhesive targets, and high inter-class visual similarity.

Preparation requirements:

- verify the exact download, licence, class mapping, and published split before writing a converter;
- do not assume that papers named “AgriPest-YOLO” used the AgriPest dataset—the 24-class light-trap dataset is a different benchmark;
- preserve full native resolution and document any tiled loading policy;
- measure object adjacency/crowding and the fraction of images whose positive-cell occupancy approaches 64/64;
- treat published mAP/mRecall numbers as contextual unless their IoU thresholds, split, and preprocessing match exactly.

Pest24 was originally scoped as the routing stress test: if oracle BPR requires nearly all cells, that is evidence about the method boundary, not a reason to remove dense images. **2026-08-16 update:** in practice it has turned out to be the strongest positive validation case for the selector fix found anywhere in this project (SS11.2.2). Both roles hold at once — neither licenses removing dense images or cherry-picking easier ones.

### 2.4 GWHD 2021

Use the full published dataset/split. Prefer GlobalWheat-WILDS when making domain-generalization claims; otherwise name the exact Global Wheat Challenge 2021 split and WDA implementation.

Preparation requirements:

- retain acquisition-domain metadata;
- use the official one-class label and published train/validation/test identities;
- compute `AP_small` by native 1024×1024 instance area;
- ignore non-small ground truths correctly during small-stratum evaluation;
- never create a private “GWHD small images” replacement benchmark.

GWHD routing admission gate:

| Preflight outcome | Decision |
|---|---|
| Oracle BPR@32 ≥ 95% and median positive occupancy well below 32/64 | Run the full A0–O matrix |
| Full set is dense but a predeclared sparse diagnostic is routable | Run official full evaluation plus the diagnostic; frame as a boundary result |
| Sparse diagnostic also requires near-dense routing | Run A0/R0/R1/R2/R3 only if informative, or use GWHD for the CIoU–SABL localization comparison |

Optional diagnostic definition must be frozen before results. A defensible example is “at least one native small GT and ≤16 GT-positive routing cells”; it is secondary and never substituted for official performance.

## 3. Data configuration and planned artifacts

The following are planned paths, not evidence that the datasets have already been integrated:

```text
configs/experiments/agri/
  agripest_yolov5m.yaml
  pest24_yolov5m.yaml
  gwhd2021_yolov5m.yaml

hesod/backends/hesod/data/
  agripest.yaml
  pest24.yaml
  gwhd2021.yaml

hesod/backends/hesod/models/cfg/esod/
  agripest_yolov5m.yaml                         # F0 / R0-R1
  agripest_yolov5m_channel_pooled_concat.yaml    # F1 control (SS6.2)
  agripest_yolov5m_reliability_gate.yaml         # F5/F6, R2-R3 (SS4.2 of the proposal)
  pest24_yolov5m.yaml
  pest24_yolov5m_channel_pooled_concat.yaml
  pest24_yolov5m_reliability_gate.yaml
  gwhd2021_yolov5m.yaml
  gwhd2021_yolov5m_channel_pooled_concat.yaml
  gwhd2021_yolov5m_reliability_gate.yaml
  # F2-F4 (unconstrained gate / uncertainty-only / low-score-rescue) can share
  # the reliability_gate.yaml architecture with a gate-mode flag rather than
  # separate files, once the gate module supports it -- do not duplicate the
  # whole architecture YAML per ablation rung.

results/agri/<dataset>/<arm>/seed_<seed>/
```

Avoid duplicating architecture YAMLs if the model parser can safely override only `nc`. If files are duplicated, an automated structural diff must prove that the baseline and proposed YAMLs differ only in the declared selector modules/arguments and class count.

Each run directory must contain:

```text
resolved_config.yaml
command.txt
environment.json
git_state.json
checkpoint_manifest.json
metrics.json
routing_metrics.json
compute.json
predictions.json           # or equivalent lossless evaluator input
selected_regions.json      # exact regions used during inference
stdout.log
```

## 4. Input-resolution policy

Tiny-object conclusions are invalid if preprocessing destroys the targets. Before choosing a training size:

1. compute the box-size distribution after the proposed resize/letterbox;
2. report the fraction with width or height below 2, 4, and 8 pixels;
3. choose a resolution/tiled loader that preserves useful target evidence;
4. keep this policy fixed across arms within a dataset.

A0 and routed methods must receive comparable native information. A low-resolution full-image baseline cannot be called compute-matched to high-resolution routed patches without an explicit processed-pixel and information analysis.

**Pest24 lock.** `IMG_SIZE=1024` means a 1024-pixel long-side working resolution: training uses a 1024×1024 mosaic canvas, while `test.py`/measure use `rect=True` and therefore evaluate native 800×600 images as 1024×768 tensors. This 1.28× interpolation preserves an exact 8×8 routing grid but does not add native detail and was not selected by a resolution sweep. An unmodified 640 switch is invalid: the training slicer fails its P3 divisibility assertion and test-time routing changes from 64 to 56 candidate cells. Current R0–R3 comparisons remain internally fair; comparisons with 640/416/224 literature values remain contextual until rerun under a common resolution.

## 5. Routing-label contract

The core study uses only box-derived selector supervision:

- deterministic target creation from the transformed ground-truth boxes;
- one documented Gaussian/coverage formulation;
- identical target generator for R0–R3;
- no SAM, pseudo-segmentation, or hybrid-label preprocessing in the core matrix.

This isolates selector evidence and box loss. Hybrid labels may be proposed later but require their own label-quality and cost ablation.

## 6. Experiment matrix

### 6.1 Core arms

| ID | Spatial allocation | Selector | Selector loss | Box loss | Retrain? | Purpose |
|---|---|---|---|---|---|---|
| A0 | Full image or exhaustive 64-cell local processing | None | N/A | CIoU | Yes | Dense accuracy reference |
| A1 | Uniform fixed-K | None | N/A | CIoU | Yes/matched detector | Compute-matched allocation control |
| R0 | Threshold and declared Top-K sweep | Semantic only (F0) | Upstream selector loss | CIoU | Yes | Reproduced HESOD baseline |
| R1 | Same as R0 | Semantic only (F0) | Upstream selector loss | SABL | Yes | Loss-only effect |
| R2 | Same as R0 | Reliability-aware residual gate (F5/F6, SS6.2) | Coverage selector loss + rescue-ranking + conditional-gate reg. | CIoU | Yes | Proposed selector-only effect |
| R3 | Same as R0 | Reliability-aware residual gate (F5/F6, SS6.2) | Coverage selector loss + rescue-ranking + conditional-gate reg. | SABL | Yes | Combined effect and interaction |
| O | Ground-truth oracle Top-K | Oracle | N/A | CIoU detector | No deployable claim | Routing upper bound |

R2/R3's selector changed from concat to the gate on 2026-08-15 (SS12). Concat is F1 in the fusion ablation ladder below, run and validated on AgriPest *before* R2/R3 are trained — per HESOD-Agri-Proposal.md SS9's falsification conditions, if F5/F6 does not beat parameter/latency-matched F1 on low-objectness-tiny-recall or textured-background false-routing, R2/R3 should be understood to fall back to F1, and that fallback must be stated explicitly wherever results are reported.

The primary factorial is:

| | CIoU | SABL |
|---|---:|---:|
| Semantic only (F0) | R0 | R1 |
| Reliability-aware gate (F5/F6) | R2 | R3 |

Interpretation:

- `R2 − R0`: selector contribution;
- `R1 − R0`: SABL effect on semantic baseline;
- `R3 − R2`: SABL effect with the proposed selector;
- `(R3 − R2) − (R1 − R0)`: selector–loss interaction.

### 6.2 Fusion ablation ladder (justifies the R2/R3 selector choice)

This ladder is not secondary/optional the way the old S1–S3 table was — it is the evidence that the gate (not concat) belongs in R2/R3. The intended order is AgriPest first, promoting to Pest24/GWHD only once F5/F6 shows the expected pattern there (HESOD-Agri-Proposal.md SS4.2/SS9). **In practice this order was not followed**: Pest24's pipeline was ready first and AgriPest preparation has not started (SS2.2 vs. SS2.3), so F1 was run and validated on Pest24 first (SS11.2.2), and F5/F6 will run there next. This is a recorded deviation, not a silent substitution — the gate still must clear the ladder before promotion into R2/R3 on whichever dataset it is tested on first, and the three-dataset claim (Proposal SS9) still requires AgriPest and GWHD results before it can be made. All rungs use CIoU box loss only (SABL is crossed with F0 and F5/F6 in the R0–R3 factorial, not with every fusion variant — do not multiply the full ladder by every loss/dataset/seed).

| ID | Fusion | Definition | Question |
|---|---|---|---|
| F0 | Objectness-only | $u_i = \mathrm{logit}(q_i)$ | Semantic baseline (= R0) |
| F1 | Concat -> MLP | $u_i = f([q_i, s_i])$ | Capacity-matched control — the previous plan's proposed method |
| F2 | Unconstrained learned gate | $a_i = \sigma(\mathrm{MLP}[q_i, s_i])$ | Does gate structure alone (no uncertainty/confidence/texture inputs) beat concat? |
| F3 | Uncertainty-only gate | $a_i = h_i$ (binary entropy of $q_i$) | Does "intervene exactly when semantic is uncertain" work on its own? |
| F4 | Low-score rescue gate | $a_i = (1-q_i)^\gamma$ | Isolates low-objectness rescue; expected to raise textured-background false-routing (predicted failure mode, not a strawman — report it even if confirmed) |
| F5 | Reliability-aware residual gate | $a_i = \sigma(\mathrm{MLP}[q_i, h_i, c_i^{spec}, t_i^{bg}])$, $u_i = \mathrm{logit}(q_i) + \alpha \cdot a_i \cdot z_i^{spec}$ | **Proposed selector** |
| F6 | F5 + rescue-ranking + conditional-gate regularization + coverage | full objective, HESOD-Agri-Proposal.md SS4.2.2 | **Full proposed method**, used in R2/R3 once validated |

Requirements for every rung (BCRS SS7.4): parameter count, channel width, input resolution, and training epochs matched across F0–F6; report new MACs and measured latency per rung, including the confidence/texture-risk heads' own cost for F5/F6, not just the gate MLP. Do not promote F5/F6 into R2/R3 (SS6.1) until this ladder confirms F5/F6 beats parameter-matched F1 on both low-objectness-tiny-recall and textured-background false-routing rate (SS8.4) — that comparison, not intuition, is what licenses the selector choice.

### 6.2.1 Gate implementation lock

Parallel to SS7's SABL implementation lock, for F5/F6:

- $h_i$ = normalized binary entropy of $q_i$ (peaks at $q_i=0.5$);
- gate MLP inputs, as implemented in `ReliabilityGateMLP`: $[q_i, h_i, c_i^{spec}, t_i^{bg}]$. Budget conditioning $e_B$ is an unimplemented future extension; dropping any of the four locked inputs requires renaming the arm;
- $u_i = \mathrm{logit}(q_i) + \alpha \cdot a_i \cdot z_i^{spec}$, evaluated in logit space, not probability space;
- $z_i^{spec}$ must be able to take negative values (background suppression), verified by inspecting its sign distribution before trusting any F5/F6 result;
- rescue-ranking pairs: $\mathcal{P}_{rescue} = \{(i,j): y_i=1, q_i<\tau_{low}, y_j=0, j\in\mathcal{B}_{tex}\}$, loss $\mathcal{L}_{rescue} = \frac{1}{|\mathcal{P}_{rescue}|}\sum \log(1+\exp(m-u_i+u_j))$;
- conditional-gate regularization: $\mathcal{L}_{cond} = \frac{1}{|\mathcal{C}_{sem}|}\sum_{i\in\mathcal{C}_{sem}} a_i + \frac{1}{|\mathcal{B}_{tex}|}\sum_{i\in\mathcal{B}_{tex}} a_i$, small weight, must not overpower coverage/ranking supervision;
- **not locked, unlike SABL's constants:** $\tau_{low}$, $m$, $\lambda_{rescue}$, $\lambda_{cond}$, $\alpha$ require a validation sweep before the primary operating point is chosen. Candidate starting points (unvalidated): $\tau_{low}=0.3$, $m=1.0$, $\lambda_{rescue}=0.5$, $\lambda_{cond}=0.1$, $\alpha=1.0$.
- gate saturation check: report the distribution of $a_i$ on tiny-object and texture-background regions separately; $a_i \to 1$ or $a_i \to 0$ almost everywhere is a failure to report, not a result to hide (HESOD-Agri-Proposal.md SS9).

### 6.3 External baselines

No paper covers all three agricultural datasets under one protocol. Keep two tables: common-protocol reruns and original-protocol literature anchors. Never rank rows across those tables.

#### Controlled reruns

| Priority | ID | Baseline | Role |
|---|---|---|---|
| Mandatory | A0 | Dense YOLOv5m | Architecture-matched no-routing control |
| Mandatory | X1 | Faster R-CNN R50-FPN | Two-stage localization control |
| Mandatory | X2a | YOLOv5m + SAHI inference | Exhaustive uniform-slicing control using A0 |
| Mandatory | X4 | RTMDet-m | Modern dense accuracy/efficiency control |
| Optional | X2b | SAHI sliced fine-tuning + inference | Stronger uniform-tile baseline |
| Optional | X3 | QueryDet R50-FPN | Learned sparse high-resolution baseline; exclude if official code cannot be ported unchanged |

All direct rows use frozen image IDs/class maps, saved predictions, the same common evaluator, the same declared working resolution, and same-hardware cost measurement.

#### Original-protocol anchors

| Dataset / method | Reported result | Protocol status | Use |
|---|---|---|---|
| AgriPest FPN / Cascade R-CNN | AP50 70.20/70.83; AP 35.21/36.54 | Official 44,716/4,991 split; COCO-style metrics | Original-protocol sanity anchors |
| Pest24 AgriPest-YOLO | AP50 71.3; AP 46.9 | 12,701/5,077/7,600; 640 px; no public official code found | Reported only; do not reconstruct |
| Pest24 Pest-YOLO | IoU-0.5 mAP 69.59; recall 77.71 | Random 70/20/10; 416 px; public legacy code but incomplete artifacts | Optional controlled port |
| Pest24 Pest-PVT | VOC07 11-point AP50 77.24; recall 81.27 | 224 px; public config, missing split/class/pretraining artifacts | Priority controlled rerun |
| Pest24 TP-YOLO | AP 42.0; AP50 66.8 | Paper §3.3 Table 4; public code/checkpoint, missing Pest24 split YAML/result log | Priority controlled rerun |
| Pest24 DAMI-YOLOv8l | AP50 74.8; AP 53.7 | Public module fragments; no complete training stack/split/checkpoint | Optional reconstruction |
| GWHD Faster R-CNN / challenge best | WDA 0.492/0.700 | Official WDA protocol, not AP | Original-protocol context only |

Minimum Pest24 evidence is A0/X1 plus HESOD R0/R2/R3. Add TP-YOLO and Pest-PVT when their releases can be adapted with a complete patch manifest. Pest-YOLO and DAMI are lower priority; AgriPest-YOLO remains literature-only because no official code was found.

Every rerun must freeze the source commit, split manifest, class order, input policy, initialization, evaluator, prediction artifact, and patch manifest. A controlled port is reported as our rerun, never as an exact reproduction unless the original protocol is fully recovered.

## 7. Training protocol

Freeze per dataset before R0–R3:

- backbone and initialization;
- optimizer, learning-rate schedule, batch/effective batch, epochs, warm-up, EMA;
- augmentations and mosaic/mixup closing epoch;
- input policy, anchor handling, NMS, checkpoint selection metric;
- selector-target generator and selector-loss weights;
- seeds: minimum `{0, 1, 2}` for claim-bearing arms.

Fairness rules:

- R0/R1 and R2/R3 differ only in `box_loss` within their selector family;
- R0/R2 and R1/R3 use the same detector components except for declared selector modules and selector loss;
- SABL requires retraining and is never toggled only at evaluation;
- never reuse a checkpoint whose module graph or training target contract does not match the resolved config;
- failed/non-finite runs remain in the ledger with a reason; they are not silently replaced;
- F0–F6 (SS6.2) match parameter count, channel width, input resolution, and training epochs — an F5/F6 win that is actually just "more parameters than F1" is not a supported claim.

SABL implementation lock for this study:

- affects `lbox` only;
- ground-truth scale `s = sqrt(w_gt * h_gt)` in input pixels;
- scale mixing `mu = exp(-(s / 32)^6)` and `C = 12`, unless a separately named sensitivity experiment changes them;
- objectness target remains on the upstream CIoU path;
- no inference-cost difference is expected.

Gate implementation lock (F5/F6): see SS6.2.1 — unlike SABL's fixed constants, the gate's hyperparameters ($\tau_{low}$, $m$, $\lambda_{rescue}$, $\lambda_{cond}$, $\alpha$) require a validation sweep before the primary operating point is chosen; do not treat the candidate starting values there as pre-validated.

## 8. Validation and budget selection

### 8.1 Threshold mode

Sweep a fixed, predeclared validation grid, for example:

```text
hm_threshold ∈ {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70}
```

The value `0.5` is a candidate operating threshold, not an initialization-only constant. Choose the final point using validation AP under a predeclared cost or BPR constraint. Report the entire Pareto curve, not only the chosen point.

### 8.2 Exact Top-K mode

Evaluate:

```text
K ∈ {4, 8, 16, 24, 32, 48, 64}
```

Top-K is a proposed budget-control analysis. It must not be attributed to an upstream fixed-threshold baseline. Ties, K=0, empty heatmaps, and fewer-than-K valid cells require deterministic documented behavior.

### 8.3 Primary operating point

Before opening test labels/results, choose one of:

- a fixed end-to-end latency budget;
- a fixed processed-area budget;
- a routing-recall constraint such as validation `BPR_box ≥ 0.95`.

The same selection rule is applied to R0 and R2. Additional thresholds/K values form secondary curves.

## 9. Evaluation pipeline audit

### 9.1 Golden unit cases

The common evaluator and `audit_failure_cases.py` must pass synthetic cases for:

1. one perfect class-correct prediction → precision/recall 1;
2. duplicate predictions for one GT → one true positive, remaining duplicates false positives;
3. correct box but wrong class → false positive and false negative;
4. prediction/GT in different images → no match;
5. zero predictions with nonzero GT → recall 0, finite metrics;
6. zero GT with predictions → false positives, no undefined dataset aggregate;
7. all-empty images → documented finite/NA behavior;
8. boxes exactly at IoU 0.50 and just below/above;
9. selected region covering 50% exactly vs strictly greater than 50% for `BPR_box`;
10. a small-only metric with large GT ignored rather than treated as background.

### 9.2 Independent cross-check

For at least one small validation subset per dataset:

- run the native test evaluator;
- export the same predictions to a reference COCO evaluator;
- compare AP/AP50/AP75 and category/image counts within numerical tolerance;
- manually inspect all disagreements in TP/FP/FN assignment.

No experiment becomes “accepted” while metric mapping, class IDs, coordinate scaling, letterbox reversal, NMS, or ignored-area handling disagrees.

### 9.3 Routing artifact audit

Metrics must use the exact selected regions consumed by inference. Never reconstruct approximate patches from a heatmap after the run. Randomly sample at least 50 images per dataset and render:

- original boxes and predictions;
- heatmap and threshold/Top-K decision;
- selected regions after merge/expansion/clipping;
- uncovered GT, classified by size and boundary condition.

## 10. Execution phases and stop/go gates

### Phase 0 — Data legality and integrity

**Pest24 status (2026-08-16): complete** -- redistribution/licence verified, raw-to-canonical manifest built (`reorganize_pest24.py`), box/class/split audits passed (SS2.1/SS2.3). **AgriPest and GWHD 2021: not started.** The checkboxes below are dataset-general and are left unchecked because they do not yet hold for all three datasets; checking them would misstate AgriPest/GWHD progress.

- [ ] Confirm download and evaluation rights for all three datasets.
- [ ] Create raw-to-canonical manifests.
- [ ] Pass box/class/split audits and visual QA.

**Stop:** if Pest24 access/split cannot be verified, do not manufacture a benchmark. Replace no dataset without revising both proposal and plan.

### Phase 1 — Routing feasibility

**Pest24 status (2026-08-16): complete** -- occupancy/size/density stats measured (SS11.2.1's Occupy=0.0815, size-bucket table); Pest24 is in scope by default (SS2.3) so no formal admission gate applied (unlike GWHD); input resolution and timing contract frozen (1024 long-side working resolution -- 1024x1024 train canvas, 1024x768 rectangular test/measure input, see SS11.2.4 -- `hyp.pest24.yaml`). **AgriPest and GWHD 2021: not started** (GWHD's own admission gate, SS2.4, has not been evaluated).

- [ ] Compute size, density, occupancy, and oracle BPR curves.
- [ ] Apply the GWHD admission gate.
- [ ] Freeze input resolution and timing contract per dataset.

**Go:** proceed to full routing only where the oracle demonstrates a meaningful sparse-compute ceiling.

### Phase 2 — Baseline reproduction

**Pest24 status (2026-08-16): R0 complete** (SS11.2.1); A0/A1 not yet run. **AgriPest and GWHD 2021: not started.**

- [ ] Train/evaluate A0, A1, and R0.
- [ ] Verify nonzero predictions, nonzero selector targets, selected-region artifacts, and metric cross-checks.
- [ ] Record gaps to literature by split, resolution, training recipe, and metric—not as an undifferentiated “reproduction gap.”

**Stop:** do not launch proposed arms if R0 is broken, mapped to the wrong classes, or using a checkpoint from a different module graph.

### Phase 3 — Core factorial

**Pest24 status (2026-08-16): R1, R2, and R3 (F1 control family) complete, single-seed only; F5 in progress** (SS11.2.1-SS11.2.3). This ran on Pest24 first, not AgriPest as the checklist below assumes -- see SS6.2's reconciling note. Three-seed replication has not started for any arm on any dataset. **AgriPest and GWHD 2021: not started.**

- [ ] Train R1, R2, R3 with three seeds on AgriPest.
- [ ] Promote the validated matrix to Pest24.
- [ ] Run the admitted GWHD matrix.
- [ ] Build threshold and exact Top-K Pareto curves.

### Phase 4 — Secondary ablations and external baselines

- [ ] Run the remaining F0–F6 fusion ablation rungs (F2/F3/F4 -- F0/F1/F5/F6 are covered by the core arms above) on AgriPest first, per SS6.2. (Renamed from the old "S1-S3" secondary-selector-ablation table when SS6.2 was rewritten around the fusion ladder on 2026-08-15; this line previously still said "S1-S3", which no longer exists anywhere else in this document.)
- [ ] Run X1 Faster R-CNN R50-FPN on all three admitted datasets.
- [ ] Run X2a SAHI inference with the exact A0 detector/checkpoint on all three; add X2b only if budget permits.
- [ ] Port and validate X3 QueryDet on AgriPest, then promote the unchanged implementation to Pest24/GWHD where feasible.
- [ ] Run X4 RTMDet-m on all three admitted datasets.
- [ ] On Pest24, prioritize controlled TP-YOLO and Pest-PVT reruns; treat Pest-YOLO/DAMI as optional ports and AgriPest-YOLO as reported-only.
- [ ] Produce separate common-evaluator and original-protocol tables; never mix their metric columns.
- [ ] Add counting/density and domain analyses needed for the intended journal.

### Phase 5 — Paper acceptance gate

The main claim is viable only if:

1. R2 or R3 improves a measured AP–compute Pareto frontier over both R0 and A1 on AgriPest;
2. the direction is supported on Pest24 or its dense-regime limitation is quantified honestly;
3. results are stable across at least three seeds;
4. AP, routing recall, and system cost artifacts all pass audit;
5. GWHD either supports cross-task transfer or provides a predeclared, useful boundary result.

## 11. Result schema and tables

### 11.1 `metrics.json`

Minimum fields:

```json
{
  "dataset": "agripest",
  "split": "official_val",
  "arm": "R2",
  "seed": 0,
  "checkpoint_sha256": "...",
  "num_images": 0,
  "num_gt": 0,
  "num_predictions": 0,
  "ap_50_95": null,
  "ap50": null,
  "ap75": null,
  "ap_small": null,
  "recall50_one_to_one": null,
  "recall75_one_to_one": null,
  "router_latency_ms": null,
  "detector_latency_ms": null,
  "tfr": null,
  "gate_activation_mean_tiny": null,
  "gate_activation_mean_texture_bg": null,
  "conditional_spectral_lift": null,
  "priority_coverage_rank_corr": null
}
```

The last six fields apply only to F2–F6 arms (SS1.3.1); leave `null` with a reason for F0/F1/A0/A1/O. Use `null` plus a reason for unavailable metrics; never silently write zero for “not computed.”

### 11.2 Main paper table

| Dataset | Arm | AP | AP50 | AP75 | APsmall | R@50 | BPRbox | Mean K | E2E ms | Peak GB | Processed pixels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AgriPest | A0 | TBD | TBD | TBD | TBD | TBD | N/A | 64/full | TBD | TBD | TBD |
| AgriPest | A1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AgriPest | R0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AgriPest | R1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AgriPest | R2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AgriPest | R3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Repeat the identical schema for Pest24 and admitted GWHD arms. Do not paste AP50 values into AP columns to fill gaps.

**Pest24 (actual, 2026-08-16):**

| Arm | AP | AP50 | R (custom, conf>=0.001) | BPRbox (Very Tiny) | BPRbox (total) | Occupy | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (semantic-only, CIoU) | 0.286 | 0.477 | 0.462 | 0.2834 | 0.8261 | 0.0815 | 40.6 | 122.2 |
| R1 (semantic-only, SABL) | TBD | TBD | TBD | 0.2905 | TBD | TBD | TBD | TBD |
| R2 (F1 concat control, CIoU) | 0.348 | 0.592 | 0.597 | 0.6276 | 0.9416 | 0.158 | 49.3 | 115.1 |
| R3 (F1 concat control, SABL) | 0.349 | 0.594 | 0.572 | 0.6686 | 0.925 | 0.157 | 49.0 | 110.5 |
| F5 (reliability gate, CIoU) | 0.348 | 0.593 | 0.576 | 0.6403 | 0.929 | 0.167 | 50.2 | 109.0 |
| Gate+SABL (reliability gate, SABL) | 0.346 | 0.592 | 0.575 | 0.6650 | 0.923 | 0.163 | 50.0 | 109.7 |

R2's and R3's rows are labeled "F1 concat control" per SS6.1/SS6.2 -- fusion-ablation controls. F5 and Gate+SABL are the proposed reliability gate (SS4.2 of the Proposal). **All six arms are now complete and audited (SS11.2.1-SS11.2.6).** All rows above are single-seed (seed 0 only) -- SS7's minimum-three-seeds-for-claim-bearing-arms bar has not been met, so none of these numbers are yet an accepted result outside this working document; they are the current best evidence, not a validated finding.

Evaluated on the official 7599/7600-image test split (`--task test`; one image dropped, see SS11.2.1's gotcha entry). Input resolution is **not** uniformly 1024x1024 -- training uses a square 1024x1024 canvas, but test/measure use rect=True and actually run at 1024x768; see SS11.2.4 for the full mechanism. `hyp.pest24.yaml` (unmodified `hyp.uavdt.yaml`, not tuned for Pest24), `pest24_yolov5m.yaml` anchors carried over from `uavdt_yolov5m_nc3.yaml`, HeatMapParser threshold=0.5 (project default, not swept for Pest24's density).

### 11.2.1 Pest24 R0 findings and a real gotcha to record

**Gotcha: `run_pest24.sh`'s eval step silently evaluated the wrong split.** `test.py` resolves its data split via `task = opt.task if opt.task in ('train','val','test') else 'val'` (test.py L164) — with no `--task` flag at all, and even with `--task measure` (which is not in that tuple), it silently falls back to **val** (5077 images), not **test** (7600 images). The first R0 pass reported "Images 5077" and its predictions.json then failed `audit_buckets.py`'s image-id check when cross-referenced against the test split's images (4724 of ~5077 val image ids not found under `images/test`, correctly rejected rather than silently trusted). Root cause confirmed by reading `test.py` directly, not inferred. Fixed by adding `--task test` to the eval step in `run_pest24.sh`; the measure step (`--task measure`, GFLOPs/FPS only) still profiles on the val split by default, which is left as-is since efficiency numbers should not differ meaningfully between two similarly-distributed held-out splits, unlike accuracy numbers. Separately, `reorganize_pest24.py`'s symlinks needed a from-raw refresh after a GPU-instance restart (stale target paths) before the corrected eval could run — an operational gotcha, not a code bug, recorded here since it blocked the same fix from landing cleanly on the first retry.

**Result, correctly on our test split: R0 AP50 = 47.7%.** This is numerically 19.2 pp below the published dense-YOLOv5m anchor of 66.89, but it is **not a direct 19.2-pp gap**: the 66.89 row originates in Pest-YOLO's random 70/20/10, 416-px, confidence-threshold-0.5 protocol, while our result uses the 12,701/5,077/7,599 manifest and 1024-px common evaluator. The cloned Pest-PVT repo confirms only that its own 77.24 is VOC07 11-point mAP@0.5; HESOD computes AP50 with 101-point interpolation, and Pest-PVT does not release the contents of `trainval.txt`, its 24-class order patch, or the private `epoch_55.pth` used by the checked-in config. Our 44.8% erroneous-val result versus 47.7% corrected-test result shows that the local eval-split bug is not the sole cause of low R0 accuracy, but only a controlled dense-YOLOv5m rerun can measure the true architecture gap. Sparse routing, untuned Pest24 anchors/hyperparameters, and the selector threshold remain live suspects.

**Size-bucket breakdown reveals the gap concentrates almost entirely in the smallest objects, not the whole dataset uniformly:**

| Size bin | GT count | Recall |
|---|---:|---:|
| Very Tiny (<16x16) | 1415 | **28.34%** |
| Tiny (16x16-32x32) | 29358 | 75.54% |
| Small (32x32-96x96) | 27420 | 92.98% |
| Medium/Large (>96x96) | 6 | 100.00% |
| **Total** | **58199** | **82.61%** |

Very Tiny recall (28.34%) is far below every other dataset's Very Tiny bucket in this project (VisDrone 78.00%, UAVDT 83.81%, TinyPerson 73.20%, HESOD-Experiment-Plan.md SS1.3/SS4.1/SS5) -- this is a materially different, sharper failure mode than anything seen on the aerial datasets, not a smaller version of the same gap. Medium/Large is essentially absent (6/58199 GT boxes total), consistent with the native-resolution box-size audit done before training (0% of boxes exceed 96x96 at native 800x600).

**Class-level recall varies from 3.68% to 98.79%, correlated with class rarity, and compounds with the size effect.** Worst: Rice planthopper 3.68% (489 GT, 28.4% of its own instances fall in the Very Tiny bin where recall is only 3.23%), Plutella xylostella 8.77% (285 GT), Nematode trench 11.32% (53 GT, the rarest class in the dataset), Rice Leaf Roller 13.75% (371 GT). Best: Anomala corpulenta 98.79% (16943 GT, the most common class), Gryllotalpa orientalis 95.70%, holotrichia parallela 96.27%. The worst-performing classes are consistently among the rarest and skew heavily toward the Very Tiny/Tiny bins in their own size distribution -- class imbalance and object size are compounding, not independent, effects here.

**Resolved: the Very Tiny weakness is selector-side, not head-side.** Coverage-vs-miss diagnostic (same method as HESOD-Experiment-Plan.md SS1.4's VisDrone analysis and SS5's UAVDT Medium/Large analysis): of the 1014 missed Very Tiny GT boxes, **85.6% (868) have no nearby prediction of any class at any confidence at all** -- the selector never gave the head a chance to see that region. Only 6.4% (65) are genuine head-side localization failures (right class, IoU<0.5) and 8.0% (81) are labeled "wrong class nearby" but at near-zero score/IoU (0.001-0.076), indistinguishable in practice from noise rather than real competing detections. Worst-performing rare classes (Rice planthopper, Plutella xylostella) dominate both non-selector-dropped failure categories, consistent with the class-rarity/size compounding already noted above.

This directly licenses R2/R3's reliability gate as the correct lever for this specific weakness -- a selector-side intervention targeting exactly the failure mode found here (real low-objectness targets dropped before the head), not a head-side or loss-tuning problem that SABL/anchor changes would be better suited for. Occupy=0.0815 (far lower than every aerial dataset: VisDrone 0.424, UAVDT 0.272, TinyPerson 0.128) is consistent with this: the selector is operating far more sparsely on Pest24 than anywhere else in this project, plausibly driven by its extreme density (7.58 objects/image at native resolution) pushing more real targets below whatever threshold/capacity the selector uses. R2/R3/F5's actual effect on this specific number (Very Tiny recall, and the selector-dropped share of misses) is the sharpest test of whether the reliability gate is earning its complexity, not just overall AP.

### 11.2.2 R1 and R2 results -- the selector fix works, and it changes what SABL needs to fix

**R1 (semantic-only, SABL) confirms the diagnostic itself is trustworthy: no change where none was expected.** Very Tiny recall 29.05% (vs. R0's 28.34%), selector-dropped share of misses 85.8% (vs. 85.6%) -- both essentially identical to R0, exactly as expected since SABL only touches `lbox`, never the selector. This is a useful negative control, not just a null result: it confirms the coverage-vs-miss diagnostic is measuring something real and stable, not noise.

**R2 (reliability-aware-gate control arm F1, channel-pooled concat) is the largest positive effect found anywhere in this project to date:**

| Metric | R0 | R2 | Δ (relative) |
|---|---:|---:|---:|
| mAP@.5 | 0.477 | 0.592 | +24.1% |
| mAP@.5:.95 | 0.286 | 0.348 | **+21.7%** |
| BPR | 0.784 | 0.924 | +17.9% |
| Occupy | 0.0815 | 0.158 | +93.9% (still far from saturation) |
| Very Tiny recall | 28.34% | 62.76% | +121% |
| GFLOPs | 40.6 | 49.3 | +21.4% |

For comparison, the largest prior positive selector-side result in this project (TinyPerson channel-pooled-concat, HESOD-Experiment-Plan.md SS4.3) was +3.2% relative AP@[.5:.95] -- this is roughly 7x that effect size. Relative to the non-direct 66.89 dense-YOLOv5m literature anchor, the numerical distance shrinks from 19.2 pp for R0 to 7.7 pp for R2 (59.2 vs. 66.89); this is useful context, not a reproduction claim, until P24-C0 is rerun under the same manifest/evaluator.

**Coverage-vs-miss diagnostic confirms the mechanism directly, not just correlationally.** Of 1415 Very Tiny GT boxes: R0 misses 1014 (868 selector-dropped, 65 head-localization-failure, 81 confusion); R2 misses only 527 (236 selector-dropped, 186 head-localization-failure, 104 confusion, 1 stolen-by-another-GT). The selector-dropped count fell by 632 in absolute terms -- this is the direct mechanism behind the Very Tiny recall jump, not a side effect of something else.

**New finding: fixing the selector bottleneck exposed the head-localization bottleneck as the next-largest failure mode.** Right-class-low-IoU cases (head sees the target, box doesn't clear IoU 0.5) grew from 65 to 186 in absolute count (and from 6.4% to 35.3% of a much smaller miss pool) -- concat is successfully delivering previously-invisible Very Tiny targets to the head, but the head's box regression on them is often still not accurate enough. **This makes R3 (concat + SABL) a materially different test than R1 (semantic-only + SABL) was**: R1 found no SABL effect because the selector, not the head, was the bottleneck at that point; R3 tests SABL against a population where head-localization failure is now a much larger share of what remains. A genuine R3-over-R2 improvement here would be real evidence of the selector x loss interaction the R0-R3 factorial was designed to detect (HESOD-Agri-Proposal.md SS5's interpretation rules), not just two independently-additive effects.

**Open caveat, not yet resolved: how much of R2's gain is "smarter selection" vs. "selecting more area"?** Occupy nearly doubled (0.0815 -> 0.158); GFLOPs grew more modestly (+21.4%) and far less than the AP gain (+21.7% to +24.1%), which is a good sign this isn't simply "keep the whole image" in disguise (contrast TinyPerson's channel-pooled-concat, which cost +64% GFLOPs for a much smaller +3.2% AP gain) -- but the equal-budget comparison required by HESOD-Agri-Proposal.md SS6.2/SS9 has not been run yet. A Top-K sweep on R0 vs. R2 (same method as HESOD-Experiment-Plan.md SS3's VisDrone sweep) would settle whether R2's advantage survives at matched Occupy, not just at each arm's own free-threshold operating point.

**If the gate does turn out to target this specific weakness, Pest24 is a better validation testbed for it than any aerial dataset tried so far** -- there is far more headroom here (72pp of missed Very Tiny recall vs. VisDrone's 22pp), and the low-objectness failure mode directly matches the gate rationale in HESOD-Agri-Proposal.md SS4.2. This is a reason to prioritize the coverage-vs-miss diagnostic and R2/R3, not a reason to assume the gate will help without checking.

### 11.2.3 R3 results -- SABL's effect is real, mechanistically clean, and invisible in the aggregate

**Aggregate AP is essentially flat: mAP@.5:.95 0.348 -> 0.349, mAP@.5 0.592 -> 0.594.** Read alone this looks like R1's null result repeating (SS11.2.2's "no SABL effect" negative control) -- but the size-bucket and coverage-vs-miss breakdowns (both from the class-aware one-to-one audit, `pest24_yolov5m_channel_pooled_concat_sabl_audit.log`) show this is an aggregation artifact, not a null effect.

| Bucket | R2 recall | R3 recall | Delta |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 62.76% (888/1415) | 66.86% (946/1415) | **+4.10pp** |
| Tiny (16x16-32x32) | 92.44% | 92.29% | -0.15pp (noise) |
| Small (32x32-96x96) | 97.61% | 97.69% | +0.08pp (noise) |
| Total | 94.16% (54798/58199) | 94.21% (54832/58199) | +0.05pp |

The gain is concentrated entirely in Very Tiny, which is only 1415/58199 = 2.4% of all GT boxes -- large enough to move the Very-Tiny-specific number by 4pp, too small to move the total by more than rounding noise. This is the same lesson SS11.2.1/SS11.2.2 already established for the selector fix, now repeating for the loss: **aggregate AP hides mechanism-targeted gains on this dataset because of Pest24's own extreme size-imbalance, and bucket-level reporting is not optional here.**

**Coverage-vs-miss diagnostic (`vt_diagnose.py pest24_yolov5m_channel_pooled_concat_sabl`) shows the gain lands exactly where SS11.2.2 predicted it should:**

| Miss category (Very Tiny) | R2 (527 missed) | R3 (469 missed) | Delta |
|---|---:|---:|---:|
| Selector-dropped (no nearby prediction) | 236 (44.8%) | 232 (49.5%) | -4 (flat, as expected -- SABL never touches the selector) |
| Head-localization failure (right class, low IoU) | 186 (35.3%) | 150 (32.0%) | **-36 (-19.4% relative)** |
| Confusion (wrong class nearby) | 104 (19.7%) | 87 (18.6%) | -17 (-16.3% relative) |
| Stolen by another GT | 1 | 0 | -1 |

Selector-dropped count is flat (as it must be -- SS6's fairness rules require R2/R3 to share every component except `box_loss`). Head-localization failure -- the exact new-dominant failure mode SS11.2.2 flagged after the selector fix -- dropped by nearly a fifth. This is the R2-vs-R3 test SS11.2.2 said would be the sharp one, and it comes back positive.

**Selector x loss interaction, using SS6.1's own formula:** `(R3-R2) - (R1-R0)` on Very Tiny recall = (66.86%-62.76%) - (29.05%-28.34%) = 4.10pp - 0.71pp = **+3.39pp**. SABL's benefit is roughly 5.8x larger once the selector is actually delivering Very Tiny targets to the head (R2/R3 pair) than it is on the semantic-only baseline (R0/R1 pair, SS11.2.2's near-null result). This is genuine evidence of the predicted interaction, not two independently-additive effects: SABL has little to work with when the selector drops most Very Tiny targets before the head ever sees them (R0/R1), and much more to work with once concat rescues them (R2/R3).

GFLOPs/FPS are unchanged from R2 within measurement noise (49.0 vs 49.3 GFLOPs, 110.5 vs 115.1 FPS) -- consistent with SS7's SABL implementation lock ("no inference-cost difference is expected").

Single-seed caveat (SS11.2's table note) applies here as everywhere else in this section -- this is the current best evidence, not a three-seed-validated claim.

### 11.2.4 Resolution protocol audit: 1024 is not uniformly 1024x1024, and is not literature-optimal

`IMG_SIZE=1024` is a long-side target, not a universal square input. Pest24 training uses a 1024×1024 mosaic canvas; standalone test/measure uses rectangular batching and runs native 800×600 images at 1024×768. At P3/8 these become 128×128 and 128×96, both yielding the intended 8×8 routing grid.

An unmodified 640 experiment is not resolution-only: training produces P3 80×80 and fails `uni_slicer`'s divisibility assertion, while 640×480 test input produces a 7×8=56-cell action space. Supporting a fair 640 arm therefore requires a slicer redesign and a separately named protocol.

The 1024 choice preserves already-small targets and the 64-cell action space, but it is interpolation from native 800×600 and was not selected by a resolution sweep. Existing R0–R3 comparisons remain internally valid; literature values at 640/416/224 remain contextual. Run A0 dense YOLOv5m under the same 1024-long-side contract before considering a resolution study. The authoritative input contract is §4; this subsection records its consequence for current results.

### 11.2.5 F5 results -- the gate beats F1 broadly but modestly, not decisively

F5 (CIoU only; gate+SABL not yet run) vs. R2 (F1 concat, CIoU): AP@.5:.95 ties at 0.348, AP50 +0.001 -- but recall improves in *every* size bucket, not just Very Tiny: Very Tiny +1.27pp (62.76%->64.03%), Tiny +0.20pp, Small +0.37pp, total +0.30pp (94.16%->94.46%, the best total recall of all five arms). Cost: +1.8% GFLOPs (49.3->50.2), -5.3% FPS (115.1->109.0).

Coverage-vs-miss confirms the mechanism is real, not noise: selector-dropped Very Tiny misses fall 236->216 (-8.5% relative), consistent with the gate rescuing more real low-objectness targets than concat's flat fusion does. But head-localization-failure rises 186->198 (+6.5%) -- expected, since F5 has no SABL yet to fix that half of the picture the way R3 did for concat (SS11.2.3).

**F5 is not yet the single best Very-Tiny arm.** Its Very Tiny recall (64.03%) sits between R2 (62.76%) and R3/concat+SABL (66.86%) -- R3 still holds that number, on the strength of SABL, not a better selector. F5's actual edge is breadth (best *total* recall across all buckets) rather than depth on the one bucket the gate was specifically designed for.

**2026-08-17 update: TFR is now measured, and F5 fails this half of the Proposal SS9 win condition.** `B_tex` was defined image-content-only (no model output involved, avoiding circularity): a P3-resolution pixel is background if no GT box covers it, texture-heavy if its Sobel edge magnitude is at/above the 75th percentile of all background pixels' scores test-set-wide; a selected region counts as "in B_tex" if more than half its area falls in that mask (`tfr_diagnose.py`, exact selected regions from `test.py --save-regions`, not reconstructed from a heatmap).

| | F1 (concat) | F5 (gate) | Delta |
|---|---:|---:|---:|
| Total selected regions | 77088 | 81050 | +5.1% |
| Regions in B_tex | 6736 | 7373 | +9.5% |
| **TFR** | **8.74%** | **9.10%** | **+0.36pp (+4.1% relative, worse)** |

F5's TFR is *higher* than F1's, not lower. Per Proposal SS9's exact wording -- "F5/F6 does not beat parameter/latency-matched F1 on low-objectness-tiny-recall **or** on textured-background false-routing rate" -- this is an OR condition, and F5 fails the TFR half even though it passed the recall half above. With ~77-81k selected regions in each arm, a 0.36pp gap is unlikely to be sampling noise, though it remains single-seed. This is exactly the failure mode BCRS's own design rationale warned an unconstrained gate could fall into: F5 selects 5.1% more regions overall (consistent with rescuing more real low-objectness targets, matching the recall gain), but the B_tex-flagged share grows faster (+9.5%) than total selections -- some of what the gate is rescuing is textured background, not just real tiny targets.

**Verdict: F5's win over F1 is mixed, not clean.** It broadly improves recall (SS11.2.5 above) but does so partly by routing more into texture-heavy background. Under the Proposal's own two-part bar this does not yet license "the gate beats the capacity-matched control" as a supported claim -- one axis improved, the other regressed.

Gate+SABL is training (SS12); its own selector x loss interaction (parallel to SS11.2.3's `(R3-R2)-(R1-R0)`) remains to be measured, but will not change this TFR finding since SABL never touches the selector.

### 11.2.6 Gate+SABL results and the final fusion-ladder verdict

**Gate+SABL: AP 0.346, AP50 0.592, Very Tiny recall 66.50% (audit) / 66.57% (`vt_diagnose.py`, 1-box difference between the two independently-implemented matchers -- expected, not a bug), total recall 94.14%, GFLOPs 50.0, FPS 109.7, TFR 8.69%.**

SABL's effect on the gate family replicates the concat family's pattern exactly: Very Tiny recall +2.54pp (64.03%->66.57%), head-localization-failure -20.2% relative (198->158), selector-dropped essentially flat (216->224, a small uptick plausibly from full-retrain noise rather than a real selector change, since SABL's loss term never touches selector parameters). The gate's own selector x loss interaction, `(Gate+SABL - F5) - (R1-R0)` on Very Tiny recall = 2.54pp - 0.71pp = **+1.83pp** -- real and positive, but about half the size of concat's interaction (+3.39pp, SS11.2.3). TFR moved slightly (9.10%->8.69%) despite SABL not touching routing -- within the same small-effect-size range as the F1-vs-F5 TFR gap itself, most plausibly retraining noise rather than a causal SABL-on-TFR effect.

**All six arms, final comparison:**

| Arm | AP@.5:.95 | AP@.5 | Very Tiny recall | Total recall | GFLOPs | FPS | TFR |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 baseline | 0.286 | 0.477 | 28.34% | 82.61% | 40.6 | 122.2 | -- |
| R1 +SABL | 0.286 | 0.476 | 29.05% | 82.22% | 40.6 | 120.2 | -- |
| R2 concat (F1) | 0.348 | 0.592 | 62.76% | 94.16% | 49.3 | 115.1 | 8.74% |
| R3 concat+SABL | **0.349** | **0.594** | **66.86%** | 94.21% | 49.0 | 110.5 | -- |
| F5 gate | 0.348 | 0.593 | 64.03% | **94.46%** | 50.2 | 109.0 | 9.10% |
| Gate+SABL | 0.346 | 0.592 | 66.50% | 94.14% | 50.0 | **109.7** | **8.69%** |

**Verdict: across the full ladder, the reliability gate does not convincingly beat the capacity-matched concat control.** R3 (concat+SABL) holds the best AP@.5:.95, AP50, and Very Tiny recall of any arm. R2 (concat, CIoU) is the cheapest and fastest of the four fusion arms and has a lower TFR than F5. F5's only clear win is total recall (94.46%, the highest of all six arms), at a modest cost (+1.8% GFLOPs over R2) -- and even that is undercut by F5 having the *worst* TFR of the six arms. Gate+SABL's TFR recovers to competitive with concat (8.69%, actually the best of all four fusion arms), but its AP@.5:.95 (0.346) is the lowest of the four fusion arms, and it does not overtake R3 on Very Tiny recall.

This is the exact scenario Proposal SS9's falsification conditions were written to catch, and it is a legitimate negative-leaning finding, not an experiment failure: the reliability gate's added architectural complexity (extra confidence head, texture-risk head, gate MLP) is not yet earning its keep over the simpler, capacity-matched concat control on Pest24. Two things this does *not* settle: whether F6's additional rescue-ranking/conditional-gate-regularization losses (SS4.2.2 of the Proposal, not yet implemented or trained) would change this picture, since F5 here is the bare gate architecture without those losses; and whether the result replicates across seeds, since every number in this table is single-seed.

### 11.2.7 F6 pilot (attempt 1) -- best-ever aggregate metrics, but fails its own primary condition

F6 adds the rescue-ranking and conditional-gate-regularization losses (Proposal SS4.2.2) on top of F5's architecture, CIoU only. Pilot config (pre-registered, HESOD-Agri-Proposal.md SS4.2.2's candidate defaults): `tau_low=0.3, margin=1.0, lambda_rescue=0.5, lambda_cond=0.1, alpha=1.0`. Implementation: `models/segmenter.py`/`models/yolo.py` (`return_extras`/`need_gate_extras` plumbing), `utils/loss.py` (`FixedTextureFilter`, `compute_rescue_loss`, `compute_cond_loss`), `scripts/esod_baseline/run_pest24_f6.sh`.

**Result: AP 0.349, AP50 0.608, total recall 95.47%, GFLOPs 54.0, FPS 101.6, TFR 6.84%, Occupy 0.200.**

| Metric | F5 | F6 (attempt 1) | Delta |
|---|---:|---:|---:|
| AP@.5:.95 | 0.348 | **0.349** | +0.001 (ties R3, the ladder's best) |
| AP50 | 0.593 | **0.608** | +1.5pp (**highest of all seven arms measured**) |
| Total recall | 94.46% | **95.47%** | +1.01pp (highest of all seven arms) |
| TFR | 9.10% | **6.84%** | -2.26pp relative -25% (lowest of all seven arms) |
| Occupy | 0.167 | 0.200 | +19.8% relative |
| GFLOPs / FPS | 50.2 / 109.0 | 54.0 / 101.6 | +7.6% GFLOPs / -6.8% FPS |
| Very Tiny recall | 64.03% (906/1415) | **59.79% (846/1415)** | **-4.24pp (-60 boxes)** |
| selector-dropped share of Very Tiny misses | 42.4% (216/509) | 46.0% (262/569) | worse, not better |

**Where the total-recall gain actually came from -- not Very Tiny, the bucket F6 was built to fix:**

| Bucket | F5 recalled | F6 recalled | Delta |
|---|---:|---:|---:|
| Very Tiny (<16x16, n=1415) | 906 (64.03%) | 846 (59.79%) | **-60 boxes** |
| Tiny (16x16-32x32, n=29358) | 27197 (92.64%) | 27516 (93.73%) | +319 boxes |
| Small (32x32-96x96, n=27420) | 26866 (97.98%) | 27192 (99.17%) | +326 boxes |
| Medium/Large (n=6) | 6 (100%) | 6 (100%) | flat |
| **Net** | | | **+585 boxes total, entirely from Tiny+Small (+645) net of a real Very Tiny loss (-60)** |

**Verdict against the four pre-registered stop/go conditions (run_pest24_f6.sh's header comment):**

- **(a) selector-dropped down >=10% relative OR Very Tiny recall up >=1pp vs. F5 -- FAILS, in the wrong direction on both halves.** Very Tiny recall fell 4.24pp; selector-dropped's share of misses rose from 42.4% to 46.0%.
- **(b) TFR <= 8.74% (F1's baseline) -- PASSES, decisively.** 6.84% is the best TFR of any arm in this project's Pest24 work.
- **(c) AP not down more than ~0.2-0.3pp vs. F5 -- PASSES, and then some.** AP ties R3's ladder-best; AP50 sets a new high.
- **(d) gate not saturated -- inconclusive from these artifacts alone**, but Occupy *rising* (not collapsing toward 0) argues against simple `a_i -> 0` saturation; a direct `a_i` histogram was not captured for this checkpoint.

Condition (a) is the one this arm exists to satisfy, and it fails outright despite every aggregate metric improving. The likely mechanism: `P_rescue`'s membership (`y_i=1, q_i<tau_low`) is not size-weighted, and Tiny+Small GT boxes outnumber Very Tiny ones roughly 40:1 (56,778 vs 1,415) -- among cells with genuinely low semantic confidence, far more of them belong to Tiny/Small objects than Very Tiny ones by sheer count, so `L_rescue`'s gradient is plausibly dominated by the much larger Tiny/Small population rather than preferentially lifting the Very Tiny cells the mechanism was motivated by. `compute_coverage_loss` (SS6.2.1) already upweights small objects for exactly this reason (`ref_area_cells / area_cells` clamped weight); `compute_rescue_loss` does not.

**Per the pre-registered rule ("at most ONE retry adjusting lambda_rescue/lambda_cond ... still failing, stop the gate line"), this is attempt 1 of at most 2.** Attempt 2 (SS11.2.8, if run): `lambda_cond` lowered (0.1 -> 0.05, less suppression pressure that may be crowding out the rescue signal) and `lambda_rescue` raised (0.5 -> 1.0, stronger pull for the mechanism attempt 1 shows is real but diluted) -- both within the rule's allowed adjustment scope (`tau_low`/`margin`/architecture unchanged). Flagged honestly: this retry cannot fix the diagnosed size-imbalance in `P_rescue` itself (that would need a coverage-loss-style area weight, out of the pre-registered retry's scope) -- it may shift the balance without resolving the root cause, and should be read as testing whether a cheap reweighting is enough, not as expected to definitely pass.

### 11.2.8 F6 pilot attempt 2 (retry) -- closer, new AP/AP50 records, still fails condition (a): gate line closed

Attempt 2 config: `lambda_cond` 0.1 -> 0.05, `lambda_rescue` 0.5 -> 1.0 (`tau_low`/`margin`/architecture unchanged, per the rule's allowed scope). Script: `scripts/esod_baseline/run_pest24_f6_retry.sh`.

**Result: AP 0.352, AP50 0.617, total recall 96.73%, GFLOPs 56.2, FPS 96.7, TFR 7.29%, Occupy 0.221.**

| Metric | F5 | F6 attempt 1 | F6 attempt 2 (retry) |
|---|---:|---:|---:|
| AP@.5:.95 | 0.348 | 0.349 | **0.352 (new best of any Pest24 arm)** |
| AP50 | 0.593 | 0.608 | **0.617 (new best)** |
| Total recall | 94.46% | 95.47% | 96.73% (new best) |
| TFR | 9.10% | 6.84% | 7.29% (still well under F1's 8.74%) |
| Occupy | 0.167 | 0.200 | 0.221 |
| GFLOPs / FPS | 50.2 / 109.0 | 54.0 / 101.6 | 56.2 / 96.7 (most expensive/slowest of the family) |
| Very Tiny recall | 64.03% (906/1415) | 59.79% (846/1415) | 62.69% (887/1415) -- gap narrows from -4.24pp to **-1.34pp** |
| selector-dropped share of Very Tiny misses | 42.4% (216/509) | 46.0% (262/569) | 42.0% (222/528) -- back to roughly F5's own rate |

**Verdict against the four pre-registered stop/go conditions:**

- **(a) selector-dropped down >=10% relative OR Very Tiny recall up >=1pp vs. F5 -- still FAILS,** though far closer than attempt 1: Very Tiny recall is -1.34pp (needs >=+1pp), selector-dropped's relative change is only -0.9% (needs <=-10%, and this is essentially neutralizing attempt 1's regression rather than a real improvement over F5).
- **(b) TFR <= 8.74% -- PASSES** (7.29%).
- **(c) AP not down more than ~0.2-0.3pp vs. F5 -- PASSES, decisively**: AP and AP50 are both new highs for any Pest24 arm in this project.
- **(d) gate not saturated -- no red flag**: Occupy keeps rising (0.167 -> 0.200 -> 0.221), inconsistent with `a_i` collapsing toward 0.

**Precise bucket-level comparison confirms the SS11.2.7 size-imbalance diagnosis, not just consistent with it:**

| Bucket | attempt 1 vs. F5 | attempt 2 vs. F5 |
|---|---:|---:|
| Very Tiny (n=1415) | -60 boxes (-4.24pp) | **-19 boxes (-1.34pp)** -- loss shrank by two-thirds |
| Tiny (n=29358) | +319 boxes (+1.09pp) | **+948 boxes (+3.23pp)** -- gain roughly tripled |
| Small (n=27420) | +326 boxes (+1.19pp) | +389 boxes (+1.42pp) |
| Total (n=58199) | +585 boxes | +1318 boxes |

Doubling `lambda_rescue` did help Very Tiny (its loss shrank meaningfully), but the Tiny bucket's gain grew nearly 3x in the same step -- exactly what a size-unweighted `P_rescue` predicts: scaling the loss's overall pull scales its (already Tiny-dominated) gradient distribution proportionally, rather than preferentially reaching Very Tiny. A real fix would need `compute_rescue_loss` to weight pairs by GT object size (mirroring `compute_coverage_loss`'s `ref_area_cells/area_cells` pattern) -- out of scope for a lambda-only retry.

**Final disposition, per the pre-registered rule ("at most ONE retry ... still failing, stop the gate line and use concat+SABL"): the retry is exhausted and condition (a) is not met. The gate line (F5/F6) is closed for Pest24.** R3 (concat+SABL, SS11.2.6: AP@.5:.95 0.349 -- now second-best behind F6 attempt 2's 0.352 -- AP50 0.594, Very Tiny recall 66.86%, the best Very Tiny recall of any arm in this project) stands as Pest24's answer to the fusion-ladder question. This is a genuinely mixed, not purely negative, result worth stating precisely rather than rounding to "the gate failed": F6's two attempts progressively improved *every aggregate metric* to new project highs (AP, AP50, total recall) and kept TFR well under control, while never closing the one gap (Very Tiny recall parity or better vs. F5) the mechanism was pre-registered to fix -- a diagnosed, size-imbalance-shaped miss, not a directionless one. Whether a size-weighted `L_rescue` would close it is now a well-motivated, out-of-scope-for-this-study open question, not an unexplained failure.

### 11.2.9 A0 (dense, no routing) vs. routed arms -- does spatial routing even earn its keep on Pest24?

Triggered by the TP-YOLO comparison (a lightweight dense architecture reporting both higher accuracy and lower GFLOPs than any routed Pest24 arm): built `models/cfg/vanilla/pest24_yolov5m.yaml` (esod/pest24_yolov5m.yaml's backbone minus the Segmenter/HeatMapParser insert, SS4's `run_pest24_resolution.sh`) and ran it at both 1024 and 640, plus a same-resolution routed control (concat+SABL@640) to isolate resolution from routing.

**Verified results (scp'd and cross-checked against the local log files, not transcribed from console output alone):**

| Arm | Resolution | AP@.5:.95 | AP50 | Total recall | Very Tiny recall | GFLOPs | FPS | TFR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 (dense) | 640 | **0.374** | **0.644** | **97.56%** | **56.40%** | 58.1 | **156.7** | N/A (no routing) |
| Concat+SABL (routed) | 640 | 0.318 | 0.559 | 95.08% | 53.50% | **21.3** | 101.5 | 12.49% (worst in this project) |
| R3 concat+SABL (routed) | 1024 | 0.349 | 0.594 | 94.21% | 66.86% | 49.0 | 110.5 | -- |
| A0 (dense) | 1024 | not yet verified locally -- train/test/measure/audit logs confirmed complete on the GPU box (SS-gotcha below), but never scp'd/cross-checked | | | | | | |

**At matched 640 resolution -- the cleanest comparison run in this project, resolution held constant -- dense beats routed on every measured axis except raw GFLOPs, and even that "win" does not survive contact with measured latency:** AP@.5:.95 +17.6% relative, AP50 +15.2% relative, total recall +2.6pp, Very Tiny recall +2.9pp, and FPS +54% *despite* A0's GFLOPs being 2.7x higher (58.1 vs 21.3) -- confirming, on a second independent same-resolution comparison, that theoretical FLOPs and measured end-to-end latency move in opposite directions for this codebase's patch-slicing pipeline, not a one-off artifact of the earlier 1024-resolution measurement (SS11.2.5's "TP-YOLO" discussion first raised this).

Concat+SABL@640's TFR (12.49%) is also the worst of any arm measured in this project (previous worst: F5's 9.10%) -- at lower resolution the selector has less fine-grained visual information to distinguish real targets from textured background, and correspondingly routes worse. Its miss-cause mix also inverts: right_class_low_iou (localization failure, 43.5%) overtakes no_nearby_prediction (selector-dropped, 34.5%) for the first time in this project's Pest24 diagnostics -- at 640 the coarser P3 grid (80x60 vs 1024's 128x96) hurts box localization on top of routing quality.

**Verdict: this is now reasonably strong, repeated, same-resolution evidence that spatial routing does not earn its computational keep on Pest24** -- not a single anomalous measurement. The leading hypothesis (SS discussion, not yet independently tested on a second dataset within this codebase): Pest24's low native resolution (800x600) and extreme object density (7.58 objects/image, Occupy already 0.15-0.22 even for the routed arms) together leave too little "safely skippable" area to amortize the patch-slicing pipeline's real, non-theoretical overhead (irregular memory access, small-batch reassembly) against. This does not indict routing as a method -- this project's *other* (non-Agri) experiment plan reports much higher Very Tiny recall on VisDrone (78.00%)/UAVDT (83.81%)/TinyPerson (73.20%) using the earlier selector fix, consistent with those being a more favorable (higher native resolution, more variable density) regime -- but no A0-equivalent dense-vs-routed head-to-head has been run on those datasets within this framework, so "routing pays off there" remains an inference from less rigorously matched prior comparisons, not a result established with the same rigor as this section's Pest24 finding.

**Gap to close before treating this as final**: scp and verify A0@1024's actual AP/AP50 numbers (never done -- see table above) to complete the 2x2 resolution-x-routing grid on Pest24.

### 11.3 Required plots

1. AP vs end-to-end latency Pareto plot, with seed uncertainty;
2. AP_small vs processed-pixel fraction;
3. BPR_box vs K and actual detector recall vs K;
4. selected-K distribution by dataset/density stratum;
5. R0–R3 factorial effect plot for AP75/AP_small;
6. GWHD domain-level plot or density-boundary plot;
7. failure taxonomy: missed routing, poor localization, confusion, duplicate, background false positive;
8. F0–F6 fusion-ablation Pareto plot (AP or low-objectness-tiny-recall vs. TFR), parameter/latency-matched, per SS6.2;
9. 2D heatmap: $q_i$ x texture-risk bucket (SS1.3.2), tiny recall and gate activation as two separate panels.

## 12. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-08-13 | Scope = AgriPest, Pest24, GWHD 2021; no private GWHD subset | Preserves the field/dense/cross-task story and official comparability |
| 2026-08-13 | Use R0–R3 selector × SABL factorial; keep hybrid/SAM labels outside it | Separates routing, localization loss, and label effects |
| 2026-08-13 | Exact Top-K is the proposed budget mode; threshold routing is the upstream-compatible baseline | Keeps method attribution explicit |
| 2026-08-13 | Separate common-protocol reruns from original-protocol literature values | AP/AP50/WDA and incompatible splits/resizes are not interchangeable |
| 2026-08-15 | Reliability-aware F5/F6 is the proposed selector; concat is F1 control | The gate must beat the capacity-matched concat control before promotion into R2/R3 |
| 2026-08-16 | Fix Pest24 evaluation to `--task test`; retain the 7,599/7,600 caveat | The prior default silently evaluated the 5,077-image validation split |
| 2026-08-16 | Pest24 R2 rescues selector-dropped Very Tiny targets; R3+SABL mainly reduces the resulting localization failures | Aggregate AP hides the single-seed mechanism signal; see §11.2.2–§11.2.3 |
| 2026-08-16 | F5/F6 gate has four implemented inputs; $e_B$ is future work | Aligns Proposal, plan, and `ReliabilityGateMLP` |
| 2026-08-16 | Pest24 uses 1024-long-side: 1024×1024 train, 1024×768 test/measure | Preserves the 8×8 grid; 640 is not a clean switch; see §4/§11.2.4 |
| 2026-08-16 | Pest24 baseline policy: TP-YOLO/Pest-PVT priority reruns; Pest-YOLO/DAMI optional; AgriPest-YOLO reported-only | Reflects actual public-code completeness and avoids speculative reconstruction |
| 2026-08-17 | F5 (gate, CIoU) beats F1/R2 across every size bucket, modestly (+0.30pp total recall, +1.27pp Very Tiny, +1.8% GFLOPs); still trails R3/concat+SABL on Very Tiny alone (66.86%) | See SS11.2.5. At the time of this entry TFR was unmeasured and gate+SABL hadn't run, so Proposal SS9's two-part win condition was only half-checked |
| 2026-08-17 | TFR measured for F1 and F5 via a content-only `B_tex` definition and a new `test.py --save-regions` audit trail; F5's TFR (9.10%) is *higher* than F1's (8.74%), not lower -- F5 fails this half of Proposal SS9's win condition even though it passed the recall half | `tfr_diagnose.py` on ~77-81k selected regions per arm; see SS11.2.5. F5 selects 5.1% more regions than F1 but its B_tex-flagged share grows 9.5% -- consistent with the gate rescuing real low-objectness targets and texture-background noise together, the exact failure mode BCRS's design rationale warned about. Verdict: F5's win over F1 is mixed, not a clean pass |
| 2026-08-17 | Gate+SABL complete; final six-arm fusion-ladder verdict: the reliability gate does not convincingly beat the capacity-matched concat control | R3 (concat+SABL) holds the best AP@.5:.95 (0.349), AP50 (0.594), and Very Tiny recall (66.86%) of any of the six arms; F5's only clear edge (total recall 94.46%) is undercut by the worst TFR of the six (9.10%); Gate+SABL's TFR recovers (8.69%, best of the four fusion arms) but its AP@.5:.95 (0.346) is the lowest of the four. See SS11.2.6. This is Proposal SS9's falsification scenario, not an experiment failure -- F6's unimplemented rescue-ranking/conditional-gate-regularization losses and three-seed replication remain open, unresolved by this result |
| 2026-08-17 | F6 rescue-ranking + conditional-gate-regularization losses implemented (`models/segmenter.py`/`yolo.py`/`utils/loss.py`, `run_pest24_f6.sh`) and trained (attempt 1, pre-registered config `tau_low=0.3 margin=1.0 lambda_rescue=0.5 lambda_cond=0.1`) | Best-ever aggregate metrics of any Pest24 arm (AP 0.349 ties R3; AP50 0.608 and TFR 6.84% are both the best of any arm), but fails its own primary pre-registered condition: Very Tiny recall fell 4.24pp (64.03%->59.79%) instead of rising, and the +585-box total-recall gain traces entirely to Tiny/Small buckets, not Very Tiny. See SS11.2.7. Likely cause: `P_rescue` membership is not size-weighted and Tiny+Small GT boxes outnumber Very Tiny ~40:1, so `L_rescue`'s gradient is plausibly diluted by the larger population. Per the pre-registered rule, one retry (adjusted lambda_rescue/lambda_cond) remains before closing the gate line |
| 2026-08-18 | F6 attempt 2 (retry: `lambda_cond` 0.1->0.05, `lambda_rescue` 0.5->1.0) complete; still fails condition (a) -- **gate line closed per the pre-registered rule, R3 (concat+SABL) is Pest24's final fusion-ladder answer** | AP 0.352/AP50 0.617 are new project-best (beating even R3), TFR 7.29% stays well under F1's 8.74%, but Very Tiny recall gap to F5 only narrowed to -1.34pp (needed >=+1pp). Bucket-level deltas confirm the size-imbalance diagnosis precisely: doubling lambda_rescue tripled the Tiny-bucket gain (+319->+948 boxes) while Very Tiny's loss only shrank by two-thirds (-60->-19 boxes) -- the size-unweighted `P_rescue` gradient stays Tiny-dominated at any lambda_rescue scale. See SS11.2.8. A size-weighted `L_rescue` (mirroring `compute_coverage_loss`'s area weight) is a well-motivated but out-of-scope follow-up, not attempted here |
| 2026-08-18 | A0 (dense, no routing) beats every routed/gated Pest24 arm on AP/AP50/total-recall/Very-Tiny-recall and measured FPS, confirmed at matched 640 resolution against concat+SABL (not just across resolutions) -- **spatial routing is not earning its computational keep on Pest24**, independent of the F5-vs-F1/F6 gate-vs-concat question already settled above | See SS11.2.9. At 640: A0 AP 0.374 vs. concat+SABL's 0.318 (+17.6% relative); FPS 156.7 vs. 101.5 (+54%) *despite* A0's GFLOPs being 2.7x higher (58.1 vs 21.3) -- theoretical FLOPs and measured latency move in opposite directions for this codebase's patch pipeline, replicated on a second independent same-resolution measurement. Leading hypothesis: Pest24's low native resolution (800x600) + extreme density (7.58 objects/image) leave too little skippable area to amortize real patch-slicing overhead against; not yet tested with the same rigor on VisDrone/UAVDT/TinyPerson, where prior (less rigorously matched) results suggest a more favorable regime. Pest24 experimentation is effectively paused pending a dataset decision |

## 13. Publication-facing additions

The experimental core is the same for every target journal. Add only the analysis needed for the chosen framing:

- **Computers and Electronics in Agriculture:** full three-dataset Pareto evidence, modern external baselines, clean novelty/complexity analysis.
- **Biosystems Engineering:** deeper end-to-end systems characterization, memory/energy proxy, fallback behavior under dense scenes, reproducible deployment protocol.
- **Precision Agriculture:** monitoring/counting consequences, density calibration, possible decision thresholds and domain robustness.
- **Plant Phenomics:** make GWHD/domain generalization and phenotype-count reliability central.
- **Pest Management Science / Crop Protection:** emphasize pest counts, species-specific errors, monitoring utility, and practical false-negative/false-positive consequences.

Quartile and category must be rechecked in the current Clarivate JCR immediately before submission.
