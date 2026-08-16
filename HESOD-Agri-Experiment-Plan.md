# HESOD-Agri Experiment Plan

> **Companion proposal:** [HESOD-Agri-Proposal.md](HESOD-Agri-Proposal.md)  
> **Method source:** [BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md](BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md) SS5.3C/SS5.4 (reliability-aware residual gate)  
> **Datasets in scope:** AgriPest, Pest24, GWHD 2021 only  
> **Primary question:** Does a reliability-aware semantic–spectral gate improve the detection accuracy–compute Pareto frontier, and does it do so *specifically* by conditioning spectral evidence on semantic uncertainty rather than treating it as a co-equal signal (plain concat)?  
> **Status:** protocol definition; no agricultural result has been accepted as a claim yet (all Pest24 numbers below are single-seed and preliminary — SS7's minimum-three-seeds bar is not yet met). 2026-08-15: R2/R3's selector redefined from concat to the reliability-aware gate; concat retained as a required F1 control, not the proposed method — see SS6.2 and the decision log (SS12). 2026-08-16: Pest24 R0, R1 (SABL, negative control), and R2 (F1 concat control) all complete and audited, see SS11.2.1/SS11.2.2. R3 (F1+SABL) and F5 (reliability gate) in progress; AgriPest and GWHD 2021 preparation has not started.

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

This ladder is not secondary/optional the way the old S1–S3 table was — it is the evidence that the gate (not concat) belongs in R2/R3. The intended order is AgriPest first, promoting to Pest24/GWHD only once F5/F6 shows the expected pattern there (BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md SS11.2 Phase 2 gate). **In practice this order was not followed**: Pest24's pipeline was ready first and AgriPest preparation has not started (SS2.2 vs. SS2.3), so F1 was run and validated on Pest24 first (SS11.2.2), and F5/F6 will run there next. This is a recorded deviation, not a silent substitution — the gate still must clear the ladder before promotion into R2/R3 on whichever dataset it is tested on first, and the three-dataset claim (Proposal SS9) still requires AgriPest and GWHD results before it can be made. All rungs use CIoU box loss only (SABL is crossed with F0 and F5/F6 in the R0–R3 factorial, not with every fusion variant — do not multiply the full ladder by every loss/dataset/seed).

| ID | Fusion | Definition | Question |
|---|---|---|---|
| F0 | Objectness-only | $u_i = \mathrm{logit}(q_i)$ | Semantic baseline (= R0) |
| F1 | Concat -> MLP | $u_i = f([q_i, s_i])$ | Capacity-matched control — the previous plan's proposed method |
| F2 | Unconstrained learned gate | $a_i = \sigma(\mathrm{MLP}[q_i, s_i])$ | Does gate structure alone (no uncertainty/confidence/texture inputs) beat concat? |
| F3 | Uncertainty-only gate | $a_i = h_i$ (binary entropy of $q_i$) | Does "intervene exactly when semantic is uncertain" work on its own? |
| F4 | Low-score rescue gate | $a_i = (1-q_i)^\gamma$ | Isolates low-objectness rescue; expected to raise textured-background false-routing (predicted failure mode, not a strawman — report it even if confirmed) |
| F5 | Reliability-aware residual gate | $a_i = \sigma(\mathrm{MLP}[q_i, h_i, c_i^{spec}, t_i^{bg}, e_B])$, $u_i = \mathrm{logit}(q_i) + \alpha \cdot a_i \cdot z_i^{spec}$ | **Proposed selector** |
| F6 | F5 + rescue-ranking + conditional-gate regularization + coverage | full objective, HESOD-Agri-Proposal.md SS4.2.2 | **Full proposed method**, used in R2/R3 once validated |

Requirements for every rung (BCRS SS7.4): parameter count, channel width, input resolution, and training epochs matched across F0–F6; report new MACs and measured latency per rung, including the confidence/texture-risk heads' own cost for F5/F6, not just the gate MLP. Do not promote F5/F6 into R2/R3 (SS6.1) until this ladder confirms F5/F6 beats parameter-matched F1 on both low-objectness-tiny-recall and textured-background false-routing rate (SS8.4) — that comparison, not intuition, is what licenses the selector choice.

### 6.2.1 Gate implementation lock

Parallel to SS7's SABL implementation lock, for F5/F6:

- $h_i$ = normalized binary entropy of $q_i$ (peaks at $q_i=0.5$);
- gate MLP inputs: $[q_i, h_i, c_i^{spec}, t_i^{bg}, e_B]$ — dropping any of these without renaming the arm is not permitted;
- $u_i = \mathrm{logit}(q_i) + \alpha \cdot a_i \cdot z_i^{spec}$, evaluated in logit space, not probability space;
- $z_i^{spec}$ must be able to take negative values (background suppression), verified by inspecting its sign distribution before trusting any F5/F6 result;
- rescue-ranking pairs: $\mathcal{P}_{rescue} = \{(i,j): y_i=1, q_i<\tau_{low}, y_j=0, j\in\mathcal{B}_{tex}\}$, loss $\mathcal{L}_{rescue} = \frac{1}{|\mathcal{P}_{rescue}|}\sum \log(1+\exp(m-u_i+u_j))$;
- conditional-gate regularization: $\mathcal{L}_{cond} = \frac{1}{|\mathcal{C}_{sem}|}\sum_{i\in\mathcal{C}_{sem}} a_i + \frac{1}{|\mathcal{B}_{tex}|}\sum_{i\in\mathcal{B}_{tex}} a_i$, small weight, must not overpower coverage/ranking supervision;
- **not locked, unlike SABL's constants:** $\tau_{low}$, $m$, $\lambda_{rescue}$, $\lambda_{cond}$, $\alpha$ require a validation sweep before the primary operating point is chosen. Candidate starting points (unvalidated): $\tau_{low}=0.3$, $m=1.0$, $\lambda_{rescue}=0.5$, $\lambda_{cond}=0.1$, $\alpha=1.0$.
- gate saturation check: report the distribution of $a_i$ on tiny-object and texture-background regions separately; $a_i \to 1$ or $a_i \to 0$ almost everywhere is a failure to report, not a result to hide (HESOD-Agri-Proposal.md SS9).

### 6.3 External baselines

#### Evidence audit

As of 2026-08-13, no verified paper reports results on AgriPest, Pest24, and GWHD 2021 together. Therefore:

- do not construct a pseudo-leaderboard by joining numbers from unrelated papers;
- build the three-dataset comparison by rerunning the same public implementations;
- keep source-paper numbers in a separate, dataset-specific protocol table;
- call a published number directly comparable only after its split, metric, resize, and test-time policy are reproduced.

Faster R-CNN is the closest common architecture-level literature anchor: Faster R-CNN-family results appear in the original/official evaluations of AgriPest, Pest24, and GWHD 2021. However, the reported metrics still differ—AgriPest emphasizes AP at IoU 0.5, Pest24 papers use paper-specific mAP/mRecall conventions, and GWHD uses WDA—so even this anchor must be rerun with the common evaluator.

#### Locked cross-dataset comparison set

| ID | Implementation | Coverage | Function in the paper | Admission rule |
|---|---|---|---|---|
| A0 | Dense YOLOv5m | All three | Same-detector matched control | Mandatory |
| X1 | [Faster R-CNN](https://arxiv.org/abs/1506.01497) R50-FPN | All three | Common source-paper architecture and two-stage localization control | Mandatory |
| X2a | YOLOv5m + [SAHI](https://arxiv.org/abs/2202.06934) inference | All three | Exhaustive regular slicing using the A0 detector | Mandatory |
| X2b | YOLOv5m + SAHI sliced fine-tuning + inference | All three | Stronger uniform-tile baseline | Run if training budget permits; never merge with X2a |
| X3 | [QueryDet](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) R50-FPN | Prefer all three; AgriPest minimum | Closest external learned sparse high-resolution method | Port official code; exclude rather than silently rewrite its algorithm |
| X4 | [RTMDet-m](https://arxiv.org/abs/2212.07784) | All three | Modern dense accuracy/efficiency control | Mandatory |

The main comparison table contains our rerun results for A0/X1/X2a/X3/X4 and HESOD R0/R2/R3. X2b is a clearly labelled stronger-cost SAHI variant. All rows are evaluated from saved predictions by the same common evaluator.

#### Dataset-specific published anchors

| Dataset | Published anchor | Original metric role | Paper usage |
|---|---|---|---|
| AgriPest | SSD512, RetinaNet, FCOS, Faster R-CNN/FPN, Cascade R-CNN in the dataset paper | Primarily AP at IoU 0.5 and condition subsets | Reproduction sanity table; only exact-protocol values may be compared directly |
| Pest24 | Faster R-CNN, SSD, YOLOv3, Cascade R-CNN in the dataset paper; [Pest-YOLO](https://pmc.ncbi.nlm.nih.gov/articles/PMC9783619/)/[Pest-PVT](https://www.sciencedirect.com/science/article/pii/S0168169924012559) as later specialized methods | Paper-specific mAP/mRecall; some later work reports AP50 and AP50:95 explicitly | Secondary Pest24 table; re-evaluate public checkpoints/predictions where possible |
| GWHD 2021 | Official Faster R-CNN reference and challenge entries | WDA/domain accuracy | Exact official split and WDA implementation required |

Pest-PVT is included only as a Pest24-specific agriculture baseline. It is not presented as a three-dataset method because its paper does not report AgriPest or GWHD 2021.

#### Audited values that may be pre-populated

These tables are literature anchors, not outputs from our common evaluator. The `Direct?` column states whether a future result may be compared numerically after reproducing the named protocol.

**AgriPest official validation protocol** — 44,716 training images and 4,991 validation images. The source paper defines AP50, AP75, and `AP@[0.50:0.05:0.95]` explicitly, so the metric mapping is usable.

| Method | AP50 | AP75 | AP@[.50:.95] | Direct? |
|---|---:|---:|---:|---|
| SSD512 | 63.38 | 26.23 | 30.85 | Yes, only with the same official validation split/protocol |
| RetinaNet | 65.03 | 28.97 | 33.45 | Yes, only with the same official validation split/protocol |
| FCOS | 66.22 | 28.72 | 33.24 | Yes, only with the same official validation split/protocol |
| Faster R-CNN | 65.58 | 27.58 | 32.26 | Yes, only with the same official validation split/protocol |
| FPN | 70.20 | 29.91 | 35.21 | Yes, only with the same official validation split/protocol |
| Cascade R-CNN | 70.83 | 32.29 | 36.54 | Yes, only with the same official validation split/protocol |

Source: AgriPest dataset paper, Tables 4-6. These numbers may be placed in an `original protocol` comparison table; they must not be copied into our rerun result fields.

**Verified against the primary source directly (2026-08-16), not re-derived from a secondary citation.** All five values above match the PDF exactly (Tables 4/5/6). Two things confirmed worth recording: (1) the paper properly separates AP50/AP75/AP[.50:.95] into three distinct tables using standard COCO-style `IoU in [0.50:0.05:0.95]` (paper's own words) -- a materially more rigorous and directly-comparable protocol than Pest-PVT's single ambiguous "mAP" column (SS6.3's Pest24 anchor above); (2) training recipe was only 12 epochs (Detectron-style 1x schedule, LR drop at 8th/11th epoch) on two NVIDIA 1080Ti -- these baselines may not represent a fully-converged upper bound, and are not directly comparable to this project's 50-epoch RTX 5090 recipe without accounting for that gap. Split is 44,716 train / 4,991 val, no separate test set -- the paper's own "validation set" is the reported evaluation split, not held out from model selection the way this project's own train/val/test three-way split is.

**Pest24 / Pest-PVT paper protocol** — the paper reports `mAP` but does not state the IoU threshold/range or the exact train/validation/test split. Its raw image description gives 2095×1944, while an architecture description states 224×224 input; this is not enough to reconstruct a direct-comparison protocol.

| Method | Reported mAP | Recall | Precision | Direct? |
|---|---:|---:|---:|---|
| Faster R-CNN | 42.67 | 54.00 | 45.58 | No; metric/split under-specified |
| YOLOv5m | 66.89 | 70.90 | 64.84 | No; metric/split under-specified |
| YOLOv7-x | 72.92 | 70.51 | 72.73 | No; metric/split under-specified |
| Pest-YOLO | 69.59 | 77.71 | 46.94 | No; imported by the paper from earlier work |
| Pest-PVT | 77.20 | 81.27 | 78.42 | No; metric/split under-specified |

Source: Pest-PVT, Table 6. The paper also gives Pest-PVT mAP as 77.24% in its ablation Table 4, versus 77.20% in Table 6/abstract. Preserve this source inconsistency; use `77.20 (reported mAP, definition not disclosed)` in prose and never map it to AP50 or AP50:95.

**GWHD 2021 official challenge protocol** — the official dataset paper reports WDA, not COCO AP.

| Method | WDA | Direct? |
|---|---:|---|
| Official Faster R-CNN reference | 0.492 | Yes, only with the Global Wheat Challenge 2021 split and exact WDA evaluator |
| Challenge best entry | 0.700 | Context only; ensemble/training recipe differs |

Source: GWHD 2021 dataset paper, Table 2. Neither value belongs in an AP/AP50 column.

#### Missing target-dataset results in the selected method papers

| Method paper | Paper datasets | AgriPest | Pest24 | GWHD 2021 | Action |
|---|---|---:|---:|---:|---|
| SAHI | VisDrone, xView | No | No | No | Rerun X2a/X2b |
| QueryDet | COCO, VisDrone | No | No | No | Rerun X3 |
| RTMDet | COCO | No | No | No | Rerun X4 |
| SSABNet/SABL | VisDrone, UAVDT | No | No | No | Use only as SABL prior; train R1/R3 |
| TP-YOLO | Own 3-class Khapra-beetle set (published numbers); lists Pest24 as a supported training dataset (450-epoch recipe) but publishes no Pest24 numbers | No | No (trains, doesn't report) | No | Context only — architecture/training-recipe reference, not a citable Pest24 comparison point; verified by reading the repo directly (2026-08-16) |

Therefore, no numerical value from SAHI, QueryDet, RTMDet, SSABNet, or TP-YOLO can be filled into an AgriPest/Pest24/GWHD result cell before running the corresponding experiment.

#### Selection and fairness rules

Every rerun baseline requires:

- open code and a licence compatible with evaluation;
- official or reproducible split;
- explicit image resolution and augmentation;
- COCO AP support or retained predictions enabling a common evaluator;
- measured inference cost on the same hardware.
- a frozen source commit and a patch manifest;
- no HESOD-specific augmentation or loss unless a separately named sensitivity arm applies it;
- both the common COCO evaluator and, where reproducible, the dataset's original metric evaluator.

If the source paper used a different dataset release, hidden test set, unavailable split, or undocumented preprocessing, its number remains literature context and the rerun is reported as a new controlled result—not a claimed exact reproduction.

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

**Pest24 status (2026-08-16): complete** -- occupancy/size/density stats measured (SS11.2.1's Occupy=0.0815, size-bucket table); Pest24 is in scope by default (SS2.3) so no formal admission gate applied (unlike GWHD); input resolution and timing contract frozen (1024x1024, `hyp.pest24.yaml`, SS11.2's Pest24 table footnote). **AgriPest and GWHD 2021: not started** (GWHD's own admission gate, SS2.4, has not been evaluated).

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

**Pest24 status (2026-08-16): R1 and R2 (F1 control) complete, single-seed only; R3 and F5 in progress** (SS11.2.2). This ran on Pest24 first, not AgriPest as the checklist below assumes -- see SS6.2's reconciling note. Three-seed replication has not started for any arm on any dataset. **AgriPest and GWHD 2021: not started.**

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
| R3 (F1 concat control, SABL) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| F5 (reliability gate, CIoU) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

R2's row is labeled "F1 concat control" per SS6.1/SS6.2 -- it is the fusion-ablation control, not yet the proposed reliability gate (F5, still training). R1's AP/AP50/R/BPRbox(total)/Occupy/GFLOPs/FPS are still TBD pending that run's full `test.py` summary line. **All rows below are single-seed** (seed 0 only) -- SS7's minimum-three-seeds-for-claim-bearing-arms bar has not been met, so none of these numbers are yet an accepted result outside this working document; they are the current best evidence, not a validated finding.

Evaluated on the official 7599/7600-image test split (`--task test`; one image dropped, see SS11.2.1's gotcha entry), 1024x1024 input, `hyp.pest24.yaml` (unmodified `hyp.uavdt.yaml`, not tuned for Pest24), `pest24_yolov5m.yaml` anchors carried over from `uavdt_yolov5m_nc3.yaml`, HeatMapParser threshold=0.5 (project default, not swept for Pest24's density).

### 11.2.1 Pest24 R0 findings and a real gotcha to record

**Gotcha: `run_pest24.sh`'s eval step silently evaluated the wrong split.** `test.py` resolves its data split via `task = opt.task if opt.task in ('train','val','test') else 'val'` (test.py L164) — with no `--task` flag at all, and even with `--task measure` (which is not in that tuple), it silently falls back to **val** (5077 images), not **test** (7600 images). The first R0 pass reported "Images 5077" and its predictions.json then failed `audit_buckets.py`'s image-id check when cross-referenced against the test split's images (4724 of ~5077 val image ids not found under `images/test`, correctly rejected rather than silently trusted). Root cause confirmed by reading `test.py` directly, not inferred. Fixed by adding `--task test` to the eval step in `run_pest24.sh`; the measure step (`--task measure`, GFLOPs/FPS only) still profiles on the val split by default, which is left as-is since efficiency numbers should not differ meaningfully between two similarly-distributed held-out splits, unlike accuracy numbers. Separately, `reorganize_pest24.py`'s symlinks needed a from-raw refresh after a GPU-instance restart (stale target paths) before the corrected eval could run — an operational gotcha, not a code bug, recorded here since it blocked the same fix from landing cleanly on the first retry.

**Result, correctly on the test split: R0 AP50 = 47.7%, a real ~19pp gap below dense YOLOv5m's reported 66.89% on the same dataset** (Pest-PVT's own repo, `pest-pvt/configs/_base_/datasets/voc0712.py`: `dataset_type='VOCDataset'`, `evaluation=dict(metric='mAP')`, no `iou_thr` override in `test.py` -> mmdetection's VOC-style single-IoU=0.5 mAP, matching this project's own `mAP@.5` column, not `mAP@.5:.95` — confirmed by reading their config directly, not inferred from the paper text). The gap held at essentially the same magnitude when first measured (incorrectly) on the val split (44.8%) as it does on the correct test split (47.7%), so it is not a split-mismatch artifact. Some gap is structurally expected -- their number is dense YOLOv5m (no patch selection at all), while R0 is YOLOv5m with ESOD's sparse patch-selection architecture, i.e., a different model that trades some accuracy for compute savings by construction -- but 19pp/~29% relative is far larger than any other dataset in this project's own aerial track has ever shown for a comparable trade-off (VisDrone's baseline sits only 2.8% below the paper's own ablation, HESOD-Experiment-Plan.md SS1.2), so "the trade-off itself" does not fully explain this, and the untuned hyperparameters/anchors/threshold above remain live suspects.

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

For comparison, the largest prior positive selector-side result in this project (TinyPerson channel-pooled-concat, HESOD-Experiment-Plan.md SS4.3) was +3.2% relative AP@[.5:.95] -- this is roughly 7x that effect size. The Pest-PVT gap also closed substantially: R0 sat 19.2pp below dense YOLOv5m's 66.89 mAP@0.5; R2 sits only 7.7pp below it (59.2 vs. 66.89).

**Coverage-vs-miss diagnostic confirms the mechanism directly, not just correlationally.** Of 1415 Very Tiny GT boxes: R0 misses 1014 (868 selector-dropped, 65 head-localization-failure, 81 confusion); R2 misses only 527 (236 selector-dropped, 186 head-localization-failure, 104 confusion, 1 stolen-by-another-GT). The selector-dropped count fell by 632 in absolute terms -- this is the direct mechanism behind the Very Tiny recall jump, not a side effect of something else.

**New finding: fixing the selector bottleneck exposed the head-localization bottleneck as the next-largest failure mode.** Right-class-low-IoU cases (head sees the target, box doesn't clear IoU 0.5) grew from 65 to 186 in absolute count (and from 6.4% to 35.3% of a much smaller miss pool) -- concat is successfully delivering previously-invisible Very Tiny targets to the head, but the head's box regression on them is often still not accurate enough. **This makes R3 (concat + SABL) a materially different test than R1 (semantic-only + SABL) was**: R1 found no SABL effect because the selector, not the head, was the bottleneck at that point; R3 tests SABL against a population where head-localization failure is now a much larger share of what remains. A genuine R3-over-R2 improvement here would be real evidence of the selector x loss interaction the R0-R3 factorial was designed to detect (HESOD-Agri-Proposal.md SS5's interpretation rules), not just two independently-additive effects.

**Open caveat, not yet resolved: how much of R2's gain is "smarter selection" vs. "selecting more area"?** Occupy nearly doubled (0.0815 -> 0.158); GFLOPs grew more modestly (+21.4%) and far less than the AP gain (+21.7% to +24.1%), which is a good sign this isn't simply "keep the whole image" in disguise (contrast TinyPerson's channel-pooled-concat, which cost +64% GFLOPs for a much smaller +3.2% AP gain) -- but a proper fixed-budget comparison (BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md SS7.8/SS9.1's equal-budget requirement) has not been run yet. A Top-K sweep on R0 vs. R2 (same method as HESOD-Experiment-Plan.md SS3's VisDrone sweep) would settle whether R2's advantage survives at matched Occupy, not just at each arm's own free-threshold operating point.

**If the gate does turn out to target this specific weakness, Pest24 is a better validation testbed for it than any aerial dataset tried so far** -- there is far more headroom here (72pp of missed Very Tiny recall vs. VisDrone's 22pp) and the failure mode (low-objectness real targets, disproportionately rare classes) matches BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md's H1 ("objectness不充分假设") more directly than anything seen on the aerial datasets. This is a reason to prioritize the coverage-vs-miss diagnostic and R2/R3, not a reason to assume the gate will help without checking.

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
| 2026-08-13 | Limit the agricultural paper to AgriPest, Pest24, and GWHD 2021 | Two pest datasets provide field/dense contrast; GWHD provides cross-task validation and a density boundary |
| 2026-08-13 | Do not create a private GWHD small-image benchmark | Preserves official comparability; small objects are evaluated as an instance stratum |
| 2026-08-13 | Use R0–R3 selector × SABL factorial | Separates routing contribution from box-regression contribution |
| 2026-08-13 | Keep hybrid/SAM labels outside the core matrix | Avoids confounding selector evidence and loss ablations |
| 2026-08-13 | Treat exact Top-K as the proposed budget mode and threshold routing as upstream-compatible baseline behavior | Prevents attributing the new budget policy to the original method |
| 2026-08-13 | Use A0/X1/X2/X3/X4 as the cross-dataset baseline set | No verified paper reports all three datasets; common code reruns provide the only defensible unified comparison |
| 2026-08-13 | Keep dataset-paper values in a separate original-protocol table | AgriPest AP50, Pest24 paper-specific mAP/mRecall, and GWHD WDA are not interchangeable |
| 2026-08-15 | Redefine R2/R3's selector from channel-pooled concat to BCRS's reliability-aware residual gate (F5/F6); demote concat to a required F1 control in a new fusion ablation ladder (SS6.2) | The prior draft's proposed method was, in BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md's own words, explicitly "a capacity-matched baseline, not the default main method" — concat has no mechanism forcing spectral evidence to activate specifically under semantic uncertainty. Independently raised by reviewer feedback converging on the same gap; BCRS SS5.3C/SS5.4 already specifies the gate, its evidence composition, and the rescue-ranking/conditional-gate-regularization losses needed to keep it from degenerating into a texture detector on Pest24 |
| 2026-08-16 | Accept Pest24 R0 as the first real result; fixed a real eval-split bug first | `run_pest24.sh`'s eval step had no `--task test`, so `test.py` silently evaluated the 5077-image val split instead of the 7600-image test split (test.py's own task-resolution logic defaults anything not in `('train','val','test')` to val). Caught because `audit_buckets.py` correctly rejected the resulting predictions rather than silently cross-checking them against the wrong split. Fixed; R0 re-evaluated cleanly on 7599 test images. See SS11.2.1 |
| 2026-08-16 | Flag Pest24's Very Tiny recall (28.34%) and rare-class recall (3.68%-98.79%, correlated with instance count) as the leading candidate explanation for R0's ~19pp AP50 gap vs. dense YOLOv5m | Confirmed via mmdetection's actual VOC-style eval config in the cloned pest-pvt repo (not inferred) that the comparison metric matches this project's `mAP@.5`; gap persisted at the same magnitude across both the (buggy) val-split and (correct) test-split measurements, ruling out split mismatch as the explanation. Whether this is selector-side (gate/concat territory) or head-side (SABL/tuning territory) is not yet determined — coverage-vs-miss diagnostic planned before R2/R3 results are over-interpreted |

## 13. Publication-facing additions

The experimental core is the same for every target journal. Add only the analysis needed for the chosen framing:

- **Computers and Electronics in Agriculture:** full three-dataset Pareto evidence, modern external baselines, clean novelty/complexity analysis.
- **Biosystems Engineering:** deeper end-to-end systems characterization, memory/energy proxy, fallback behavior under dense scenes, reproducible deployment protocol.
- **Precision Agriculture:** monitoring/counting consequences, density calibration, possible decision thresholds and domain robustness.
- **Plant Phenomics:** make GWHD/domain generalization and phenotype-count reliability central.
- **Pest Management Science / Crop Protection:** emphasize pest counts, species-specific errors, monitoring utility, and practical false-negative/false-positive consequences.

Quartile and category must be rechecked in the current Clarivate JCR immediately before submission.
