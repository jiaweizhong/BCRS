# BCRS Experiment Plan

**Source:** `BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md`  
**Status:** In Progress (Phase 0 Baseline Reproduction Verified)  
**Primary question:** Can semantic and spectral evidence allocate a fixed inference budget better than objectness alone while protecting tiny-object recall and producing real end-to-end speedups?

## 0. Baseline Benchmark & Execution Tracking

### Verification Summary (50-Epoch VisDrone ESOD Baseline)

| Metric | Target / Claim | Verified Result | Notes |
|---|---|---|---|
| **mAP@0.5** | $\ge 0.360$ (Paper 640p) | **0.5580 (55.8%)** | Evaluated at 1536x1536 resolution |
| **mAP@0.5:0.95**| Baseline | **0.3290 (32.9%)** | Standard COCO mAP metric |
| **BBox Precision (P)**| Baseline | **0.6204 (62.0%)** | Clean predictions, 0 false deadlocks |
| **BBox Recall (R)** | Baseline | **0.5374 (53.7%)** | Target for BCRS context-refinement improvement |
| **Patch BPR ($BPR_{box}$)**| $\ge 0.950$ | **0.9744 (97.4%)** | 97.44% GT boxes covered by selected patches |
| **Inference Latency** | $< 20.0\text{ms}$ | **16.5ms / img** | Batch size 1 on RTX 5090 / PyTorch 2.8+cu128 |

---

### Phase 0 — Infrastructure, reproduction, and problem confirmation

| ID | Experiment | Runs/variables | Required outputs | Status |
|---|---|---|---|---|
| E0.1 | Data and metric validation | AI-TOD, VisDrone; dense detector | Dataset manifests, visual annotation audit, official metric parity | **COMPLETED** |
| E0.2 | ESOD reproduction | Original and high-resolution dense baselines; ESOD; 50 epochs | AP/APt/APvt, FLOPs, latency, variance, checkpoints | **COMPLETED** |
| E0.3 | Selector failure audit | Objectness quantiles × size/density/texture/light bins | Object-level coverage curves and low-objectness tiny prevalence | **IN PROGRESS** |
| E0.4 | Oracle headroom | Random, objectness, GT coverage, semantic+spectral GT oracle × budget | Selector recall/AP upper-bound curves | **IN PROGRESS** |
| E0.5 | Cost calibration | Patch size/count, input size, batch size, downstream modules | Latency lookup table and predicted-vs-measured residuals | Pending |
| E0.6 | Module microbenchmarks | FFT, Sobel/Laplacian, learned depthwise, DCT/wavelet, fusion, top-k, dispatch | Median/P95 latency, memory, kernels, break-even curves | Pending |

**Go:** ESOD reproduction is within the locked tolerance; low-objectness regions contain a meaningful number of tiny targets; the GT oracle produces material headroom; and at least one spectral proxy has a plausible break-even point.  
**Stop:** There is no oracle headroom or low-objectness tiny subgroup.  
**Repair before proceeding:** Metric mismatch, unstable baseline variance, or failed timing validation.

At this gate, write `claim-thresholds.yaml`, `budget-grid.yaml`, and `latency-lookup.json`.

### Phase 1 — Fixed-budget semantic MVP and recall constraint

| ID | Comparison | Swept variables | Primary readout |
|---|---|---|---|
| E1.1 | Semantic learned priority vs objectness top-k | Four budgets | Selector recall and APt at exact top-k |
| E1.2 | No coverage loss vs coverage loss | `lambda_cov` screen `{0.1, 0.3, 1.0}`; lock one value | Miss rate, P10 coverage, violation rate, background ratio |
| E1.3 | Fixed top-k vs threshold routing | Calibrated thresholds | Budget drift and latency jitter |
| E1.4 | Pseudo-label audit | Gaussian, SAM, hybrid labels | Tiny-target coverage bias by size/objectness bin |

**Go:** At fixed patch count, learned semantic priority beats reproduced objectness and the coverage objective reduces miss risk without increasing retained actions.  
**No-Go:** If semantic priority cannot beat objectness, do not add spectral or structured routing complexity; diagnose labels, action granularity, or oracle headroom.

### Phase 2 — Dual-evidence mechanism

Use a two-stage funnel to avoid an uncontrolled Cartesian product.

**Screen with one seed and the two middle budgets:**

- spectral implementations: image/patch FFT oracle, fixed Sobel/Laplacian depthwise filters, learned multi-kernel depthwise filters, lightweight DCT/wavelet proxy;
- fusion: concatenation, gated fusion, and parameter-matched convolution;
- evidence: semantic-only, spectral-only, semantic + spectral;
- supervision: objectness target, direct GT coverage target, size-weighted target, and difficulty-weighted target.

**Confirm only the two best deployable variants plus controls** with three seeds, all budgets, and both primary datasets.

| ID | Decisive comparison | Control rule | Pass evidence |
|---|---|---|---|
| E2.1 | Semantic + spectral vs semantic-only | Match selector params/FLOPs | Low-objectness tiny recall gain with positive CI |
| E2.2 | Fused vs spectral-only/objectness/random | Exact same top-k | Better selector-recall and APt frontier |
| E2.3 | Spectral proxy vs ordinary convolution | Match width/params/training | Spectral-specific benefit; otherwise reject the spectral claim |
| E2.4 | Fusion variants | Match output action and budget | Gain justifies added measured latency |
| E2.5 | Hard-case diagnostics | Locked bins | Gain concentrates in predicted low-objectness/high-texture or low-light subgroups |

**Go:** Equal-budget gain is reproducible and a deployable spectral/fusion implementation reaches break-even.  
**No-Go:** If a matched ordinary convolution performs equally, retain the engineering improvement if useful but drop the spectral-mechanism claim. If only FFT works but no lightweight proxy breaks even, report an oracle result and stop the efficiency path.

### Phase 3 — Single-model multi-budget routing

| ID | Experiment | Comparison | Pass evidence |
|---|---|---|---|
| E3.1 | Budget embedding | Unified model vs four independently trained budget models | Per-budget APt gap within the locked non-inferiority margin |
| E3.2 | Budget sampling | Uniform vs edge-biased vs curriculum | Stable coverage and monotonic realized cost |
| E3.3 | Unseen budget interpolation | Evaluate intermediate budgets not sampled in training | Smooth, monotonic AP-cost curve without coverage collapse |
| E3.4 | Calibration | Requested vs realized cost/latency | Low budget violation and calibrated latency lookup |

Failure of H5 does not invalidate H2/H3. Fall back to per-budget models and narrow the claim.

### Phase 4 — Equal-budget end-to-end efficiency

| ID | Experiment | Alignment | Required output |
|---|---|---|---|
| E4.1 | Accuracy frontier | Fixed patch count and FLOPs | AP/APt/APvt, selector recall, non-dominated points |
| E4.2 | Deployment frontier | Fixed measured latency | AP-latency curves with confidence bands |
| E4.3 | Equal-accuracy speed | Match APt and total AP | Median/P95 speedup and memory |
| E4.4 | Resolution/density scaling | Input resolution × density bin × batch size | Break-even surface and failure regions |
| E4.5 | Overhead decomposition | All pipeline modules | Selector overhead, downstream saving, net saving |

A FLOP reduction without lower wall-clock latency falsifies the real-speedup claim. Keep it as a compute result only.

### Phase 5 — QueryDet cross-backend validation and transfer

First adapt BCRS priority to QueryDet's query candidates while preserving its CSQ context size and sparse implementation. Compare at exact query count and measured latency.

| ID | Experiment | Dataset | Primary metrics |
|---|---|---|---|
| E5.1 | QueryDet reproduction | VisDrone | APt, query count, query-center coverage, latency |
| E5.2 | Query Adapter | VisDrone | Equal-query/equal-latency delta vs original Query Head |
| E5.3 | Context sensitivity | Locked context radii, including the reproduced default | Coverage, APt, latency |
| E5.4 | Unseen-data ranking transfer | Train/calibrate on AI-TOD or VisDrone; evaluate UAVDT without fine-tuning | Rank correlation, coverage, budget violation, APt |
| E5.5 | Recalibration-only transfer | Temperature/threshold/cost recalibration, no weight update | Recovery relative to zero-shot transfer |

**Strong success:** ESOD and QueryDet show same-direction gains under their own matched budgets.  
**Boundary:** Recalibration-only success supports interface transfer, not zero-shot weight transfer.

### Phase 6 — Optional structured actions and CEASC

Only run after the core ESOD and QueryDet claims are secure.

- Joint patch/FPN/context actions vs fixed-cost patches;
- lookup-based latency cost vs FLOP cost;
- CEASC Mask Adapter at equal layer-wise activation budget;
- optional TinyPerson external validation.

Report CEASC GT-positive activation coverage, per-layer mask ratio, CE-GN global-path cost, APt, and measured latency. Do not let this phase alter the core success thresholds.

## 7. Claim thresholds

Lock numeric thresholds after Phase 0 baseline-variance measurement. Use these proposal-derived defaults unless Phase 0 demonstrates they are below ordinary noise:

- tiny selector recall: at least **+1.0 percentage point**, or relative miss-rate reduction of at least **15%**;
- APt/APvt: positive paired 95% CI on AI-TOD and VisDrone at claim-bearing budgets;
- total AP non-inferiority: lower 95% CI above `-max(0.2 AP, baseline repeatability margin)`;
- budget: zero violation for exact top-k; for latency budgets, P95 realized latency no more than 5% above request;
- generality: gain under at least two of fixed action count, fixed FLOPs, and fixed measured latency;
- overhead: added selector latency at most 5% of total and at most 10% of downstream latency saved;
- multi-budget model: APt within `max(0.3 AP, baseline repeatability margin)` of the per-budget oracle at at least three budgets;
- real acceleration: positive net median-latency saving with positive paired 95% CI, with P95 not materially worse;
- robustness: primary result holds on both AI-TOD and VisDrone and is not driven by one density bin or class.

Thresholds may become stricter after Phase 0, but must never be weakened after treatment results are inspected.

## 8. Metrics and required plots

### Detection

- AP, AP50, AP75, APvt/APt/APs as supported by the dataset;
- AR and class-wise AP;
- total-AP non-inferiority relative to the matched detector.

### Selector

- object-level selector recall;
- BPRbox and BPRcenter;
- low-objectness tiny recall;
- retained background and foreground ratios;
- coverage per GFLOP and per millisecond;
- Spearman rank correlation between learned priority and GT coverage target;
- per-image P10 coverage and catastrophic-miss rate.

Backend-specific metrics:

- ESOD: box coverage, target truncation rate, patch count;
- QueryDet: query-center coverage, context-radius coverage, per-FPN-level query count;
- CEASC: GT-positive activation coverage, per-level mask ratio, CE-GN global cost.

### Efficiency and safety

- predicted FLOPs, measured median/P95 latency, peak memory, parameter count, kernel count;
- selector overhead ratio and net latency saving;
- budget violation frequency and magnitude;
- Pareto hypervolume and the number of non-dominated operating points.

Required figures:

1. objectness quantile vs tiny miss/coverage;
2. selector recall and APt vs action count/FLOPs/latency;
3. retained ratio vs selector overhead, downstream saving, and net latency;
4. coverage CDF with emphasis on the lower tail;
5. difficult-subgroup forest plot with confidence intervals;
6. predicted vs measured latency calibration;
7. AI-TOD-to-VisDrone/UAVDT ranking-transfer plot;
8. qualitative true recoveries, false spectral triggers, truncations, and catastrophic misses.

## 9. Artifact and run contract

Every run writes to `artifacts/<experiment_id>/<run_id>/`:

| Artifact | Minimum contents |
|---|---|
| `manifest.yaml` | experiment ID, git commit, config hashes, seed, dataset checksum, host/device, start/end status |
| `config.yaml` | fully resolved training, data, model, selector, budget, and loss configuration |
| `environment.txt` | OS, Python, framework, CUDA/cuDNN, compiler, sparse kernels, package lock hash |
| `metrics.json` | dataset-level detection, selector, budget, and efficiency metrics |
| `objects.parquet` | one row per GT object with bins, priority, selected/covered flags, and matched detection |
| `images.parquet` | per-image action count, cost, latency components, coverage, density, fallback state |
| `latency.json` | protocol, warm-up, session repetitions, median/P95/IQR, raw sample location |
| `checkpoint.*` | model state plus the exact resolved config |
| `stdout.log` | complete run log and failure reason |

Maintain a central `results/registry.parquet` keyed by experiment ID, run ID, seed, dataset, backend, selector, budget, and commit. Analysis scripts must read this registry rather than hand-copied table values.

Suggested experiment ID format:

`E<phase>.<number>_<backend>_<dataset>_<selector>_b<budget>_s<seed>`

## 10. Execution schedule and resource control

| Milestone | Indicative duration | Exit artifact |
|---|---:|---|
| M0: harness, data, metric and timing validation | 1–2 weeks | Reproducible dense/ESOD baseline |
| M1: H1 audit, oracles, microbenchmarks | 1 week | Phase 0 gate report and locked thresholds |
| M2: semantic MVP and coverage constraint | 1–2 weeks | Phase 1 decision memo |
| M3: spectral/fusion screen and confirmation | 2–3 weeks | H2/H3 evidence package |
| M4: multi-budget and end-to-end profiling | 1–2 weeks | H5–H7 evidence package |
| M5: QueryDet and UAVDT transfer | 2 weeks | H8 evidence package |
| M6: optional CEASC/external validation | 1–2 weeks | Extension appendix |

Keep compute bounded with the screen/confirm funnel. Before launching any grid, estimate full-training equivalents and stop variants that are both accuracy-dominated and slower after the screening seed. Preserve controls and negative results even when stopping early.

## 11. Decision record after each phase

Each gate produces a short immutable memo containing:

1. configs and run IDs included/excluded, with reasons;
2. predefined criteria and observed confidence intervals;
3. decision: Go, Repair, Narrow Claim, or Stop;
4. falsified hypotheses and retained hypotheses;
5. the next phase's locked variants and compute budget;
6. any deviation from this plan, timestamped before the affected result is viewed.

The project succeeds minimally only if the equal-budget selector improvement is reproducible on AI-TOD and VisDrone, total AP remains non-inferior, and added selector cost has a credible break-even point. It succeeds strongly if the improvement forms a better measured-latency frontier, transfers to QueryDet, and remains stable under dataset shift or recalibration. Negative results listed in the proposal—no matched spectral benefit, no priority/coverage correlation, gains caused only by retaining more background, no oracle headroom, no real speedup, unstable multi-budget behavior, or single-bin dependence—must be reported as falsification rather than hidden by metric selection.
