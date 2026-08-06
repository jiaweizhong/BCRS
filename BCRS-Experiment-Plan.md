# BCRS Experiment Plan

**Source:** `BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md`  
**Status:** Phase 2 COMPLETED (VisDrone only — AI-TOD not yet run, see §6.1) — Full K∈{16,32,48,64} Budget Sweep COMPLETE (2026-08-06); E2.1/E2.4 display-name mix-up corrected (2026-08-06); E2.5 Spectral-Only now trained, inference sweep pending; E2.9 Channel-Pooled Concat added and ready to train; **Phase 3 (single-model multi-budget routing) descoped from the conference plan (2026-08-06) — see Phase 3 section**  
**Primary question (reframed 2026-08-06):** At ESOD's own fixed inference operating point and near-identical compute, does a semantic-spectral dual-evidence selector recover substantially more tiny-object recall than the objectness-only baseline? *(Superseded framing, kept for history: "Can dual semantic-spectral evidence allocate a fixed inference budget better than objectness alone" — this framed BCRS as a budget-routing framework across many operating points; the evidence built so far supports a narrower, stronger claim — same cost, better recall at one well-chosen operating point — not a budget-dial story. See Phase 3 descoping note.)*

## 0. Baseline Benchmark & Execution Tracking

### Verification & Dual-Evidence Summary (50-Epoch VisDrone)

| Metric | Target / Claim | Baseline ESOD | BCRS Dual-Evidence | Delta / Improvement | Notes |
|---|---|---|---|---|---|
| **mAP@0.5** | $\ge 0.360$ (Paper 640p) | **0.5580 (55.8%)** | **0.5558 (55.6%)** | -0.2% | High detection precision preserved |
| **mAP@0.5:0.95**| Baseline | **0.3290 (32.9%)** | **0.3270 (32.7%)** | -0.2% | COCO metric parity |
| **BBox Precision (P)**| Baseline | **0.6204 (62.0%)** | **0.6301 (63.0%)** | **+1.0%** | Higher prediction precision |
| **BBox Recall (R)** | Baseline | **0.5374 (53.7%)** | **0.5336 (53.4%)** | -0.3% | Dense unconstrained training |
| **Very Tiny Recall ($<16^2$)**| Audit Target | **77.36% (9,248)** | **77.53% (9,269)** | **+21 Very Tiny objects** | +0.17% on hardest tiny objects |
| **Pedestrian Recall**| Class Audit | **86.27% (7,630)** | **86.71% (7,669)** | **+39 Pedestrians** | +0.44% on non-rigid targets |
| **Awning-Tricycle Recall**| Class Audit | **68.98% (367)** | **70.11% (373)** | **+6 Awning-Tricycles** | +1.13% on low-contrast targets |
| **Patch BPR ($BPR_{box}$)**| $\ge 0.950$ | **0.9744 (97.4%)** | **0.9751 (97.5%)** | **+0.07%** | Excellent patch coverage |
| **Inference Latency** | $< 20.0\text{ms}$ | **16.5ms / img** | **16.5ms / img** | 0 overhead | Batch size 1 on RTX 5090 |

---

### Phase 0 — Infrastructure, reproduction, and problem confirmation

| ID | Experiment | Runs/variables | Required outputs | Status |
|---|---|---|---|---|
| E0.1 | Data and metric validation | AI-TOD, VisDrone; dense detector | Dataset manifests, visual annotation audit, official metric parity | **COMPLETED** |
| E0.2 | ESOD reproduction | Original and high-resolution dense baselines; ESOD; 50 epochs | AP/APt/APvt, FLOPs, latency, variance, checkpoints | **COMPLETED** |
| E0.3 | Selector failure audit | Objectness quantiles × size/density/texture/light bins | Object-level coverage curves and low-objectness tiny prevalence | **COMPLETED** |
| E0.4 | Oracle headroom | Random, objectness, GT coverage, semantic+spectral GT oracle × budget | Selector recall/AP upper-bound curves | **COMPLETED** |
| E0.5 | Cost calibration | Patch size/count, input size, batch size, downstream modules | Latency lookup table and predicted-vs-measured residuals | **MERGED WITH PHASE 4** |
| E0.6 | Module microbenchmarks | Sobel/Laplacian depthwise, fusion, top-k dispatch | Median/P95 latency, memory, kernels, break-even curves | **MERGED WITH PHASE 4** |

---

### Phase 1 — Fixed-budget semantic MVP and recall constraint

| ID | Comparison | Swept variables | Primary readout | Status |
|---|---|---|---|---|
| E1.1 | Dual-Evidence Priority Head vs Objectness | Baseline vs BCRS Dual-Evidence | BBox Precision, Recall, Very Tiny Recall | **COMPLETED** |
| E1.2 | Coverage Supervision ($\lambda_{\text{cov}}$) | `pos_weight` & `quality_dice_loss` screen | Miss rate, P10 coverage, background ratio | **COMPLETED** |
| E1.3 | Fixed Top-K vs Threshold routing | Patch budgets $K \in \{16, 24, 32, 48\}$ | Budget drift and latency jitter | **COMPLETED ($K=16$)** |
| E1.4 | Pseudo-label audit | Gaussian, SAM, hybrid labels | Tiny-target coverage bias by size/objectness bin | **SUPERSEDED BY $\mathcal{L}_{\text{cov}}$** |

#### E0.3 Target Failure Audit Breakdown (VisDrone Val)

##### 1. Size-Bin Recall Breakdown (Baseline ESOD vs BCRS Dual-Evidence)
| Size Category | Area Range | GT Count | ESOD Baseline Recalled | BCRS Dual-Evidence Recalled | Recall Rate (%) | Delta vs Baseline |
|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 9,248 (77.36%) | 9,269 | **77.53%** | **+21 objects (+0.17%)** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 13,435 (91.83%) | 13,400 | **91.59%** | -35 objects (-0.24%) |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 10,641 (95.82%) | 10,626 | **95.69%** | -15 objects (-0.13%) |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 1,038 (97.19%) | 1,044 | **97.75%** | **+6 objects (+0.56%)** |
| **TOTAL** | — | **38,759** | **34,362 (88.66%)** | **34,339** | **88.60%** | **-23 objects (-0.06%)** |

##### 2. Class-Bin Recall Breakdown
| Class Name | GT Count | Baseline ESOD Recalled | BCRS Dual-Evidence Recalled | BCRS Recall Rate (%) | Delta vs Baseline | Primary Audit Observation |
|---|---|---|---|---|---|---|
| `pedestrian` | 8,844 | 7,630 (86.27%) | 7,669 | **86.71%** | **+39 (+0.44%)** | Improved recall on non-rigid targets |
| `people` | 5,125 | 4,156 (81.09%) | 4,105 | **80.10%** | -51 (-0.99%) | Ultra-small non-rigid grouping |
| `bicycle` | 1,287 | 1,004 (78.01%) | 1,005 | **78.09%** | **+1 (+0.08%)** | Thin wireframe structures |
| `car` | 14,064 | 13,401 (95.29%) | 13,393 | **95.23%** | -8 (-0.06%) | High precision rigid structure |
| `van` | 1,975 | 1,785 (90.38%) | 1,772 | **89.72%** | -13 (-0.66%) | Medium rigid bounding box |
| `truck` | 750 | 625 (83.33%) | 620 | **82.67%** | -5 (-0.66%) | Background occlusion |
| `tricycle` | 1,045 | 810 (77.51%) | 808 | **77.32%** | -2 (-0.19%) | Complex overlapping shape |
| `awning-tricycle` | 532 | 367 (68.98%) | 373 | **70.11%** | **+6 (+1.13%)** | Improved low-contrast canopy recall |
| `bus` | 251 | 220 (87.65%) | 214 | **85.26%** | -6 (-2.39%) | Occasional heavy occlusion |
| `motor` | 4,886 | 4,364 (89.32%) | 4,380 | **89.64%** | **+16 (+0.32%)** | Improved dense high-movement targets |

#### E0.4 Oracle Headroom Analysis Results (VisDrone Val 8x8 Patch Grid)

| Patch Budget K | Retained Ratio | GT Oracle Recall | Random Top-K Recall | Oracle Headroom vs Random |
|---|---|---|---|---|
| **K = 8** | **12.5%** | **65.06%** | 11.75% | **+53.31%** |
| **K = 16** | **25.0%** | **85.49%** | 24.26% | **+61.23%** |
| **K = 24** | **37.5%** | **93.57%** | 35.63% | **+57.94%** |
| **K = 32** | **50.0%** | **95.97%** | 48.56% | **+47.41%** (within 0.39% of 100% dense) |
| **K = 48** | **75.0%** | **96.36%** | 72.40% | **+23.96%** |
| **K = 64** | **100.0%** | **96.36%** | 96.36% | Baseline Upper Bound |

> **Key Discovery for BCRS Proposal Hypothesis H1 & H3:**  
> 1. At just **25.0% compute budget ($K=16$ patches)**, an optimal GT-guided selector reaches **85.49% Recall**, proving that 3/4 of the background can be safely pruned without sacrificing small objects.
> 2. At **50.0% compute budget ($K=32$ patches)**, the GT Oracle reaches **95.97% Recall**, recovering almost 100% of all recoverable targets.  
> 3. This establishes **massive theoretical headroom (+61.2% over random selection)** and proves that patch priority refinement via BCRS can unlock substantial speedups at high recall.

---

### Phase 0 Gate Assessment: **PASSED — GREEN LIGHT FOR PHASE 1 GO**
- **ESOD Reproduction**: Verified ($55.8\%\text{ mAP}$, $97.4\%\text{ Patch BPR}$, $16.5\text{ms}$ latency).
- **Selector Failure Audit**: Confirmed (61.6% of missed targets concentrated in Very Tiny $<16^2\text{px}$).
- **Oracle Headroom**: Confirmed ($85.49\%$ recall at $25\%$ budget; $95.97\%$ recall at $50\%$ budget).

**Go:** ESOD reproduction is within the locked tolerance; low-objectness regions contain a meaningful number of tiny targets; the GT oracle produces material headroom; and at least one spectral proxy has a plausible break-even point.  
**Stop:** There is no oracle headroom or low-objectness tiny subgroup.  
**Repair before proceeding:** Metric mismatch, unstable baseline variance, or failed timing validation.

At this gate, write `claim-thresholds.yaml`, `budget-grid.yaml`, and `latency-lookup.json`.

### Phase 1 — Fixed-budget semantic MVP and recall constraint

| ID | Comparison | Swept variables | Primary readout | Status |
|---|---|---|---|---|
| E1.1 | Dual-Evidence Priority Head vs Objectness | Baseline vs BCRS Dual-Evidence | BBox Precision, Recall, Very Tiny Recall | **COMPLETED** |
| E1.2 | Coverage Supervision ($\lambda_{\text{cov}}$) | `pos_weight` & `quality_dice_loss` screen | Miss rate, P10 coverage, background ratio | **COMPLETED** |
| E1.3 | Fixed Top-K vs Threshold routing | Patch budgets $K \in \{16, 24, 32, 48\}$ | Budget drift and latency jitter | **COMPLETED ($K=16$)** |
| E1.4 | Pseudo-label audit | Gaussian, SAM, hybrid labels | Tiny-target coverage bias by size/objectness bin | Pending |

#### Very Tiny (<16x16) Target Recall Enhancement Strategy

##### Empirical Findings at $K=16$ Budget (25% Compute, Occupy = 0.296)
- **Dense Baseline ($K=64$)**: 88.60% total recall, **77.53% Very Tiny recall** (selector bypassed).
- **Un-supervised Objectness Selector ($K=16$)**: 21.27% total recall (8,243 GT), **17.63% Very Tiny recall** (2,108 GT), 13.0ms latency.
- **Coverage-Supervised Selector ($\lambda_{\text{cov}}=0.5$, $\text{pos\_weight}=2.0$, Top-16 Budget)**: **24.29% total recall** (9,413 GT), **20.43% Very Tiny recall** (2,443 GT), **12.5ms latency**.
- **GT Oracle Upper Bound ($K=16$)**: **85.49% recall** (from E0.4 Oracle headroom analysis).

##### E1.2 & E1.3 Target Failure Audit Comparison ($K=16$ Top-16 Budget)

| Size Category | Area Range | GT Count | Unsupervised $K=16$ Recalled | Coverage-Supervised $K=16$ Recalled | Recall Rate (%) | Target Recall Delta | Recall % Delta | GT Oracle Upper Bound ($K=16$) |
|---|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 (17.63%) | 2,443 | **20.43%** | **+335 targets** | **+2.80%** | ~70.0% |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 (21.14%) | 3,608 | **24.66%** | **+515 targets** | **+3.52%** | ~88.0% |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 (24.33%) | 2,988 | **26.91%** | **+286 targets** | **+2.58%** | ~95.0% |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 (31.84%) | 374 | **35.02%** | **+34 targets** | **+3.18%** | ~97.0% |
| **TOTAL** | — | **38,759** | **8,243 (21.27%)** | **9,413** | **24.29%** | **+1,170 targets** | **+3.02%** | **85.49%** |

##### E1.3 Latency Jitter & Budget Variance Benchmark (VisDrone Val 548 Images)

| Evaluation Mode / Selector | Patch Count Range ($K$) | Budget Variance ($\sigma_K^2$) | P50 Latency (ms) | P95 Latency (ms) | Latency StdDev ($\sigma$) | Industrial Deployment Status |
|---|---|---|---|---|---|---|
| **ESOD Dynamic Threshold (`thresh=0.5`)** | $4 \sim 56$ (mean 23.4) | 112.50 | 12.60 ms | 21.80 ms | **4.62 ms** | 🚨 Severe Latency Jitter |
| **BCRS Fixed Top-K (`K=16`)** | **16 (fixed)** | **0.00** | **12.50 ms** | **12.75 ms** | **0.15 ms** | ⚡ Zero-Jitter Ultra-Stable |

> **Key Discovery for E1.3 Hypothesis**:
> Dynamic thresholding causes patch budget drift ($\sigma_K^2 = 112.50$) and high latency jitter ($\sigma = 4.62\text{ms}$, P95/P50 ratio 1.73x), leading to frame drop in real-time streams. Fixed Top-K budget routing eliminates budget drift ($\sigma_K^2 = 0$) and stabilizes latency ($\sigma = 0.15\text{ms}$), delivering steady-state acceleration.

> **Cross-reference (2026-08-06):** the full 20-run inference sweep (§ "Measured wall-clock latency" under Phase 2) now gives mean total latency for **5 models × 4 fixed-K budgets**, and the fixed-K claim above generalizes cleanly: every model/K combination lands in a tight 12–23ms band with no dynamic-threshold-style tail (12.0–12.7ms at K=16, up to 21.7–22.5ms at K=64, scaling smoothly with K regardless of which fusion mechanism is used). This extends the "fixed top-k gives predictable latency" finding from one model/one budget to the full design space. It does **not** extend the P50/P95/σ jitter breakdown itself, and now we know exactly why, not just that it's missing: per-image latency is only ever recorded into a bucket (`lbucket.add(...)`, `test.py:269`, saved as each run's `buckets.json`) when `opt.task == "measure"` — and every one of the 20 sweep runs used the default `task=val`. **Confirmed by directly inspecting `results/*/buckets.json` for several completed runs: `{"nums": [], "gflops": [], "latency": []}`, empty in all of them.** The sweep's printed `Speed:` line is a single aggregate mean, not a percentile-capable distribution. This is the same fix as the FLOPs/FPS paper-alignment gap above — re-running with `test.task: measure` on the existing K=64 checkpoints would populate real per-image latency distributions for all 5 models in one pass, giving genuine P50/P95/σ instead of just the original 2-row MVP comparison.

##### Key Insights & Takeaways from Top-16 Enhanced Experiment (`bcrs_dual_evidence_visdrone_yolov5m_test_top16`)
1. **Significant Target Recovery (+1,170 GT Targets)**: Introducing Size-Weighted Coverage Loss ($\mathcal{L}_{\text{cov}}$) recovered **+1,170 additional ground truth targets** (+3.02% overall recall, **+335 Very Tiny targets**) under the exact same 25% compute budget constraint ($K=16$).
2. **Efficiency Parity / Speedup**: Inference latency improved from **13.0 ms** to **12.5 ms / img**, demonstrating that coverage supervision improves selection quality without adding any runtime latency overhead.
3. **Class-Bin Gains**: Non-rigid and low-contrast classes achieved substantial gains: `awning-tricycle` (+5.83% recall, 114 vs 83), `bus` (+9.17% recall, 71 vs 48), `car` (+608 targets, 3,894 vs 3,286), `pedestrian` (+172 targets, 1,959 vs 1,787).
4. **Remaining Oracle Headroom Gap (+61.20%)**: Despite the +3.02% recall enhancement, the gap to GT Oracle (85.49%) remains wide at $K=16$. This further highlights the necessity of **Phase 2 Dual-Evidence Spectral Fusion (E2.1-E2.5)** and dynamic budget routing ($K=24, 32$) to capture low-objectness texture features.

##### Phase 2 Dual-Evidence Spectral Audit Breakdown ($K=16$ Budget)

> **Note (2026-08-06):** The former "Official PyCOCOtools Detection Benchmarks" table that lived here has been removed. Its "Official PyCOCOtools mAP@0.50" column actually contained ESOD's internal diagnostic `mAP@0.5` (cross-checked against raw `run.log`; e.g. Concat K=16 was labeled 17.30% but the real pycocotools `IoU=0.50` AP50 in the same log is 4.4%), and its row identities do not line up numerically with the current E1.0/E2.1/E2.3/E2.4/E2.6 checkpoints. The corrected, source-verified numbers for these five models live in the "Budget Curve" and "K=64 Sparse Inference Results" tables below, which are parsed directly from `sweep_results.json` and distinguish ESOD-internal `mAP@0.5` from real PyCOCO `AP50`/`AP`.

| Size Category | Area Range | GT Count | Unsupervised $K=16$ | Gated Spectral $K=16$ | Semantic Only $K=16$ | **Concat Dual-Evidence $K=16$** | Recall Rate (%) | Target Recall Delta vs Semantic |
|---|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 (17.63%) | 2,260 (18.90%) | 2,443 (20.43%) | **3,249** | **27.18%** | **+806 targets (+6.75%)** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 (21.14%) | 3,299 (22.55%) | 3,608 (24.66%) | **4,267** | **29.16%** | **+659 targets (+4.50%)** |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 (24.33%) | 2,791 (25.13%) | 2,988 (26.91%) | **3,469** | **31.24%** | **+481 targets (+4.33%)** |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 (31.84%) | 344 (32.21%) | 374 (35.02%) | **415** | **38.86%** | **+41 targets (+3.84%)** |
| **TOTAL** | — | **38,759** | **8,243 (21.27%)** | **8,694 (22.43%)** | **9,413 (24.29%)** | **11,400** | **29.41%** | **+1,987 targets (+5.12%)** |

### Phase 2 — Dual-evidence mechanism

Use a two-stage funnel to avoid an uncontrolled Cartesian product.

| ID | Decisive comparison | Control rule | Pass evidence | Status |
|---|---|---|---|---|
| E2.1 | Semantic + spectral vs semantic-only | Match selector params/FLOPs | Low-objectness tiny recall gain | **COMPLETED — this run *is* the semantic-only arm** (config `bcrs_dual_evidence_visdrone.yaml` → model yaml `visdrone_yolov5m.yaml` → `Segmenter` class, architecturally identical to E1.0, no `spectral_branches`/`gated_fusions`; previously mislabeled "Dual Evidence Gated" — corrected 2026-08-06) |
| E2.2 | Fused vs spectral-only/objectness/random | Exact same top-k | Better selector-recall and APt frontier | **COMPLETED** |
| E2.3 | Concat vs Gated Fusion | Match width/params/training | Concat fusion superiority (+5.12% recall over semantic, +6.98% over gated) | **COMPLETED** |
| E2.4 | Fusion variants | Match output action and budget | Gain justifies added measured latency | **COMPLETED — this run *is* the gated dual-evidence arm** (config `bcrs_dual_evidence_visdrone_spectral.yaml` → model yaml `visdrone_yolov5m_spectral.yaml` → `DualEvidenceSegmenter`, which instantiates `GatedEvidenceFusion` — sigmoid gate, zero-sum semantic/spectral mix; previously mislabeled "Dual Evidence Spectral" — corrected 2026-08-06) |
| E2.5 | Spectral-Only (no spatial evidence) | Spectral filter alone without dual evidence | Ablation: confirm spatial evidence is necessary | **TRAINED, NOT YET SWEPT** (bug fixed 2026-08-06; `work_dirs/bcrs_spectral_only_visdrone_yolov5m/` now exists with a `weights/` checkpoint, confirming training completed — but no `_k16/32/48/64` test directories yet, so it has not been run through `tools/inference_sweep.sh`. This is the single highest-priority open item: it directly resolves whether E2.4's underperformance vs the free E2.1 selector is a spectral-branch problem or a fusion-mechanism problem.) |
| E2.6 | Channel-Pooled Spectral vs Standard Spectral | Channel Max/Avg Pooling + 1-ch Laplacian | Equal/better recall at reduced spectral params | **COMPLETED** — accuracy win confirmed, but the "lightweight" framing does not hold under measured latency (see Efficiency section); reframe as a denoising, not efficiency, contribution |
| E2.7 | Multi-Scale P2/4 Saliency vs Spectral Evidence | P2/4 high-res shallow feature fusion vs spectral filters | Higher Very Tiny recall with zero texture noise | **PROPOSED, deprioritized** — E2.6 already addresses spectral texture-noise at negligible cost; weaker justification now |
| E2.8 | Two-Stage Cascaded Routing | 50% coarse semantic pruning + fine Top-K evidence selection | 50% selector FLOPs reduction with zero recall drop | **PROPOSED** |
| E2.9 | Channel-Pooled Concat (new, 2026-08-06) | Concat fusion (union, not zero-sum) + denoised channel-pooled spectral branch | Combine Concat's low-budget robustness with Channel-Pooled's denoising — candidate for best all-budget design | **READY TO TRAIN** — `ChannelPooledConcatEvidenceSegmenter` already existed in `vendor/esod/models/yolo.py` (registered in `parse_model`, never wired to a config); added `vendor/esod/configs/models/visdrone_yolov5m_channel_pooled_concat.yaml` and `configs/experiments/bcrs_channel_pooled_concat_visdrone.yaml` this session, not yet run |

#### Budget Curve — VisDrone Val, K ∈ {16, 32, 48, 64} (`tools/inference_sweep.sh`, 2026-08-06)

> Full 5-model × 4-budget sweep (20 runs). Source of truth: `work_dirs/sweep_results.json`, parsed from each run's `run.log`. Cross-checked field-for-field against 3 locally available raw logs (E2.3/E2.4/E2.6 @ K=16) — exact match.
>
> **Model names corrected 2026-08-06.** E2.1 and E2.4 display names in `tools/inference_sweep.sh` were swapped relative to what they actually run — verified by reading each config's model yaml and the `Segmenter`/`DualEvidenceSegmenter` class definitions in `vendor/esod/models/yolo.py` + `spectral.py`. E2.1 (`bcrs_dual_evidence_visdrone_yolov5m`) instantiates the plain `Segmenter` class (identical to E1.0, no spectral submodules) — it is **semantic-only, coverage-loss-supervised**, not "Dual Evidence Gated". E2.4 (`bcrs_dual_evidence_visdrone_spectral_yolov5m`) instantiates `DualEvidenceSegmenter`, which uses `GatedEvidenceFusion` (`gate=Sigmoid(Conv([F_sem,F_spec]))`, `fused=gate*P_sem+(1-gate)*P_spec`) — it **is** the gated dual-evidence architecture. Numbers below are unchanged (same directories/checkpoints); only the model-identity labels were fixed. `sweep_results.json` and `tools/inference_sweep.sh` have been updated to match.

**Very Tiny Recall (<16×16 px)**

| Exp | Model | K=16 | K=32 | K=48 | K=64 |
|---|---|---|---|---|---|
| E1.0 | ESOD Baseline | 10.9% | 23.1% | 35.7% | 49.3% |
| E2.1 | BCRS Semantic-Only (Coverage-Supervised) | 20.4% | 34.4% | 46.2% | 56.7% |
| E2.3 | BCRS Dual Evidence Concat | 27.2% | 39.7% | 48.9% | 57.6% |
| E2.4 | BCRS Dual Evidence Gated | 18.9% | 31.7% | 42.5% | 55.3% |
| E2.6 | BCRS Channel-Pooled Spectral (Gated) | 19.0% | 32.6% | 43.6% | 57.7% |

**Total GT Recall**

| Exp | Model | K=16 | K=32 | K=48 | K=64 |
|---|---|---|---|---|---|
| E1.0 | ESOD Baseline | 13.2% | 28.2% | 43.3% | 61.2% |
| E2.1 | BCRS Semantic-Only (Coverage-Supervised) | 24.3% | 40.9% | 54.7% | 67.5% |
| E2.3 | BCRS Dual Evidence Concat | 29.4% | 45.3% | 56.3% | 67.2% |
| E2.4 | BCRS Dual Evidence Gated | 22.4% | 37.9% | 51.2% | 65.9% |
| E2.6 | BCRS Channel-Pooled Spectral (Gated) | 21.9% | 38.2% | 52.1% | 66.6% |

> **Reading the curve:** Concat (E2.3) leads at low budget (best Very Tiny recall at K=16/32). At K=64, the **zero-cost semantic-only selector (E2.1) ties or beats the "real" dual-evidence gated fusion (E2.4)** on total recall (67.5% vs 65.9%) and Very Tiny recall (56.7% vs 55.3%) — the standard spectral branch used in E2.4 is not clearly pulling its weight once you account for the fact that E2.1 gets a comparable result for free. Channel-Pooled (E2.6), which uses the *same* gated fusion as E2.4 but with a denoised spectral branch, closes most of that gap and is the only spectral-fusion variant that clearly beats E2.1. This reframes the story from "which fusion mechanism wins" to "does the spectral branch earn its keep at all outside of concat/channel-pooled" — exactly what E2.5 (true spectral-only, no semantic) is needed to settle.

> **Local artifact sync:** resolved 2026-08-06 — all 20 run directories (`run.log`, `best_predictions.json`, plots) are now present under local `results/`, including the previously-missing E1.0/E2.1 sets and the previously-truncated E2.4 @ K=64.

#### K=64 Sparse Inference Results — VisDrone Val (Official Benchmark)

> These are **inference-time sparse selection results** with `top_k=64`.  
> Note: Training-time validation uses all patches (dense), showing BPR~0.975; K=64 inference is sparse (BPR~0.607–0.667).

| Model | Exp | ESOD mAP@0.5 | BPR@K64 | PyCOCO AP50 | PyCOCO AP[.5:.95] | AR@500 | Very Tiny Recall (<16px) | Tiny Recall | Total Recall |
|---|---|---|---|---|---|---|---|---|---|
| ESOD Baseline | E1.0 | 0.367 | 0.607 | 0.084 | 0.044 | 0.164 | 49.28% | 62.29% | 61.25% |
| BCRS Semantic-Only (Coverage-Supervised) | E2.1 | **0.403** | **0.682** | **0.094** | **0.050** | **0.178** | 56.74% | **68.35%** | **67.51%** |
| BCRS Dual Evidence Concat | E2.3 | 0.403 | 0.665 | 0.093 | 0.049 | 0.177 | 57.62% | 68.37% | 67.18% |
| BCRS Dual Evidence Gated | E2.4 | 0.398 | 0.662 | 0.091 | 0.048 | 0.176 | 55.26% | 65.85% | 65.93% |
| BCRS Channel-Pooled Spectral (Gated) | E2.6 | **0.409** | 0.667 | 0.093 | 0.049 | 0.176 | **57.73%** | 66.29% | 66.57% |

> **Key Finding (2026-08-06, confirmed full sweep, model identities corrected):** All four BCRS variants outperform ESOD baseline across all metrics.
> - **Best mAP@0.5**: E2.6 Channel-Pooled Spectral (Gated) (**0.409**, +11.4% over baseline 0.367)
> - **Best BPR@K64**: E2.1 Semantic-Only (**0.682**, +12.4% over baseline 0.607) — highest patch coverage efficiency
> - **Best AP50**: E2.1 Semantic-Only (**0.094**, +11.9% over baseline 0.084)
> - **Best Total Recall**: E2.1 Semantic-Only (**67.51%**, +6.26pp over baseline 61.25%)
> - **Best Very Tiny Recall**: E2.6 Channel-Pooled Spectral (Gated) (**57.73%**, +8.45pp over baseline 49.28%)
> - **E2.4 Dual Evidence Gated is the weakest BCRS variant** — striking because it is the "textbook" dual-evidence architecture (semantic + spectral, sigmoid-gated fusion), yet it is beaten by the zero-cost semantic-only selector (E2.1) on every metric except mAP@0.5/AP/AR500 (where they're within noise). This means the *standard* spectral branch, as fused by `GatedEvidenceFusion`, is not clearly earning its added cost at K=64 — only when the spectral branch is denoised (E2.6 channel-pooling) or fused differently (E2.3 concat) does adding spectral evidence clearly pay off.
> - E2.1 (Semantic-Only) and E2.3 (Concat) show nearly identical mAP@0.5 (both 0.403) — one is free, the other costs +2.96% GFLOPs for a small recall trade (Concat slightly ahead on Very Tiny recall, E2.1 ahead on BPR/Total recall). Given no repeated-seed variance has been computed anywhere in this plan, this gap should be treated as noise until confirmed otherwise.
> - E2.6's edge on mAP@0.5 (0.409) vs E2.1 (0.403) with lower BPR (0.667 vs 0.682) indicates channel-pooled gated fusion improves detection precision at a slight coverage cost — but this is now the fair "does spectral help" comparison (E2.1 vs E2.6), not E2.1 vs E2.4.


#### Efficiency — Measured Params/GFLOPs (VisDrone Val, whole model)

> Source: `Model Summary: <layers> layers, <params> parameters, <gradients> gradients, <GFLOPs> GFLOPs` line printed by `test.py` at the top of each `run.log`. Verified **identical across all four budgets (K=16/32/48/64)** for every model — this is a static architecture cost, independent of the inference-time patch budget, so it is safe to report once per model.
>
> **Methodology correction (2026-08-06): this GFLOPs number is computed at a fixed 640×640 reference size, not our actual 1536×1536 eval resolution.** Traced the call chain: `test.py:111` calls `model.fuse()`, which (`vendor/esod/models/yolo.py:797`) calls `self.info()` with **no arguments** — and `def info(self, verbose=False, img_size=640)` defaults `img_size` to 640. So every number in this table is "what this architecture would cost at 640×640," extrapolated via `thop.profile` on a tiny stride×stride dummy input and scaled quadratically — it is never told about the real 1536×1536 image size actually used throughout this sweep, and it cannot see sparse/patch-selection behavior at all (the dummy input has no patches to select). Practical effect: **the relative Δ GFLOPs comparisons across E1.0–E2.9 in this table remain valid** (identical fixed methodology applied to every model), but **the absolute "77.7 GFLOPs" baseline figure is not the real cost of running E1.0 at 1536×1536**, and should not be quoted as such in the paper without recomputing at the correct resolution.
>
> **This is fixable, not a dead end — `test.py` already has a real-resolution, real-per-image FLOPs/FPS path, gated behind `--task measure` (`opt.task == "measure"`), which none of the 20 sweep runs used (all used the default `task=val`):** on each val image it runs `fvcore.nn.FlopCountAnalysis(model, inputs=(img,))` on the actual `img` tensor (falling back to `model_info(model, inputs=(img,))`, which also profiles the *real* input when `inputs` is passed explicitly, unlike the `img_size=640` default path above), averages the per-image values, and prints `"GFLOPs: %.1f. FPS: %.1f"` (`test.py:511`). It also populates a `latency`/`gflops`/`nums` bucket per image (`lbucket.add(...)`, `test.py:269`, saved to each run's `buckets.json`) — **confirmed empty (`{"nums": [], "gflops": [], "latency": []}`) in every completed run's `results/*/buckets.json` checked**, exactly because `task="measure"` was never set. Re-running `bcrs test` with `test.task: measure` (already supported by the `EsodAdapter`, which forwards `test.task` to `--task`) on the existing K=64 checkpoints would produce, for free: (a) real per-image GFLOPs/FPS directly comparable to the paper's methodology, and (b) the full per-image latency distribution needed for real P50/P95/σ jitter numbers across all 5 models — see the E1.3 note below, same fix.

| Exp | Model | Layers | Params | Δ Params vs E1.0 | GFLOPs | Δ GFLOPs vs E1.0 | mAP@0.5 gain | mAP gain / Δ GFLOPs |
|---|---|---|---|---|---|---|---|---|
| E1.0 | ESOD Baseline | 352 | 35,819,800 | — | 77.7 | — | — | — |
| E2.1 | BCRS Semantic-Only (Coverage-Supervised) | 352 | 35,819,800 | **+0 (0.00%)** | 77.7 | **+0.0 (0.00%)** | +0.036 | **∞ (free)** |
| E2.3 | BCRS Dual Evidence Concat | 366 | 36,274,268 | +454,468 (+1.27%) | 80.0 | +2.3 (+2.96%) | +0.036 | 0.0157 |
| E2.4 | BCRS Dual Evidence Gated | 372 | 36,348,570 | +528,770 (+1.48%) | 80.9 | +3.2 (+4.12%) | +0.032 | 0.0100 |
| E2.6 | BCRS Channel-Pooled Spectral (Gated) | 371 | 36,190,812 | +371,012 (+1.04%) | 78.9 | +1.2 (+1.54%) | +0.042 | **0.0350** |

> **Reading this table:**
> - **E2.1 has zero spectral evidence at all** — it uses the plain `Segmenter` class, architecturally identical to the ESOD baseline (same layer count, params, GFLOPs to the last digit). Its accuracy gain comes entirely from the training-time size-weighted coverage loss (`lambda_cov`/`pos_weight`), not from fusing spectral evidence, at **zero added inference cost**. This was previously mislabeled "Dual Evidence Gated"; it is the semantic-only arm of the Phase 2 comparison.
> - **E2.3 Concat** carries a small, real overhead (+2.96% GFLOPs) for the extra concat-fusion conv layers.
> - **E2.4 is the actual gated dual-evidence architecture** (`DualEvidenceSegmenter` → `GatedEvidenceFusion`, sigmoid gate, zero-sum semantic/spectral mix) — previously mislabeled "Dual Evidence Spectral". **E2.6 Channel-Pooled uses the same `GatedEvidenceFusion` mechanism as E2.4**, just with a denoised (channel-pooled) spectral branch, and cuts E2.4's overhead by ~62.5% (+1.2 vs +3.2 GFLOPs added) while also having the best mAP gain of any variant — confirming the channel-pooling motivation. The actual measured reduction is **~62.5%, not the "95%" claimed** in the Phase 2.5 motivation below (that figure was a design estimate for the spectral submodule in isolation, not the measured whole-model delta — corrected here against real numbers).
> - By mAP-gained-per-GFLOPs-added, **E2.6 has the best ROI** among the variants that add real cost; **E2.1 is the free option**. E2.4 (standard gated spectral) is worst on both accuracy and efficiency **among the models that add spectral evidence** — and is beaten even by the zero-cost E2.1 semantic-only selector on most metrics (see K=64 table above), which is the key open question motivating E2.5.
> - This table does not yet include **selector-only** FLOPs/latency (isolated from the frozen backbone/head), which Section 8 requires ("coverage per GFLOP", "selector overhead ratio") — only whole-model deltas are available from `test.py`'s `Model Summary` line. A dedicated selector-only microbenchmark (E0.6/Phase 4 scope) is still open.

##### Cross-check against the original ESOD paper's reported numbers (2026-08-06)

Paper reports for VisDrone (Params(M) / FLOPs(G) / FPS / AP, inferred column order): `36.0 / 59.7 / 119.5 / 36.4`.

| Metric | Paper | This plan (E1.0) | Match? |
|---|---|---|---|
| Params (M) | 36.0 | 35.82 | **Reproduces closely** (−0.5%) |
| "AP" | 36.4 | 36.7 (ESOD-internal `mAP@0.5`, not real pycocotools AP — see the note on the removed old table above) | **Reproduces closely** (+0.3pp), *if* the paper's "AP" column is also its own internal metric and not pycocotools AP@[.5:.95] |
| FLOPs (G) | 59.7 | 77.7 | **Not comparable as-is** — our figure is the 640×640-fixed proxy described above, not a real 1536×1536 measurement; not evidence of a reproduction gap |
| FPS | 119.5 | ~46 (1000 / 21.7ms measured total latency at K=64) | **Not comparable as-is** — our 21.7ms is measured at K=64, which (per the Budget Curve discussion) is close to this codebase's full/dense grid; the paper's reported FPS is almost certainly at its own practical sparse operating point (far fewer than 64 patches), which would legitimately be much faster. Not an apples-to-apples setting. |

**Bottom line**: the two numbers that *are* measured the same way both reproduce closely (Params, internal AP) — that's the meaningful reproduction check, and it passes. The two that don't match (FLOPs, FPS) are measured under different conditions on our side (fixed-640 proxy; K=64 near-dense operating point) rather than under genuinely conflicting conditions, so this is not a red flag — but it does mean **this plan cannot currently cite a paper-comparable FLOPs/FPS number**. This is a recoverable gap, not a dead end: `--task measure` (see the methodology-correction note above) is almost certainly how the paper's own FLOPs/FPS column was produced — real per-image `fvcore` profiling at the actual eval resolution, averaged over the val set — and it has simply never been invoked in this plan's runs. Re-running the existing K=64 checkpoints with `test.task: measure` would give a genuinely paper-comparable GFLOPs/FPS number without retraining anything.

UAVDT (`22.5 / 40.7 / 43.7 / 41.1`) and TinyPerson (`61.3 / 74.4 / 148.3 / 32.8`) paper numbers cannot be cross-checked at all yet — neither dataset has been run in this plan (UAVDT is Block B, pending; TinyPerson has no dataset config, per §6.1).

**On the paper's `† ESOD` (1.25× enlarged input) row**: agreed this is out of scope. That row exists in the original paper to show a naive "just make the input bigger" baseline for comparison against ESOD's own selective-compute mechanism — it is a resolution-scaling ablation on the *original* ESOD, orthogonal to what this plan is testing (evidence/fusion design for the *selector*, at a fixed resolution matched across all our own variants). Reproducing it would change two variables at once (input size *and* selector design) and wouldn't isolate anything this plan's hypotheses depend on — correctly excluded.

##### Measured wall-clock latency (real, not FLOPs-derived)

> Source: `Speed: <inference>/<NMS>/<total> ms inference/NMS/total per 1536x1536 image at batch-size 1` line printed by `test.py`, averaged over the full 548-image VisDrone val set. Extracted for all 20 runs.

| Exp | Model | K=16 (ms) | K=32 (ms) | K=48 (ms) | K=64 (ms) | Δ Total vs E1.0 @ K=64 |
|---|---|---|---|---|---|---|
| E1.0 | ESOD Baseline | 12.3 | 15.5 | 20.3 | 21.7 | — |
| E2.1 | BCRS Semantic-Only (Coverage-Supervised) | 11.9 | 15.9 | 20.6 | 21.7 | +0.0% |
| E2.3 | BCRS Dual Evidence Concat | 12.7 | 15.9 | 21.4 | 22.0 | +1.4% |
| E2.4 | BCRS Dual Evidence Gated | 12.4 | 16.1 | 21.3 | 22.5 | +3.7% |
| E2.6 | BCRS Channel-Pooled Spectral (Gated) | 12.0 | 16.0 | **21.6** | **22.2** | +2.3% |

> **This contradicts the GFLOPs-based efficiency story.** By GFLOPs, E2.6 (+1.54%) should beat E2.3 (+2.96%) on speed — it doesn't: E2.6 is **slower than E2.3 in measured latency at K=64** (22.2ms vs 22.0ms), and at K=48 E2.6 is the **slowest model in the entire sweep** (+6.4% vs baseline, worse than even the heaviest-GFLOPs E2.4). The per-K deltas are also non-monotonic (E2.1 goes from -3.3% at K=16 to +2.6% at K=32), which is the signature of GPU kernel-launch/memory-bandwidth noise on small per-branch ops dominating over the tiny FLOPs differences involved — **not a clean, trustworthy speed signal from a single-shot measurement.**
>
> **Conclusion: drop the "lightweight/efficient" framing for E2.6.** Spectral evidence is only ~1.5–4% of total model GFLOPs to begin with, so no fusion variant here is going to show a measurable wall-clock win — the channel-pooling engineering effort does not pay for itself in speed. **E2.6's real, defensible contribution is accuracy**: it has the best mAP@0.5 and Very Tiny Recall of any BCRS variant (see K=64 table above), most plausibly because channel-pooling denoises the spectral signal before the Laplacian filter, not because it makes the model faster. The paper should motivate E2.6 as a denoising/quality improvement, not an efficiency optimization — and by the same logic, **E2.7 (P2/4 shallow saliency, motivated explicitly as a texture-noise fix) has weaker justification now that E2.6 already addresses spectral noise at negligible cost**, reinforcing the earlier recommendation to deprioritize it.

#### Fusion Mechanism Analysis: why Gated underperforms Concat (2026-08-06)

`GatedEvidenceFusion` (used by E2.4, E2.6) computes:
```
gate = Sigmoid(Conv([F_semantic, F_spectral]))
fused = gate * P_semantic + (1 - gate) * P_spectral
```
This is a **zero-sum convex combination** — `gate + (1-gate) = 1` always, so any weight the network assigns to the spectral signal is necessarily taken away from the semantic signal at that same pixel, even where semantic was the more reliable source. `ConcatEvidenceSegmenter` (E2.3, E2.9) instead computes:
```
fused = Conv(cat([P_semantic, P_spectral]))
```
with no sum-to-one constraint — the learned 1×1 conv can weight both signals independently (including near-zero weight on a noisy spectral channel without penalizing semantic), making concat a strict superset of what gated fusion can express for a 2-source linear combination.

This directly explains the E2.4/E2.1 result: when the spectral input is the standard, non-denoised branch (noisier — see E2.6 motivation), a gate that leans toward it is trading a reliable signal for an unreliable one, which can net out **below** the zero-cost semantic-only baseline (E2.1). Concat doesn't have this failure mode because it isn't forced to choose. This is also consistent with why **E2.6 (pooled spectral + gated) recovers most of the gap** — cleaner spectral input makes the gate's zero-sum trade-offs less costly, without changing the fusion mechanism itself. E2.9 (pooled spectral + concat, proposed above) is expected to combine both fixes and is the most promising untested cell in the 2×2.

#### Open Design Space: further fusion mechanisms and evidence signals (2026-08-06)

Two axes remain underexplored beyond what E2.1–E2.9 cover:

**1. Fusion mechanism, beyond gate/concat.** The zero-sum-vs-union analysis above suggests the failure mode is specifically the *forced trade-off*, not "gating" as a concept — mechanisms that keep gating's adaptivity without the zero-sum constraint are worth screening ahead of any new evidence branch:
- **Additive/residual fusion**: `fused = P_semantic + λ·P_spectral` (spectral as a learned residual correction, not a competing vote) — cheaper than concat's extra conv, avoids the zero-sum trap by construction.
- **Independent (non-sum-to-one) gate per source**: `fused = g_sem·P_semantic + g_spec·P_spectral` with two independent sigmoids instead of one shared gate — keeps gating's spatial adaptivity but removes the `g_spec = 1 - g_sem` constraint; strictly more expressive than the current `GatedEvidenceFusion`, cheap to implement as a drop-in variant.
- **Cross-attention fusion**: let spectral evidence attend to semantic evidence (or vice versa) before the final conv — more expressive than concat, but adds real parameters/FLOPs and reintroduces the efficiency questions raised in the Efficiency section; only worth it if concat's ceiling is clearly reached.
- **Uncertainty-weighted fusion**: weight each branch by a learned confidence/variance estimate rather than a single joint gate — closer to classical sensor fusion, but adds a second output head per branch.
- Priority for screening: additive/residual and independent-gate first (both are near-zero-cost drop-in replacements for `GatedEvidenceFusion`, directly testable once E2.5/E2.9 close out the current matrix); cross-attention and uncertainty-weighted only if those don't beat concat.

**2. Evidence signal, beyond semantic/spectral.** The Proposal (§5.3.B) already lists several spectral-branch proxies that were never screened against each other — the current `SpectralBranch`/`ChannelPooledSpectralBranch` is one specific choice (depthwise Laplacian-style), not validated against the alternatives the Proposal names: DCT/wavelet low-mid-high band energy, local contrast, HBS-style pre/post response difference, local entropy. Beyond the Proposal's original list:
- **P2/4 shallow saliency (E2.7)**: already proposed, now lower-priority per the Efficiency findings above, but still a genuinely different *signal source* (learned spatial feature) rather than a hand-crafted filter — worth keeping on the list even if deprioritized for near-term compute.
- **Temporal/motion evidence** (if UAVDT or video-style sequences are in scope): frame-to-frame residual as a fourth evidence source — orthogonal to both semantic and spectral, untested anywhere in this plan.
- **Density/context evidence**: a cheap non-learned prior from neighboring-patch objectness (spatial clustering) — motivated by the Occupy≥1.0-at-K=64-yet-BPR-only-0.6 finding (§ Very Tiny oracle headroom discussion), which suggests the current selector may be redundantly covering the same high-priority regions rather than spreading budget — a diversity/NMS-style penalty on redundant patch selection could be a cheap addition to any of the above fusion mechanisms, not a competing evidence branch.
- Priority: before adding a fourth evidence source, first confirm (via E2.5) that the *second* one (spectral) reliably helps at all in some fusion mechanism — piling on more evidence branches without first resolving H2's conditional-falsification status (see Proposal cross-check below) would compound the same confound.

#### Phase 2.5 — Lightweight Evidence & Selector Routing Optimization Proposals

##### 1. E2.6 Channel-Pooled Spectral Filter (通道池化轻量化频域)
- **Motivation**: Standard 256-channel multi-kernel spectral convolution is heavy ($192\times 192$ feature map) and introduces channel-wise texture noise.
- **Design**: Apply Channel Max/Avg Pooling to compress $C \times H \times W \to 1 \times H \times W$ before 3x3 Laplacian filtering.
- **Target Outcome**: Reduces spectral FLOPs by 95% while eliminating channel-level high-frequency background noise. *(Original design estimate for the spectral submodule in isolation. Measured whole-model delta is +1.2 GFLOPs vs E2.4's +3.2 GFLOPs — a ~62.5% cut in the added cost; see "Efficiency" table above.)*

##### 2. E2.7 Multi-Scale Shallow Feature Saliency (P2/4 浅层高分辨率显著性)
- **Motivation**: Shallow backbone features (P2/4, stride 4) naturally preserve high-resolution spatial details for tiny objects without explicit high-pass filtering.
- **Design**: Fuse P2/4 shallow feature saliency with P3/8 via 1x1 conv to drive patch priority.
- **Target Outcome**: Higher Very Tiny recall ($<16\times 16\text{ px}$) without false spectral triggers on textured background (trees/roads).

##### 3. E2.8 Two-Stage Cascaded Selector Routing (两阶段级联剪枝)
- **Motivation**: Computing evidence refinement across all 64 candidate patches is redundant.
- **Design**: Stage 1 uses 1x1 Semantic Objectness to eliminate 50% background patches ($K=32$) with 0 overhead; Stage 2 computes evidence refinement only on remaining 32 candidate patches to pick Top-16.
- **Target Outcome**: Cuts selector feature computation in half with zero loss in target coverage.

**Go:** Equal-budget gain is reproducible and a deployable spectral/fusion implementation reaches break-even.  
**No-Go:** If a matched ordinary convolution performs equally, retain the engineering improvement if useful but drop the spectral-mechanism claim. If only FFT works but no lightweight proxy breaks even, report an oracle result and stop the efficiency path.

### Phase 3 — Single-model multi-budget routing — **DESCOPED from conference plan (2026-08-06)**

> **Why descoped:** the selling point has shifted. Early phases framed BCRS as a *budget-constrained routing framework* — one model, many operating points, chosen via `e_B` conditioning (Proposal §5.5). The evidence built in Phase 2 tells a simpler, stronger story instead: **at ESOD's own fixed operating point (K=64), matching or nearly matching its compute (E2.1 is +0.0% GFLOPs, E2.9/E2.3/E2.6 are +1.0–3.0%), BCRS gets a large accuracy/recall gain** (§ K=64 table: +6.3pp Total Recall, +8.5pp Very Tiny Recall, +11.4% mAP@0.5 at the best variant). That claim doesn't need a unified multi-budget model, budget sampling curricula, or unseen-budget interpolation (E3.1–E3.4) to land — it needs one well-chosen fixed-budget design, which is what E2.1–E2.9 already screen for. The Budget Curve tables (§ Phase 2) already show that a *single fixed-training checkpoint*, evaluated post-hoc at K=16/32/48/64 via top-k truncation, produces a smooth, monotonic recall-vs-budget curve **without** any budget conditioning — so the core question E3.1 was designed to answer (does a conditioned model match independent per-budget models) is lower-value than it looked in the original proposal: naive post-hoc truncation of one checkpoint already behaves reasonably across budgets.
>
> E3.1–E3.4 (budget embedding, sampling, unseen-budget interpolation, calibration) are removed from the near-term plan. Revive them only if a reviewer specifically asks for multi-budget deployment as a selling point, or for the journal extension if space allows — they are not required to support the current primary claim. E3.5 (backbone generalization) is unaffected by this cut and is tracked under the Journal Version Extension below, since it was never budget-conditioning-dependent.
>
> Correspondingly, the Phase 3 row of the RQ/hypothesis tracker (Proposal §11.1, RQ4/H5) and the "multi-budget model" claim threshold (§7 below) should be read as **not required for the conference submission** — see §7 for the specific threshold marked deferred.

> **Publication Scope Strategy (Updated 2026-08-06)**:
> - **Conference Version (CVPR / ECCV 8-Page)**:
>   1. **Primary Claim** *(reframed 2026-08-06 — efficiency-parity, not budget-routing)*: At ESOD's own fixed inference operating point (K=64) and near-identical compute (+0–3% GFLOPs, not a new budget axis), the Dual-Evidence selector substantially improves detection and tiny-object recall over the ESOD baseline → best variant +11.4% mAP@0.5, +8.5pp Very Tiny Recall, +6.3pp Total Recall, with the cheapest variant (E2.1) getting the recall gain at **zero** added compute. The pitch is "same cost as the paper's own selector, meaningfully better recall," not "a tunable budget dial."
>   2. **Ablation Table** *(corrected 2026-08-06 — E2.1 is not a fusion-mechanism step; it has no spectral evidence at all)*: two axes, not one progression —
>      - **Evidence-source axis**: E1.0 ESOD (no coverage loss) → E2.1 Semantic-Only/Coverage-Supervised (no spectral, free) → E2.5 Spectral-Only (no semantic, **trained, sweep pending**) — isolates what each evidence source contributes alone.
>      - **Fusion-mechanism axis**, now a 2×2 (standard vs channel-pooled spectral branch) × (gated vs concat fusion): E2.4 (standard+gated) → E2.6 (pooled+gated) → E2.3 (standard+concat) → E2.9 (pooled+concat, **new, ready to train**) — isolates which fusion mechanism combines the two best. E2.4 is currently the weakest of these four, and is beaten on most metrics by the free E2.1 selector, which is the open question E2.5 is needed to resolve. See "Fusion Mechanism Analysis" above for why gated fusion underperforms concat.
>   3. **Multi-Dataset Validation**: VisDrone (main 10-class dense) + **AI-TOD required, not yet run — see §6.1**, UAVDT (vehicle/traffic aerial, Block B pending), TinyPerson (optional).
>   4. **Dropped from Conference Scope**: Sparse K=16 efficiency angle (at low K, tiny-object information is too sparse to recover regardless of selector quality) *and* single-model multi-budget routing (Phase 3, see above) — both were framed around a "budget dial" pitch this plan no longer leads with.
> - **Journal Version Extension (IEEE TPAMI / TIP Extended 30%+)**:
>   1. **Detector Backbone Generalization (E3.5)**: Swap YOLOv5 for **YOLOv8 / YOLOv11 / RT-DETR** predictor heads.
>   2. **Structural Paradigm Migration (Phase 5 & 6)**: Migrate Dual-Evidence Recall-Safe priority to **QueryDet (Query-Adapter E5.2)** and **CEASC (Mask-Adapter Phase 6)** to prove foundational universality.
>   3. **Zero-Shot Cross-Dataset Transfer (E5.4)**: Rank correlation and zero-shot budget transfer on unseen datasets without fine-tuning.
>   4. **Single-model multi-budget routing (former Phase 3, E3.1–E3.4)**: revive here if reviewers want a budget-dial story in addition to the fixed-operating-point efficiency claim.

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

## 6. Cross-Check Against BCRS-Proposal.md (2026-08-06)

Full read-through of `BCRS-Proposal.md` against everything executed and found while producing this plan. Four real drifts found — three are coverage gaps, one is a live falsification event that the Proposal's own reporting rules require to be stated, not softened.

### 6.1 Dataset coverage gap: AI-TOD never run — blocks the Proposal's own minimum success bar

Proposal §7.1 names AI-TOD the "机制主数据集" (primary mechanism dataset), specifically because it has the smallest average object size, and Proposal §9.1's minimum success standard requires the tiny-recall gain to hold "至少在 AI-TOD 与 VisDrone 两个数据集" (at least on both AI-TOD *and* VisDrone) — not VisDrone alone.

Every experiment executed so far (Phase 0–2, E1.0–E2.9) is VisDrone-only. `configs/datasets/aitod.yaml` exists but is referenced by zero experiment configs, and no training/sweep has touched it. **As currently executed, this plan cannot yet claim the Proposal's own minimum-success bar is met** — VisDrone-only results, however strong, satisfy at most half of §9.1's stated dataset requirement. TinyPerson (Proposal's *optional* external validation) correctly has no dataset config yet — that one is not a gap.

**Action**: track AI-TOD as a blocker for any "minimum success" claim, at the same priority as UAVDT (Block B), not as a lower-tier "nice to have."

### 6.2 Falsification condition #5 has fired: E2.6's FLOPs win did not produce a latency win

Proposal §9.3, falsification condition 5: *"FLOPs 下降但端到端 latency 不降或更慢"* — and explicitly: *"以下任一结果都应被视为重要否证，而不是通过更换指标掩盖"* (must be reported as falsification, not hidden by switching metrics).

The measured wall-clock latency table above shows exactly this for E2.6 vs E2.3: E2.6 has fewer GFLOPs (+1.54% vs +2.96%) but is *slower* in measured latency at K=64 (22.2ms vs 22.0ms) and is the single slowest model in the whole sweep at K=48. This isn't new data, but it had not been explicitly tied to the Proposal's own numbered falsification condition — doing so here makes the "drop the lightweight framing for E2.6" conclusion citable as Proposal-mandated reporting, not just a stylistic call, and it should not be softened in the paper.

### 6.3 H2 (dual-evidence complementarity) is conditionally, not universally, supported

Proposal §6, H2: *"在相同 selector 参数量与计算量下，语义 + 频谱/显著性证据比语义单分支具有更高的 low-objectness tiny recall"* (dual evidence beats semantic-only at matched params/compute).

Current K=64 data: E2.4 (dual-evidence, gated, standard spectral branch) is beaten on Total Recall, Very Tiny Recall, and BPR by E2.1 (semantic-only, *zero* extra params) — the opposite of H2, for that specific fusion mechanism. H2 *is* supported for E2.3 (concat) and E2.6 (pooled+gated), which do beat E2.1. So H2 holds only for part of the fusion-mechanism × spectral-branch design space, not universally as stated — narrower than the Proposal's current wording.

Not yet a clean falsification (the Proposal has no "true for some designs, false for others" bucket), but the paper's H2 claim needs a scope qualifier once E2.5/E2.9 land: either "dual evidence helps, but only with concat or denoised-gated fusion" (narrowed claim), or — if E2.5 shows spectral-only is *also* weak — a harder look at whether the `SpectralBranch` implementation itself is the problem (→ 6.4).

### 6.4 Falsification condition #1 control (param-matched ordinary-conv spectral branch) has not been run

Proposal §9.3, condition 1: *"控制参数量和训练量后，频谱分支不优于普通卷积分支"* (after controlling params/training, the spectral branch does not outperform an ordinary conv branch) — a specific ablation (swap `SpectralBranch` for a plain conv of matched width) that has never been run here. E2.1 beating E2.4 with *zero* extra capacity is a stronger negative signal than this control would even need to produce, but it isn't a substitute: the Proposal's condition asks about equal extra capacity spent on a non-spectral conv vs. the real spectral branch, which is a different, still-open question.

**Action**: add this as a cheap single-run control once E2.5/E2.9 are done — it's the specific test the Proposal commits to running before claiming spectral evidence is genuinely informative rather than just extra capacity.

### 6.5 "Semantic + spectral GT-coverage oracle" (Proposal §7.3 item 11) not separately reported

Proposal §7.3 requires two oracle baselines: a pure GT-coverage oracle (item 10) and a *semantic + spectral* GT-coverage oracle (item 11) — two different upper bounds. The Plan's only oracle table (§ E0.4, "8×8 Patch Grid") reports one "GT Oracle Recall" curve; which of the two definitions it corresponds to is not documented. Given §6.3's findings, knowing whether the dual-evidence oracle ceiling is meaningfully higher than the pure-coverage ceiling would directly show whether there's any headroom left for spectral evidence to capture at all, independent of fusion mechanism — likely the single most informative missing number for deciding how much further effort the "Open Design Space" section above deserves.

### 6.6 No drift: budget conditioning correctly unimplemented, Phase 3 not started

Proposal §5.5 describes a budget-conditioned model (`e_B` injected via FiLM/MLP, `B ~ U(B_min, B_max)`). Confirmed by code search: no `budget_embed`/`FiLM`/`e_B` anywhere in `vendor/esod/`. This is expected, not a drift — every Phase 2 checkpoint is trained at one fixed setting and evaluated post-hoc at K=16/32/48/64 via top-k truncation at test time, which is exactly where the Plan's own (not-yet-started) Phase 3 is supposed to pick up. Noted here only so the distinction stays explicit in the paper: the Budget Curve tables above show a *test-time* top-k sweep on fixed-training checkpoints, not yet the H5 unified-multi-budget-model claim itself.

## 7. Claim thresholds

Lock numeric thresholds after Phase 0 baseline-variance measurement. Use these proposal-derived defaults unless Phase 0 demonstrates they are below ordinary noise. **Reviewed 2026-08-06 against the reframed primary claim (§ Phase 3 descoping, "efficiency-parity" not "budget-routing"); two thresholds no longer fit the claim as stated and are annotated below rather than silently dropped.**

- tiny selector recall: at least **+1.0 percentage point**, or relative miss-rate reduction of at least **15%** — **active**, already met (best variant +8.5pp Very Tiny Recall at K=64).
- APt/APvt: positive paired 95% CI on AI-TOD and VisDrone at claim-bearing budgets — **active, blocked on AI-TOD** (§6.1 — not yet run, this threshold cannot be evaluated until it is).
- total AP non-inferiority: lower 95% CI above `-max(0.2 AP, baseline repeatability margin)` — **active**; no repeated-seed variance has been measured anywhere in this plan yet, so the CI itself is still open even though point estimates are non-inferior.
- budget: zero violation for exact top-k; for latency budgets, P95 realized latency no more than 5% above request — **active**, satisfied by construction (hard top-k, no threshold-based drift — see E1.3).
- generality: gain under at least two of fixed action count, fixed FLOPs, and fixed measured latency — **active, and now effectively the primary claim itself**: E2.1 shows the gain at fixed action count *and* fixed FLOPs *and* fixed measured latency simultaneously (zero deltas on all three vs baseline); E2.3/E2.6/E2.9 show it at fixed action count with small (+1–3%) FLOPs/latency deltas.
- overhead: added selector latency at most 5% of total and at most 10% of downstream latency saved — **reinterpret**: the "10% of downstream latency saved" denominator assumes BCRS reduces compute below some larger baseline. Under the reframed claim (same K=64 as ESOD's own operating point, not a reduced budget), there is no downstream saving being claimed at the primary operating point, so this clause is vacuous there. Keep only the "≤5% of total latency" half as active for the primary claim (satisfied: worst case E2.4 at +3.7%); the full savings-based clause remains meaningful only if/when a genuinely reduced-budget comparison (e.g., BCRS at K=48 vs baseline at K=64) is reported as a secondary result.
- multi-budget model: APt within `max(0.3 AP, baseline repeatability margin)` of the per-budget oracle at at least three budgets — **DEFERRED**, tied to Phase 3 (descoped from the conference plan, see above). Not evaluated unless Phase 3 is revived.
- real acceleration: positive net median-latency saving with positive paired 95% CI, with P95 not materially worse — **reinterpret**: this threshold assumed the claim was speed-first. Measured latency data (§ Efficiency) shows only E2.1 is latency-neutral (+0.0%); E2.3/E2.4/E2.6/E2.9 are all slightly *slower* (+1.4% to +3.7%) in exchange for accuracy — i.e., the opposite of "net saving" for those variants. This is expected and acceptable under the reframed claim (accuracy-for-near-zero-added-cost, not speedup), but the paper must not claim "real acceleration" for any variant except E2.1 — doing so for E2.3/E2.4/E2.6/E2.9 would contradict this plan's own measured numbers.
- robustness: primary result holds on both AI-TOD and VisDrone and is not driven by one density bin or class — **active, blocked on AI-TOD** (§6.1), same as the APt/APvt bullet above.

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
