# BCRS Experiment Plan — Audited

**Source of research requirements:** `BCRS-Proposal.md`

**Audit date:** 2026-08-07

**Audited artifacts:** `results/**/run.log`, `results/**/best_predictions.json`, `results/**/buckets.json`, the ESOD validation/inference path, and the post-hoc failure-audit path.
**Current status:** the 7-model × 4-requested-budget VisDrone inference sweep is complete, but the metric/data validation gate is reopened. E2.9 is the best model **inside the local sweep**; paper-parity, SOTA, official COCO, strict fixed-budget, and size-bin recall claims are not yet validated.

---

## 0. Audit outcome and corrected headline

The previous plan mixed two different metric columns:

- local `test.py` variable `map50` / diagnostic `mAP@0.5` is **AP50**;
- the following native table column `mAP@.5:.95` / variable `map` is **AP@[.5:.95]**.

The ESOD paper defines the columns the same way: `AP` is averaged over IoU 0.50:0.95 and `AP50` uses IoU 0.50. Therefore the local E1.0 value `0.3669` is local **AP50**, not paper `AP=0.360`. The local E1.0 AP is `0.215`, so the current run does **not** reproduce the paper result (`AP=0.360`, `AP50=0.597`). See the [ESOD evaluation protocol](https://arxiv.org/html/2407.16424#S4.SS2).

### Corrected K=64 comparison

| Auditable metric | E1.0 ESOD | E2.9 channel-pooled concat | Delta | Status |
|---|---:|---:|---:|---|
| Native AP@[.5:.95] | 0.215 | 0.299 | +0.084 (+39.1% rel) | Valid local comparison; log is rounded to 3 decimals |
| Native AP50 | 0.3669 | 0.5071 | +0.1402 (+38.2% rel) | Valid local comparison |
| Patch BPRbox | 0.6065 | 0.8540 | +0.2475 (+40.8% rel) | Valid selector box-coverage comparison |
| Effective patches, mean [min,max] | 55.94 [54,56] | 55.95 [54,56] | +0.01 | Requested `K=64` is not 64 executed patches |
| fvcore GFLOPs, artifact mean | 258.63 | 259.56 | +0.93 (+0.36%) | Artifact is internally usable; exact code revision is not recorded |
| Model-forward latency P50/P95 | 19.07/19.86 ms | 19.50/20.92 ms | +0.43/+1.06 ms | Forward only, not end-to-end |
| Mean inference + NMS from log | 20.8 ms | 21.7 ms | +0.9 ms (+4.3%) | Aggregate mean, not a percentile distribution |

**Correct claim:** E2.9 improves local native AP, AP50, and BPR over E1.0 at approximately equal executed patch count and +0.36% artifact GFLOPs, with a +0.43 ms model-forward P50 and +0.9 ms mean inference+NMS cost.

**Claims withdrawn pending repair:** “paper baseline reproduced,” “50.7 AP / SOTA,” official PyCOCO AP/AP50/AR, 70.2% Very-Tiny recall, 81.1% total recall, “exact fixed Top-K,” “zero budget violation,” and end-to-end speedup.

### 2026-08-07 inference repair status

- E1.0 now follows upstream ESOD inference: fixed `hm_threshold=0.5`, no `top_k` / `patch_budget`, and a dynamic 0–64 patch count. The canonical metric run uses batch 8 and does not enable SparseHead, matching the upstream README command.
- BCRS Top-K is now an explicit, separate route. It performs stable coarse-cell ranking and emits exactly K patches for every feasible `K∈[1,64]`, including tied scores. Top-K no longer implicitly enables SparseHead.
- The non-upstream low-score threshold relaxation and “empty SparseHead → full patch” fallbacks were removed. Zero selected patches are once again a valid consequence of the upstream threshold.
- The current and upstream E1.0 model YAML/hyperparameters are byte-identical. Fresh model construction produces the same 35,842,600 parameters, 581 state-dict entries, identical state-key/shape hash, and identical 28-module class sequence. The repair changes no parameter name or tensor shape, so the existing E1.0 `best.pt` is reusable.
- All 28 saved K-sweep rows below remain **legacy artifacts from the pre-repair router**. They are retained for audit history, not as repaired fixed-budget or ESOD-reproduction results; E1.0 must be rerun once with threshold routing, and the six BCRS models must be rerun with the corrected Top-K router.

---

## 1. Locked metric contract

| Name used in this plan | Definition | Current source | Current validity |
|---|---|---|---|
| Native AP | mean class AP over IoU thresholds 0.50:0.05:0.95 | native `all` row, `mAP@.5:.95` | Valid for within-sweep comparison |
| Native AP50 | mean class AP at IoU 0.50 | diagnostic `mAP@0.5` / native `mAP@.5` | Valid for within-sweep comparison |
| Native P/R | per-class P/R sampled at the confidence index maximizing mean F1 | `ap_per_class()` | Valid, but not threshold-free recall |
| Patch BPRbox | fraction of GT boxes with intersection-over-smaller-area ≥ 0.5 against at least one selected patch | `cluster_recall(..., mode="bbox")` | Valid for the executed routing output |
| Effective patches | number of patches emitted by `HeatMapParser` | `buckets.json: nums` | Threshold route: dynamic 0–64. Repaired Top-K route: exactly K. Saved 28-run artifacts used the legacy non-exact router. |
| Forward latency | model forward only | `buckets.json: latency` | Valid for same-machine relative comparison; excludes NMS/data pipeline |
| Mean inference+NMS | accumulated model forward plus NMS | `run.log: Speed` | Valid aggregate mean; not end-to-end and not P50/P95 |
| PyCOCO AP/AP50/AR | COCOeval on aligned predictions | `run.log: Average Precision/Recall` | **Invalid: category mapping bug** |
| Size/class GT hit rate | fraction of YOLO-label GT boxes with any same-class IoU≥0.5 saved detection | `audit_failure_cases.py` output | **Invalid/provisional: wrong image-size fallback; also not one-to-one detection recall or selector recall** |

Publication tables must always label both axes explicitly:

- `AP@[.5:.95]` for the 10-IoU mean;
- `AP50` for IoU 0.50;
- never use bare `mAP` when the IoU range is not in the header.

The paper and local native implementations share the nominal IoU definitions, but paper parity still requires the same split, preprocessing, max detections, checkpoint/training protocol, and evaluator behavior.

---

## 2. Source-code metric audit

### 2.1 Critical findings

| Severity | Location | Finding | Impact | Required repair |
|---|---|---|---|---|
| P0 | `BCRS/vendor/esod/test.py`, JSON alignment | Raw non-COCO predictions use classes 0–9. The alignment code increments a class only when it is absent from GT ids 1–10; consequently only class 0 is shifted, classes 1–9 stay off by one, and class 10 is never predicted. | All logged PyCOCO metrics are invalid; observed PyCOCO AP50 is only 22.5–27.3% of native AP50 across the 28 runs. | Use an explicit dataset mapping (`0→1, …, 9→10`) or a class-name-to-category-id map; assert mapped ids are a subset of GT ids. |
| P0 | `BCRS/tools/audit_failure_cases.py` + `inference_sweep.sh` | The sweep passes only `labels/val`. Auto-discovery checks `labels/images[/val]` instead of the dataset sibling `images/val`, then silently falls back to 1920×1080. VisDrone images are not guaranteed to share that size. | Existing size bins, class hit rates, VTiny/Tiny/Total hit rates, and their counts cannot be used. | Require `--images-dir`; fail closed if an image is missing; alternatively read width/height and GT boxes directly from COCO JSON. |
| P0 | `HeatMapParser.ada_slicer_fast()` | Legacy “Top-K” took the K-th score, applied `>=`, then local-max/grid filtering. Ties emitted more than K and filtering emitted fewer. At requested K=64 the old rectangular-grid path emitted only 54–56 patches. | **SOURCE REPAIRED; RESULTS STALE.** Stable explicit cell indices now emit exact K and tests cover ties/K={1,16,32,64}. The 28 saved runs still contain the old behavior and cannot support fixed-budget claims. | Rerun the six BCRS variants; do not include threshold-based E1.0 in the Top-K sweep. |
| P1 | `BCRS/vendor/esod/test.py`, diagnostic counter | `num_correct_hits = stats[0].sum()` sums correctness over all 10 IoU thresholds but is printed as “Correct IoU>0.5 Hits.” | Diagnostic count can exceed GT count and is mislabeled. | Use `stats[0][:, 0].sum()` for IoU 0.50 or relabel it as the sum over IoU thresholds. |
| P1 | `BCRS/tools/inference_sweep.sh` summary parser | The generated summary records `map50` but omits native AP; it records the already-invalid PyCOCO fields. | This omission enabled AP/AP50 drift in the plan. | Parse and store both `native_ap` and `native_ap50`; quarantine COCO fields until mapping parity passes. |
| P1 | Timing labels | `buckets.json` times only model forward; log “total” is inference+NMS and still excludes data loading/preprocessing. | Previous “end-to-end” and deployment wording was too strong. | Add synchronized per-image preprocess, forward, NMS/merge, and true end-to-end distributions with warm-up metadata. |
| P1 | Result provenance | `results/` is gitignored. Runs have no manifest, git SHA, resolved config, seed, environment, or evaluator version. Current `test.py` caches first-image fvcore after the first iteration, while saved buckets contain varying per-image GFLOPs, proving source/results revision drift. | Exact reproduction of the saved artifacts is not guaranteed. | Persist a manifest and code/config hashes with every run; preserve the exact evaluator source revision. |

### 2.2 Components that are logically sound

- Native AP/AP50 routing is internally consistent: IoUs 0.50:0.05:0.95 are computed, `ap[:,0]` is AP50, and the mean over 10 IoUs is AP@[.5:.95].
- Native class matching uses the same 0–9 ids for labels and predictions.
- AP50 is greater than or equal to AP for every one of the 28 VisDrone runs.
- All 28 VisDrone logs completed without traceback or evaluator exception.
- All 28 VisDrone prediction JSON files are parseable; categories cover 0–9, scores are in [0,1], boxes have non-negative widths/heights, and JSON record counts match the logs.
- Each VisDrone `buckets.json` contains 548 latency/patch/GFLOPs samples.

### 2.3 Meaning of the post-hoc “recall” audit

Even after image-size repair, `audit_failure_cases.py` measures a **GT detection hit rate at IoU≥0.5 and saved-prediction confidence floor 0.001**. It does not enforce one-to-one matching, and it operates on post-NMS detections rather than selected patches. It must not be called selector recall, standard detector recall, or BPR. If retained, rename it `GT-hit@0.5` and report the confidence floor and matching rule.

---

## 3. Result inventory and trust boundary

### 3.1 Complete artifacts

- 28 VisDrone runs: 7 variants × requested K ∈ {16,32,48,64}.
- Every run has `run.log`, `best_predictions.json`, `buckets.json`, plots, and qualitative batches.
- There is no `results/sweep_results.json`; the script writes `work_dirs/sweep_results.json`, which was not copied into the audited result bundle.

### 3.2 Incomplete artifact

`results/esod_uavdt_yolov5m_test/` contains predictions and plots but no `run.log`, no `buckets.json`, and no metrics/manifest. It is not an auditable UAVDT result and does not establish cross-dataset validation.

### 3.3 Trust tiers

- **Tier A — usable now:** native AP, native AP50, BPRbox, effective patch count, saved forward latency, saved GFLOPs for local relative comparisons.
- **Tier B — usable after relabeling:** native P/R and mean inference+NMS timing.
- **Tier C — quarantined until re-evaluation:** all PyCOCO numbers and all size/class/VTiny/Tiny/Total post-hoc hit rates.
- **Not present:** official AI-TOD metrics, complete UAVDT metrics, TinyPerson metrics, repeated seeds/confidence intervals, true end-to-end timing, parameter-matched ordinary-conv control, low-objectness/density/texture/light audits, priority/GT-coverage correlation, and auditable oracle outputs.

---

## 4. Complete audited VisDrone sweep

Values below are read directly from the 28 `run.log` and `buckets.json` artifacts. “K request” is the requested setting, not an exact executed patch count. Latency is model-forward only. Invalid PyCOCO and post-hoc size/class hit-rate fields are intentionally excluded.

| Exp | Variant | K request | Native AP | Native AP50 | BPRbox | Patches mean [min,max] | GFLOPs mean | Forward P50/P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1.0 | ESOD baseline | 16 | 0.041 | 0.071 | 0.123 | 15.95 [15,16] | 113.2 | 10.57/12.16 |
| E1.0 | ESOD baseline | 32 | 0.089 | 0.154 | 0.262 | 31.93 [30,32] | 171.4 | 13.97/15.43 |
| E1.0 | ESOD baseline | 48 | 0.143 | 0.248 | 0.412 | 47.90 [46,48] | 229.4 | 19.21/20.47 |
| E1.0 | ESOD baseline | 64 | 0.215 | 0.367 | 0.607 | 55.94 [54,56] | 258.6 | 19.07/19.86 |
| E2.1 | Semantic-only + coverage | 16 | 0.081 | 0.142 | 0.222 | 15.97 [15,16] | 113.3 | 10.93/12.40 |
| E2.1 | Semantic-only + coverage | 32 | 0.142 | 0.245 | 0.386 | 31.94 [30,32] | 171.4 | 13.73/14.80 |
| E2.1 | Semantic-only + coverage | 48 | 0.189 | 0.325 | 0.534 | 47.93 [46,49] | 229.5 | 18.46/19.76 |
| E2.1 | Semantic-only + coverage | 64 | 0.236 | 0.403 | 0.682 | 55.97 [55,56] | 258.8 | 19.30/20.24 |
| E2.3 | Standard spectral + concat | 16 | 0.098 | 0.170 | 0.272 | 15.96 [15,16] | 120.1 | 10.78/12.11 |
| E2.3 | Standard spectral + concat | 32 | 0.158 | 0.271 | 0.430 | 31.91 [30,32] | 178.3 | 14.13/15.16 |
| E2.3 | Standard spectral + concat | 48 | 0.198 | 0.338 | 0.544 | 47.89 [46,50] | 236.7 | 19.64/21.07 |
| E2.3 | Standard spectral + concat | 64 | 0.236 | 0.403 | 0.665 | 55.94 [54,56] | 266.1 | 19.81/21.23 |
| E2.4 | Standard spectral + gated | 16 | 0.074 | 0.129 | 0.206 | 15.95 [15,16] | 123.2 | 10.83/11.79 |
| E2.4 | Standard spectral + gated | 32 | 0.132 | 0.228 | 0.355 | 31.93 [31,32] | 181.5 | 14.51/15.22 |
| E2.4 | Standard spectral + gated | 48 | 0.179 | 0.308 | 0.490 | 47.91 [46,48] | 239.8 | 19.66/20.91 |
| E2.4 | Standard spectral + gated | 64 | 0.232 | 0.399 | 0.662 | 55.96 [55,56] | 269.2 | 19.87/21.16 |
| E2.5 | Spectral-only | 16 | 0.058 | 0.100 | 0.155 | 15.99 [15,16] | 120.2 | 11.61/12.87 |
| E2.5 | Spectral-only | 32 | 0.100 | 0.172 | 0.274 | 31.94 [30,32] | 178.4 | 14.96/16.67 |
| E2.5 | Spectral-only | 48 | 0.139 | 0.241 | 0.384 | 47.95 [46,49] | 236.9 | 19.62/21.65 |
| E2.5 | Spectral-only | 64 | 0.208 | 0.353 | 0.566 | 55.95 [55,56] | 266.1 | 20.29/21.52 |
| E2.6 | Channel-pooled spectral + gated | 16 | 0.073 | 0.129 | 0.200 | 15.96 [15,16] | 116.7 | 11.43/12.99 |
| E2.6 | Channel-pooled spectral + gated | 32 | 0.129 | 0.222 | 0.358 | 31.93 [30,32] | 175.0 | 14.14/14.71 |
| E2.6 | Channel-pooled spectral + gated | 48 | 0.186 | 0.318 | 0.503 | 47.89 [45,48] | 233.2 | 18.76/20.00 |
| E2.6 | Channel-pooled spectral + gated | 64 | 0.239 | 0.409 | 0.667 | 55.94 [55,56] | 262.6 | 19.58/20.66 |
| E2.9 | Channel-pooled spectral + concat | 16 | 0.151 | 0.258 | 0.421 | 15.96 [15,16] | 113.6 | 10.54/11.47 |
| E2.9 | Channel-pooled spectral + concat | 32 | 0.216 | 0.368 | 0.613 | 31.92 [31,32] | 171.8 | 14.11/15.72 |
| E2.9 | Channel-pooled spectral + concat | 48 | 0.260 | 0.442 | 0.750 | 47.89 [45,48] | 230.1 | 18.52/19.40 |
| E2.9 | Channel-pooled spectral + concat | 64 | 0.299 | 0.507 | 0.854 | 55.95 [54,56] | 259.6 | 19.50/20.92 |

### 4.1 Conclusions supported by these artifacts

1. E2.9 is the local winner for native AP, AP50, and BPR at every requested K.
2. E2.9's K=64 improvement over E1.0 is large within the local evaluator, while compute and forward-latency increases are small but non-zero.
3. E2.5 spectral-only is worse than E1.0 at K=64 (`AP 0.208 vs 0.215`, `AP50 0.353 vs 0.367`, `BPR 0.566 vs 0.607`), so spectral evidence alone is not sufficient in this implementation.
4. E2.9 strongly suggests value from channel-pooled spectral + concat, but it does not by itself prove the Proposal's H2: H2 requires matched parameters/compute and low-objectness-tiny analysis, neither of which is complete.
5. The prior statement that E2.6 triggered falsification condition #5 is retracted. Audited K=64 forward P50 is 19.58 ms for E2.6 versus 19.81 ms for E2.3, and K=48 is 18.76 versus 19.64 ms. The old 22.2/22.0 ms comparison is not present in the audited artifacts.
6. E2.9 is not a demonstrated speedup over E1.0: forward P50 and mean inference+NMS are both higher.

---

## 5. Proposal-to-execution drift matrix

| Proposal/previous-plan requirement | Audited state | Correct status |
|---|---|---|
| Source document | Previous plan named a nonexistent `BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md` | Fixed to `BCRS-Proposal.md` |
| E0.1 data and official metric parity | COCO mapping is wrong; paper mapping was wrong in the plan | **REOPENED / BLOCKER** |
| E0.2 ESOD reproduction | The saved E1.0 rows forced the legacy Top-K router and are not ESOD reproduction runs. Fixed-threshold source/config is repaired; repaired metric run is pending. | **SOURCE/CONFIG REPAIRED; RERUN REQUIRED** |
| E0.3 selector failure audit | Only size/class post-NMS GT-hit tables exist, and they used an unsafe size fallback; required objectness/density/texture/light analyses are absent | **INVALID + INCOMPLETE** |
| E0.4 random/objectness/oracle headroom | No auditable oracle artifact exists in `results/` | **NOT EVIDENCED** |
| E1.1/E1.2 local model comparison | Native AP/AP50/BPR deltas are present | **POINT-ESTIMATE EVIDENCE ONLY** |
| E1.3 exact Top-K vs threshold | Exact Top-K source/tests are repaired and E1.0 is separated as the fixed-threshold control; no post-repair artifacts exist yet. | **IMPLEMENTED / RERUN REQUIRED** |
| E1.4 pseudo-label audit | Required Gaussian/SAM/hybrid bias artifacts are absent | **NOT EVIDENCED** |
| Phase 2 VisDrone inference sweep | 28 legacy runs are present, but they used the non-exact router and incorrectly included E1.0 at K. The repaired sweep is six BCRS models × four budgets; E1.0 runs once as a threshold control. | **LEGACY COMPLETE / REPAIRED RERUN REQUIRED** |
| H2 dual-evidence complementarity | E2.9 is better, but matched-param control and valid low-objectness tiny audit are absent | **PROMISING, NOT CONFIRMED** |
| Falsification #1 ordinary-conv control | Not run | **OPEN** |
| AI-TOD minimum-success dataset | Config exists; no result | **BLOCKER** |
| UAVDT | Predictions/plots only; no log or metrics | **INCOMPLETE** |
| TinyPerson | No result | **NOT STARTED / OPTIONAL** |
| Phase 3 budget-conditioned single model | No budget embedding/training | **DESCOPED; explicit Proposal deviation** |
| Phase 4 equal-budget/end-to-end efficiency | No true end-to-end percentile timing or paired CI | **NOT STARTED** |
| QueryDet/CEASC transfer | No result | **NOT STARTED** |
| Repeated seeds and confidence intervals | No seed metadata or repeats | **NOT STARTED** |

---

## 6. Repaired execution plan

### Gate A — Metric and artifact repair (must complete before any paper table)

| ID | Action | Acceptance test | Status |
|---|---|---|---|
| A1 | Replace heuristic COCO category alignment with explicit dataset mapping | Synthetic classes 0–9 map exactly to GT ids 1–10; no unmapped id | TODO |
| A2 | Add native-vs-COCO parity test on a synthetic fixture | Same predictions/GT produce consistent AP/AP50 within a documented tolerance | TODO |
| A3 | Log native AP and AP50 as named JSON fields | No parsing of positional console columns; schema test passes | TODO |
| A4 | Repair failure audit to require COCO GT or real images | Missing dimensions hard-fail; 548 images all resolve to exact width/height | TODO |
| A5 | Rename post-hoc metric to `GT-hit@0.5` and implement one-to-one option | Metric metadata includes confidence, IoU, class rule, and matching rule | TODO |
| A6 | Implement exact emitted-patch Top-K with deterministic tie-breaking | For every image and feasible K, emitted count equals K; adapter rejects explicit budgets outside `[1,64]` | **DONE IN SOURCE; 22 TARGETED TESTS PASS** |
| A7 | Record git SHA, dirty diff hash, resolved configs, checkpoint hash, seed, dataset hash, environment, evaluator version | Every run has a complete manifest | TODO |
| A8 | Generate `results/sweep_results.json` from artifacts and copy it with the run bundle | Aggregate contains native AP/AP50, validation flags, and provenance | TODO |

Most corrections do not require re-running inference: saved predictions can be re-evaluated with the correct COCO mapping and exact GT metadata. Exact-Top-K results do require new inference after the router is repaired.

### Gate B — Baseline parity and repeatability

1. Re-evaluate E1.0 saved predictions after Gate A.
2. Resolve the gap to paper AP/AP50 by checking checkpoint, split, preprocessing, max detections, NMS, input scaling, and training schedule.
3. Run at least three seeds for E1.0, E2.1, and E2.9 at the claim-bearing budget.
4. Report paired bootstrap confidence intervals for AP, AP50, BPR, and latency.
5. Do not call E1.0 “reproduced” until both metric identity and protocol parity are documented.

### Gate C — Proposal-critical evidence

| Priority | Experiment | Required outputs |
|---|---|---|
| P0 | AI-TOD E1.0/E2.1/E2.9 | AP/AP50/APvt/APt, BPR, exact actions, true end-to-end latency, repeated seeds |
| P0 | Valid VisDrone subgroup audit | low-objectness, density, texture, illumination, size, class; paired confidence intervals |
| P0 | Matched ordinary-conv control | same params/training/compute as the spectral branch |
| P0 | Exact Top-K vs threshold control | requested/executed actions, violation rate, P50/P95, AP/BPR frontier |
| P1 | Oracle headroom | random, objectness top-k, GT coverage oracle, semantic+spectral oracle with saved per-image outputs |
| P1 | True end-to-end profiling | preprocess + selector + slicing/gather/scatter + head + merge/NMS, warm-up and raw samples |
| P1 | UAVDT completion | complete log, buckets, GT evaluator, manifest, and metrics |
| P2 | QueryDet adapter | equal-query/equal-latency AP and coverage |
| P2 | CEASC/TinyPerson | extension evidence only after core gates pass |

### Phase status after audit

| Phase | Status | Exit condition |
|---|---|---|
| Phase 0 — validation/reproduction | **REOPENED** | Gate A+B complete |
| Phase 1 — semantic/coverage MVP | **PARTIAL** | valid subgroup audit, exact-budget control, repeatability |
| Phase 2 — dual-evidence sweep | **INFERENCE COMPLETE; CLAIMS PARTIAL** | matched controls and valid subgroup evidence |
| Phase 3 — budget-conditioned model | **DESCOPED** | revive only with an explicit scope decision |
| Phase 4 — end-to-end efficiency | **NOT STARTED** | paired end-to-end latency frontier |
| Phase 5 — QueryDet transfer | **NOT STARTED** | equal-budget cross-backend result |
| Phase 6 — CEASC/optional external | **NOT STARTED** | only after core claims are secure |

---

## 7. Claim policy

### May be stated now

- Within the audited local VisDrone native evaluator, E2.9 is best among the seven tested variants at every requested K for AP, AP50, and BPR.
- At requested K=64, E2.9 changes native AP from 0.215 to 0.299 and AP50 from 0.3669 to 0.5071, with +0.93 artifact GFLOPs and +0.43 ms forward P50.
- Spectral-only E2.5 underperforms E1.0 at K=64.

### Must not be stated yet

- local `0.3669` matches paper `AP=0.360`;
- E2.9 has `50.7 AP` or is VisDrone SOTA;
- any logged PyCOCO AP/AP50/AR value is official;
- 70.2% VTiny or 81.1% total selector/detection recall is validated;
- the legacy 28-run artifacts have exact K, zero budget variance, or zero violation (the repaired source does, but must be rerun);
- E2.9 accelerates E1.0 end to end;
- the Proposal's minimum success standard is met;
- H2 is fully confirmed or the Laplacian mechanism is isolated;
- results transfer to AI-TOD, UAVDT, TinyPerson, QueryDet, or CEASC.

External competitor tables must remain out of the claim-bearing section until metric, split, resolution, detector, and compute protocol are aligned. Paper-reported FPS from other hardware must not be ranked against local RTX 5090 timing.

---

## 8. Artifact contract going forward

Each run must write to `results/<experiment_id>/<run_id>/`:

| Artifact | Required contents |
|---|---|
| `manifest.yaml` | git SHA, dirty-diff hash, run/config/checkpoint/dataset hashes, seed, host/device, timestamps, completion status |
| `config.yaml` | fully resolved data/model/training/evaluation/routing configuration |
| `environment.txt` | OS, Python, PyTorch, CUDA/cuDNN, fvcore, pycocotools, compiler/package-lock hashes |
| `metrics.json` | named native AP/AP50, official evaluator metrics, BPR, executed actions, latency scope and validation flags |
| `predictions.raw.json` | internal class ids with explicit schema |
| `predictions.coco.json` | explicitly mapped COCO ids, retained for audit |
| `objects.parquet` | per-GT size/class/objectness/subgroup, selected/covered/matched flags |
| `images.parquet` | per-image requested/executed actions, violations, cost and latency components |
| `latency.json` | warm-up, repetitions, synchronization, P50/P95/IQR, raw samples, included pipeline stages |
| `run.log` | complete stdout/stderr and evaluator summaries |

Maintain `results/sweep_results.json` or `results/registry.parquet` as the only table-generation input. Hand-copied values are not a source of truth.

---

## 9. Minimum success standard (unchanged from the Proposal)

The project reaches minimum success only when, on both AI-TOD and VisDrone and under at least two aligned budget definitions:

- tiny selector recall improves by at least 1 percentage point or relative miss rate drops by at least 15%;
- APt/APvt improves stably;
- total AP is non-inferior beyond measured repeatability noise;
- added selector latency is within the stated overhead/break-even rule;
- the result is reproducible with valid metrics, exact action accounting, and recorded provenance.

Until Gates A–C are complete, the current artifacts are strong local screening evidence for E2.9, not a completed paper-level validation.
