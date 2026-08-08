# HESOD Experiment Plan — VisDrone E1.0 Baseline Audit

**Status:** first post-repair VisDrone baseline (E1.0, upstream loss, threshold routing) trained and evaluated end-to-end via `esod/` + `scripts/esod_baseline/`. Audited against the ESOD paper (TIP 2024) and the run's own training log. Superseded the deleted `BCRS-Experiment-Plan.md` for this line of work; see `ESOD-Baseline-Patches.md` for the environment patches this run depended on.

---

## 1. Run provenance

| Field | Value |
|---|---|
| Arm | E1.0 upstream baseline (`Segmenter`, plain weighted BCE, fixed-threshold routing) |
| Code | `esod/` (pristine upstream + environment-compat patches only, see `ESOD-Baseline-Patches.md`); git short-hash printed by the run: `0ccce47` |
| Hardware | 1x RTX 5090 (32GB), torch 2.8.0+cu128 |
| Config | `models/cfg/esod/visdrone_yolov5m.yaml`, `data/hyps/hyp.visdrone.yaml`, batch 8, img 1536, 50 epochs, SGD lr0=0.01 cosine |
| Dataset | VisDrone2019-DET, 6471 train / 548 val images, 38759 val GT boxes |
| Selector masks | Gaussian-only fallback (`gen_masks.py`, no SAM installed) — **not** the paper's official hybrid Gaussian+SAM pseudo-mask |
| Seeds | 1 (no repeats) |
| Artifacts audited | `results/visdrone_yolov5m_baseline/{*_train.log, *_test.log, *_measure.log, *_audit.log, buckets.json, best_predictions.json, PR_curve.png, F1_curve.png}` |

Batch=8 (`test.py --save-json`) and batch=1 (`test.py --task measure`) runs on the same `best.pt` produced **identical** P/R/mAP@.5/mAP@.5:.95/BPR/Occupy (0.615/0.553/0.560/0.336/0.972/0.433) — no batch-size sensitivity, both readings trustworthy.

---

## 2. Audited results vs. paper

Table I's headline ESOD number (AP 36.0) is the paper's **hybrid Gaussian+SAM** mask setting. This run used **Gaussian-only** masks (no SAM installed). The paper's own Table V ablates exactly this: Gaussian-only scores AP 0.357/AP50 0.595, 0.3/0.2 points below the hybrid headline. That is the correct apples-to-apples comparator for this run, not Table I directly.

| Metric | Paper, hybrid (Table I) | Paper, Gaussian-only (Table V) | This run, Gaussian-only (RTX 5090) | Delta vs. Gaussian-only |
|---|---:|---:|---:|---:|
| AP@[.5:.95] | 0.360 | 0.357 | 0.336 | −0.021 (−5.9%) |
| AP50 | 0.597 | 0.595 | 0.560 | −0.035 (−5.9%) |
| GFLOPs | 119.5 | not reported | 140.2 | — |
| FPS | 36.4 | not reported | 95.4 | not comparable — different GPU generation (Proposal SS7.13); not a claim |
| BPRbox | not reported at this operating point | not reported | 0.972 | selector coverage looks healthy in isolation |
| Occupy (patch retention) | not reported numerically | not reported | 0.433 | see SS5 |

**Verdict: not a precise reproduction, and the gap is mostly not explained by the Gaussian-vs-hybrid mask choice.** Even against the paper's own Gaussian-only ablation (not the hybrid headline), AP and AP50 are still ~5.9% relatively low. Mask choice alone (Gaussian vs. hybrid, the paper's own Table V delta) accounts for only 0.3 of the paper's 2.4-point hybrid-vs-this-run gap — roughly an eighth of it. **Most of the gap is currently unexplained; see SS5.**

---

## 3. Training curve (from `visdrone_yolov5m_baseline_train.log`, all 50 epochs parsed)

- **Losses (box/obj/cls/lpixl) decreased smoothly and monotonically for all 50 epochs**, no divergence, no NaN, no instability. `larea`/`ldist` stayed at 0 throughout, confirming `selector_loss=upstream` ran with no coverage/dice/focal term active (correct — this is the E1.0 baseline).
- **Validation mAP@[.5:.95] trajectory:** 0.054 (ep0) → 0.208 (ep3) → 0.295 (ep15) → 0.323 (ep27) → 0.332 (ep39) → **0.337 (ep49, final)**.
- **Last 15 epochs (35→49): 0.329 → 0.337, i.e. +0.008 over 15 epochs.** The curve has visibly flattened; this is a converged run under its 50-epoch cosine schedule, not one cut short. **Training duration is not the explanation for the gap.**
- **Occupy (fraction of image kept as patches) stabilized in the 0.41-0.43 range from roughly epoch 10 onward** and finished at 0.415-0.433 across the three evaluation runs. This is a stable, converged property of the trained selector, not measurement noise.
- **BPRbox rose from 0.995 (epoch 0, selector barely discriminates, keeps almost everything) down to 0.958-0.974 as the selector sparsified, settling at 0.972.** The adaptive slicer is including >97% of GT boxes in some selected patch — gross selector coverage failure is not the bottleneck.

---

## 4. Bucket recall audit (`audit_buckets.py`, conf>=0.001, IoU>=0.5, one-to-one matching)

| Size bin | GT | Recalled | Recall |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 11955 | 9190 | 76.87% |
| Tiny (16x16-32x32) | 14631 | 13205 | 90.25% |
| Small (32x32-96x96) | 11105 | 10582 | 95.29% |
| Medium/Large (>96x96) | 1068 | 1033 | 96.72% |
| **Total** | **38759** | **34010** | **87.75%** |

This is a low-confidence-threshold recall ceiling (how many GT could be found at all, not the deployment operating point) — it is **not** meant to match the native R=0.553 reported at the best-F1 confidence threshold; the two numbers measuring different things is expected, not a bug.

The size-bin pattern itself (Very Tiny clearly lowest at 76.87%, Tiny/Small/Medium-Large all >=90%) is the textbook shape for any detector on a tiny-object-heavy dataset, not a symptom of something wrong with this run or the audit methodology.

**What this ceiling does *not* establish: that AP should be high.** AP is a ranking-sensitive, area-under-precision-recall-curve statistic computed across confidence and (for AP@[.5:.95]) across IoU 0.5-0.95 -- a GT box counted as "recalled" by this audit (some prediction reached IoU>=0.5 at any confidence>=0.001) contributes to AP only in proportion to how highly that prediction is *ranked* relative to false positives, and only at the IoU thresholds its localization actually clears. AP50 uses the same IoU>=0.5 bar this audit uses, and it is *also* ~5.9% low against the paper's Gaussian-only number (SS2) -- meaning the "87.75% recall ceiling exists" fact does not by itself explain why AP50 came out low; a plausible added contributor is confidence-score calibration/ranking (true positives not scored clearly above the false-positive tail), not pure coverage. This reframes the ranked list in SS5.

Per-class AP50 (from `PR_curve.png`, cross-checked against per-class recall in the audit log — both rank classes in the same order): `car` 0.884, `bus` 0.723, `pedestrian` 0.682, `motor` 0.637, `van` 0.592, `people` 0.540, `truck` 0.487, `tricycle` 0.438, `bicycle` 0.368, `awning-tricycle` 0.249. The weakest classes (`awning-tricycle` 532 GT, `bicycle` 1287 GT) are also VisDrone's rarest classes — consistent with class-imbalance being the driver here, not something specific to this reproduction.

### 4.1 Confusion matrix and P-curve (`confusion_matrix.png`, `P_curve.png`) — quantifies SS5's #1 hypothesis directly

`confusion_matrix.png` is column-normalized (per true class) at YOLOv5's own default confusion-matrix operating point (conf=0.25, iou=0.45) -- a stricter, more deployment-realistic threshold than the audit's conf>=0.001. Its `background FN` row (true object, nothing survived at conf>=0.25) is **10-45% depending on class**: pedestrian 27%, people 38%, bicycle 45%, car 10%, van 12%, truck 30%, tricycle 35%, awning-tricycle 38%, bus 20%, motor 30%.

This is now a directly quantified confirmation, not an inference: a large fraction of objects that the audit's near-zero-threshold recall ceiling (87.75%, SS4) says *are* found by some prediction do not survive a realistic confidence cutoff. That gap between "found at conf>=0.001" and "found at conf>=0.25" is exactly the confidence-ranking/calibration signature SS5 already ranked #1 -- this confirms it with a concrete number instead of an inference from AP50 alone.

The confusion matrix also shows a second, additive effect: **genuine cross-class confusion** among visually similar VisDrone categories -- `bicycle`->`motor` 12%, `tricycle`<->`awning-tricycle` 19%/7% both directions, `car`<->`van`/`truck`/`bus` 3-11%. A detection assigned the wrong class label counts as a false positive for the predicted class and a false negative for the true one, directly costing AP for the confused classes. Not yet separable from this artifact alone: how much of this confusion is inherent VisDrone difficulty (plausibly present in the paper's own numbers too) vs. specific to this reproduction's training.

`P_curve.png` (precision vs. confidence threshold, not recall) rises smoothly from ~0.10-0.12 at confidence~0 to ~1.0 by confidence~0.94, with no sharp discontinuities, noise cliffs, or otherwise pathological shape -- consistent with an under-calibrated-but-not-broken model, not evidence of a distinct bug.

### 4.2 Selector-coverage breakdown (`dump_selected_patches.py` + `audit_selector_coverage.py`) -- directly tests the "selector drops tiny objects" hypothesis

For every GT box, checked whether its box center fell inside any patch `HeatMapParser` actually selected at inference -- using the Segmenter's own predicted mask, not GT masks (the same no-oracle setting `best_predictions.json` was produced under, i.e. `--use-gt` was never passed) -- then cross-tabulated against SS4's recall match (same conf>=0.001, IoU>=0.5, one-to-one matching):

| Size bin | Total GT | Selector-dropped | Covered, head missed | Covered, recalled |
|---|---:|---:|---:|---:|
| Very Tiny (<16x16) | 11955 | 326 (2.7%) | 2457 (20.6%) | 9172 (76.7%) |
| Tiny (16x16-32x32) | 14631 | 291 (2.0%) | 1174 (8.0%) | 13166 (90.0%) |
| Small (32x32-96x96) | 11105 | 220 (2.0%) | 344 (3.1%) | 10541 (94.9%) |
| Medium/Large (>96x96) | 1068 | 22 (2.1%) | 19 (1.8%) | 1027 (96.2%) |

`Selector-dropped` is 2.0-2.7% across every bin -- flat, essentially size-independent, and consistent with SS3's aggregate BPRbox=0.972. **This directly tests and rules out, for this VisDrone run, the specific hypothesis that the selector disproportionately drops Very Tiny objects before the detection head ever sees them** -- `HeatMapParser`'s own patch selection is size-agnostic here, not just healthy in aggregate.

The entire size-graded shape of SS4's recall gap lives in `Covered, head missed`: 20.6% for Very Tiny down to 1.8% for Medium/Large, a clean monotonic trend -- roughly an 11x higher miss rate for Very Tiny than Medium/Large among objects the selector handled identically well. Even at the most permissive possible confidence threshold (conf>=0.001, same as SS4, where SS4.1's calibration effect cannot yet apply), the detection head still fails to produce any IoU>=0.5 match far more often for small objects. This is a **detection-head localization/regression failure specific to small objects within an already-correctly-selected patch** -- distinct from SS4.1's calibration finding (that one measures conf>=0.001-to-conf>=0.25 survival *among boxes already matched at IoU>=0.5*; this one measures whether a matching box is produced at IoU>=0.5 *at all*, at the most permissive threshold). Both are real, both sit downstream of selection, but they act at different pipeline stages and are not the same mechanism.

---

## 5. Gap diagnosis

Revised after computing the correct apples-to-apples comparator (SS2: this run vs. the paper's own Gaussian-only ablation, not the hybrid headline), re-examining what the recall-ceiling audit does and doesn't establish (SS4), and quantifying the confusion matrix / P-curve (SS4.1).

**Ruled out:**
- **Training under-convergence.** SS3 — the run converged normally under its 50-epoch schedule; last-15-epoch mAP gain was +0.008.
- **Gross selector coverage failure — including the specific, size-biased version of it.** SS3/SS4 gave the aggregate signal (BPRbox 0.972); SS4.2 confirms it holds uniformly per size bin too (selector-dropped rate is 2.0-2.7%, flat, from Very Tiny through Medium/Large) — directly disproving the hypothesis that the selector disproportionately drops small objects. Whatever is producing the AP gap sits entirely downstream of object selection, at the detection-head/confidence-ranking stages (SS4.1, SS4.2).

**Real but small, already priced into the comparison:**
- **Gaussian-only vs. hybrid masks.** Table V's own delta (35.7/59.5 Gaussian-only vs. 36.0/59.7 hybrid) is 0.3/0.2 AP/AP50 points. SS2's table already uses the Gaussian-only row as the comparator, so this factor is *not* available to explain the remaining 5.9% relative gap -- it's already subtracted out. Occupy stabilizing at 0.42-0.43 (higher patch retention than the paper's implied ratio, given GFLOPs came out 17% above Table I's 119.5) is still consistent with Gaussian masks producing broader/blurrier activation than SAM-assisted ones, but that's a compute-efficiency observation, not an AP explanation on its own.

**Confirmed contributors to the still-unexplained ~5.9% relative AP/AP50 gap (ranked by evidence strength):**
1. **Detection-head localization failure on small objects within correctly-selected patches — confirmed by SS4.2, newly identified.** Even at conf>=0.001 (a threshold permissive enough that SS4.1's calibration effect cannot yet apply), 20.6% of Very Tiny GT boxes inside a selector-chosen patch still get no IoU>=0.5 match at all, vs. 1.8% for Medium/Large — an ~11x gap that SS4.2 rules out attributing to the selector (its drop rate is flat, 2.0-2.7%, across every bin). This points at the SparseHead's own box regression/objectness for small objects as the mechanism, not object selection — i.e. architecture/loss changes at the detection-head level (anchor design, small-box regression loss weighting) look like a more promising lever for closing this specific part of the gap than selector-only interventions (e.g. coverage loss, which targets selection, not head localization).
2. **Confidence-score ranking/calibration — confirmed by SS4.1, not just inferred.** The confusion matrix's `background FN` row shows 10-45% of GT objects (class-dependent) never survive conf>=0.25, despite the 87.75% ceiling at conf>=0.001 saying they *are* found by some prediction. A meaningful share of true detections sit in a low-but-nonzero confidence band that any reasonable operating threshold discards. `P_curve.png`'s smooth, non-pathological shape suggests this is under-calibration, not a distinct bug. Still open: *why* calibration is weaker than the paper's — candidates below.
3. **Genuine cross-class confusion — confirmed by SS4.1, newly identified.** `bicycle`->`motor` 12%, `tricycle`<->`awning-tricycle` 19%/7%, `car`<->`van`/`truck`/`bus` 3-11%. Directly costs AP for the confused classes (wrong-class detections are FPs for the predicted class and FNs for the true one). Not yet separable: how much is inherent VisDrone difficulty vs. this reproduction specifically — would need the paper's own confusion matrix (not published) or a second independent run to compare against.
4. **Single seed, no repeats.** The paper doesn't report variance either; a single run landing on the low end of natural noise can't be ruled out.
5. **Exact Gaussian mask generation parameters.** "Gaussian-only" is not a single fully-pinned specification -- sigma/threshold choices affect mask sharpness, and mask quality could plausibly feed into the selector's own confidence signal propagating downstream. This run used the official `gen_mask()` from the released repo, which should match the paper's own Gaussian recipe since it's the same code, but that assumption hasn't been independently verified against whatever produced Table V's specific numbers.
6. **Hyperparameter/library-version drift** from the paper's original (~2021-era) stack to this run's PyTorch 2.8/CUDA 12.8 stack. The smooth, textbook-shaped loss/mAP curves in SS3 give no direct evidence this happened, so this stays last-ranked, not eliminated.

---

## 6. Claim policy

**May be stated now:**
- E1.0 baseline trained and evaluated end-to-end on real VisDrone data with valid Gaussian-fallback selector masks (fail-closed checked, see `ESOD-Baseline-Patches.md`), on the current environment stack.
- The run converged normally; the observed AP/AP50/GFLOPs gap vs. the paper is not attributable to premature stopping or training instability.
- Selector-level coverage (BPRbox, low-threshold recall ceiling) is healthy; the shortfall is concentrated in overall detection quality/compute efficiency, not gross object loss at the selection stage. **This now holds at the size-bin level too, not just in aggregate** (SS4.2: selector-dropped rate is 2.0-2.7%, flat, from Very Tiny through Medium/Large) — the selector does not disproportionately drop small objects on this run.
- A confidence-calibration gap, genuine cross-class confusion, and a size-graded detection-head localization failure (objects the selector *did* route correctly still failing to get an IoU>=0.5 match) are all **confirmed present** (SS4.1, SS4.2, quantified from `confusion_matrix.png`/`P_curve.png` and the new selector-coverage cross-tabulation) and together are the best-evidenced description of where the AP/AP50 gap sits. What is *not* yet established is *why* calibration/localization are weaker than the paper's, or how much of the confusion is VisDrone-inherent vs. reproduction-specific — those remain hypotheses (SS5 #4-#6).

**Must not be stated yet:**
- "Paper reproduced" or any framing implying AP/AP50 match — the gap is real (~5.9% relative *against the paper's own Gaussian-only ablation*, not just the hybrid headline).
- Any claim that Gaussian-only masks are the/a major cause of the AP gap — SS5 shows the paper's own Gaussian-vs-hybrid delta (0.3 AP) is already priced into the comparator and does not cover the remaining ~5.9% relative shortfall. Mask choice may still matter for the GFLOPs/Occupy gap specifically, and remains an open (not yet ruled out) contributor to the calibration gap too; neither is demonstrated.
- Any claim that the 87.75% bucket-recall ceiling indicates the selector/detector is "mostly working" at the AP level — SS4/SS4.1 show this ceiling number does not predict AP, and the confusion matrix directly quantifies why (10-45% of objects found at conf>=0.001 don't survive conf>=0.25).
- That a selector-side intervention (e.g. `hesod/backends/hesod`'s coverage loss) is *the* fix for VisDrone's Very Tiny recall gap — SS4.2 shows this run's gap sits at the detection head, not the selector, for VisDrone specifically. **Whether this generalizes is not established**: UAVDT's aggregate selector coverage (BPRbox 0.807, SS1-adjacent audit still in progress) is far worse than VisDrone's 0.972, so a selector-side fix may matter more there — its own per-size-bin breakdown (SS4.2-equivalent) has not been run yet.
- A root cause for *why* calibration/localization are weaker than the paper's, or a VisDrone-inherent-vs-reproduction-specific split of the class confusion — both are open questions, not conclusions (SS7).
- Any cross-arm (E2.x) comparison — no other roster arm has been retrained yet under this environment.
- Anything about UAVDT/TinyPerson as a completed audit — those runs are separate and not fully covered by this document yet. UAVDT's own E1.0 baseline has finished all 50 epochs and its eval step has completed (P=0.277/R=0.384/AP50=0.186/AP=0.106/BPR=0.807/Occupy=0.0674, confusion matrix and PR/P curves reviewed informally in-session), but its own bucket recall audit and selector-coverage breakdown have not yet been formally run and folded into this document.

---

## 7. Next steps to close the gap

1. ~~Inspect the PR curve / confusion matrix for a ranking-vs-coverage signature.~~ **Done (SS4.1)** — confirmed via `confusion_matrix.png` + `P_curve.png`, no retraining needed. Confirmed both a calibration gap and genuine cross-class confusion; see SS5 #2-#3.
2. ~~Test whether the selector itself (not just detection quality overall) disproportionately drops small objects — the specific mechanism `hesod/backends/hesod`'s coverage loss targets.~~ **Done for VisDrone (SS4.2)**, via new standalone tools `scripts/esod_baseline/dump_selected_patches.py` + `audit_selector_coverage.py` (no retraining, reuses `best.pt`). Ruled out for this run — selector-dropped rate is flat (2.0-2.7%) across all size bins; the size-graded gap sits at the detection head instead (SS5 #1). **Not yet run for UAVDT** — its much lower aggregate BPRbox (0.807 vs VisDrone's 0.972) makes it plausible the answer differs there; run the same two scripts once UAVDT's own artifacts are fully collected (item 3 below).
3. UAVDT's E1.0 baseline has finished training and its AP/AP50 eval step (`best_predictions.json`, confusion matrix, PR/P curves) is done — but its own bucket recall audit (`audit_buckets.py`, blocked earlier by a since-fixed degenerate-bbox crash) and selector-coverage breakdown (item 2) still need to be run and formally folded into this document, matching SS1-SS4.2's treatment of VisDrone. Then wait for TinyPerson's first-round E1.0 baseline too, and look at all three side by side before deciding whether VisDrone specifically needs a retrain — a single-dataset gap could be VisDrone-specific (e.g. this Gaussian mask recipe, this class-confusion structure); a gap that shows up on all three points more toward something systemic (schedule, calibration-affecting hyperparameter, library-version drift).
4. Repeat E1.0 with >=1 additional seed to establish whether the ~5.9% relative AP/AP50 gap (vs. the paper's own Gaussian-only ablation) is inside or outside natural run-to-run variance.
5. Install SAM and regenerate hybrid masks for a controlled Gaussian-vs-hybrid comparison, holding everything else fixed — direct test for the GFLOPs/Occupy gap, and now also a candidate for the calibration gap (SS5 #5) since mask quality could plausibly propagate into the selector's confidence signal.
6. Once E1.0 is trusted (or its gap is understood and accepted as a documented baseline), proceed to the locked five-arm HESOD roster (`HESOD-Proposal.md` SS7.3.1) using this same audited pipeline (`scripts/esod_baseline/`, `hesod/backends/hesod/`'s new `--selector-loss coverage`/`--top-k` flags) so every arm is compared against a baseline with a known, documented gap rather than an assumed-exact one. **SS4.2's finding is a live input to this**: if UAVDT confirms VisDrone's pattern (selector coverage healthy, gap at the head), coverage loss alone may not be the highest-leverage arm to prioritize first — worth weighing against the dual-evidence/concat arms, which also change what the head receives, not just what the selector routes.

---

## 8. Local patches applied during this audit

- Fixed a real bug found while auditing the (still in-progress) UAVDT run and confirmed present in this VisDrone run's code path too: `test.py`'s per-class label count (`nt`) was gated behind `stats[0].any()`, so any validation pass with zero IoU>0.5 hits displayed `Labels: 0` and `BPR: nan` even though the true label count was never zero (matches `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` #2's already-documented fix for the separate vendored fork). Applied identically to `esod/`, `hesod/backends/esod/`, `hesod/backends/hesod/`. Display-only — doesn't change any number in this document, since this VisDrone run never actually hit the zero-hit branch (epoch 0's validation already had real matches, see SS3).
