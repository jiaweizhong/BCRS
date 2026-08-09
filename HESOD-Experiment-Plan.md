# HESOD Experiment Plan — VisDrone E1.0 Baseline Audit

**Status:** first post-repair VisDrone baseline (E1.0, upstream loss, threshold routing) trained and evaluated end-to-end via `esod/` + `scripts/esod_baseline/`. Audited against the ESOD paper (TIP 2024) and the run's own training log. Superseded the deleted `BCRS-Experiment-Plan.md` for this line of work; see `ESOD-Baseline-Patches.md` for the environment patches this run depended on.

**UAVDT status: paused, likely a data-source problem, not a model/method problem.** UAVDT's own E1.0 baseline finished training and eval (P=0.277/R=0.384/AP50=0.186/AP=0.106/BPR=0.807/Occupy=0.0674) and got audited the same way (`audit_buckets.py`, `list_examples.py`, confusion matrix) before this was found. Its size-bin recall was non-monotonic (Medium/Large=52.13%, *worse* than Very Tiny=61.60%), traced via `list_examples.py` to the `car` class specifically cratering in that bin (51.16% vs. 92.91% in Small) while `bus` in the same bin scored 99.30% — and 90% (18/20) of a random sample of "missed, Medium/Large car" GT boxes had literally no overlapping prediction of any class at all. A random `--status recalled` sample of the same size/class showed video sequence `M0802` at 0% (vs. 55% of the `missed` sample) — a clean signal that isn't explained by `M0802` simply containing more large-car frames. Visual inspection of two `M0802` frames confirmed why: the offending GT boxes (~130-190px wide) sit on top of a dense parking lot of many small, individually-parked cars, not a single large vehicle — almost certainly a labeling artifact (one oversized box drawn over a cluster of cars), not something any model should be expected to match. Contrasted against a genuinely large, cleanly-boxed, correctly-recalled car in a different sequence (`M0205`, score=0.872, IoU=0.924), the difference is visually obvious. `/root/autodl-tmp/UAVDT_processed` is understood to be sourced from a third-party Kaggle repackaging of UAVDT (not verified against the official UAVDT release's own annotations) — consistent with this kind of box-quality defect. **Decision: pause further UAVDT diagnosis until/unless an official-source UAVDT conversion is available to compare against; do not draw conclusions about the HESOD method from UAVDT's current numbers.** Focus returns to VisDrone (this document's main subject) until that's resolved. TinyPerson has not been started; its own data source should be verified *before* investing GPU time, given this precedent.

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

**IoU-sensitivity check on `Covered, head missed`** — re-ran the same cross-tabulation at `--iou-thres 0.3` (diagnostic only; the paper-comparable AP/AP50 numbers in SS2 stay at the standard IoU 0.5/0.5:0.95, this does not change any reported metric) to test how much of the 20.6% is "near miss, IoU-metric harshness for small absolute pixel counts" (SS4's original framing) vs. genuinely unrecoverable at any reasonable localization tolerance:

| Size bin | Covered, missed @ IoU>=0.5 | Covered, missed @ IoU>=0.3 | Absolute drop | Relative drop |
|---|---:|---:|---:|---:|
| Very Tiny (<16x16) | 20.6% | 10.8% | −9.8pp | −47.6% |
| Tiny (16x16-32x32) | 8.0% | 5.2% | −2.8pp | −35.0% |
| Small (32x32-96x96) | 3.1% | 2.3% | −0.8pp | −25.8% |
| Medium/Large (>96x96) | 1.8% | 1.2% | −0.6pp | −33.3% |

Two things are both true at once: **(1)** loosening IoU 0.5→0.3 helps Very Tiny by far the most in absolute terms (−9.8pp vs. −0.6 to −2.8pp for the other bins), confirming the metric-harshness mechanism from SS2/earlier discussion is real and disproportionately affects small boxes, as predicted; **(2)** the size-graded pattern still fully persists at IoU>=0.3 (10.8% vs. 1.2%, still a ~9x gap) — roughly **half** of Very Tiny's original 20.6% "head missed" figure is IoU-threshold near-misses (real detections, just not tightly localized enough for the 0.5 bar), and roughly **half survives even a much more permissive bar**, meaning it is not merely a measurement artifact. **Revised reading:** the detection-head localization weakness for small objects (SS5 #1) is real but was overstated by the raw 20.6% figure — the evidence now supports a real, size-graded head-localization/box-regression-precision issue of roughly half that magnitude, plus a metric-harshness component that inflates the raw recall-gap number but does not reflect an actual model deficiency. Not yet separated out of the remaining ~10.8%: how much is genuine complete misses (no usable candidate box at all) vs. class-confusion bleeding into this class-aware match (a correctly-located but wrong-class prediction counts as a miss for the true class here, same mechanism as SS4.1's confusion finding) — a class-agnostic version of this same cross-tabulation would separate the two and hasn't been run yet.

---

## 5. Gap diagnosis

Revised after computing the correct apples-to-apples comparator (SS2: this run vs. the paper's own Gaussian-only ablation, not the hybrid headline), re-examining what the recall-ceiling audit does and doesn't establish (SS4), quantifying the confusion matrix / P-curve (SS4.1), and — the largest single revision — confirming via a controlled rerun (SS9) that the VisDrone data-conversion pipeline itself was a major contributor. The ranked list below still describes the SS1 run's ~5.9% relative gap; SS9's rerun already closed about half of the AP gap (down to ~2.8%) and three-quarters of the AP50 gap (down to ~1.7%) via item 1 alone, so items 2+ below should now be read as candidates for the smaller *remaining* gap (SS9), not the original 5.9%.

**Ruled out:**
- **Training under-convergence.** SS3 — the run converged normally under its 50-epoch schedule; last-15-epoch mAP gain was +0.008.
- **Gross selector coverage failure — including the specific, size-biased version of it.** SS3/SS4 gave the aggregate signal (BPRbox 0.972); SS4.2 confirms it holds uniformly per size bin too (selector-dropped rate is 2.0-2.7%, flat, from Very Tiny through Medium/Large) — directly disproving the hypothesis that the selector disproportionately drops small objects. Whatever is producing the AP gap sits entirely downstream of object selection, at the detection-head/confidence-ranking stages (SS4.1, SS4.2).

**Real but small, already priced into the comparison:**
- **Gaussian-only vs. hybrid masks.** Table V's own delta (35.7/59.5 Gaussian-only vs. 36.0/59.7 hybrid) is 0.3/0.2 AP/AP50 points. SS2's table already uses the Gaussian-only row as the comparator, so this factor is *not* available to explain the remaining 5.9% relative gap -- it's already subtracted out. Occupy stabilizing at 0.42-0.43 (higher patch retention than the paper's implied ratio, given GFLOPs came out 17% above Table I's 119.5) is still consistent with Gaussian masks producing broader/blurrier activation than SAM-assisted ones, but that's a compute-efficiency observation, not an AP explanation on its own.

**Confirmed contributors to the SS1 run's ~5.9% relative AP/AP50 gap (ranked by evidence strength):**
1. **VisDrone data-conversion pipeline mismatch — confirmed by SS9 via a controlled rerun, the strongest-evidence item on this list.** SS1's data (`/root/autodl-tmp/VisDrone`) turned out to be Ultralytics' convenience conversion of VisDrone2019-DET, not ESOD's own `prepare_visdrone()` — the two differ in truncation-based GT filtering (negligible in practice, box counts matched almost exactly) and, materially, in pixel-masking "ignored regions"/"others" areas to gray before saving (46.7%/64.1% of train/val images affected). Retraining E1.0 identically except for this one variable (SS9) closed about half the AP gap and three-quarters of the AP50 gap vs. the paper's Gaussian-only ablation. This is the only item on this list backed by a controlled A/B rerun rather than inference from a single run's artifacts, and it now explains more of the original gap than every other item below combined.
2. **Detection-head localization failure on small objects within correctly-selected patches — confirmed by SS4.2, newly identified, magnitude refined by its own IoU-sensitivity check.** At conf>=0.001, IoU>=0.5, 20.6% of Very Tiny GT boxes inside a selector-chosen patch get no match at all, vs. 1.8% for Medium/Large. Re-running the same check at IoU>=0.3 (SS4.2) shows roughly **half** of that 20.6% (down to 10.8%) is IoU-threshold near-misses — real detections just not tightly localized enough for the 0.5 bar — while the other half persists even at the looser threshold, so a real, size-graded head-localization issue remains (~9x Very Tiny vs. Medium/Large at IoU>=0.3 too), just roughly half the magnitude the raw 20.6% figure suggested. SS4.2 rules out attributing either component to the selector (its drop rate is flat, 2.0-2.7%, across every bin). This still points at the SparseHead's own box regression/objectness for small objects as the mechanism, not object selection — architecture/loss changes at the detection-head level (anchor design, small-box regression loss weighting) look like a more promising lever for closing this specific part of the gap than selector-only interventions (e.g. coverage loss, which targets selection, not head localization) — but the achievable ceiling from head-only fixes should be sized against ~half of 20.6%, not the full figure.
3. **Confidence-score ranking/calibration — confirmed by SS4.1, not just inferred.** The confusion matrix's `background FN` row shows 10-45% of GT objects (class-dependent) never survive conf>=0.25, despite the 87.75% ceiling at conf>=0.001 saying they *are* found by some prediction. A meaningful share of true detections sit in a low-but-nonzero confidence band that any reasonable operating threshold discards. `P_curve.png`'s smooth, non-pathological shape suggests this is under-calibration, not a distinct bug. Still open: *why* calibration is weaker than the paper's — candidates below. Not yet re-checked against SS9's official-data run (SS9 flags this as a pending diagnostic).
4. **Genuine cross-class confusion — confirmed by SS4.1, newly identified.** `bicycle`->`motor` 12%, `tricycle`<->`awning-tricycle` 19%/7%, `car`<->`van`/`truck`/`bus` 3-11%. Directly costs AP for the confused classes (wrong-class detections are FPs for the predicted class and FNs for the true one). Not yet separable: how much is inherent VisDrone difficulty vs. this reproduction specifically — would need the paper's own confusion matrix (not published) or a second independent run to compare against.
5. **Single seed, no repeats.** The paper doesn't report variance either; a single run landing on the low end of natural noise can't be ruled out.
6. **Exact Gaussian mask generation parameters.** "Gaussian-only" is not a single fully-pinned specification -- sigma/threshold choices affect mask sharpness, and mask quality could plausibly feed into the selector's own confidence signal propagating downstream. This run used the official `gen_mask()` from the released repo, which should match the paper's own Gaussian recipe since it's the same code, but that assumption hasn't been independently verified against whatever produced Table V's specific numbers.
7. **Hyperparameter/library-version drift** from the paper's original (~2021-era) stack to this run's PyTorch 2.8/CUDA 12.8 stack. The smooth, textbook-shaped loss/mAP curves in SS3 give no direct evidence this happened, so this stays last-ranked, not eliminated.

---

## 6. Claim policy

**May be stated now:**
- E1.0 baseline trained and evaluated end-to-end on real VisDrone data with valid Gaussian-fallback selector masks (fail-closed checked, see `ESOD-Baseline-Patches.md`), on the current environment stack.
- The run converged normally; the observed AP/AP50/GFLOPs gap vs. the paper is not attributable to premature stopping or training instability.
- Selector-level coverage (BPRbox, low-threshold recall ceiling) is healthy; the shortfall is concentrated in overall detection quality/compute efficiency, not gross object loss at the selection stage. **This now holds at the size-bin level too, not just in aggregate** (SS4.2: selector-dropped rate is 2.0-2.7%, flat, from Very Tiny through Medium/Large) — the selector does not disproportionately drop small objects on this run.
- A confidence-calibration gap, genuine cross-class confusion, and a size-graded detection-head localization failure (objects the selector *did* route correctly still failing to get an IoU>=0.5 match) are all **confirmed present** (SS4.1, SS4.2, quantified from `confusion_matrix.png`/`P_curve.png` and the new selector-coverage cross-tabulation) and together are the best-evidenced description of where the AP/AP50 gap sits. What is *not* yet established is *why* calibration/localization are weaker than the paper's, or how much of the confusion is VisDrone-inherent vs. reproduction-specific — those remain hypotheses (SS5 #4-#6, renumbered after SS9).
- **The VisDrone data-conversion pipeline (Ultralytics-style vs. ESOD's own `prepare_visdrone()`) is a confirmed, large contributor to the gap** (SS9) — verified via a controlled rerun, not inference. Retraining on `prepare_visdrone()`-converted data, with every other variable held fixed, closed about half the AP gap and three-quarters of the AP50 gap vs. the paper's Gaussian-only ablation (AP 0.336→0.347 vs. paper's 0.357; AP50 0.560→0.585 vs. paper's 0.595).
- **SS9's rerun is now fully audited** (bucket recall, confusion matrix/P-curve, selector-coverage breakdown all repeated on the official-data run). Two independent confirmations carried over unchanged: selector coverage is still not the bottleneck (drop rate flat ~2.1-2.7% across all size bins, matching SS4.2 exactly), and the head-localization gap is still the dominant confirmed contributor to what recall gap remains (barely moved). One new finding: confidence calibration improved consistently across **every** class (background-FN rate down 1-6pp each, zero exceptions) — the first real, quantified candidate mechanism for SS5's previously-open "why is calibration weaker than the paper's" question.

**Must not be stated yet:**
- "Paper reproduced" or any framing implying AP/AP50 match — a real gap remains even on the official-data rerun (SS9: ~2.8% relative on AP, ~1.7% on AP50, against the paper's own Gaussian-only ablation).
- Any claim that Gaussian-only masks are the/a major cause of the AP gap — SS5 shows the paper's own Gaussian-vs-hybrid delta (0.3 AP) is already priced into the comparator and does not cover the remaining ~5.9% relative shortfall. Mask choice may still matter for the GFLOPs/Occupy gap specifically, and remains an open (not yet ruled out) contributor to the calibration gap too; neither is demonstrated.
- Any claim that the 87.75% bucket-recall ceiling indicates the selector/detector is "mostly working" at the AP level — SS4/SS4.1 show this ceiling number does not predict AP, and the confusion matrix directly quantifies why (10-45% of objects found at conf>=0.001 don't survive conf>=0.25).
- That a selector-side intervention (e.g. `hesod/backends/hesod`'s coverage loss) is *the* fix for VisDrone's Very Tiny recall gap — SS4.2 shows this run's gap sits at the detection head, not the selector, for VisDrone specifically. **Whether this generalizes is not established**: UAVDT's aggregate selector coverage (BPRbox 0.807, SS1-adjacent audit still in progress) is far worse than VisDrone's 0.972, so a selector-side fix may matter more there — its own per-size-bin breakdown (SS4.2-equivalent) has not been run yet.
- A root cause for *why* calibration/localization are weaker than the paper's, or a VisDrone-inherent-vs-reproduction-specific split of the class confusion — both are open questions, not conclusions (SS7).
- Any cross-arm (E2.x) comparison — no other roster arm has been retrained yet under this environment.
- A root cause for *why* the pixel-masking/data-conversion fix improved calibration specifically — the mechanism (removing noisy unmasked-ignored-region supervision) is plausible and consistent with the direction of the effect, but not independently isolated (e.g. by toggling pixel-masking and truncation-filtering as two separate variables).
- Anything about UAVDT/TinyPerson as a completed audit — those runs are separate and not fully covered by this document yet. UAVDT's own E1.0 baseline has finished all 50 epochs and its eval step has completed (P=0.277/R=0.384/AP50=0.186/AP=0.106/BPR=0.807/Occupy=0.0674, confusion matrix and PR/P curves reviewed informally in-session), but its own bucket recall audit and selector-coverage breakdown have not yet been formally run and folded into this document.

---

## 7. Next steps to close the gap

1. ~~Inspect the PR curve / confusion matrix for a ranking-vs-coverage signature.~~ **Done (SS4.1)** — confirmed via `confusion_matrix.png` + `P_curve.png`, no retraining needed. Confirmed both a calibration gap and genuine cross-class confusion; see SS5 #2-#3.
2. ~~Test whether the selector itself (not just detection quality overall) disproportionately drops small objects — the specific mechanism `hesod/backends/hesod`'s coverage loss targets.~~ **Done for VisDrone (SS4.2)**, via new standalone tools `scripts/esod_baseline/dump_selected_patches.py` + `audit_selector_coverage.py` (no retraining, reuses `best.pt`). Ruled out for this run — selector-dropped rate is flat (2.0-2.7%) across all size bins; the size-graded gap sits at the detection head instead (SS5 #1). **Not repeated for UAVDT** — see item 3, UAVDT is paused pending a data-source concern, so this is deprioritized rather than skipped for lack of interest.
3. ~~Wait for UAVDT's first-round E1.0 baseline and audit it the same way as VisDrone.~~ **Done, and paused.** UAVDT's baseline finished and got audited (`audit_buckets.py`, `list_examples.py`, confusion matrix) — see the "UAVDT status" note at the top of this document. Its size-bin recall was non-monotonic (Medium/Large *worse* than Very Tiny) and root-caused to a likely GT labeling defect (oversized boxes over dense parking lots) in `/root/autodl-tmp/UAVDT_processed`, suspected to trace back to its data source (a third-party Kaggle repackaging, not verified against the official UAVDT release). Decision: do not use UAVDT's current numbers to draw conclusions about VisDrone or about the method generally; do not start TinyPerson without first verifying its data source given this precedent. The original plan to compare all three datasets side-by-side before deciding on a VisDrone retrain (SS7, prior text) is deferred until this is resolved — a single-dataset (VisDrone) gap diagnosis has to stand on its own for now.
4. ~~Confirm the VisDrone data-conversion pipeline (Ultralytics-style vs. ESOD's own `prepare_visdrone()`) as a gap contributor.~~ **Done (SS9)** — controlled rerun on `VisDrone_v2`, closed about half the AP gap and three-quarters of the AP50 gap. This is now the reference VisDrone dataset going forward; `/root/autodl-tmp/VisDrone` (SS1's Ultralytics-converted data) is superseded but kept for comparison, not deleted.
5. ~~Pull SS9's `confusion_matrix.png`/`P_curve.png` and rerun the SS4.2-equivalent selector-coverage breakdown against `VisDrone_v2`.~~ **Done (SS9)** — calibration improved consistently across every class; selector coverage and head-localization findings both held essentially unchanged from SS4.2.
6. Repeat E1.0 (on `VisDrone_v2` now, not the superseded SS1 data) with >=1 additional seed to establish whether the remaining ~2.8%/1.7% relative gap (vs. the paper's own Gaussian-only ablation) is inside or outside natural run-to-run variance.
7. Install SAM and regenerate hybrid masks **on `VisDrone_v2`** for a controlled Gaussian-vs-hybrid comparison, holding everything else fixed — direct test for the GFLOPs/Occupy gap, and now also a candidate for the calibration gap (SS5 #6) since mask quality could plausibly propagate into the selector's confidence signal. (If this was already in progress against the old SS1 data/masks, redo it against `VisDrone_v2` instead — the SAM masks should be regenerated on the newly re-converted images so the "ignored regions" pixel-masking and hybrid-mask fixes aren't accidentally conflated.)
8. Once E1.0 on `VisDrone_v2` is trusted (or its remaining gap is understood and accepted as a documented baseline), proceed to the locked five-arm HESOD roster (`HESOD-Proposal.md` SS7.3.1) using this same audited pipeline (`scripts/esod_baseline/`, `hesod/backends/hesod/`'s new `--selector-loss coverage`/`--top-k` flags) so every arm is compared against a baseline with a known, documented gap rather than an assumed-exact one. **SS4.2's finding is a live input to this**: if UAVDT (once its data source is resolved) confirms VisDrone's pattern (selector coverage healthy, gap at the head), coverage loss alone may not be the highest-leverage arm to prioritize first — worth weighing against the dual-evidence/concat arms, which also change what the head receives, not just what the selector routes.

---

## 8. Local patches applied during this audit

- Fixed a real bug found while auditing the (still in-progress) UAVDT run and confirmed present in this VisDrone run's code path too: `test.py`'s per-class label count (`nt`) was gated behind `stats[0].any()`, so any validation pass with zero IoU>0.5 hits displayed `Labels: 0` and `BPR: nan` even though the true label count was never zero (matches `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` #2's already-documented fix for the separate vendored fork). Applied identically to `esod/`, `hesod/backends/esod/`, `hesod/backends/hesod/`. Display-only — doesn't change any number in this document, since this VisDrone run never actually hit the zero-hit branch (epoch 0's validation already had real matches, see SS3).

---

## 9. E1.0 rerun on official-source VisDrone data (`VisDrone_v2`) — confirms the data-conversion pipeline was a real, large contributor

**Motivation.** SS1's run used `/root/autodl-tmp/VisDrone`, whose provenance turned out to be the Ultralytics YOLOv5/v8 convenience conversion of VisDrone2019-DET (confirmed: its original `VisDrone.yaml` was in plain Ultralytics format before this project rewrote it to ESOD format), not ESOD's own `scripts/data_prepare.py::prepare_visdrone()`. Diffing the two conversion routines found two real differences: (1) ESOD excludes GT boxes with `truncation>=2` (>50% truncated), Ultralytics' conversion does not appear to; (2) ESOD physically blanks "ignored regions"/"others"-category pixels to gray (value 85) in the image itself before saving (`_masked.jpg`), Ultralytics' conversion never touches image pixels, only the label file.

**Provenance of the new run.**

| Field | Value |
|---|---|
| Data | `/root/autodl-tmp/VisDrone_v2` — raw VisDrone2019-DET (Ultralytics-hosted zips of the official 2019-DET release; aiskyeye.com confirms "the data set for Object Detection [is] the same as VisDrone2019") re-converted with ESOD's own `prepare_visdrone()`, then reorganized into this project's standard `images/`\|`labels/`\|`masks/` layout via `scripts/esod_baseline/reorganize_visdrone.py` (handles the `_masked.jpg`/`_masked.txt` filename substitution correctly via `split/{train,val}.txt`, not a directory glob). |
| Label count sanity check | val: 38759 vs. 38759 (exact match to SS1's data). train: 343205 vs. 343200 (5-box difference, noise-level). **Box counts are essentially identical between the two conversions** — the truncation filter evidently almost never fires on this data; the real difference is the pixel-masking (confirmed present: 3020/6471 train images and 351/548 val images — 46.7% and 64.1% respectively — have `_masked.jpg` pixel content). |
| Everything else | Identical to SS1: same `models/cfg/esod/visdrone_yolov5m.yaml`, same `data/hyps/hyp.visdrone.yaml`, same batch 8 / img 1536 / 50 epochs, same Gaussian-only masks (no SAM), same code (`hesod/backends/esod/`, git hash `cabcca6`). This isolates the data-conversion variable from mask-quality (SAM vs. Gaussian) and hyperparameters. |
| Run name | `visdrone_yolov5m_official_data` (kept separate from SS1's `visdrone_yolov5m_baseline`, which is untouched, so both remain directly comparable). |

**Headline comparison:**

| Metric | Paper, Gaussian-only (Table V) | SS1 (Ultralytics data) | This run (official data) | SS1 gap vs. paper | This run's gap vs. paper |
|---|---:|---:|---:|---:|---:|
| AP@[.5:.95] | 0.357 | 0.336 | **0.347** | −0.021 (−5.9%) | **−0.010 (−2.8%)** |
| AP50 | 0.595 | 0.560 | **0.585** | −0.035 (−5.9%) | **−0.010 (−1.7%)** |
| P | not reported | 0.615 | 0.662 | — | — |
| R | not reported | 0.553 | 0.552 | — | — |
| BPR | not reported | 0.972 | 0.972 | — | — |
| Occupy | not reported | 0.433 | 0.424 | — | — |
| GFLOPs | 119.5 | 140.2 | 138.5 | — | — |

**The AP gap vs. the paper's own Gaussian-only ablation shrank from −5.9% to −2.8% relative (about half closed); the AP50 gap shrank from −5.9% to −1.7% relative (about three-quarters closed).** Precision improved meaningfully too (0.615→0.662). BPR and Occupy stayed essentially flat, consistent with SS4.2's earlier finding that selector coverage was never the bottleneck — this fix acted elsewhere (very likely by removing the noisy/ambiguous unmasked-ignored-region supervision signal SS9's motivation paragraph describes, though that specific causal step has not been isolated further, e.g. by testing pixel-masking and truncation-filtering as two independently-toggled variables).

**Bucket recall audit (`audit_buckets.py`, same conf>=0.001/IoU>=0.5 protocol as SS4):**

| Size bin | SS1 (Ultralytics data) | This run (official data) | Δ |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 76.87% | 78.00% | +1.13pp |
| Tiny (16x16-32x32) | 90.25% | 90.85% | +0.60pp |
| Small (32x32-96x96) | 95.29% | 95.16% | −0.13pp (noise) |
| Medium/Large (>96x96) | 96.72% | 97.00% | +0.28pp |
| **Total** | **87.75%** | **88.29%** | +0.54pp |

Same monotonic shape as SS4 (no new anomaly), broadly small positive shift across every bin, largest at Very Tiny — consistent with the pixel-masking fix disproportionately helping the hardest-to-see objects, though this is a plausible reading, not independently isolated. The per-size-bin class-composition breakdown (same tool used to root-cause UAVDT's anomaly) was also checked here specifically looking for a UAVDT-style single-class collapse within a bin — **none found**: every class's recall within every bin is in a normal, unremarkable range, no red flags of the kind that flagged UAVDT's data quality.

**SS4.2-equivalent selector-coverage breakdown, now done:**

| Size bin | SS4.2 (Ultralytics data) covered,missed | This run (official data) covered,missed | Selector-dropped (both runs) |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 20.6% | 19.5% | ~2.7%, unchanged |
| Tiny (16x16-32x32) | 8.0% | 7.4% | ~2.1%, unchanged |
| Small (32x32-96x96) | 3.1% | 3.2% | ~2.1%, unchanged |
| Medium/Large (>96x96) | 1.8% | 1.8% | ~2.1%, unchanged |

**SS4.2's conclusion holds, essentially unchanged, on the official data.** Selector-dropped rate is still flat (~2.1-2.7%) across every bin — the data-conversion fix did not touch selector behavior. The head-localization gap (`covered,missed`) improved only slightly for the two smallest bins (Very Tiny −1.1pp, Tiny −0.6pp) and is flat for Small/Medium-Large — it is still the dominant, size-graded, confirmed contributor to what recall gap remains, now on a smaller base. This is a second, independent confirmation (different dataset conversion, same model/hyperparameters) that selector coverage is not VisDrone's bottleneck.

**SS4.1-equivalent confusion-matrix/P-curve check, now done:**

`background FN` row, same class order, same conf=0.25/iou=0.45 operating point as SS4.1:

| Class | SS4.1 (Ultralytics data) | This run (official data) | Δ |
|---|---:|---:|---:|
| pedestrian | 27% | 24% | −3pp |
| people | 38% | 35% | −3pp |
| bicycle | 45% | 41% | −4pp |
| car | 10% | 9% | −1pp |
| van | 12% | 9% | −3pp |
| truck | 30% | 24% | −6pp |
| tricycle | 35% | 31% | −4pp |
| awning-tricycle | 38% | 33% | −5pp |
| bus | 20% | 18% | −2pp |
| motor | 30% | 27% | −3pp |

**Every single class's background-FN rate improved, by 1-6 percentage points, with no exceptions.** This is a clean, consistent signal that the data-conversion fix improved confidence calibration specifically, not just raw AP — directly answering one of SS5's open questions ("why is calibration weaker than the paper's") with a real, quantified candidate mechanism: the unmasked "ignored regions" pixels in the old data were plausibly injecting noisy/ambiguous supervision that degraded calibration, and removing them (via the official pixel-masking) measurably helped. `P_curve.png` remains smooth and non-pathological (same shape/characterization as SS4.1), consistent with under-calibration rather than a distinct bug, on both runs.

Cross-class confusion spot-checks (limited to pairs SS4.1's text explicitly quoted, since that entry summarized some pairs as a range rather than exact figures): `bicycle`->`motor` still exactly 12% (unchanged); `awning-tricycle`->`tricycle` still exactly 7% (unchanged); `tricycle`->`awning-tricycle` improved 19%->17%. The full `car`/`van`/`truck`/`bus` cross-confusion block was not precisely diff-able against SS4.1 (that entry recorded only an aggregate "3-11%" range, not the exact per-pair values needed for a fair before/after comparison) — this run's exact values are now on record in this document's underlying `confusion_matrix.png` for any future comparison.

**Revises SS5's gap diagnosis:** the VisDrone-specific data-conversion pipeline (Ultralytics vs. ESOD's own `prepare_visdrone()`) is now a **confirmed, large contributor** — larger, by this measurement, than any single item previously in SS5's ranked list — not a ruled-out or minor factor. This was tested directly via a controlled rerun (only the data-conversion variable changed), which is stronger evidence than any of SS5's other items currently have. See SS5 for the updated ranking.
