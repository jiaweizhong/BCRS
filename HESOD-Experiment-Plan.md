# HESOD Experiment Plan

**Reset note (2026-08-09):** this document was rewritten from scratch to drop the original VisDrone baseline audit, which ran on a mis-converted copy of VisDrone2019-DET (Ultralytics' convenience conversion, not ESOD's own `prepare_visdrone()`) and is superseded in every respect by the official-data rerun below. The old material's only lasting value — that the two conversions differ and which one to use — is folded into §1's provenance note. Full history is still in this file's git log if ever needed.

**Status.** VisDrone: baseline reproduced close to the paper's own Gaussian-only ablation (§1), two roster arms (dual-evidence concat, channel-pooled concat) tested and both negative including under a hard patch-budget sweep (§2-3). TinyPerson: baseline retrained on the corrected hyp file, gap closed by more than half (§4.1-4.2); channel-pooled-concat is a genuine positive result (first roster arm anywhere in this project to move AP, not just recall ceiling), box-regression size-weighting is negative (§4.3). UAVDT: re-fetched from the official source and being reconverted with the corrected single-class (nc=1) protocol; baseline training in progress (§5).

---

## 1. VisDrone baseline

### 1.1 Provenance

| Field | Value |
|---|---|
| Data | `/root/autodl-tmp/VisDrone_v2` — raw VisDrone2019-DET converted with ESOD's own `scripts/data_prepare.py::prepare_visdrone()`, reorganized via `scripts/esod_baseline/reorganize_visdrone.py`. 6471 train / 548 val images, 38759 val GT boxes. |
| Code | `hesod/backends/esod/` (pristine upstream + environment-compat patches only, `ESOD-Baseline-Patches.md`), git hash `cabcca6` |
| Config | `models/cfg/esod/visdrone_yolov5m.yaml`, `data/hyps/hyp.visdrone.yaml`, batch 8, img 1536, 50 epochs, SGD lr0=0.01 cosine |
| Selector masks | Gaussian-only fallback (`gen_masks.py`, no SAM installed) — **not** the paper's official hybrid Gaussian+SAM pseudo-mask |
| Seeds | 1 (no repeats) |
| Run name | `visdrone_yolov5m_official_data` |

**Operational note — data source matters, use `prepare_visdrone()`, not a generic Ultralytics-style conversion.** The common Ultralytics YOLOv5/v8 convenience conversion of VisDrone2019-DET never pixel-masks "ignored regions"/"others"-category areas to gray before saving images; ESOD's own `prepare_visdrone()` does. Confirmed via a controlled A/B rerun (identical code/hyperparameters, only the data conversion changed): the pixel-masking fix closed about half the AP gap and three-quarters of the AP50 gap to the paper, and improved confidence calibration (background-FN rate at conf=0.25) for every one of VisDrone's 10 classes, no exceptions. If VisDrone is ever re-converted for any reason, re-verify `_masked.jpg` pixel content is present (46.7%/64.1% of train/val images should show it) before trusting a new baseline against this document's numbers.

### 1.2 Results vs. paper

Table I's headline ESOD number (AP 36.0) is the paper's **hybrid Gaussian+SAM** mask setting; this project has no SAM masks, so Table V's **Gaussian-only ablation row** (AP 35.7/AP50 59.5) is the correct comparator, not Table I.

| Metric | Paper, Gaussian-only (Table V) | This run | Gap |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.357 | 0.347 | −0.010 (−2.8%) |
| AP50 | 0.595 | 0.585 | −0.010 (−1.7%) |
| P | not reported | 0.662 | — |
| R | not reported | 0.552 | — |
| BPR | not reported | 0.972 | — |
| Occupy | not reported | 0.424 | — |
| GFLOPs | 119.5 | 138.5 | — |

**Not a precise reproduction, but close** — about half the original (pre-data-fix) gap remained after fixing the data pipeline; the rest is diagnosed in §1.3-1.6 below.

### 1.3 Bucket recall audit (`audit_buckets.py`, conf>=0.001, IoU>=0.5, one-to-one matching)

| Size bin | GT | Recalled | Recall |
|---|---:|---:|---:|
| Very Tiny (<16x16) | 11955 | 9334 | 78.00% |
| Tiny (16x16-32x32) | 14631 | 13293 | 90.85% |
| Small (32x32-96x96) | 11105 | 10568 | 95.16% |
| Medium/Large (>96x96) | 1068 | 1036 | 97.00% |
| **Total** | **38759** | **34231** | **88.29%** |

This is a low-confidence-threshold recall *ceiling* (can the GT box be found at all, at any confidence), not the deployment operating point — it is not meant to match R=0.552 at the best-F1 threshold, and it does not by itself predict AP (§1.6).

### 1.4 Selector-coverage breakdown (`dump_selected_patches.py` + `audit_selector_coverage.py`) — is the selector dropping small objects?

For every GT box: is its center inside any patch the trained selector actually chose at inference (no GT-mask oracle), cross-tabulated against §1.3's recall match:

| Size bin | Total GT | Selector-dropped | Covered, head missed | Covered, recalled |
|---|---:|---:|---:|---:|
| Very Tiny (<16x16) | 11955 | ~2.7% | 19.5% | ~77.8% |
| Tiny (16x16-32x32) | 14631 | ~2.1% | 7.4% | ~90.5% |
| Small (32x32-96x96) | 11105 | ~2.1% | 3.2% | ~94.7% |
| Medium/Large (>96x96) | 1068 | ~2.1% | 1.8% | ~96.1% |

**Selector-dropped rate is flat (~2.1-2.7%) across every size bin — the selector does not disproportionately drop small objects.** All of the size-graded gap lives in `Covered, head missed`: 19.5% for Very Tiny down to 1.8% for Medium/Large. A follow-up IoU-sensitivity check (re-matching at IoU>=0.3 instead of 0.5, diagnostic only, does not change any reported AP/AP50) found roughly half of Very Tiny's head-missed figure is IoU-threshold near-misses (real detections, just not tightly localized enough for 0.5), and roughly half persists even at the looser bar — i.e. a real, size-graded **detection-head localization weakness**, genuinely present but of roughly half the magnitude the raw 19.5% figure suggests on its own.

### 1.5 Confusion matrix and confidence calibration (`confusion_matrix.png`, `P_curve.png`, conf=0.25/iou=0.45)

`background FN` row (true object, nothing survived at conf>=0.25) by class: pedestrian 24%, people 35%, bicycle 41%, car 9%, van 9%, truck 24%, tricycle 31%, awning-tricycle 33%, bus 18%, motor 27%. A meaningful share of the objects §1.3's near-zero-threshold recall ceiling says *are* found never survive a realistic confidence cutoff — a **confidence-ranking/calibration gap**, additive with §1.4's head-localization finding, not the same mechanism (this one is measured among boxes already matched at IoU>=0.5; §1.4 measures whether a match exists at all).

The matrix also shows genuine **cross-class confusion** among visually similar categories: `bicycle`→`motor` 12%, `tricycle`↔`awning-tricycle` 17%/7%, `car`↔`van`/`truck`/`bus` several points each. `P_curve.png` rises smoothly with no pathological shape — consistent with under-calibration, not a distinct bug.

### 1.6 Gap diagnosis

Ranked by evidence strength, for the ~2.8%/1.7% relative AP/AP50 gap that remains after §1.1's data-pipeline fix:

1. **Detection-head localization on small objects, within correctly-selected patches (§1.4).** Confirmed via a controlled IoU-sensitivity check, not just inference. Points at the SparseHead's own box regression/objectness for small objects, not object selection — an architecture/loss lever at the head level (anchor design, small-box regression loss weighting — see `--box-loss size_weighted` in `hesod/backends/hesod/utils/loss.py`, built but not yet run) looks more promising than selector-only interventions for this specific part of the gap.
2. **Confidence-score calibration (§1.5).** Real, quantified, but *why* it's weaker than the paper's is still open.
3. **Genuine cross-class confusion (§1.5).** Not separable yet from "inherent VisDrone difficulty" vs. reproduction-specific, since the paper doesn't publish its own confusion matrix.
4. **Single seed, no repeats.** Can't rule out landing on the low end of natural run-to-run variance.
5. **Exact Gaussian mask generation parameters / no SAM.** Uses the official `gen_mask()`, should match the paper's recipe since it's the same code, but not independently verified against whatever produced Table V specifically.
6. **Hyperparameter/library-version drift** (paper's ~2021-era stack vs. this project's PyTorch 2.8/CUDA 12.8). No direct evidence this is happening (loss/mAP curves are smooth and textbook-shaped), stays last-ranked, not eliminated.

**Ruled out:** training under-convergence (50-epoch curve fully flattened, last-15-epoch mAP gain +0.008) and gross/size-biased selector coverage failure (§1.4: selector-dropped rate flat 2.1-2.7% across every bin).

---

## 2. VisDrone roster arms: dual-evidence concat & channel-pooled concat — both negative

**Provenance.** `hesod/backends/hesod/` (git hash `570a2e7`), trained on `VisDrone_v2` (§1's data), `--selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0`, otherwise identical recipe to §1 (same hyp, batch 8, img 1536, 50 epochs). Dual-evidence: `models/cfg/esod/visdrone_yolov5m_dual_evidence_concat.yaml`. Channel-pooled: same recipe, cfg swapped to `models/cfg/esod/visdrone_yolov5m_channel_pooled_concat.yaml`.

**Headline (batch=1 `--task measure` numbers for GFLOPs/FPS):**

| Metric | §1 baseline | Dual-evidence concat | Channel-pooled concat |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.347 | 0.344 | 0.347 |
| AP50 | 0.585 | 0.584 | 0.586 |
| P | 0.662 | 0.684 | 0.665 |
| R | 0.552 | 0.539 | 0.548 |
| BPR | 0.972 | 0.986 | 0.986 |
| Occupy | 0.424 | 0.557 (+31%) | 0.569 (+34%) |
| GFLOPs | 138.5 | 171.2 (+23.6%) | 167.0 (+20.6%) |
| FPS | 94.9 | 84.9 | 86.6 |

**Neither arm improved AP or AP50 over baseline — both cost meaningfully more compute, and channel-pooling barely recovered any of it** (167.0 vs. 171.2 GFLOPs, −2.5%: GFLOPs here is driven mainly by how much image area the selector decides to keep, which pooling doesn't touch, not by the selector's own internal compute, which it does).

**Bucket recall — both arms raise the recall ceiling slightly (channel-pooled the most, largest gains at Very Tiny) but neither moves AP:**

| Size bin | Baseline | Dual-evidence | Channel-pooled |
|---|---:|---:|---:|
| Very Tiny | 78.00% | 78.62% | 79.44% |
| Tiny | 90.85% | 91.18% | 91.20% |
| Small | 95.16% | 95.31% | 95.66% |
| Medium/Large | 97.00% | 97.10% | 97.19% |
| **Total** | **88.29%** | **88.65%** | **89.01%** |

Channel-pooled has the best recall ceiling of all three arms and ties baseline (not beats it) on AP — a third confirmation that recall-ceiling gains and AP are not moving together on VisDrone (§1.6 already established this mechanism). Confusion-matrix deltas were mixed (some classes better, some flat, one or two worse) rather than the clean universal improvement §1.1's data-pipeline fix produced — consistent with a diffuse, roughly-cancelling effect on calibration rather than a real fix.

**Conclusion: on VisDrone specifically, selector-side interventions (coverage loss, dual-evidence fusion, pooled or not) are not the highest-leverage lever** — §1.6 already places the remaining gap at head-localization and calibration, not selector coverage, and these two arms only ever targeted the selector. Scoped to VisDrone: UAVDT's much worse aggregate selector coverage (BPRbox 0.807 vs. VisDrone's 0.97+, §5) means this should not be assumed to transfer there, and see §4 for TinyPerson.

---

## 3. VisDrone Top-K patch-budget sweep — does the roster's disadvantage reverse under a tight compute budget?

**Motivation.** An earlier, now-deleted version of this experiment line (`BCRS-Experiment-Plan.md`, recovered from git history) ran under a hard Top-K router and found channel-pooled dual-evidence's relative AP gain over baseline *increasing* sharply as K tightened (K=16: +268% relative AP; K=64: +39%, computed from the recovered table). §2's arms were tested under the paper's normal *threshold* routing, which has no fixed budget and so never actually tests that specific hypothesis. This sweep reruns it directly on §1/§2's own checkpoints via `test.py --top-k K` (`run_topk_sweep.sh`, no retraining), forcing exactly K of 64 P3-grid cells regardless of the model's own threshold, K∈{16,32,48,64}.

**AP@[.5:.95] by arm × K, all on `VisDrone_v2`:**

| K | Baseline | Dual-evidence concat | Channel-pooled concat |
|---|---:|---:|---:|
| 16 | 0.276 | 0.275 (−0.4%) | 0.272 (−1.4%) |
| 32 | 0.336 | 0.330 (−1.8%) | 0.331 (−1.5%) |
| 48 | 0.338 | 0.333 (−1.5%) | 0.334 (−1.2%) |
| 64 | 0.338 | 0.333 (−1.5%) | 0.334 (−1.2%) |

(Occupy is identical across arms at a given K by construction — forced Top-K overrides each model's own patch selection — so this is the first true matched-compute comparison between the three arms.)

**Clean negative result, and it contradicts the old recovered pattern rather than merely failing to replicate it.** At every K from 16 to 64, both roster arms are flat-to-slightly-worse than baseline, with no trend of narrowing (let alone reversing) as the budget tightens — the old +268%→+39% curve does not appear in any form here. **Most defensible read: the old sweep's budget-dependent gains were an artifact of that run's weak baseline or degraded (pre-fix) VisDrone data, not a genuine latent advantage of dual-evidence fusion under tight patch budgets.** No further budget-constrained follow-up planned on VisDrone; UAVDT or TinyPerson would be more informative venues if this is worth retesting, since VisDrone's selector was never the bottleneck to begin with (§1.4).

*Side note:* baseline's forced K=64 AP (0.338, Occupy≈1.0, literally the whole image) is measurably below its own free-threshold result (0.347, Occupy=0.424) — the SparseHead's performance is not simply monotonic in how much area it's given.

---

## 4. TinyPerson

### 4.1 Baseline + official APt50/APs50 evaluation

**Provenance.** Data: official TinyPerson release (`ucas-vg/PointTinyBenchmark`), reorganized via `scripts/esod_baseline/reorganize_tinyperson.py` (full official `trainval.txt`/`test.txt` splits). Config: `models/cfg/esod/tinyperson_yolov5m.yaml`, img-size=2048 (confirmed correct — paper text: *"the larger side of input size is set to ... 2,048 for TinyPerson"*), trained with `data/hyps/hyp.tinyperson.scratch.yaml`.

**Operational note — hyp file matters, use `scratch`, not `finetune`.** The first baseline used `hyp.tinyperson.finetune.yaml` on the reasoning "we init from the pretrained COCO checkpoint, so finetune applies" — but `hyp.visdrone.yaml`/`hyp.uavdt.yaml` (both close to the paper on their own datasets) also init from that same checkpoint and are both scratch-style, contradicting that reasoning; `hyp.tinyperson.finetune.yaml`'s header is unmodified upstream `ultralytics/yolov5` boilerplate ("VOC finetuning"), never adapted to TinyPerson, unlike the scratch file. A controlled A/B rerun (identical data/code/img-size, only the hyp file changed) confirmed this: AP50 0.534→0.605 (+13.3% relative), AP@[.5:.95] 0.185→0.222 (+20% relative), and the official APt50/APs50 gap to the paper roughly halved (below). **`hyp.tinyperson.scratch.yaml` is now the standard choice for all TinyPerson arms.**

**This project's generic metrics** (`audit_buckets.py` convention — Very Tiny/Tiny/Small/Medium-Large by pixel size, not the same bins as the official table below):

| Metric | Value |
|---|---:|
| P | 0.712 |
| R | 0.574 |
| AP50 | 0.605 |
| AP@[.5:.95] | 0.222 |
| BPR | 0.931 |
| Occupy | 0.128 |
| GFLOPs | 146.8 |

| Size bin | Recall |
|---|---:|
| Very Tiny | 73.20% |
| Tiny | 86.50% |
| Small | 84.50% |
| Medium-Large | 84.02% |
| **Total** | **79.09%** |

**Official evaluation** (`scripts/esod_baseline/tinyperson_eval/eval_tinyperson_official.py`, vendored+patched `tinyperson_cocoeval.py` from `ucas-vg/PointTinyBenchmark`, run locally against the pulled-down GT (`tiny_set_test_all.json`); `Params.EVAL_STRANDARD='tiny'`, `ignore_uncertain=True`, `use_iod_for_ignore=True` — confirmed correct, matches `esod/evaluation/tiny_benchmark/MyPackage/tools/evaluate/evaluate_tiny.py` line 112's hardcoded call exactly):

| Area bin | AP @IoU=0.25 | AP @IoU=0.50 | AP @IoU=0.75 | AR @IoU=0.25 | AR @IoU=0.50 | AR @IoU=0.75 |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.7726 | 0.5953 | 0.1200 | 0.8701 | 0.7462 | 0.2650 |
| tiny (1-400px²) | 0.7493 | **0.5546** | 0.0819 | 0.8548 | 0.7075 | 0.2133 |
| ↳ tiny1 | 0.5999 | 0.3601 | 0.0256 | 0.7407 | 0.5036 | 0.0896 |
| ↳ tiny2 | 0.7949 | 0.5971 | 0.0829 | 0.8855 | 0.7394 | 0.2102 |
| ↳ tiny3 | 0.8341 | 0.6744 | 0.1248 | 0.9068 | 0.8105 | 0.2923 |
| small (400-1024px²) | 0.8483 | **0.7104** | 0.1843 | 0.9010 | 0.8233 | 0.3421 |
| reasonable (>1024px²) | 0.8427 | 0.7008 | 0.2218 | 0.9036 | 0.8133 | 0.3799 |

| Metric | Paper Table II | This run | Gap |
|---|---:|---:|---:|
| APt50 (IoU=0.50, area=tiny) | 61.3% | 55.46% | −5.84pp (−9.5% relative) |
| APs50 (IoU=0.50, area=small) | 74.4% | 71.04% | −3.36pp (−4.5% relative) |

Gap roughly halved from the finetune-hyp run (was −21%/−13% relative) and now the same order of magnitude as VisDrone's own remaining gap (§1.2: −2.8%/−1.7% relative) rather than a clear outlier. tiny1 is still visibly the weakest sub-bin (AP 0.36 vs. reasonable's 0.70 at IoU=0.5) — consistent with §1's VisDrone finding that a size-graded head-localization/calibration gap is a real, if secondary, effect, not fully explained by the hyp-file fix alone.

### 4.2 Protocol investigation — what did ESOD actually do for TinyPerson?

Checked directly against the paper's PDF full text and the `esod/` repo (including the officially-vendored `ucas-vg/TinyBenchmark` framework at `esod/evaluation/tiny_benchmark/`) rather than inferring:

- **Confirmed correct:** full-image img-size=2048 training, no external image cropping. TinyBenchmark's own 640×512 "corner" sub-window tiling protocol (`erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json`, documented in `esod/evaluation/tiny_benchmark/README.md`) is real, but is for the Faster-RCNN/RetinaNet-FPN baselines in the paper's comparison table, not for ESOD itself — the paper's own text states one image size per dataset, `prepare_tinyperson()` deliberately reads the full-image annotation file (not the corner one sitting next to it), and ESOD's own official converter `data_convert.py::darknet2tinyperson()` builds predictions keyed off full, uncut test images. Matches the paper's own framing of its contribution ("replace image-level cropping with feature-level slicing").
- **Confirmed correct:** `ignore_uncertain=True, use_iod_for_ignore=True, eval_standard='tiny'` in our own eval script — matches `evaluate_tiny.py`'s hardcoded call exactly.
- **Confirmed the fix: the training hyperparameter file** (§4.1's operational note) — was the largest single contributor found so far.
- **Loose end, low priority, not chased:** `data_convert.py::darknet2tinyperson()` discards any prediction with area > 30×30px before scoring, which taken literally would zero out the "small"/"reasonable" bins — inconsistent with the paper's real APs50=74.4. Most likely a narrow tool for isolating AP_tiny during development, not the Table II-generating path.
- **`ratio=16` tested (both plain baseline and full-spectral concat) — negative, contradicts the paper's own stated guidance.** Moves AP50/AP/BPR/bucket-recall/APt50 all further from the paper vs. `ratio=8` (§4.1's baseline), not closer; only APs50 nudged up marginally. Not supported in this implementation/data setup despite the paper's text suggesting it. Not pursuing further; detailed run artifacts not kept.

### 4.3 Roster arms — channel-pooled concat positive, box-size-weighting negative

Two single-variable arms on the same corrected recipe (`scripts/esod_baseline/run_tinyperson_scratch_hyp.sh`), each changing exactly one thing vs. §4.1's baseline: channel-pooled dual-evidence concat (selector-side coverage loss, same recipe as §2's VisDrone arm) and box-regression size-weighting (`--box-loss size_weighted --box-weight-ref-area 4.0 --box-weight-max 5.0`, head-side, targets §1.6's #1 VisDrone gap item — kept as its own arm so it's never confounded with the selector-side change).

| Metric | Baseline (§4.1) | Channel-pooled concat | Box-size-weighted |
|---|---:|---:|---:|
| P | 0.712 | 0.726 | 0.706 |
| R | 0.574 | 0.572 | 0.573 |
| AP50 (generic) | 0.605 | 0.619 | 0.596 |
| AP@[.5:.95] (generic) | 0.222 | 0.229 | 0.217 |
| BPR | 0.931 | 0.987 | 0.924 |
| Occupy | 0.128 | 0.392 (+206%) | 0.127 |
| GFLOPs | 146.8 | 241.1 (+64%) | 146.5 |
| FPS | 93.8 | 73.2 | 99.6 |
| Bucket recall, Total | 79.09% | 81.95% | 78.32% |
| **APt50 (official)** | 55.46% | **56.24%** (+1.4% rel.) | 54.99% (−0.8% rel.) |
| **APs50 (official)** | 71.04% | **72.95%** (+2.7% rel.) | 69.75% (−1.8% rel.) |
| tiny1 AP@IoU=0.5 (official) | 0.3601 | 0.3715 | **0.3755** |

**Channel-pooled concat: a genuine positive result, unlike on VisDrone (§2).** Every metric improves — bucket recall across all size bins (Very Tiny +1.9pp through Medium/Large +6.5pp), and critically the *official, paper-comparable* APt50/APs50 both improve too, not just the recall ceiling. Cost is real: Occupy triples (0.128→0.392) and GFLOPs jump 64%. Best mechanistic read: TinyPerson's baseline selector is far more budget-constrained than VisDrone's (Occupy 0.128 vs. 0.424) — closer to the tight-K regime where the recovered old `BCRS-Experiment-Plan.md` sweep (§3's motivation) originally saw dual-evidence pay off, and where §3's own VisDrone Top-K sweep found nothing. This is the first roster-arm result anywhere in this project where a coverage-loss/dual-evidence arm actually moved AP, not just the recall ceiling.

**Box-size-weighting: a negative result, opposite of what §1.6 predicted from VisDrone.** Every aggregate metric is flat-to-worse (AP50 −1.5%, AP@[.5:.95] −2.3%, bucket recall −0.8pp, APt50 −0.8% relative, APs50 −1.8% relative) despite being free in compute (GFLOPs/FPS essentially unchanged, as expected for a loss-only change). One narrow bright spot matching its design intent: tiny1 (the smallest official sub-bin) AP@IoU=0.5 is highest of all three arms (0.3755) — the reweighting does measurably help the very smallest objects specifically — but this doesn't survive aggregation once tiny2/tiny3/small/reasonable all move the other way. **Selector-side and head-side interventions gave opposite-sign results on TinyPerson**, worth remembering before assuming either lever generalizes across datasets: VisDrone's diagnosis (§1.6) ranked head-localization as the most promising untested lever there, but the one dataset this was actually tried on says otherwise, at least at these hyperparameters (`ref_area=4.0, max=5.0`, untuned for TinyPerson specifically).

---

## 5. UAVDT — re-fetched from official source, baseline in progress

**Old baseline (superseded, third-party data):** `/root/autodl-tmp/UAVDT_processed`, P=0.277/R=0.384/AP50=0.186/AP=0.106/BPR=0.807/Occupy=0.0674. Size-bin recall was non-monotonic (Medium/Large=52.13%, *worse* than Very Tiny=61.60%). Root-caused via `list_examples.py` + direct visual inspection (not inference): the `car` class specifically craters in the Medium/Large bin (51.16% vs. 92.91% in Small) while `bus` in the same bin scores 99.30%; 90% (18/20) of a random sample of "missed, Medium/Large car" GT boxes had no overlapping prediction of any class at all; recalled-vs-missed samples showed video sequence `M0802` at 0% recalled vs. 55% of the missed sample. Visually inspecting `M0802` frames: the offending GT boxes (~130-190px wide) sit on top of a dense parking lot of many small, individually-parked cars — one oversized box drawn over a cluster, not a real large vehicle. `/root/autodl-tmp/UAVDT_processed` was a third-party Kaggle repackaging of UAVDT, not verified against the official release's own annotations — consistent with this kind of box-quality defect.

**Re-fetched from the official source** (`UAV-benchmark-M.zip`, `M_attr.zip`, `UAV-benchmark-MOTD_v1.0.zip`), converted with ESOD's own `prepare_uavdt()` and reorganized via the new `scripts/esod_baseline/reorganize_uavdt.py` (video-subfolder raw layout, same in-place-suffix-swap style as TinyPerson's reorganizer, but with video-name-prefixed flattening since `imgXXXXXX.jpg` repeats across every video). 22321 train / 16580 test images, 100% mask coverage both splits.

**Class-count fix, resolved via code not text.** `prepare_uavdt()` was read end to end: it hardcodes every GT box to class 0 regardless of raw category (car/truck/bus), no flag changes this. This project's configs had `nc: 3` (`car`,`truck`,`bus`), previously set on the reasoning that the ESOD paper's text (§4.B) describes UAVDT as "3 categories to detect" — the paper's text and its own released code disagree here. Flipped back to `nc: 1` (names=`['vehicle']`) across all three trees (`esod/`, `hesod/backends/esod/`, `hesod/backends/hesod/`) since nc=3 against single-class-0 labels would have silently starved 2 of 3 classes of any positive example — going with what the code demonstrably does, since that's what's actually reproducible. `run_baseline.sh`'s `uavdt` case updated to point at `/root/autodl-tmp/UAVDT_v2.yaml`/`UAVDT_v2/`.

**Status: baseline training in progress.** Not yet auditable — no results to report until the run completes.

---

## 6. Next steps

1. **UAVDT baseline** (§5) — training in progress on official-source data with the corrected nc=1 protocol. Once done: full audit (bucket recall, selector coverage, confusion matrix) same as VisDrone/TinyPerson, and a direct comparison against the old third-party-data numbers to see how much of the old non-monotonic-recall anomaly was really the data defect vs. something else.
2. Once UAVDT's baseline is trusted, decide whether to test the roster arms there too (§1.6/§2's VisDrone conclusion and §4.3's TinyPerson conclusion actively disagree on which lever — selector-side vs. head-side — helps, so a third data point matters) and whether TinyPerson's "budget-constrained selector benefits more from dual-evidence fusion" hypothesis (§4.3) replicates on UAVDT, whose old (third-party-data) selector coverage (BPRbox 0.807) was the most budget-constrained of the three datasets tested so far.
3. Lower priority, not scheduled: repeat VisDrone baseline with an additional seed to size natural run-to-run variance against the remaining ~2.8%/1.7% gap; install SAM and regenerate hybrid masks for a controlled Gaussian-vs-hybrid comparison on `VisDrone_v2`; retune box-size-weighting's hyperparameters (`ref_area`, `max_weight`) for TinyPerson specifically before concluding the mechanism itself doesn't work there (§4.3 only tested one hyperparameter setting, carried over from the VisDrone-oriented default).

---

## 7. Known gotchas

- **VisDrone data conversion** — see §1.1. Must go through ESOD's own `prepare_visdrone()`; a generic Ultralytics-style conversion silently skips pixel-masking ignored regions and costs real AP/AP50/calibration.
- **TinyPerson hyp file** — `hyp.tinyperson.yaml` (the name `train.sh`'s default formula expects) does not exist; must explicitly pick `scratch` or `finetune`. **Use `scratch`** — confirmed via controlled A/B (§4.1): `finetune` cost ~13-20% relative AP/AP50 and roughly doubled the official-protocol gap to the paper vs. `scratch`.
- **TinyPerson data paths** — the dataset root is `/root/autodl-tmp/TinyPerson_v1` with a **sibling** config `/root/autodl-tmp/TinyPerson_v1.yaml` (same convention as `VisDrone_v2`/`VisDrone_v2.yaml` — config next to the data dir, not nested inside it). `run_baseline.sh` briefly had this wrong (`TinyPerson/tinyperson.yaml`, nested, wrong dir name) — fixed; if a "required file not found" error mentions a `TinyPerson` path, check it's using `TinyPerson_v1` first.
- **`--top-k` / `--selector-loss` / `--box-loss` only exist in `hesod/backends/hesod/`** (the dev tree), not the pristine `esod/`/`hesod/backends/esod/` mirrors — cross-tree checkpoint loads work fine (architecture is identical for the plain baseline), but roster-arm-specific flags will simply not be recognized if pointed at the wrong tree.
- **`test.py`'s per-class label count (`nt`)** was gated behind `stats[0].any()`, so a validation pass with zero IoU>0.5 hits anywhere displayed `Labels: 0`/`BPR: nan` even when the true label count wasn't zero. Fixed identically in `esod/`, `hesod/backends/esod/`, `hesod/backends/hesod/`. Display-only.
- **UAVDT class count** — `prepare_uavdt()` hardcodes every GT box to class 0; use `nc: 1` (`names: ['vehicle']`), not `nc: 3` (§5). Raw data layout for `prepare_uavdt()`: `<raw-root>/UAV-benchmark-M/<video>/imgXXXXXX.jpg`, `<raw-root>/M_attr/{train,test}/*.txt`, `<raw-root>/UAV-benchmark-MOTD_v1.0/GT/<video>_gt_{whole,ignore}.txt` — official zips (`UAV-benchmark-M.zip`, `M_attr.zip`, `UAV-benchmark-MOTD_v1.0.zip`) extract into exactly this shape. `reorganize_uavdt.py` (mirrors `reorganize_tinyperson.py`'s in-place-suffix-swap style, not `reorganize_visdrone.py`'s directory-swap style) flattens it, prefixing filenames with the video name since `imgXXXXXX.jpg` repeats across every video.
- **`--batch-size` is the GLOBAL batch size, not per-GPU — confirmed in `train.py`** (`opt.total_batch_size = opt.batch_size` before the DDP split, then `opt.batch_size = opt.total_batch_size // opt.world_size`, with `assert opt.batch_size % opt.world_size == 0`). The paper trained "on two Nvidia V100 GPUs... The batch size is 8" — if that followed this same repo's own `train.sh` (`torch.distributed.launch --nproc_per_node 2`), their *per-GPU* batch was 4; every arm in this document has run single-GPU with `--batch-size 8`, i.e. per-GPU 8, double theirs. The dominant training-dynamics knob (gradient-accumulation target, `nbs=64` in `train.py`) is computed from the *global* `total_batch_size`, which is 8 either way (4×2 GPUs vs. 8×1 GPU) — so this is not a wholesale recipe mismatch. The real, narrower difference is per-GPU BatchNorm statistics (computed from whatever batch actually lands on each GPU; the paper's own `sync_bn` defaults to off, so even their two V100s never synchronized BN stats with each other either) — a real, previously-undocumented difference, plausibly a minor contributor to the residual gaps throughout this document, magnitude not evaluated. Directly relevant to the TinyPerson `ratio=16` + full-spectral-concat arm's OOM at `--batch-size 8` on a single RTX 5090 (§4.3-adjacent, `tinyperson_yolov5m_concat_ratio16`) — the `--batch-size 2` workaround used there fixes the memory ceiling but does not reproduce the paper's per-GPU batch of 4 either; nothing single-GPU can.
- Environment-compat patches (NumPy 2.x, PyTorch 2.6+ `torch.load`, etc.) are tracked separately in `ESOD-Baseline-Patches.md` — that file, not this one, is the source of truth for "why did X break and how was it fixed."
