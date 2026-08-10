# HESOD Experiment Plan

**Reset note (2026-08-09):** this document was rewritten from scratch to drop the original VisDrone baseline audit, which ran on a mis-converted copy of VisDrone2019-DET (Ultralytics' convenience conversion, not ESOD's own `prepare_visdrone()`) and is superseded in every respect by the official-data rerun below. The old material's only lasting value — that the two conversions differ and which one to use — is folded into §1's provenance note. Full history is still in this file's git log if ever needed.

**Status.** VisDrone: baseline reproduced close to the paper's own Gaussian-only ablation (§1), two roster arms (dual-evidence concat, channel-pooled concat) tested and both negative including under a hard patch-budget sweep (§2-3). TinyPerson: baseline trained, official APt50/APs50 evaluated, gap larger than VisDrone's and not yet explained — leading candidate is a wrong training-hyperparameter file, untested (§4). UAVDT: paused on a data-source problem (§5).

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

**Provenance.** Data: official TinyPerson release (`ucas-vg/PointTinyBenchmark`), reorganized via `scripts/esod_baseline/reorganize_tinyperson.py` (full official `trainval.txt`/`test.txt` splits). Config: `models/cfg/esod/tinyperson_yolov5m.yaml`, img-size=2048 (confirmed correct — paper text: *"the larger side of input size is set to ... 2,048 for TinyPerson"*), trained with `data/hyps/hyp.tinyperson.finetune.yaml` — **now suspected wrong, see §4.2**.

**This project's generic metrics** (`audit_buckets.py` convention — Very Tiny/Tiny/Small/Medium-Large by pixel size, not the same bins as the official table below):

| Metric | Value |
|---|---:|
| P | 0.667 |
| R | 0.504 |
| AP50 | 0.534 |
| AP@[.5:.95] | 0.185 |
| BPR | 0.91 |
| Occupy | 0.132 |

| Size bin | Recall |
|---|---:|
| Very Tiny | 68.59% |
| Tiny | 83.74% |
| Small | 82.23% |
| Medium-Large | 76.92% |
| **Total** | **75.37%** |

**Official evaluation** (`scripts/esod_baseline/tinyperson_eval/eval_tinyperson_official.py`, vendored+patched `tinyperson_cocoeval.py` from `ucas-vg/PointTinyBenchmark`, `Params.EVAL_STRANDARD='tiny'`; `ignore_uncertain=True`, `use_iod_for_ignore=True` — **now confirmed correct**, see §4.2):

| Area bin | AP @IoU=0.25 | AP @IoU=0.50 | AP @IoU=0.75 | AR @IoU=0.25 | AR @IoU=0.50 | AR @IoU=0.75 |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.7068 | 0.5256 | 0.0840 | 0.8407 | 0.7065 | 0.2314 |
| tiny (1-400px²) | 0.6813 | **0.4839** | 0.0528 | 0.8192 | 0.6616 | 0.1781 |
| ↳ tiny1 | 0.5418 | 0.2916 | 0.0205 | 0.6687 | 0.4201 | 0.0647 |
| ↳ tiny2 | 0.7449 | 0.5278 | 0.0518 | 0.8611 | 0.7001 | 0.1634 |
| ↳ tiny3 | 0.7780 | 0.6060 | 0.0853 | 0.8880 | 0.7829 | 0.2591 |
| small (400-1024px²) | 0.7938 | **0.6494** | 0.1473 | 0.8781 | 0.7972 | 0.3195 |
| reasonable (>1024px²) | 0.7942 | 0.6274 | 0.1702 | 0.8868 | 0.7823 | 0.3412 |

| Metric | Paper Table II | This run | Gap |
|---|---:|---:|---:|
| APt50 (IoU=0.50, area=tiny) | 61.3% | 48.39% | −12.9pp (−21% relative) |
| APs50 (IoU=0.50, area=small) | 74.4% | 64.94% | −9.5pp (−13% relative) |

Substantially larger than VisDrone's post-fix gap (§1.2: −2.8%/−1.7% relative). Two things inside the table itself point at a real detection problem for the smallest objects, not just weaker confidence ranking: the tiny1→tiny3 gradient is steep (AP 0.29 vs. 0.63 at IoU=0.5), and **recall, not just AP, collapses for tiny1** (AR 0.42 vs. 0.78) — a pure calibration problem would depress AP while leaving the recall ceiling closer to intact.

### 4.2 Protocol investigation — what did ESOD actually do for TinyPerson?

Directly checked against the paper's PDF full text and the `esod/` repo (including the officially-vendored `ucas-vg/TinyBenchmark` framework at `esod/evaluation/tiny_benchmark/`) rather than inferring:

- **Confirmed correct:** full-image img-size=2048 training, no external image cropping. TinyBenchmark's own 640×512 "corner" sub-window tiling protocol (`erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json`, documented in `esod/evaluation/tiny_benchmark/README.md`) is real, but confirmed to be for the Faster-RCNN/RetinaNet-FPN baselines in the paper's comparison table, not for ESOD itself — the paper's own text states one image size per dataset, `prepare_tinyperson()` deliberately reads the full-image annotation file (not the corner one sitting next to it), and ESOD's own official converter `data_convert.py::darknet2tinyperson()` builds predictions keyed off full, uncut test images. This also matches the paper's own framing of its contribution ("replace image-level cropping with feature-level slicing"). An initial version of this document flagged corner-tiling as the leading suspect; that is retracted.
- **Confirmed correct:** `ignore_uncertain=True, use_iod_for_ignore=True, eval_standard='tiny'` — `esod/evaluation/tiny_benchmark/MyPackage/tools/evaluate/evaluate_tiny.py` line 112 hardcodes exactly this for its AP path. Previously an unverified assumption in our own `eval_tinyperson_official.py`; now confirmed to match the paper's own tooling.
- **Leading open suspect: the training hyperparameter file.** `hyp.tinyperson.finetune.yaml` was picked on the reasoning "we init from the pretrained COCO checkpoint, so finetune applies" — but `hyp.visdrone.yaml`/`hyp.uavdt.yaml` (both close to the paper on their own datasets) **also** init from that same checkpoint and are both scratch-style (header: "COCO training from scratch", lr0=0.01), which contradicts that reasoning. `hyp.tinyperson.scratch.yaml` is the structural analogue of those two (own tuned augmentation, includes the ESOD-specific `pixl` mask-loss gain); `hyp.tinyperson.finetune.yaml`'s header is unmodified upstream `ultralytics/yolov5` boilerplate ("VOC finetuning"), never adapted to TinyPerson. Cheap, direct test queued (§6).
- **Loose end, low priority:** `data_convert.py::darknet2tinyperson()` discards any prediction with area > 30×30px before scoring, which taken literally would zero out the "small"/"reasonable" bins — inconsistent with the paper's real APs50=74.4. Most likely a narrow tool for isolating AP_tiny during development, not the Table II-generating path. Not investigating further unless the hyp-file test doesn't close the gap.

---

## 5. UAVDT — paused, data-source problem

**Baseline** (`hesod/backends/esod/`, `models/cfg/esod/uavdt_yolov5m.yaml`, `data/hyps/hyp.uavdt.yaml`, img 1280): P=0.277/R=0.384/AP50=0.186/AP=0.106/BPR=0.807/Occupy=0.0674.

Size-bin recall was non-monotonic (Medium/Large=52.13%, *worse* than Very Tiny=61.60%). Root-caused via `list_examples.py` + direct visual inspection (not inference): the `car` class specifically craters in the Medium/Large bin (51.16% vs. 92.91% in Small) while `bus` in the same bin scores 99.30%; 90% (18/20) of a random sample of "missed, Medium/Large car" GT boxes had no overlapping prediction of any class at all; recalled-vs-missed samples showed video sequence `M0802` at 0% recalled vs. 55% of the missed sample. Visually inspecting `M0802` frames: the offending GT boxes (~130-190px wide) sit on top of a dense parking lot of many small, individually-parked cars — one oversized box drawn over a cluster, not a real large vehicle. Contrasted against a genuinely large, cleanly-boxed, correctly-recalled car in sequence `M0205` (score=0.872, IoU=0.924), the difference is visually obvious.

`/root/autodl-tmp/UAVDT_processed` is a third-party Kaggle repackaging of UAVDT, not verified against the official release's own annotations — consistent with this kind of box-quality defect. Separately, ESOD's own official `prepare_uavdt()` hardcodes **all** vehicle classes to a single class 0, meaning the paper's real UAVDT protocol is likely single-class, not the 3-class (car/truck/bus) setup currently configured (`nc: 3`) — flagged, not yet acted on.

**Decision: paused until an official-source UAVDT conversion is available; do not draw method conclusions from UAVDT's current numbers.** Re-fetching from the official source (Google Drive links in the ESOD README) was blocked once on a quota error; a Baidu-Pan-style workaround (as used for TinyPerson) has not yet been attempted.

---

## 6. Next steps

1. **TinyPerson: retrain baseline + channel-pooled-concat with `hyp.tinyperson.scratch.yaml`** (§4.2's leading suspect) — `scripts/esod_baseline/run_tinyperson_scratch_hyp.sh`, not yet run.
2. **Box-regression size-weighting ablation** (`--box-loss size_weighted` in `hesod/backends/hesod/utils/loss.py`, targets §1.6's #1 VisDrone gap item) — implemented, not yet run.
3. **UAVDT official-source re-fetch** — try the Baidu Pan route that worked for TinyPerson; re-run `prepare_uavdt()` once raw data is in hand, and resolve the single-class-vs-3-class question at the same time.
4. Once TinyPerson's hyp-file question is resolved, decide whether to also test the roster arms (dual-evidence, channel-pooled) there and on UAVDT — deprioritized behind item 1, since a moving baseline makes roster comparisons meaningless.
5. Lower priority, not scheduled: repeat VisDrone baseline with an additional seed to size natural run-to-run variance against the remaining ~2.8%/1.7% gap; install SAM and regenerate hybrid masks for a controlled Gaussian-vs-hybrid comparison on `VisDrone_v2`.

---

## 7. Known gotchas

- **VisDrone data conversion** — see §1.1. Must go through ESOD's own `prepare_visdrone()`; a generic Ultralytics-style conversion silently skips pixel-masking ignored regions and costs real AP/AP50/calibration.
- **TinyPerson hyp file** — `hyp.tinyperson.yaml` (the name `train.sh`'s default formula expects) does not exist; must explicitly pick `scratch` or `finetune`. Current best guess is `scratch` (§4.2), not yet validated by a rerun.
- **`--top-k` / `--selector-loss` / `--box-loss` only exist in `hesod/backends/hesod/`** (the dev tree), not the pristine `esod/`/`hesod/backends/esod/` mirrors — cross-tree checkpoint loads work fine (architecture is identical for the plain baseline), but roster-arm-specific flags will simply not be recognized if pointed at the wrong tree.
- **`test.py`'s per-class label count (`nt`)** was gated behind `stats[0].any()`, so a validation pass with zero IoU>0.5 hits anywhere displayed `Labels: 0`/`BPR: nan` even when the true label count wasn't zero. Fixed identically in `esod/`, `hesod/backends/esod/`, `hesod/backends/hesod/`. Display-only.
- Environment-compat patches (NumPy 2.x, PyTorch 2.6+ `torch.load`, etc.) are tracked separately in `ESOD-Baseline-Patches.md` — that file, not this one, is the source of truth for "why did X break and how was it fixed."
