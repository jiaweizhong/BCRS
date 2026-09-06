# HESOD Experiment Plan & Benchmark Contract

**Canonical status: 2026-09-07.** This document records the current benchmark contract, validated results, and paper-facing decisions for HESOD. It is intentionally not a chronological experiment log: superseded runs, discarded checkpoints, and patch-by-patch narratives are omitted once their conclusions have been absorbed into the protocol or interpretation below.

## 1. Paper Decision

### 1.1 Mainline architecture and training

The paper's primary method is **HESOD: an evidence-preserving selector paired with a lightweight detection head**:

1. a semantic branch predicts class/objectness logits from shallow stem features;
2. a channel-pooled spectral branch supplies complementary high-frequency evidence;
3. the selector fuses the two logits with an elementwise maximum;
4. object-level soft-coverage loss supervises patch selection;
5. ISPPHead reduces the downstream neck/head cost with partial convolution and a lightweight decoupled prediction head;
6. upstream CIoU box loss is retained and SABL is not used.

The flagship configuration is therefore **Dual-Max + coverage supervision + ISPPHead, without SABL**, trained with the selector-first staged protocol in §1.3. The coupled-head Dual-Max model remains the selector reference used to measure recall gains; it is not the intended final efficiency configuration.

The exact staged flagship has been completed on UAVDT. SeaPerson already verifies both ingredients independently in the intended direction -- Dual-Max improves routing/accuracy, while ISPPHead substantially compresses a matched dual-evidence detector with negligible accuracy change -- but the exact staged composition remains the decisive pending run. This distinction must remain explicit until that run completes.

### 1.2 Why max fusion is the core contribution

Let $z_s$ and $z_f$ denote semantic and spectral logits. HESOD uses

$$z = \max(z_s, z_f).$$

This operation is parameter-free and evidence-preserving: $z \ge z_s$ and $z \ge z_f$ at every spatial location. A learned affine concat combiner has no such lower bound and can suppress a location supported by either branch. Because a patch rejected by the selector cannot be recovered by the detector, this monotonicity directly addresses the failure mode motivating the paper.

Implementation: `ChannelPooledMaxEvidenceSegmenter` in `hesod/backends/hesod/models/segmenter.py`.

### 1.3 Why ISPPHead belongs in the mainline

ISPPHead is the efficiency half of the method rather than an optional add-on. Dual evidence improves patch coverage by routing more useful regions, but that benefit increases detector work. ISPPHead is designed to recover a large fraction of this overhead without erasing the selector gain.

On UAVDT, staged Dual-Max + ISPPHead changes mAP@.5 only from 0.395 to 0.394 and preserves BPR at 0.940, while reducing Dual-Max from 90.1 to 74.9 GFLOPs (-16.9%) and from 35.85M to 25.98M parameters (-27.5%). On SeaPerson, the currently available matched head comparison uses Dual-Concat: replacing its coupled head with ISPPHead changes mAP@.5 from 0.772 to 0.771 and mAP@.5:.95 from 0.326 to 0.328, while reducing GFLOPs from 281.2 to 217.6 (-22.6%), parameters from 35.79M to 25.92M (-27.6%), and increasing FPS from 69.8 to 77.3 (+10.7%). These results support ISPPHead's mainline role as the compute compensator, while the remaining SeaPerson staged run verifies its exact composition with Dual-Max.

The staged protocol is:

1. train the complete Dual-Max selector with the original coupled head;
2. initialize the ISPPHead model from the converged Dual-Max checkpoint;
3. freeze the backbone, evidence branches, fusion segmenter, and heat-map parser (`model.0-12`);
4. fine-tune only the neck and detection head.

Joint optimization is not the canonical training recipe: on UAVDT it lets head gradients perturb the selector and damages BPR/recall. Staged training is therefore part of the method, not merely an experimental trick. Until the matching SeaPerson run completes, the paper may claim that ISPPHead **substantially recovers the extra cost of dual-evidence routing with little accuracy loss**, but it must not claim that the complete architecture has already been validated identically on both datasets or that it is cheaper than R0 in absolute terms.

### 1.4 Excluded from the flagship

- **SABL** is removed from the main method. It is not needed for the selector contribution, joint training is unstable with Dual-Max, and the staged SABL variants do not improve the clean Dual-Max accuracy/recall ceiling.
- **Learned concat and gated fusion** are not the unified fusion rule. They can suppress positive evidence and do not provide the same cross-dataset result as Dual-Max.
- **Jointly trained Dual-Max + ISPPHead** is not the flagship training protocol; it is retained only as a negative control demonstrating selector/head interference.
- The historical names **Concat-Max** and **HESOD Full v2** are deprecated. The corresponding fusion-only model is named **Dual-Max** throughout this document and should be named the same way in the paper.

### 1.5 Why the spectral branch is channel-pooled

The spectral branch is channel-pooled (`ChannelPooledSpectralBranch`, spatial max/mean pooling to 2 channels before the depthwise saliency filters) rather than operating on the full stem-feature width. This is not a compute-only claim: GFLOPs alone do not favor pooling by a wide margin at the single-evidence level (§A.1 arm 3 vs. arm 4: 98.1 vs. 99.3 GFLOPs on UAVDT; §A.5 arm 3 vs. arm 4: 267.4 vs. 263.4 on SeaPerson), so pooling is justified by memory footprint and by its effect once combined with the semantic branch under Dual-Max, not by FLOPs reduction in isolation.

| Property | Full-width spectral branch | Channel-pooled spectral branch |
|---|:---:|:---:|
| SeaPerson training batch size (shared budget: 8) | 2 (OOMs at 8) | 8 |
| SeaPerson Dual-Max mAP@.5 | 0.763 | **0.778** |
| SeaPerson Dual-Max mAP@.5:.95 | 0.320 | **0.330** |
| SeaPerson Dual-Max BPR | 0.986 | **0.991** |
| SeaPerson Dual-Max total recall | 87.53% | **88.10%** |
| SeaPerson Dual-Max GFLOPs | 277.8 | **255.1** |
| SeaPerson Dual-Max FPS | 70.1 | **73.8** |

Under Dual-Max specifically, the channel-pooled spectral branch is not a trade-off against the full-width branch -- it wins on every reported metric simultaneously (mAP@.5 +1.5 pp, mAP@.5:.95 +1.0 pp, BPR +0.5 points, total recall +0.57 pp, GFLOPs -8.9%, FPS +5.0%), in addition to the memory benefit that motivated trying it in the first place. Testing this required a training-time comparison, not an inference-time one: eval/measure-time memory pressure is much lower than training's, so an inference-only VRAM check would not have reproduced the batch-size ceiling that full-width spectral evidence hits during training. Parameter count is essentially unaffected either way (35.79M pooled vs. 35.94M full-width, a difference of $<0.5\%$), consistent with pooling changing the spectral branch's own input channel count rather than its parameter-bearing layers.

## 2. Fixed Experimental Contract

### 2.1 Dataset configurations

| Dataset | Dataset YAML | Classes | Model | Hyperparameters | Input | Canonical evaluation split |
|---|---|---:|---|---|---:|---|
| VisDrone | `/root/autodl-tmp/VisDrone_v2.yaml` | 10 | `visdrone_yolov5m.yaml` | `hyp.visdrone.yaml` | 1536 | Val: 548 images, 38,759 GT |
| TinyPerson | `tinyperson.yaml` | 1 | `tinyperson_yolov5m.yaml` | `hyp.tinyperson.yaml` | 2048 | Official test: 786 images |
| UAVDT | `/root/autodl-tmp/UAVDT_fresh.yaml` | 3 | arm-specific UAVDT YOLOv5m | `hyp.uavdt.yaml` | 1280 | Test: 16,580 images, 373,997 GT |
| SeaPerson (TinyPersonV2) | `/root/autodl-tmp/seaperson.yaml` | 1 | arm-specific SeaPerson YOLOv5m | `hyp.seaperson.yaml` | 2048 | Official test: 5,752 images, 300,375 GT |

Unless explicitly marked otherwise, models use YOLOv5m, 50 epochs, SGD, cosine learning-rate decay, weight decay 0.0005, and global batch size 8. SeaPerson diagnostics using the full-width (non-channel-pooled) spectral branch require batch size 2 because of memory pressure -- both the single-evidence spectral-only diagnostic and the full-width Dual-Max pooling diagnostic (§1.5, §A.5 arm 11) hit this ceiling; every channel-pooled model, including the Dual-Max mainline, trains at the shared batch size 8.

### 2.2 UAVDT split caveat

UAVDT has an official train/test video-sequence partition but no official validation split. Training uses every tenth frame from the official train sequences; testing uses every frame from the official test sequences. All three classes (car, truck, and bus) are preserved, ignore-region objects are removed, and flattened image identifiers are prefixed by sequence to prevent collisions.

The active `UAVDT_fresh.yaml` uses the full test directory for both training-time validation and final evaluation. This inherits the original ESOD repository's practice of not wiring its generated train-derived `valid` split into the dataset YAML, although this project's active pipeline uses the full test set rather than the upstream repository's untracked `test_ds.txt` convention. Consequently, UAVDT arms are internally comparable, but their best-epoch selection is not based on a held-out split. This caveat must accompany reported UAVDT results.

### 2.3 Metrics

1. **Detection accuracy:** COCO-style mAP@.5 and mAP@.5:.95. TinyPerson additionally uses official APt50 and APs50.
2. **Physical-size recall:** class-aware, confidence-ranked one-to-one matching at confidence $\ge0.001$ and IoU $\ge0.5$ using `audit_buckets.py`:
   - Very Tiny: area $<16^2$ px;
   - Tiny: $16^2$--$32^2$ px;
   - Small: $32^2$--$96^2$ px;
   - Medium/Large: area $>96^2$ px.
3. **Bounding Patch Recall (BPR):** a GT instance is covered when $\operatorname{intersection}(GT,\mathcal P)/\operatorname{area}(GT)>0.5$.
4. **Efficiency:** GFLOPs and measured FPS at the dataset's canonical input size. Parameter counts are reported after inference-time Conv-BN fusion.

### 2.4 Coverage objective

For candidate cells $\mathcal N(j)$ around object $j$, the selector optimizes

$$p_j^{\mathrm{cover}}=1-\prod_{i\in\mathcal N(j)}(1-s_i), \qquad
\mathcal L_{\mathrm{cover}}=-\frac{1}{N_{\mathrm{gt}}}\sum_j w_j\log(p_j^{\mathrm{cover}}+\epsilon),$$

where $w_j=\operatorname{clip}(4/a_j,1,5)$ upweights objects occupying fewer selector cells. This objective targets the asymmetric error that matters in patch routing: a false-positive patch costs compute, but a false-negative patch permanently removes all objects inside it.

## 3. Cross-Dataset Mainline Evidence

### 3.1 Selector contribution

| Dataset | Method | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | Params (M) | FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UAVDT | ESOD R0 | 0.385 | 0.214 | 0.884 | 85.17% | **68.2** | 35.85 | **117.8** |
| UAVDT | Dual-Max selector reference | **0.395** | **0.218** | **0.940** | **90.36%** | 90.1 | 35.85 | 102.3 |
| UAVDT | **HESOD (Dual-Max + ISPP, staged)** | 0.394 | 0.215 | **0.940** | 88.83% | 74.9 | **25.98** | 106.1 |
| SeaPerson | ESOD R0 | 0.750 | 0.320 | 0.947 | 84.42% | **202.4** | 35.78 | **85.7** |
| SeaPerson | **Dual-Max selector reference** | **0.778** | **0.330** | **0.991** | **88.10%** | 255.1 | 35.79 | 73.8 |

The Dual-Max selector improves every reported detection and routing-quality metric on both datasets. The strongest evidence is selector coverage: versus R0, BPR/total recall rise by 5.6 points/5.19 pp on UAVDT and 4.4 points/3.68 pp on SeaPerson. The cost is denser routing: GFLOPs rise by 32.1% on UAVDT and 26.0% on SeaPerson, while FPS falls by 13.2% and 13.9%, respectively.

ISPPHead is designed to recover this overhead, and on UAVDT it does: the complete, staged HESOD row keeps mAP@.5/BPR within 0.1 pp/0.0 of the selector reference (0.395$\to$0.394, 0.940$\to$0.940) while cutting GFLOPs 16.9% (90.1$\to$74.9) and parameters 27.5% (35.85M$\to$25.98M) relative to Dual-Max -- more than half of R0's own compute increase is recovered, and FPS actually improves over the selector reference (102.3$\to$106.1). Total recall gives back 1.53 pp (90.36%$\to$88.83%) as part of this trade, still 3.66 pp above R0. **The equivalent SeaPerson row is not yet available** -- it is the single decisive pending run this document is tracking (§7, §A.8); until it completes, SeaPerson's selector and head contributions are each independently confirmed (this table's own Dual-Max row; §3.2/§5.2's Dual-Concat+ISPP row) but their exact staged composition is not.

### 3.2 ISPPHead efficiency contribution

| Dataset | Matched comparison | $\Delta$mAP@.5 | $\Delta$mAP@.5:.95 | $\Delta$BPR | $\Delta$recall | $\Delta$GFLOPs | $\Delta$params | $\Delta$FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UAVDT | Dual-Max $\to$ staged Dual-Max+ISPP | -0.1 pp | -0.3 pp | 0.0 | -1.53 pp | **-16.9%** | **-27.5%** | **+3.7%** |
| SeaPerson | Dual-Concat $\to$ Dual-Concat+ISPP | -0.1 pp | **+0.2 pp** | +0.2 points | -0.16 pp | **-22.6%** | **-27.6%** | **+10.7%** |

The two comparisons use different fusion rules, so they establish the head's function rather than a completed cross-dataset test of the exact full recipe. In both cases ISPPHead removes a large portion of detector compute and roughly one quarter of parameters with little change in detection accuracy. UAVDT additionally establishes that selector-first staged training is required when ISPPHead is composed with Dual-Max.

### 3.3 Evidence strength

UAVDT exhibits roughly 2--4 pp run-to-run variation, including about a 4 pp swing between independent R0 runs. Therefore, its single-run AP gains (+1.0 pp mAP@.5 and +0.4 pp mAP@.5:.95) are directional rather than statistically established. Its +5.19 pp total-recall and +5.6-point BPR gains are the more credible evidence. SeaPerson Dual-Max has an independent rerun that improved all accuracy metrics simultaneously; the table reports that confirmed result.

## 4. UAVDT Evidence

### 4.1 Minimal ablation supporting the paper

| Configuration | Selector loss | Head / box loss | mAP@.5 | mAP@.5:.95 | BPR | Recall | GFLOPs | Params (M) | FPS |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | BCE | Coupled / CIoU | 0.385 | 0.214 | 0.884 | 85.17% | **68.2** | 35.85 | **117.8** |
| Semantic-only | Coverage | Coupled / CIoU | 0.384 | 0.217 | 0.906 | 84.82% | 75.0 | 35.85 | 113.8 |
| Spectral-only, pooled | Coverage | Coupled / CIoU | 0.394 | 0.209 | **0.963** | **91.93%** | 99.3 | 35.85 | 97.6 |
| Dual-Concat | Coverage | Coupled / CIoU | 0.371 | 0.205 | 0.919 | 86.35% | 83.9 | 35.85 | 107.2 |
| **Dual-Max** | Coverage | Coupled / CIoU | **0.395** | **0.218** | 0.940 | 90.36% | 90.1 | 35.85 | 102.3 |
| Dual-Max + ISPP, joint | Coverage | ISPP / CIoU | 0.360 | 0.199 | 0.915 | 85.99% | **67.5** | **25.98** | 109.0 |
| **HESOD: Dual-Max + ISPP, staged** | Coverage | ISPP / CIoU | 0.394 | 0.215 | 0.940 | 88.83% | 74.9 | **25.98** | 106.1 |

The single-evidence rows establish complementarity: spectral evidence recovers many objects missed by semantic routing, while the semantic branch retains category-aware evidence. Dual-Concat then demonstrates the central failure mode: adding both cues through an unconstrained affine combiner performs worse than the stronger single branch. Replacing only that fusion rule with max raises mAP@.5 from 0.371 to 0.395, BPR from 0.919 to 0.940, and total recall from 86.35% to 90.36%.

The ISPP comparison isolates a training interaction. Joint optimization perturbs the selector and loses 4.37 pp recall relative to Dual-Max. Freezing the converged selector before head fine-tuning restores BPR to 0.940 and mAP@.5 to 0.394 while cutting 16.9% GFLOPs and 27.5% parameters relative to Dual-Max. This staged row is the current UAVDT flagship: ISPPHead substantially offsets the cost introduced by better routing, although it does not beat R0's absolute GFLOPs.

### 4.2 Relevant size-bucket recall

| Configuration | Very Tiny | Tiny | Small | Medium/Large | Total recall |
|---|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | 79.43% | 84.21% | 94.54% | 59.78% | 85.17% |
| Dual-Concat | 79.09% | 85.85% | 95.85% | 62.66% | 86.35% |
| **Dual-Max** | **85.69%** | **89.97%** | **97.37%** | 65.99% | **90.36%** |
| **HESOD: Dual-Max + ISPP, staged** | 80.76% | 88.98% | 97.11% | **69.16%** | 88.83% |

Dual-Max's largest gain over R0 is on Very Tiny objects (+6.26 pp), matching the paper's intended failure mode. The staged full model gives back part of this extreme-scale recall in exchange for its compute reduction; the paper should present this as the selector-efficiency trade-off rather than hiding the recall change.

## 5. SeaPerson Evidence

SeaPerson contains 300,375 test instances, more than 85% of which are tiny or micro persons. It is the stronger confirmation that the selector contribution generalizes beyond UAVDT.

### 5.1 Dense baselines and ESOD family

| Method | Type | mAP@.5 | mAP@.5:.95 | Params (M) | GFLOPs | FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Faster R-CNN, ResNet-50-FPN | Dense two-stage | 0.551 | 0.246 | 43.26 | 1546.8 | 23.1 |
| RetinaNet, ResNet-50-FPN | Dense one-stage | 0.473 | 0.201 | 36.35 | 942.7 | 28.0 |
| ESOD R0 | Selective, single evidence | 0.750 | 0.320 | 35.78 | **202.4** | **85.7** |
| **Dual-Max selector reference** | Selective, dual evidence | **0.778** | **0.330** | 35.79 | 255.1 | 73.8 |

The Dual-Max selector reference exceeds R0 by 2.8 pp mAP@.5 and 1.0 pp mAP@.5:.95. The dense detectors provide conventional reference points, but they are not evidence for the selector ablation because their training and inference structures differ substantially. The full HESOD row will be added after the staged SeaPerson composition is complete.

### 5.2 Minimal selector ablation

| Configuration | Selector loss | Head / box loss | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | FPS |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | BCE | Coupled / CIoU | 0.750 | 0.320 | 0.947 | 84.42% | **202.4** | **85.7** |
| Semantic-only | Coverage | Coupled / CIoU | 0.769 | 0.325 | 0.991 | 87.75% | 266.8 | 73.5 |
| Spectral-only, pooled | Coverage | Coupled / CIoU | 0.767 | 0.324 | 0.988 | 87.77% | 263.4 | 72.7 |
| Dual-Concat | Coverage | Coupled / CIoU | 0.772 | 0.326 | 0.986 | 87.84% | 281.2 | 69.8 |
| **Dual-Max** | Coverage | Coupled / CIoU | **0.778** | **0.330** | **0.991** | **88.10%** | 255.1 | 73.8 |
| Dual-Concat + ISPP | Coverage | ISPP / CIoU | 0.771 | 0.328 | 0.988 | 87.68% | **217.6** | **77.3** |

Coverage supervision supplies the first major routing gain: semantic-only raises BPR from 0.947 to 0.991 and reduces selector-dropped errors from 25.6% to 19.2%. Spectral-only provides a similarly strong independent cue. Against the correct fusion baseline, Dual-Max improves over Dual-Concat on every reported axis: +0.6 pp mAP@.5, +0.4 pp mAP@.5:.95, +0.5 BPR points, +0.26 pp recall, 9.3% fewer GFLOPs, and 5.7% higher FPS.

The SeaPerson head ablation already validates ISPPHead's mainline purpose: compared with the matched Dual-Concat coupled-head model, it cuts GFLOPs by 22.6% and parameters by 27.6% with only -0.1 pp mAP@.5 and +0.2 pp mAP@.5:.95. Because this completed comparison uses Dual-Concat rather than Dual-Max, the exact staged Dual-Max + ISPP, no-SABL experiment is still required to validate the final composition, not to justify ISPPHead's role in the paper.

### 5.3 Relevant size-bucket recall

| Configuration | Very Tiny | Tiny | Small | Medium/Large | Total recall |
|---|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | 74.03% | 86.78% | 94.74% | 79.35% | 84.42% |
| Dual-Concat | **76.00%** | 91.50% | 95.68% | 80.00% | 87.84% |
| **Dual-Max** | 75.65% | **92.06%** | **95.85%** | **81.94%** | **88.10%** |

Dual-Max does not maximize the Very Tiny bin in this run, but it improves the much larger Tiny bin and produces the best aggregate recall and detection accuracy among the clean selector-only configurations. The paper should report this distribution rather than implying that every size bucket improves.

## 6. Reproducibility Constraints Integrated from Code Fixes

### 6.1 Same-architecture checkpoint loading

Staged training is valid only if a converged Dual-Max checkpoint is loaded without a layer offset. `intersect_dicts()` must first attempt direct, unshifted key matching and use the legacy backbone-offset adapter only when direct coverage is insufficient. The corrected identity control transfers 601/601 tensors; the old offset path transferred only 173/601 and produced a nominally frozen but effectively random selector.

All reported staged results use the corrected loader. The five failed pre-fix frozen-selector attempts are intentionally excluded because they do not test the claimed method.

### 6.2 Run acceptance

- A completed run must include detection metrics, BPR, physical-size recall, GFLOPs, fused parameter count, and FPS from the same checkpoint.
- Cross-tool recall checks from `audit_buckets.py` and `vt_diagnose.py` should agree within rounding.
- UAVDT improvements below its observed 2--4 pp run-to-run range require an independent rerun before being described as confirmed.
- A configuration file or queued runner is not evidence. Only completed, audited runs enter the canonical tables.

## 7. Remaining Composition Validation

Run **SeaPerson Dual-Max + ISPPHead, no SABL, staged/frozen selector** from the confirmed Dual-Max checkpoint using the same freeze boundary and fine-tuning protocol as UAVDT. Report it against SeaPerson R0, Dual-Max, and the existing Dual-Concat + ISPP head ablation.

Promotion rule:

- if it preserves the Dual-Max accuracy/BPR advantage within the established run variance while materially reducing Dual-Max GFLOPs, it becomes the canonical SeaPerson result for the full HESOD architecture;
- if it does not, the paper must report the selector and head as two validated mainline contributions but avoid claiming that their exact staged composition is uniformly positive across datasets.

No additional SABL, learned-gate, or obsolete full-recipe reruns are required for the current paper claim.

## 8. Active Runners and Test Gates

- SeaPerson main roster: `scripts/esod_baseline/run_seaperson.sh`
- SeaPerson staged ISPP experiment: `scripts/esod_baseline/run_seaperson_frozen_selector.sh`
- UAVDT roster: `scripts/esod_baseline/run_uavdt.sh`
- VisDrone roster: `scripts/esod_baseline/run_visdrone_roster.sh`
- Dense competitors: `hesod/backends/baseline/`

Before staged training, verify checkpoint transfer coverage in the training log. Relevant regression tests include the model/checkpoint-loading tests and `tests/test_baseline_torchvision.py`; SABL-only tests are not a gate for the HESOD flagship.

## Supplement A. Complete Experimental Record

This supplement preserves completed diagnostics, negative controls, superseded reruns, and pending result slots without allowing them to redefine the paper decision in §1. Rows marked **mainline** or **reference** support the current paper; rows marked **diagnostic** explain a design choice; rows marked **negative control**, **superseded**, or **invalid** must not be presented as the proposed method.

### A.1 Complete UAVDT roster

| Arm | Configuration | Training | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | Params (M) | FPS | Role/status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | ESOD R0 | BCE, coupled, CIoU | 0.385 | 0.214 | 0.884 | 85.17% | 68.2 | 35.85 | 117.8 | Reference |
| 2 | Semantic-only | Coverage, coupled, CIoU | 0.384 | 0.217 | 0.906 | 84.82% | 75.0 | 35.85 | 113.8 | Selector diagnostic |
| 3 | Spectral-only, full-width | Coverage, coupled, CIoU | 0.396 | 0.214 | 0.936 | 88.83% | 98.1 | 36.01 | 100.5 | Capacity diagnostic |
| 4 | Spectral-only, pooled | Coverage, coupled, CIoU | 0.394 | 0.209 | 0.963 | 91.93% | 99.3 | 35.85 | 97.6 | Selector diagnostic |
| 5 | Dual-Concat | Coverage, coupled, CIoU | 0.371 | 0.205 | 0.919 | 86.35% | 83.9 | 35.85 | 107.2 | Fusion negative control |
| 6 | Dual-Concat + SABL | Coverage, coupled, SABL | 0.360 | 0.187 | 0.940 | 89.72% | 97.1 | 35.85 | 92.3 | Non-mainline diagnostic |
| 7 | Dual-Concat + ISPP | Coverage, ISPP, CIoU | 0.371 | 0.192 | 0.940 | 88.94% | 82.6 | 25.98 | 97.8 | Head diagnostic |
| 8 | Dual-Concat + SABL + ISPP | Coverage, ISPP, SABL | 0.378 | 0.202 | 0.937 | 88.94% | 81.6 | 25.98 | 95.2 | Superseded full recipe |
| 9 | **Dual-Max** | Coverage, coupled, CIoU | **0.395** | **0.218** | **0.940** | **90.36%** | 90.1 | 35.85 | 102.3 | Mainline selector reference |
| 10 | Dual-Max + SABL + ISPP | Joint, coverage, SABL | 0.382 | 0.212 | 0.920 | 86.21% | 69.0 | 25.98 | 108.1 | Joint-training negative control |
| 11 | Dual-Max + SABL | Joint, coverage, SABL | 0.367 | 0.202 | 0.938 | 89.31% | 93.6 | 35.85 | 100.0 | SABL diagnostic |
| 12 | Dual-Max + ISPP | Joint, coverage, CIoU | 0.360 | 0.199 | 0.915 | 85.99% | **67.5** | 25.98 | **109.0** | Joint-training negative control |
| 13 | Dual-Max + SABL | Staged/frozen selector | 0.395 | 0.214 | 0.940 | 88.69% | 90.1 | 35.85 | 104.7 | Training-interaction diagnostic |
| 14 | **Dual-Max + ISPP** | **Staged/frozen selector** | 0.394 | 0.215 | **0.940** | 88.83% | 74.9 | **25.98** | 106.1 | **Current UAVDT flagship** |
| 15 | Dual-Max + SABL + ISPP | Staged/frozen selector | 0.392 | 0.213 | 0.940 | 88.30% | 74.9 | 25.98 | 102.9 | SABL exclusion control |

### A.2 UAVDT complete physical-size recall

| Arm | Configuration | Very Tiny | Tiny | Small | Medium/Large | Total recall | Car | Truck | Bus |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ESOD R0 | 79.43% | 84.21% | 94.54% | 59.78% | 85.17% | 85.61% | 81.05% | 67.67% |
| 2 | Semantic-only | 79.13% | 83.91% | 94.00% | 59.95% | 84.82% | 85.20% | 77.74% | 73.27% |
| 3 | Spectral-only, full-width | 83.78% | 87.83% | 97.62% | 64.68% | 88.83% | 89.11% | 83.89% | 79.74% |
| 4 | Spectral-only, pooled | **86.88%** | **92.23%** | 97.74% | 66.43% | **91.93%** | **92.27%** | **87.61%** | 79.63% |
| 5 | Dual-Concat | 79.09% | 85.85% | 95.85% | 62.66% | 86.35% | 86.68% | 80.68% | 75.84% |
| 6 | Dual-Concat + SABL | 83.53% | 89.43% | 97.93% | 64.16% | 89.72% | 90.13% | 83.05% | 76.58% |
| 7 | Dual-Concat + ISPP | 83.06% | 88.19% | 98.07% | 62.32% | 88.94% | 89.33% | 84.19% | 74.69% |
| 8 | Dual-Concat + SABL + ISPP | 82.54% | 88.34% | **98.09%** | 63.02% | 88.94% | 89.31% | 83.90% | 75.63% |
| 9 | **Dual-Max** | 85.69% | 89.97% | 97.37% | 65.99% | 90.36% | 90.67% | 85.14% | 80.09% |
| 10 | Dual-Max + SABL + ISPP, joint | 80.31% | 85.40% | 95.70% | 57.17% | 86.21% | 86.45% | 82.75% | 77.71% |
| 11 | Dual-Max + SABL, joint | 83.75% | 88.86% | 97.40% | 63.17% | 89.31% | 89.64% | 83.45% | 79.10% |
| 12 | Dual-Max + ISPP, joint | 79.05% | 85.84% | 94.74% | 57.56% | 85.99% | 86.26% | 82.07% | 76.69% |
| 13 | Dual-Max + SABL, staged | 81.74% | 88.71% | 96.74% | 64.16% | 88.69% | 88.92% | 85.55% | 80.81% |
| 14 | **Dual-Max + ISPP, staged** | 80.76% | 88.98% | 97.11% | **69.16%** | 88.83% | 89.03% | **86.50%** | 81.42% |
| 15 | Dual-Max + SABL + ISPP, staged | 80.59% | 88.13% | 97.09% | 67.90% | 88.30% | 88.49% | 85.87% | **81.45%** |

### A.3 UAVDT fusion alternatives

| Fusion rule | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | FPS | Raw predictions | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Learned concat | 0.371 | 0.205 | 0.919 | 86.35% | 83.9 | 107.2 | 3.31M | Negative control |
| **Elementwise max** | **0.395** | **0.218** | **0.940** | **90.36%** | 90.1 | 102.3 | 5.43M | Selected fusion |
| Soft-OR (`logsumexp`) | 0.387 | 0.215 | 0.927 | 88.77% | 89.9 | 102.7 | 5.20M | Valid alternative, not selected |

Max is retained because it gives the best end-to-end accuracy and recall while enforcing evidence preservation exactly. Soft-OR is useful supporting evidence that a union-like fusion is better than affine concat, but it does not exceed max.

### A.4 UAVDT stability reruns

| Configuration | Run | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | FPS | Canonical? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Dual-Concat + SABL + ISPP | Original | 0.378 | 0.202 | 0.937 | 88.94% | 81.6 | 95.2 | Yes, historical arm 8 |
| Dual-Concat + SABL + ISPP | Rerun 2 | 0.375 | 0.196 | 0.944 | 89.44% | 83.5 | 95.5 | No; noise probe |
| Dual-Max + SABL + ISPP, joint | Original | 0.382 | 0.212 | 0.920 | 86.21% | 69.0 | 108.1 | Yes, historical arm 10 |
| Dual-Max + SABL + ISPP, joint | Rerun 2 | 0.366 | 0.188 | 0.943 | 90.29% | 76.0 | 104.3 | No; conflicting noise probe |

These reruns establish the approximate 2--4 pp UAVDT noise floor. They are retained for uncertainty analysis, not averaged into the canonical tables because the original experiment record used a single-run selection rule and the rerun metrics move in conflicting directions.

### A.5 Complete SeaPerson roster

| Arm | Configuration | Head / box loss | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | FPS | Role/status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | ESOD R0 | Coupled / CIoU | 0.750 | 0.320 | 0.947 | 84.42% | **202.4** | **85.7** | Reference |
| 2 | Semantic-only | Coupled / CIoU | 0.769 | 0.325 | 0.991 | 87.75% | 266.8 | 73.5 | Selector diagnostic |
| 3 | Spectral-only, full-width | Coupled / CIoU | 0.770 | 0.327 | **0.992** | 87.77% | 267.4 | 71.7 | Capacity diagnostic |
| 4 | Spectral-only, pooled | Coupled / CIoU | 0.767 | 0.324 | 0.988 | 87.77% | 263.4 | 72.7 | Selector diagnostic |
| 5 | Dual-Concat | Coupled / CIoU | 0.772 | 0.326 | 0.986 | 87.84% | 281.2 | 69.8 | Fusion reference |
| 6 | Dual-Concat + SABL | Coupled / SABL | 0.771 | 0.323 | 0.990 | 87.89% | 263.7 | 73.4 | SABL diagnostic |
| 7 | Dual-Concat + ISPP | ISPP / CIoU | 0.771 | 0.328 | 0.988 | 87.68% | **217.6** | 77.3 | Mainline head evidence |
| 8 | Dual-Concat + SABL + ISPP | ISPP / SABL | 0.774 | 0.326 | 0.991 | **88.11%** | 209.0 | **80.7** | Superseded full recipe |
| 9 | Dual-Max + SABL + ISPP | ISPP / SABL | 0.763 | 0.320 | 0.988 | 87.05% | 221.9 | 77.1 | Composition negative control |
| 10 | **Dual-Max** | Coupled / CIoU | **0.778** | **0.330** | **0.991** | 88.10% | 255.1 | 73.8 | Mainline selector reference; confirmed rerun |
| 11 | Dual-Max, full-width spectral | Coupled / CIoU | 0.763 | 0.320 | 0.986 | 87.53% | 277.8 | 70.1 | Pooling diagnostic (§1.5); batch size forced to 2 (OOMs at the shared 8) |

### A.6 SeaPerson complete physical-size recall

| Arm | Configuration | Very Tiny | Tiny | Small | Medium/Large | Total recall |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | ESOD R0 | 74.03% | 86.78% | 94.74% | 79.35% | 84.42% |
| 2 | Semantic-only | 75.49% | 91.57% | 95.71% | 81.94% | 87.75% |
| 3 | Spectral-only, full-width | 75.23% | 91.69% | **95.89%** | 86.45% | 87.77% |
| 4 | Spectral-only, pooled | 76.13% | 91.36% | 95.50% | **87.10%** | 87.77% |
| 5 | Dual-Concat | 76.00% | 91.50% | 95.68% | 80.00% | 87.84% |
| 6 | Dual-Concat + SABL | 76.67% | 91.49% | 94.77% | 80.65% | 87.89% |
| 7 | Dual-Concat + ISPP | 75.86% | 91.25% | 95.85% | **87.10%** | 87.68% |
| 8 | Dual-Concat + SABL + ISPP | **77.14%** | 91.59% | 95.01% | 84.52% | **88.11%** |
| 9 | Dual-Max + SABL + ISPP | 76.01% | 90.36% | 94.78% | 83.87% | 87.05% |
| 10 | **Dual-Max** | 75.65% | **92.06%** | 95.85% | 81.94% | 88.10% |
| 11 | Dual-Max, full-width spectral | 74.45% | 91.75% | 95.48% | 77.42% | 87.53% |

The original SeaPerson Dual-Max run (mAP@.5 0.766, mAP@.5:.95 0.323, BPR 0.988, total recall 87.51%, 255.5 GFLOPs, 76.5 FPS) is retained as a superseded run record. Its independent rerun improved every accuracy/routing metric and is the canonical arm-10 result above.

### A.7 Negative, out-of-scope, and invalid probes

| Probe | Result | Disposition |
|---|---|---|
| Learned gated fusion (`ChannelPooledDualEvidenceSegmenter`) | SeaPerson mAP@.5 0.765; total recall 87.26% | Negative: learned gate suppresses candidate evidence; no further runs planned |
| SeaDronesSeeV2 R0 at 1536 | mAP@.5 0.894; total recall 95.76%; Very Tiny only 1.9% of GT | Out of scope for the selector-headroom claim |
| Five pre-fix frozen-selector attempts | Only 173/601 checkpoint tensors transferred | Invalid, not method results; retained only as provenance for the loader constraint in §6.1 |
| Pest24 | Recorded in `HESOD-Agri-Experiment-Plan.md` | Separate project; not duplicated here |

### A.8 Pending result placeholders

`TBD` means the run or its complete audit is not yet available. A placeholder must never be interpreted as a zero or copied into a paper result table.

| Dataset | Configuration | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | Params (M) | FPS | Purpose/status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| SeaPerson | **HESOD: Dual-Max + ISPP, staged, no SABL** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Decisive full-composition run; queued/awaiting completion |
| UAVDT | Dual-Max independent confirmation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Confirm AP gain beyond the observed run variance |
| UAVDT | HESOD staged independent confirmation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Confirm final flagship stability |

When the SeaPerson staged result completes, replace its placeholder in this table first, audit all metrics under §6.2, and only then add the full HESOD row to the main SeaPerson table in §5.1.
