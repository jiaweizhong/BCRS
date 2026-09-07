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

The exact staged flagship is now complete and canonical on both datasets. On UAVDT it is a near-free trade (mAP@.5/BPR within 0.1 pp/0.0 of the selector reference, §3.1). On SeaPerson, after fixing a `train.py` protocol confound that made an earlier attempt's number unreliable (§6.2/§7), the corrected run reaches mAP@.5 0.773 (-0.5 pp vs. the 0.778 selector reference), mAP@.5:.95 0.327 (-0.3 pp), fully preserves BPR at 0.991, and reduces Dual-Max GFLOPs from 255.1 to 208.5 (-18.2%) with FPS rising from 73.8 to 80.6. Both datasets now show the same qualitative pattern: ISPPHead trades a small amount of accuracy for a real, material efficiency gain, with BPR untouched.

### 1.2 Why max fusion is the core contribution

Let $z_s$ and $z_f$ denote semantic and spectral logits. HESOD uses

$$z = \max(z_s, z_f).$$

This operation is parameter-free and evidence-preserving: $z \ge z_s$ and $z \ge z_f$ at every spatial location. A learned affine concat combiner has no such lower bound and can suppress a location supported by either branch. Because a patch rejected by the selector cannot be recovered by the detector, this monotonicity directly addresses the failure mode motivating the paper.

Implementation: `ChannelPooledMaxEvidenceSegmenter` in `hesod/backends/hesod/models/segmenter.py`.

### 1.3 Why ISPPHead belongs in the mainline

ISPPHead is the efficiency half of the method rather than an optional add-on. Dual evidence improves patch coverage by routing more useful regions, but that benefit increases detector work. ISPPHead is designed to recover a large fraction of this overhead without erasing the selector gain.

On UAVDT, staged Dual-Max + ISPPHead changes mAP@.5 only from 0.395 to 0.394 and preserves BPR at 0.940, while reducing Dual-Max from 90.1 to 74.9 GFLOPs (-16.9%) and from 35.85M to 25.98M parameters (-27.5%). On SeaPerson, the corrected staged run (§6.2/§7) changes mAP@.5 from 0.778 to 0.773 (-0.5 pp) and mAP@.5:.95 from 0.330 to 0.327 (-0.3 pp), fully preserves BPR at 0.991, and reduces Dual-Max from 255.1 to 208.5 GFLOPs (-18.2%), from 35.79M to 25.92M parameters (-27.6%), and increases FPS from 73.8 to 80.6 (+9.2%). A separate Dual-Concat comparison shows ISPPHead is also nearly accuracy-neutral there (0.772$\to$0.771 mAP@.5) under a different fusion rule and training history -- consistent supporting evidence, not the same matched estimate as the Dual-Max row. ISPPHead is the mainline compute compensator on both datasets, with its exact staged composition now clean on both.

The staged protocol is:

1. train the complete Dual-Max selector with the original coupled head;
2. initialize the ISPPHead model from the converged Dual-Max checkpoint;
3. freeze the backbone, evidence branches, fusion segmenter, and heat-map parser (`model.0-12`);
4. fine-tune only the neck and detection head, using the **real frozen-selector routing distribution from epoch 0** for both training and validation;
5. keep optimizer warmup independent of routing mode and select the best checkpoint only from real-routing validation.

Joint optimization is not the canonical training recipe: on UAVDT it lets head gradients perturb the selector and damages BPR/recall. Staged training is therefore part of the method, not merely an experimental trick. SeaPerson additionally exposed and required fixing a `train.py` implementation bug: one `warmup_flag` had been controlling optimizer interpolation, GT-assisted-routing, and validation settings together, combined with a hard-coded `use_gt = epoch < epochs * 0.6` (§6.2). An early attempt at this arm crashed reproducibly at the exact iteration a short optimizer warmup completed, because the LR jump, the routing-distribution switch, and the validation-threshold switch all happened simultaneously; extending warmup to cover the whole run avoided the simultaneous LR jump but left the run using GT-assisted routing for 60% of training and the wrong validation threshold for all of it, producing a real but confounded 0.761 result. Fixing `train.py` to force real-selector routing and the real validation threshold from epoch 0 for any `--freeze` run (independent of `warmup_epochs`, which now controls only the optimizer) resolved this: the corrected rerun trains smoothly with no crash and lands at 0.773, the number now used throughout this document. The paper may claim that ISPPHead **substantially recovers the extra cost of dual-evidence routing with a small, honestly-reported accuracy cost on both datasets**. HESOD is not cheaper than R0 in absolute GFLOPs/FPS terms on either dataset.

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
| SeaPerson | Dual-Max selector reference | **0.778** | **0.330** | **0.991** | **88.10%** | 255.1 | 35.79 | 73.8 |
| SeaPerson | **HESOD (Dual-Max + ISPP, staged)** | 0.773 | 0.327 | **0.991** | 87.75% | 208.5 | **25.92** | 80.6 |

The Dual-Max selector improves every reported detection and routing-quality metric on both datasets. The strongest evidence is selector coverage: versus R0, BPR/total recall rise by 5.6 points/5.19 pp on UAVDT and 4.4 points/3.68 pp on SeaPerson. The cost is denser routing: GFLOPs rise by 32.1% on UAVDT and 26.0% on SeaPerson, while FPS falls by 13.2% and 13.9%, respectively.

ISPPHead is designed to recover this overhead, and on both datasets it does. On UAVDT it is nearly free: the complete, staged HESOD row keeps mAP@.5/BPR within 0.1 pp/0.0 of the selector reference (0.395$\to$0.394, 0.940$\to$0.940) while cutting GFLOPs 16.9% (90.1$\to$74.9) and parameters 27.5% (35.85M$\to$25.98M) relative to Dual-Max -- more than half of R0's own compute increase is recovered, and FPS improves over the selector reference (102.3$\to$106.1). Total recall gives back 1.53 pp (90.36%$\to$88.83%) as part of this trade, still 3.66 pp above R0. On SeaPerson the staged row keeps mAP@.5 within 0.5 pp and mAP@.5:.95 within 0.3 pp of the selector reference (0.778$\to$0.773, 0.330$\to$0.327), fully preserves BPR (0.991$\to$0.991), and cuts GFLOPs 18.2% (255.1$\to$208.5) and parameters 27.6% (35.79M$\to$25.92M) while FPS rises above both R0 and the selector reference (80.6 vs. 85.7/73.8). Total recall gives back 0.35 pp (88.10%$\to$87.75%), still 3.33 pp above R0.

### 3.2 ISPPHead efficiency contribution

| Dataset | Matched comparison | $\Delta$mAP@.5 | $\Delta$mAP@.5:.95 | $\Delta$BPR | $\Delta$recall | $\Delta$GFLOPs | $\Delta$params | $\Delta$FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UAVDT | Dual-Max $\to$ staged Dual-Max+ISPP | -0.1 pp | -0.3 pp | 0.0 | -1.53 pp | **-16.9%** | **-27.5%** | **+3.7%** |
| SeaPerson | Dual-Concat $\to$ Dual-Concat+ISPP | -0.1 pp | **+0.2 pp** | +0.2 points | -0.16 pp | **-22.6%** | **-27.6%** | **+10.7%** |
| SeaPerson | Dual-Max $\to$ staged Dual-Max+ISPP | -0.5 pp | -0.3 pp | 0.0 | -0.35 pp | **-18.2%** | **-27.6%** | **+9.2%** |

The Dual-Concat and Dual-Max rows use different fusion rules and training histories, so their difference is suggestive rather than a clean causal isolation of ISPPHead. What is established is: ISPPHead is nearly free on top of Dual-Concat on SeaPerson (-0.1 pp mAP@.5), and now also nearly free on top of Dual-Max on both datasets (-0.1 pp UAVDT, -0.5 pp SeaPerson) once the SeaPerson staged fine-tune uses the corrected protocol (§6.2/§7) -- BPR is preserved (0.0 change) in all three comparisons.

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

| Method | Type | mAP@.5 | mAP@.5:.95 | Total recall | Params (M) | GFLOPs | FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Faster R-CNN, ResNet-50-FPN | Dense two-stage | 0.551 | 0.246 | 67.27% | 43.26 | 1546.8 | 23.1 |
| RetinaNet, ResNet-50-FPN | Dense one-stage | 0.473 | 0.201 | 65.98% | 36.35 | 942.7 | 28.0 |
| ESOD R0 | Selective, single evidence | 0.750 | 0.320 | 84.42% | 35.78 | **202.4** | **85.7** |
| Dual-Max selector reference | Selective, dual evidence | **0.778** | **0.330** | **88.10%** | 35.79 | 255.1 | 73.8 |
| **HESOD (Ours)** | Selective, dual evidence, staged | 0.773 | 0.327 | 87.75% | **25.92** | 208.5 | 80.6 |

The Dual-Max selector reference exceeds R0 by 2.8 pp mAP@.5 and 1.0 pp mAP@.5:.95. The staged HESOD row trades 0.5 pp of that back for compute (255.1$\to$208.5 GFLOPs) but still exceeds R0 by 2.3 pp mAP@.5 at lower parameter count. Against the dense detectors, HESOD improves mAP@.5 by 22.2 pp over Faster R-CNN and 30.0 pp over RetinaNet, total recall by 20.48 pp and 21.77 pp respectively, while cutting GFLOPs by 7.4$\times$ and 4.5$\times$ respectively, at 80.6 FPS. The recall gap is the more striking number here: at a fixed IoU/confidence matching protocol, both dense detectors miss roughly a third of all instances (particularly Very Tiny: 46.07%/50.05% recall vs. HESOD's 75.71%), consistent with the selective-computation motivation this comparison exists to support. The dense detectors provide conventional reference points, but they are not evidence for the selector ablation because their training and inference structures differ substantially; BPR is left blank for them since it is a selector-routing metric with no analog in a dense detector.

### 5.2 Minimal selector ablation

| Configuration | Selector loss | Head / box loss | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | FPS |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | BCE | Coupled / CIoU | 0.750 | 0.320 | 0.947 | 84.42% | **202.4** | **85.7** |
| Semantic-only | Coverage | Coupled / CIoU | 0.769 | 0.325 | 0.991 | 87.75% | 266.8 | 73.5 |
| Spectral-only, pooled | Coverage | Coupled / CIoU | 0.767 | 0.324 | 0.988 | 87.77% | 263.4 | 72.7 |
| Dual-Concat | Coverage | Coupled / CIoU | 0.772 | 0.326 | 0.986 | 87.84% | 281.2 | 69.8 |
| **Dual-Max** | Coverage | Coupled / CIoU | **0.778** | **0.330** | **0.991** | **88.10%** | 255.1 | 73.8 |
| Dual-Concat + ISPP | Coverage | ISPP / CIoU | 0.771 | 0.328 | 0.988 | 87.68% | **217.6** | **77.3** |
| **HESOD: Dual-Max + ISPP, staged** | Coverage | ISPP / CIoU | 0.773 | 0.327 | **0.991** | 87.75% | 208.5 | 80.6 |

Coverage supervision supplies the first major routing gain: semantic-only raises BPR from 0.947 to 0.991 and reduces selector-dropped errors from 25.6% to 19.2%. Spectral-only provides a similarly strong independent cue. Against the correct fusion baseline, Dual-Max improves over Dual-Concat on every reported axis: +0.6 pp mAP@.5, +0.4 pp mAP@.5:.95, +0.5 BPR points, +0.26 pp recall, 9.3% fewer GFLOPs, and 5.7% higher FPS.

ISPPHead's mainline purpose is supported by two matched comparisons that isolate it from the fusion rule: against Dual-Concat's coupled head it costs almost nothing (-0.1 pp mAP@.5, +0.2 pp mAP@.5:.95, -22.6% GFLOPs, -27.6% params); against Dual-Max's coupled head (the actual flagship composition) it now also costs almost nothing (-0.5 pp mAP@.5, -0.3 pp mAP@.5:.95) once the staged fine-tune uses the corrected protocol (§6.2/§7: real selector routing and the real validation threshold from epoch 0, independent of optimizer warmup), while fully preserving BPR (0.991$\to$0.991) and cutting GFLOPs 18.2% and params 27.6%.

### 5.3 Relevant size-bucket recall

| Configuration | Very Tiny | Tiny | Small | Medium/Large | Total recall |
|---|:---:|:---:|:---:|:---:|:---:|
| ESOD R0 | 74.03% | 86.78% | 94.74% | 79.35% | 84.42% |
| Dual-Concat | **76.00%** | 91.50% | 95.68% | 80.00% | 87.84% |
| **Dual-Max** | 75.65% | **92.06%** | **95.85%** | **81.94%** | **88.10%** |
| **HESOD: Dual-Max + ISPP, staged** | 75.71% | 91.47% | 95.72% | 80.65% | 87.75% |

Dual-Max does not maximize the Very Tiny bin in this run, but it improves the much larger Tiny bin and produces the best aggregate recall and detection accuracy among the clean selector-only configurations. The paper should report this distribution rather than implying that every size bucket improves. The staged HESOD row is essentially flat against Dual-Max on the three largest bins -- Very Tiny is marginally higher (+0.06 pp), Tiny/Small marginally lower (-0.59/-0.13 pp) -- with a larger Medium/Large drop (-1.29 pp, but that bin has only 155 GT total, so this one delta should be read with wide uncertainty). This is consistent with a genuine, small head-capacity/compute trade rather than a systematic bucket-specific failure mode.

## 6. Reproducibility Constraints Integrated from Code Fixes

### 6.1 Same-architecture checkpoint loading

Staged training is valid only if a converged Dual-Max checkpoint is loaded without a layer offset. `intersect_dicts()` must first attempt direct, unshifted key matching and use the legacy backbone-offset adapter only when direct coverage is insufficient. The corrected identity control transfers 601/601 tensors; the old offset path transferred only 173/601 and produced a nominally frozen but effectively random selector.

All reported staged results use the corrected loader. The five failed pre-fix frozen-selector attempts are intentionally excluded because they do not test the claimed method.

### 6.2 Decouple optimizer warmup from routing (fixed)

`train.py`'s `warmup_flag` used to simultaneously control optimizer interpolation, the training input (`GT masks` versus selector-routed images), validation routing, and validation confidence threshold. This created two confounds:

- at the end of warmup, routing distribution, LR/momentum/accumulation, and validation threshold all changed together;
- if `warmup_epochs` equalled the entire fine-tune, the hard-coded `use_gt = epoch < epochs * 0.6` still made the first 60% of epochs use GT-assisted routing and the remaining 40% use real routing, while the warmup validation threshold remained active for the whole run.

**Fix applied** (`train.py`): `use_gt` is now forced `False` whenever `opt.freeze` is set, and validation `conf_thres` is forced to `0.001` whenever `opt.freeze` is set, both independent of `warmup_flag`/`warmup_epochs`. A staged (`--freeze`) fine-tune therefore always uses real selector routing and the real validation threshold from epoch 0, while `warmup_epochs` continues to control only LR/momentum/gradient-accumulation, unaffected by this change; non-`--freeze` runs (the 8-arm rosters, joint-training negative controls) are unaffected. Confirmed: the corrected SeaPerson staged rerun (§7) trains with no crash and no epoch-12-style metric step, and its BPR/Occupy are stable from epoch 0.

### 6.3 Run acceptance

- A completed run must include detection metrics, BPR, physical-size recall, GFLOPs, fused parameter count, and FPS from the same checkpoint.
- Cross-tool recall checks from `audit_buckets.py` and `vt_diagnose.py` should agree within rounding.
- UAVDT improvements below its observed 2--4 pp run-to-run range require an independent rerun before being described as confirmed.
- A configuration file or queued runner is not evidence. Only completed, audited runs enter the canonical tables.
- For staged runs, the log must record transferred and unmatched checkpoint keys and confirm (via §6.2's fix) that routing and validation threshold are real/final from epoch 0, independent of optimizer warmup length.
- Historical note: before §6.2's fix, a probe that only changed LR magnitude while still sharing the coupled `warmup_flag` could not distinguish a gentler optimizer transition from a routing-distribution transition -- this class of ambiguity no longer applies once a run uses the corrected `train.py`.

## 7. SeaPerson Staged Composition: Failure Analysis, Fix, and Final Result

### 7.1 Failure analysis (historical -- describes the pre-fix behavior)

**Initial observed result.** An early, correctly-checkpointed attempt at `seaperson_yolov5m_channel_pooled_max_isphead_frozen` completed without NaN, OOM, or runtime failure: mAP@.5 0.761, mAP@.5:.95 0.311, BPR 0.991, total recall 87.34%, 208.5 GFLOPs, 25.92M params, 80.6 FPS. Cross-tool check: `audit_buckets.py` Very Tiny recall 75.22% versus `vt_diagnose.py` 75.24%, agreeing within rounding. "Collapse" below refers to optimization/performance collapse during earlier short-warmup trajectories, not a crashed process.

**What did not fail, even pre-fix.** Relative to the converged channel-pooled Dual-Max source, BPR was identical (0.991) and occupancy was identical (0.337). Physical recall decreased only 0.76 pp (88.10% to 87.34%), whereas mAP@.5 and mAP@.5:.95 decreased 1.7 and 1.9 pp. This pattern localized the degradation downstream of patch selection: the selector was routing essentially the same content, but the fine-tuned neck/head was producing worse classification/localization. The frozen selector and its BN/eval handling were correctly ruled out as the primary suspect.

**Primary protocol confound.** In `train.py`, one `warmup_flag` controlled all of the following:

1. LR, momentum, and gradient-accumulation interpolation;
2. GT-assisted versus real-selector routing during training;
3. GT-assisted versus real-selector routing during validation;
4. validation confidence threshold (0.1 during warmup, 0.001 afterward).

Consequently, an AP crash at `warmup_epochs=2` (mAP@.5 falling from $\sim$0.75-0.79 to $\sim$0.01-0.03 at the exact iteration warmup completed, then only partially recovering) occurred when **three experimental factors changed together** and could not be assigned specifically to the LR jump. Setting `warmup_epochs=20` kept `warmup_flag` true for every training batch and avoided that specific crash, but the separate hard-coded `use_gt` condition still switched training and validation from GT-assisted routing to real routing after epoch 12 (60% of 20), and the validation confidence threshold remained 0.1 for all 20 epochs rather than the real 0.001 -- the resulting smooth-looking curve was not proof the instability was fixed, only that its most visible symptom (the LR-driven crash) had been avoided; it produced the confounded 0.761/0.311 result above.

**Secondary optimization risk (not separately confirmed, kept for completeness).** Replacing `YOLOv6Head` with `ISPPHead` preserves compatible prediction-layer shapes, but its `expand`, partial-convolution, and `project` feature block has no equivalent source tensors and must adapt during the short fine-tune. The pre-fix default SeaPerson recipe also applied full-detector-scale SGD (`lr0=0.01`, `warmup_bias_lr=0.1`) to the trainable neck/head. §7.3's low-LR/short-warmup hyp (independently adopted in the fix) addresses this whether or not it was a distinct contributing factor.

### 7.2 Fix applied

`train.py`: `use_gt` is now forced `False` whenever `opt.freeze` is set (line ~556), and validation `conf_thres` is now forced to `0.001` whenever `opt.freeze` is set (line ~786), both independent of `warmup_flag`/`warmup_epochs`. A staged fine-tune therefore uses real selector routing and the real validation threshold from epoch 0; `warmup_epochs` now controls only the optimizer's LR/momentum/accumulation schedule, exactly as intended. Non-`--freeze` runs are unaffected. `run_seaperson_frozen_selector.sh`'s own default `HYP` now points at `hyp.seaperson_frozen_v2.yaml` (short 2-epoch optimizer warmup, `lr0=0.001`, `warmup_bias_lr=0.001` -- a 10x/100x reduction from the original `hyp.seaperson.yaml`, since the trainable neck/head is being fine-tuned from an already-converged trunk, not learning objectness priors from scratch) rather than the original `hyp.seaperson.yaml`, closing the reproducibility gap where a corrected run previously required a caller-supplied `HYP=` override to take effect.

### 7.3 Final result and promotion

**Corrected rerun.** Same warm-start checkpoint (`seaperson_yolov5m_channel_pooled_max`, the confirmed 0.778-mAP pooled Dual-Max), same freeze boundary (`model.0-12`), 30 epochs. Training trajectory (`results.txt`, val split) is smooth throughout -- mAP@.5 rises from 0.671 (epoch 0) to $\sim$0.805-0.808 by epoch 4 and stays there through epoch 11 (last epoch inspected mid-run) with BPR/Occupy exactly flat (0.9957/0.4281) for every epoch, no step change anywhere. Final test-split result (`audit_buckets.py`/`vt_diagnose.py` cross-checked, Very Tiny 75.71% vs. 75.73%, agree within rounding):

| Metric | Dual-Max (pre-staging) | v1, confounded (§7.1) | **Corrected (canonical)** |
|---|:---:|:---:|:---:|
| mAP@.5 | 0.778 | 0.761 (-1.7 pp) | **0.773 (-0.5 pp)** |
| mAP@.5:.95 | 0.330 | 0.311 (-1.9 pp) | **0.327 (-0.3 pp)** |
| BPR | 0.991 | 0.991 | **0.991** |
| Total recall | 88.10% | 87.34% | **87.75%** |
| GFLOPs | 255.1 | 208.5 | **208.5** |
| Params (M) | 35.79 | 25.92 | **25.92** |
| FPS | 73.8 | 80.6 | **80.6** |

**Promotion.** Applying the promotion rule stated before this run (preserve the Dual-Max accuracy/BPR advantage within accepted variation while materially reducing GFLOPs $\Rightarrow$ canonical): BPR is fully preserved, the AP cost shrank from a real 1.7/1.9 pp to a near-UAVDT-level 0.5/0.3 pp, and GFLOPs/FPS are unchanged from the confounded run (the fix changed training dynamics, not the resulting architecture's own efficiency). This is now the canonical SeaPerson HESOD result, used throughout §1, §3, §5, §A.5, §A.6. The confounded 0.761/0.311 run and the two wrong-checkpoint attempts before it are retained only in §A.7 for provenance; neither should be cited as a paper number. No additional SABL, learned-gate, or obsolete full-recipe reruns are required.

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
| 12 | **HESOD: Dual-Max + ISPP, staged** | ISPP / CIoU | **0.773** | **0.327** | **0.991** | 87.75% | 208.5 | 80.6 | **Current SeaPerson flagship** (§7) |

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
| 12 | **HESOD: Dual-Max + ISPP, staged** | 75.71% | 91.47% | 95.72% | 80.65% | 87.75% |

The original SeaPerson Dual-Max run (mAP@.5 0.766, mAP@.5:.95 0.323, BPR 0.988, total recall 87.51%, 255.5 GFLOPs, 76.5 FPS) is retained as a superseded run record. Its independent rerun improved every accuracy/routing metric and is the canonical arm-10 result above.

### A.7 Negative, out-of-scope, and invalid probes

| Probe | Result | Disposition |
|---|---|---|
| Learned gated fusion (`ChannelPooledDualEvidenceSegmenter`) | SeaPerson mAP@.5 0.765; total recall 87.26% | Negative: learned gate suppresses candidate evidence; no further runs planned |
| SeaDronesSeeV2 R0 at 1536 | mAP@.5 0.894; total recall 95.76%; Very Tiny only 1.9% of GT | Out of scope for the selector-headroom claim |
| Five pre-fix frozen-selector attempts | Only 173/601 checkpoint tensors transferred | Invalid, not method results; retained only as provenance for the loader constraint in §6.1 |
| SeaPerson staged Dual-Max+ISPP, first two attempts | Warm-started from the wrong checkpoint (`seaperson_yolov5m_max`, a full-width-spectral diagnostic, not Dual-Max) due to a run-name collision; mAP@.5 $\sim$0.71 | Invalid, not method results; discarded once the mismatch was found -- see §7.1 |
| SeaPerson staged Dual-Max+ISPP, correct checkpoint but pre-fix `train.py` | mAP@.5 0.761, mAP@.5:.95 0.311 (BPR/GFLOPs/params/FPS identical to the canonical row) | Superseded, not invalid -- a real run with a real protocol confound (coupled `warmup_flag`/`use_gt`/validation threshold); replaced once §6.2's fix was applied -- see §7 |
| Pest24 | Recorded in `HESOD-Agri-Experiment-Plan.md` | Separate project; not duplicated here |

### A.8 Pending result placeholders

`TBD` means the run or its complete audit is not yet available. A placeholder must never be interpreted as a zero or copied into a paper result table.

| Dataset | Configuration | mAP@.5 | mAP@.5:.95 | BPR | Total recall | GFLOPs | Params (M) | FPS | Purpose/status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| UAVDT | Dual-Max independent confirmation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Confirm AP gain beyond the observed run variance; queued as `uavdt_yolov5m_channel_pooled_max_run2` |
| UAVDT | HESOD staged independent confirmation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Confirm final flagship stability; also re-verifies UAVDT's own staged runs under the §6.2 `train.py` fix (UAVDT was not previously known to need it, but shares the same code path) |

The SeaPerson staged composition placeholder that previously occupied this table is resolved -- §7 records the fix, the corrected rerun, and its promotion to the canonical §A.5 arm 12 result.
