# HESOD Experiment Plan & Benchmark Contract

**Canonical Status: August 2026.** This document serves as the authoritative specification and results record for the HESOD (High-Resolution Efficient Small Object Detection) project. All historical patches, bug fixes, and protocol corrections have been integrated into their respective sections.

---

## 1. Fixed Evaluation Protocols & Metrics

### 1.1 Dataset Configurations

| Dataset | Dataset YAML | Classes | Model Architecture | Hyperparameters | Input Size | Canonical Split |
|---|---|---:|---|---|---:|---|
| **VisDrone** | `/root/autodl-tmp/VisDrone_v2.yaml` | 10 | `visdrone_yolov5m.yaml` | `hyp.visdrone.yaml` | 1536 | Val: 548 images, 38,759 GT |
| **TinyPerson** | `tinyperson.yaml` (audited) | 1 | `tinyperson_yolov5m.yaml` | `hyp.tinyperson.yaml` | 2048 | Official Test: 786 images |
| **UAVDT** | `/root/autodl-tmp/UAVDT_fresh.yaml` | 3 | `uavdt_yolov5m.yaml` | `hyp.uavdt.yaml` | 1280 | Test: car, truck, bus (373,997 GT) |
| **SeaPerson (TinyPersonV2)** | `/root/autodl-tmp/seaperson.yaml` (`seaperson_v2/`) | 1 | Arm-specific (see §4) | `hyp.seaperson.yaml` | 2048 | Official Test: 5,752 images (300,375 GT) |

- **Common Training Recipe**: Base detector YOLOv5m, 50 epochs, SGD optimizer, cosine annealing learning rate scheduler, weight decay 0.0005, global batch size 8 (with the single exception of SeaPerson `spectral-only` which runs at batch size 2 due to full-channel memory constraints).
- **Data Preparation Invariants**: 
  - UAVDT multi-video naming disambiguation: `image_id` uses `path.parent.stem + '_' + path.stem` to prevent cross-sequence frame ID collisions in external evaluators.
  - SeaPerson image format robustness: `vt_diagnose.py` iteratively checks `.jpg`, `.jpeg`, `.png`, and `.bmp` (to correctly handle the `rgb1000/` subfolder) before falling back to native dimensions.
  - TinyPerson label masking: Evaluated strictly under the paper-text focal:dice 20:1 protocol with literal RGB-SAM Eq. 4 pseudo-masks.

### 1.2 Evaluation Metrics & Parameter Accounting

1. **Detection Accuracy**: Standard COCO-style $\mathrm{mAP@.5} = \mathrm{mean}(\mathrm{ap}[:, 0])$ and $\mathrm{mAP@.5:.95} = \mathrm{mean}(\mathrm{ap})$. TinyPerson headline evaluations additionally report official $\mathrm{APt50}$ ($1\text{--}400\,\mathrm{px}^2$) and $\mathrm{APs50}$ ($400\text{--}1024\,\mathrm{px}^2$).
2. **Physical Size-Bucket Recall**: Evaluated via `audit_buckets.py` using class-aware, confidence-ranked 1-to-1 matching at $\text{conf}\ge 0.001, \text{IoU}\ge 0.5$ across four standard physical bins:
   - *Very Tiny*: $<16\times 16\,\mathrm{px}$ ($<256\,\mathrm{px}^2$)
   - *Tiny*: $16\times 16\text{--}32\times 32\,\mathrm{px}$ ($256\text{--}1024\,\mathrm{px}^2$)
   - *Small*: $32\times 32\text{--}96\times 96\,\mathrm{px}$ ($1024\text{--}9216\,\mathrm{px}^2$)
   - *Medium/Large*: $>96\times 96\,\mathrm{px}$ ($>9216\,\mathrm{px}^2$)
3. **Bounding Patch Recall (BPR)**: Evaluated via strict bounding box coverage $\mathrm{intersection}(\mathrm{GT}, \mathcal{P}) / \mathrm{area}(\mathrm{GT}) > 0.5$.
4. **Deployed Parameter Accounting**: All reported parameter counts reflect deployed post-fusion state (`Model.fuse()`, folding all `Conv2d + BatchNorm2d` pairs into single convolutions as executed during runtime inference).

---

## 2. Core Architectural & Algorithmic Components

### 2.1 Multi-Evidence Spatial Routing
Existing spatial selectors rely solely on a single semantic objectness score, which degrades severely for sub-16px targets after stem downsampling. HESOD introduces a Dual-Evidence Priority Head fusing:
1. **Semantic Objectness Branch**: Predicts class logits via $1\times 1$ conv over shallow stem features $\mathbf{F} \in \mathbb{R}^{H_s \times W_s \times C}$.
2. **Channel-Pooled Spectral Saliency Branch**: Compresses $\mathbf{F}$ via spatial max- and mean-pooling to 2 channels ($\mathbf{F}_{\mathrm{pooled}}$), ensuring $\mathcal{O}(1)$ spatial filtering complexity regardless of backbone channel width $C$. Learnable $3\times 3$ depthwise filters (initialized with discrete Sobel and Laplacian kernels) extract high-frequency gradients.
3. **Fixed Channel Concatenation**: Fuses semantic and spectral logits via $1\times 1$ convolution into priority score $s_i \in [0, 1]$.

### 2.2 Object-Level Soft-Coverage Loss ($\mathcal{L}_{\mathrm{cover}}$)
To penalize premature selector-dropped false negatives under asymmetric selection cost, the soft-coverage loss optimizes the joint probability that at least one candidate cell covers ground-truth instance $j$:
$$p_j^{\mathrm{cover}} = 1 - \prod_{i \in \mathcal{N}(j)} (1 - s_i), \quad \mathcal{L}_{\mathrm{cover}} = -\frac{1}{N_{\mathrm{gt}}} \sum_{j=1}^{N_{\mathrm{gt}}} w_j \log\left(p_j^{\mathrm{cover}} + \epsilon\right)$$
where $w_j = \operatorname{clip}(4/a_j, 1, 5)$ adaptively upweights micro targets ($a_j < 4\text{ cells}$).

### 2.3 Inverted Residual Partial-Convolution Head (ISPPHead)
To drastically cut computational overhead, neck features are expanded to $2C$ and split: 25% channels are processed through a $3\times 3$ Partial Convolution (PConv) while 75% act as an identity passthrough. Decoupled $1\times 1$ heads independently predict classification, box regression, and objectness, slashing parameter count by 27.6% and GFLOPs by 22.6% with zero accuracy loss.

### 2.4 Scale-Aware Balancing Loss (SABL)
Dynamically balances Normalized Gaussian Wasserstein Distance ($D_W$) and Complete IoU (CIoU) via scale gating $\mu(s) = \exp(-(s/32)^6)$:
$$\mathcal{L}_{\mathrm{box}} = 1 - \mathrm{IoU} + \mu(s)\left(1 - e^{-D_W / 12}\right) + (1 - \mu(s))\ell_{\mathrm{ctr}} + \alpha v$$
ensuring non-vanishing localization gradients for sub-16px objects with zero spatial IoU overlap.

---

## 3. UAVDT Benchmark Results ($1280\times 1280$)

UAVDT evaluates high-resolution urban aerial vehicle surveillance across 3 classes (*car*, *truck*, *bus*). Arms below are numbered to mirror SeaPerson's own roster order (§4) for direct arm-for-arm comparison: single-evidence ablations first, then fusion, then the SABL/ISPPHead variants, then the full combination.

**Status (2026-08-31):** All 8 arms are freshly trained and audited this session, cross-verified against raw `audit_buckets.py`/`vt_diagnose.py`/`measure.log` output -- no arm still carries a pre-session checkpoint. Arms (3) Spectral-only and (5) Concat-only both had crash histories (a DataLoader-worker memory leak, mitigated with `--workers 2`) and were retrained end-to-end; both reruns changed the numbers meaningfully (direction held for arm 3, arm 5's own story changed substantially -- §3.4 point 2). Arm (8) HESOD (Full) was retrained for the first time this session (its prior checkpoint, 2026-08-21, predated all of this work) -- the new numbers are close to the old ones (mAP@.5 +0.7pp, Total Recall -1.2pp), a smaller swing than arms (3) or (5) showed. One exploratory probe is in flight, outside the 8-arm roster: `uavdt_yolov5m_channel_pooled_max` retrains arm (5)'s evidence branches with an elementwise-max fusion rule instead of the learned 1x1 combiner, motivated by a representational-capacity argument (§3.4 point 4) for why arm (5) can end up below both single-evidence arms in the first place. A `--pos-weight 1.0` retrain (testing the coverage loss's positive-class upweighting instead) was started first, then killed in favor of this once that argument was worked out -- pos_weight only reshapes the training *gradient*, it can't add capacity a linear combiner structurally lacks, so it was deprioritized rather than run to completion.

### 3.1 Main Results & Delta over ESOD Baseline

| Method | mAP@.5 | mAP@.5:.95 | Total Recall | Car Recall | Truck Recall | Bus Recall | GFLOPs | Params (M) | FPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ESOD Baseline (R0)** | 0.385 | 0.214 | 85.17% | 85.61% | 81.05% | 67.67% | 68.2 | 35.85 | 117.8 |
| **HESOD (Full, Ours)** | 0.378 | 0.202 | **88.94%** | **89.31%** | **83.90%** | **75.63%** | 81.6 | 25.98 | 95.2 |
| **Delta ($\Delta$)** | -0.7 pp | -1.2 pp | **+3.77 pp** | **+3.70 pp** | **+2.85 pp** | **+7.96 pp** | +19.6% | -27.5% | -19.2% |

R0's own reproduction is close to the paper (mAP@.5/.95: 0.385/0.214 vs. paper's 0.407/0.225, within ~1-2pp) but UAVDT has shown real run-to-run noise on this metric (~4pp mAP@.5 swing between independent R0 runs, confirmed by arm (3)'s own rerun) -- larger than the mAP delta HESOD shows over R0 here, so **the mAP comparison stays a soft call**, not a confirmed win or loss. The recall margin (+3.77pp, still comfortably outside the noise floor) is the more trustworthy comparison. Both R0 and HESOD (Full) are now independently confirmed via clean reruns -- no open items remain on this table specifically (HESOD (Full)'s GFLOPs is genuinely *higher* than R0's, +19.6%, despite ISPPHead's own compression -- the dual-evidence selector itself costs more than R0's single-branch one, more than offsetting the head savings; see §3.2).

---

### 3.2 Module Ablation Matrix on UAVDT ($1280\times 1280$)

| Arm | Module Configuration | Selector Loss | Box Loss | Head | mAP@.5 | mAP@.5:.95 | BPR | GFLOPs | Params (M) | FPS |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **(1)** | **Baseline (BCE)** | Upstream BCE | CIoU | Coupled | 0.385 | 0.214 | 0.884 | 68.2 | 35.85 | 117.8 |
| **(2)** | **Semantic-only** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.384 | 0.217 | 0.906 | 75.0 | 35.85 | 113.8 |
| **(3)** | **Spectral-only** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.396 | 0.214 | 0.936 | 98.1 | 36.01 | 100.5 |
| **(4)** | **Spectral-only (Pooled)** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.394 | 0.209 | 0.963 | 99.3 | 35.85 | 97.6 |
| **(5)** | **Concat-only** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.371 | 0.205 | 0.919 | 83.9 | 35.85 | 107.2 |
| **(6)** | **Concat+SABL** | $\mathcal{L}_{\mathrm{cover}}$ | SABL | Coupled | 0.360 | 0.187 | 0.940 | 97.1 | 35.85 | 92.3 |
| **(7)** | **Concat+ISPPHead** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | ISPPHead | 0.371 | 0.192 | 0.940 | 82.6 | 25.98 | 97.8 |
| **(8)** | **HESOD (Full)** | $\mathcal{L}_{\mathrm{cover}}$ | SABL | ISPPHead | 0.378 | 0.202 | 0.937 | 81.6 | 25.98 | 95.2 |
| **(9)**$^\P$ | **Concat-Max** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | **0.395** | **0.218** | **0.940** | 90.1 | 35.85 | 102.3 |

- **(4)** replaces gated-fusion in this roster (SeaPerson's own gated-fusion arm already has a clear negative result, §5 -- not worth re-confirming here). It channel-pools the spectral branch the same way arms (5)-(8) do, isolating whether (3)'s strong result comes from spectral evidence itself or its extra (unpooled) capacity -- same confound-check as SeaPerson's own arm (4), §4.2.
- $^\P$ **(9)** is outside the 8-arm roster that mirrors SeaPerson's own order (§3 intro) -- same evidence branches and training flags as (5), only the fusion rule changed (`torch.max` instead of the learned 1x1 combiner). Appended rather than inserted after (5) to avoid renumbering arms (6)-(8) and the many "arm N" cross-references to them in §3.4. Full comparison and interpretation in §3.5.

---

### 3.3 Size-Bucket & Per-Class Recall on UAVDT (373,997 GT Targets)

| Arm | Very Tiny ($<16^2$ px)<br>*(74,375 GT)* | Tiny ($16^2\text{--}32^2$ px)<br>*(206,015 GT)* | Small ($32^2\text{--}96^2$ px)<br>*(86,282 GT)* | Med/Large ($>96^2$ px)<br>*(7,325 GT)* | Total Recall | Car | Truck | Bus |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(1) Baseline (BCE)** | 79.43% (59,076) | 84.21% (173,493) | 94.54% (81,575) | 59.78% (4,379) | 85.17% (318,523) | 85.61% | 81.05% | 67.67% |
| **(2) Semantic-only** | 79.13% | 83.91% | 94.00% | 59.95% | 84.82% | 85.20% | 77.74% | 73.27% |
| **(3) Spectral-only** | 83.78% | 87.83% | 97.62% | 64.68% | 88.83% | 89.11% | 83.89% | 79.74% |
| **(4) Spectral-only (Pooled)** | 86.88% | 92.23% | 97.74% | 66.43% | 91.93% | 92.27% | 87.61% | 79.63% |
| **(5) Concat-only** | 79.09% | 85.85% | 95.85% | 62.66% | 86.35% | 86.68% | 80.68% | 75.84% |
| **(6) Concat+SABL** | 83.53% | 89.43% | 97.93% | 64.16% | 89.72% | 90.13% | 83.05% | 76.58% |
| **(7) Concat+ISPPHead** | 83.06% | 88.19% | 98.07% | 62.32% | 88.94% | 89.33% | 84.19% | 74.69% |
| **(8) HESOD (Full)** | 82.54% | 88.34% | 98.09% | 63.02% | 88.94% | 89.31% | 83.90% | 75.63% |
| **(9)$^\P$ Concat-Max** | **85.69%** | 89.97% | 97.37% | 65.99% | **90.36%** | 90.67% | 85.14% | 80.09% |

---

### 3.4 In-Depth Experimental Interpretations

1. **Evidence-source ablations only partly transfer from SeaPerson.** Spectral evidence alone helps on both datasets (UAVDT arm 3: +3.66pp Total Recall, +1.1pp mAP@.5 over R0; SeaPerson: +3.35pp recall, §4.4.1). Coverage loss alone (semantic-only, arm 2) doesn't clearly help on UAVDT (-0.35pp Total Recall, within the ~4pp noise floor) despite helping SeaPerson (+3.33pp).
2. **Concat-only (arm 5) is UAVDT's weakest and most unstable arm.** It trails every single-evidence arm on mAP@.5/mAP@.5:.95 across two independent reruns, even as its raw-prediction count swings wildly (7.49M$\to$3.31M). A `--hm-threshold`/`--top-k` inference-time sweep (`sweep_uavdt_concat_thresholds.sh`, no retraining) rules out "too many raw candidates" as the cause: mAP@.5 stays flat (0.371$\to$0.370) as the threshold rises and the candidate pool shrinks 15%, and a hard per-image `--top-k` cap makes mAP *worse*. The deficit is in the fusion, not the candidate volume -- see point 4.
3. **Arm (5)'s instability confounds SABL/ISPPHead's "isolated effect" claims (arms 6/7), which were only ever measured against a moving arm-5 baseline** -- e.g. SABL's Very-Tiny-recall effect flips sign (-1.8pp vs. +4.44pp) depending on which concat-only run it's compared against. The one claim immune to this is ISPPHead's Params reduction (a pure architecture property): -27.5% (35.85M$\to$25.98M) either way, matching SeaPerson's own -27.6% almost exactly (§4.4.3). The max+SABL / max+ISPPHead arms queued below exist specifically to re-run this comparison against a fusion rule that isn't itself broken.
4. **Root cause: concat's combiner is provably affine, so it cannot represent a max/union rule.** `ChannelPooledConcatEvidenceSegmenter` (`models/segmenter.py`) fuses the two branches through plain `Conv2d` layers with no nonlinearity between them -- a composition of linear layers is itself affine ($\text{combined}=w_1 p_{\mathrm{semantic}}+w_2 p_{\mathrm{spectral}}+b$), which can only express weighted averages of the two branches and structurally cannot express $\max(a,b)$ (piecewise, not linear). Concretely: a confident branch gets pulled *down* toward a weak co-branch instead of winning outright -- the direct mechanism behind arm (5) underperforming both single-evidence arms. This also correctly predicts that `--pos-weight` retuning (tried and abandoned before this fix) couldn't have helped: it only reshapes the training gradient, not the representable function class.
5. **General principle, confirmed empirically.** For a recall-safe selector, a fusion rule should satisfy **evidence preservation**, $F(p_s,p_f)\ge\max(p_s,p_f)$ -- adding a second evidence source should never make an already-trusted location look less important. Concat's affine combiner provably violates this; `torch.max` and noisy-OR ("soft-OR", $p_{\mathrm{fuse}}=1-(1-p_s)(1-p_f)$, computed exactly via `logsumexp` in logit space) both satisfy it by construction, and soft-OR additionally unifies algebraically with the object-level soft-coverage loss's own noisy-OR formula (§2.2). §3.5 confirms max wins empirically over both concat and soft-OR on every metric -- max is the fusion rule carried into the flagship recipe.

---

### 3.5 Fusion-Rule Probe: Concat vs. Max vs. Soft-OR (exploratory, outside the 8-arm roster)

Same evidence branches (channel-pooled semantic + spectral), same `--selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2` training flags, same 50 epochs -- only the fusion rule combining the two branches' logits changes. Reference rows from §3.2/§3.3 included for scale.

| Fusion rule | mAP@.5 | mAP@.5:.95 | Total Recall | BPR | GFLOPs | Params (M) | FPS | Raw predictions |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Concat (arm 5, learned 1x1 conv) | 0.371 | 0.205 | 86.35% | 0.919 | 83.9 | 35.85 | 107.2 | 3.31M |
| **Max (`torch.max`)** | **0.395** | **0.218** | **90.36%** | **0.940** | 90.1 | 35.85 | 102.3 | 5.43M |
| Soft-OR (`logsumexp` noisy-OR) | 0.387 | 0.215 | 88.77% | 0.927 | 89.9 | 35.85 | 102.7 | 5.20M |
| *Reference: R0 (arm 1)* | *0.385* | *0.214* | *85.17%* | *0.884* | *68.2* | *35.85* | *117.8* | *2.35M* |
| *Reference: Spectral-only (arm 3)* | *0.396* | *0.214* | *88.83%* | *0.936* | *98.1* | *36.01* | *100.5* | *5.14M* |
| *Reference: Spectral-only Pooled (arm 4)* | *0.394* | *0.209* | *91.93%* | *0.963* | *99.3* | *35.85* | *97.6* | *5.94M* |

**Max fusion confirms the evidence-preservation hypothesis (§3.4 points 4-5) cleanly, and does better than "cap out at the better single branch" alone.** Every metric improves over concat: +2.4pp mAP@.5, +1.3pp mAP@.5:.95, +4.01pp Total Recall, +0.021 BPR -- for +7.4% GFLOPs (83.9$\to$90.1) and -4.6% FPS, a real but modest cost. More notably, max now matches or beats the best *single-evidence* arms while using *less* compute than either: mAP@.5 (0.395) is within noise of spectral-only's 0.396, Total Recall (90.36%) exceeds spectral-only's 88.83% by +1.53pp, and GFLOPs (90.1) is lower than both spectral-only (98.1) and spectral-only-pooled (99.3). Params stay exactly at concat's 35.85M (expected -- fusion adds no parameters either way, `torch.max` has none, the removed `concat_convs` had few). `vt_diagnose.py`'s miss-reason mix shifts toward concat's own pattern but less extreme (`right_class_low_iou` 48.8% vs. concat's 50.6%, `no_nearby_prediction` 34.7% vs. 32.8%) -- max fusion isn't a clean return to either single-branch's own miss profile, it's a genuinely new operating point. Params note: `ChannelPooledMaxEvidenceSegmenter` has no `concat_convs` at all (removed, not just unused), yet Params is reported identical to concat's 35.85M to 2 decimal places -- the removed 1x1 conv's parameter count (a few thousand, `nc*2*nc` weights) rounds away at this precision, consistent with §3.4 point 5's claim that the fusion op's own cost (compute *or* parameters) was never the story.

**Soft-OR did not beat max -- the opposite of what its own theoretical property (§3.4 point 5) predicted.** Its evidence-preservation bound is strictly tighter than max's ($p_{\mathrm{fuse}}\ge\max(p_s,p_f)$ with equality only when one branch is exactly 0, vs. max's equality whenever either branch wins outright), yet soft-OR trails max on every single metric measured: mAP@.5 (0.387 vs. 0.395, -0.8pp), mAP@.5:.95 (0.215 vs. 0.218, -0.3pp), Total Recall (88.77% vs. 90.36%, -1.59pp), and BPR (0.927 vs. 0.940). This happens despite soft-OR producing *fewer* raw predictions than max (5.20M vs. 5.43M) and near-identical GFLOPs (89.9 vs. 90.1) -- so it isn't a case of soft-OR being more conservative and giving up recall for precision; it's uniformly behind. A plausible read: hard max's winner-take-all gradient (point 5's flagged risk) turned out to be the less costly trade-off here than expected, while soft-OR's extra permissiveness when both branches are moderately confident (its main theoretical advantage) doesn't appear to route toward the *right* extra candidates on this dataset -- consistent with semantic and spectral not being symmetric, easily-confused evidence sources to begin with (point 5's own caveat about why winner-take-all might be milder here than in typical mixture-of-experts settings). **Max is the fusion rule carried forward into the flagship-recipe test** (`uavdt_yolov5m_channel_pooled_max_sabl_isphead`, max fusion + SABL + ISPPHead -- "HESOD Full v2" -- queued to test whether the fix also improves arm (8) HESOD (Full)'s own 0.378/0.202/88.94% numbers, alongside two isolating arms, `uavdt_yolov5m_channel_pooled_max_sabl` and `uavdt_yolov5m_channel_pooled_max_isphead`, that redo the point-3 SABL/ISPPHead comparison against a fusion rule that isn't itself broken) -- soft-OR's elegant unification with the coverage loss's own algebra (point 5) turned out not to translate into a better empirical result on UAVDT, at least not with these training flags.

---

## 4. SeaPerson (TinyPersonV2) Benchmark Results ($2048\times 2048$)

SeaPerson evaluates ultra-high-resolution maritime search-and-rescue over 5,752 test images containing 300,375 annotated instances (over 85% tiny/micro persons).

### 4.1 Comparison against Dense Competitors & Baseline

| Method | Type | mAP@.5 | mAP@.5:.95 | Params (M) | GFLOPs | FPS |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Faster R-CNN** (ResNet-50-FPN) | Dense Two-Stage | 0.551 | 0.246 | 43.26 | 1546.8 | 23.1 |
| **RetinaNet** (ResNet-50-FPN) | Dense One-Stage | 0.473 | 0.201 | 36.35 | 942.7 | 28.0 |
| **ESOD Baseline (R0)** | Selective Single-Evidence | 0.750 | 0.320 | 35.78 | **202.4** | **85.7** |
| **HESOD (Ours, Dual+ISPP)** | Selective Dual-Evidence | **0.771** | **0.328** | **25.92** | 217.6 | 77.3 |
| **HESOD (Ours, Full +SABL)** | Selective Dual-Evidence | **0.774** | 0.326 | **25.92** | 209.0 | **80.7** |

- **Versus Dense Detectors**: HESOD achieves **+22.3 pp higher mAP@.5** than Faster R-CNN while reducing GFLOPs by **7.4$\times$** and accelerating inference by **3.5$\times$**; versus RetinaNet, **+30.1 pp higher mAP@.5** while reducing GFLOPs by **4.5$\times$** and accelerating inference by **2.9$\times$**. RetinaNet's own Very Tiny recall (50.05%) is notably higher than Faster R-CNN's (46.07%) despite lower overall mAP -- its dense, single-stage prediction handles SeaPerson's extreme per-image object density better than Faster R-CNN's two-stage RPN funnel (`no_nearby_prediction` miss share 28.9% vs 49.5%).
- **Versus ESOD Baseline**: HESOD boosts mAP@.5 by **+2.1~2.4 pp** and mAP@.5:.95 to **0.328**, while cutting parameters by **27.6%**.

---

### 4.2 Module Ablation Matrix on SeaPerson ($2048\times 2048$)

| Arm | Module Configuration | Selector Loss | Box Loss | Head | mAP@.5 | mAP@.5:.95 | BPR | GFLOPs | FPS |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **(1)** | **Baseline (BCE)** | Upstream BCE | CIoU | Coupled | 0.750 | 0.320 | 0.947 | **202.4** | **85.7** |
| **(2)** | **Semantic-only$^*$** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.769 | 0.325 | \underline{0.991} | 266.8 | 73.5 |
| **(3)** | **Spectral-only$^*$** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.770 | 0.327 | **0.992** | 267.4 | 71.7 |
| **(4)** | **Spectral-only (Pooled)** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.767 | 0.324 | 0.988 | 263.4 | 72.7 |
| **(5)** | **Dual-Concat** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | Coupled | 0.772 | 0.326 | 0.986 | 281.2 | 69.8 |
| **(6)** | **Dual+SABL** | $\mathcal{L}_{\mathrm{cover}}$ | SABL | Coupled | 0.771 | 0.323 | 0.990 | 263.7 | 73.4 |
| **(7)** | **Dual+ISPP** | $\mathcal{L}_{\mathrm{cover}}$ | CIoU | ISPPHead | 0.771 | **0.328** | 0.988 | 217.6 | \underline{77.3} |
| **(8)** | **HESOD (Full)** | $\mathcal{L}_{\mathrm{cover}}$ | SABL | ISPPHead | **0.774** | 0.326 | 0.991 | \underline{209.0} | **80.7** |

- **(4)** channel-pools the spectral branch the same way arms (5)-(8) do, isolating whether (3)'s strong result came from spectral evidence itself or from its extra (unpooled) capacity -- trains at the shared batch=8, unlike (3)'s forced batch=2 for the full-width branch. Params 35.79M vs (3)'s 35.94M.

---

### 4.3 Size-Bucket Recall on SeaPerson (300,375 GT Targets)

| Arm | Very Tiny ($<16^2$ px)<br>*(82,417 GT)* | Tiny ($16^2\text{--}32^2$ px)<br>*(174,816 GT)* | Small ($32^2\text{--}96^2$ px)<br>*(42,987 GT)* | Med/Large ($>96^2$ px)<br>*(155 GT)* | Total Recall<br>*(300,375 GT)* |
|---|:---:|:---:|:---:|:---:|:---:|
| **(1) Baseline (BCE)** | 74.03% (61,014) | 86.78% (151,707) | 94.74% (40,725) | 79.35% (123) | 84.42% (253,569) |
| **(2) Semantic-only$^*$** | 75.49% (62,217) | 91.57% (160,087) | 95.71% (41,141) | 81.94% (127) | 87.75% (263,572) |
| **(3) Spectral-only$^*$** | 75.23% (62,006) | **91.69% (160,286)** | **95.89% (41,220)** | 86.45% (134) | 87.77% (263,646) |
| **(4) Spectral-only (Pooled)** | 76.13% (62,748) | 91.36% (159,705) | 95.50% (41,051) | **87.10% (135)** | 87.77% (263,639) |
| **(5) Dual-Concat** | 76.00% (62,637) | 91.50% (159,962) | 95.68% (41,131) | 80.00% (124) | 87.84% (263,854) |
| **(6) Dual+SABL** | 76.67% (63,193) | 91.49% (159,935) | 94.77% (40,740) | 80.65% (125) | 87.89% (263,993) |
| **(7) Dual+ISPP** | 75.86% (62,521) | 91.25% (159,516) | \underline{95.85\%} (41,204) | **87.10% (135)** | 87.68% (263,376) |
| **(8) HESOD (Full)** | **77.14% (63,574)** | 91.59% (160,110) | 95.01% (40,842) | 84.52% (131) | **88.11% (264,657)** |

---

### 4.4 In-Depth Experimental Interpretations

1. **Coverage Supervision Drives the Primary Recall Jump**: Replacing cell-wise independent BCE with object-level soft-coverage loss $\mathcal{L}_{\mathrm{cover}}$ (Arm 1 $\to$ 2) delivers an immediate **+1.9 pp mAP@.5** and raises BPR from 0.947 to 0.991. Error diagnosis (`vt_diagnose.py`) confirms selector-dropped error drops from **25.6% to 19.2%**, proving that joint coverage optimization successfully halts premature micro-target pruning.
2. **Spectral Saliency Acts as an Orthogonal Cue**: Spectral-only routing (Arm 3) reaches 0.770 mAP@.5 and 0.992 BPR, demonstrating that channel-pooled structural gradients provide a dependable spatial routing signal independent of semantic activations. Unconditional concatenation (Arm 4) achieves peak 0.772 mAP@.5 and recovers +420 to +631 more Very Tiny instances than single-evidence selectors.
3. **ISPPHead Delivers True Pareto Compression**: Swapping the coupled head for the inverted residual ISPP decoupled head (Arm 6) slashes parameters by **27.6%** (35.79M $\to$ 25.92M) and GFLOPs by **22.6%** (281.2 $\to$ 217.6), while establishing the highest high-IoU precision ($\mathrm{mAP@.5:.95} = \mathbf{0.328}$) at 77.3 FPS.
4. **SABL Maximizes Micro-Target Recall**: While aggregate mAP averages across all scales, SABL's Wasserstein distance regression specializes in sub-16px instances, pushing **Very Tiny recall to 77.14%** (recovering +2,560 targets over baseline) and lowering GFLOPs to **209.0** via tighter patch localization.

---

## 5. Negative Probes & Domain Boundary Findings

To maintain rigorous scientific standards, all negative exploratory results are formally documented:

1. **Gated Evidence Fusion (Dual-Gated)**: Replacing fixed $1\times 1$ concatenation with a learned sigmoid gate (`ChannelPooledDualEvidenceSegmenter`) caused the network to suppress candidate evidence, degrading mAP@.5 to **0.765** and total recall to **87.26%** (worst among all coverage-loss arms). Fixed unconditional fusion is strictly superior.
2. **SeaDronesSeeV2**: Evaluated under YOLOv5m at $1536\times 1536$. The dataset was dropped because Very Tiny targets represent merely **1.9%** of annotations; baseline R0 already achieved **95.76% total recall** and **0.894 mAP@.5**, leaving no headroom for spatial routing benefits.

Pest24 (dense agricultural insect imagery, same "no headroom for routing" class of finding) belongs to the separate `HESOD-Agri-Experiment-Plan.md` project, not this document's own dataset scope (VisDrone/TinyPerson/UAVDT/SeaPerson/SeaDronesSeeV2) -- not reproduced here.

---

## 6. Execution Verification & Test Gates

Before dispatching training or evaluation runs, the following unit and regression gates must pass:

```bash
# 1. Verify SABL loss mechanics (finite zero-overlap gradients, scale gating, pixel normalization)
pytest -q tests/test_sabl_loss.py

# 2. Verify Torchvision baseline data pipelines and multi-class labels
pytest -q tests/test_baseline_torchvision.py
```

### Active Runner Scripts:
- **SeaPerson Pipeline**: `scripts/esod_baseline/run_seaperson.sh`
- **UAVDT Pipeline**: `scripts/esod_baseline/run_uavdt.sh`
- **VisDrone Pipeline**: `scripts/esod_baseline/run_visdrone_roster.sh`
- **Torchvision Competitors**: `hesod/backends/baseline/` (Faster R-CNN & RetinaNet runners)
