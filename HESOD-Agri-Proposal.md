# HESOD-Agri Proposal

> **Working title:** HESOD-Agri: Budget-Adaptive Semantic–Spectral Routing for Tiny Agricultural Object Detection  
> **Scope lock:** AgriPest, Pest24, and GWHD 2021 only  
> **Primary target:** *Computers and Electronics in Agriculture*  
> **Status:** prospective study; no unverified result in this document is a claim

## 1. Research problem

Agricultural monitoring images combine three properties that make conventional full-frame detection inefficient and fragile:

1. the objects can occupy only a tiny fraction of a high-resolution image;
2. target density varies sharply between images and domains;
3. localization errors dominate when boxes are small, crowded, or weakly separated from the background.

The proposed study asks whether an image-conditioned router can allocate expensive high-resolution local inference only where it is useful, while preserving small-object recall. The detector is HESOD; the proposed selector fuses a low-cost semantic signal with channel-pooled spectral evidence through a **reliability-aware residual gate**, not plain concatenation — see §4.2 and [BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md](BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md) §5.3C for the source design this study transfers to agriculture. A scale-adaptive box loss (SABL) is tested as an orthogonal localization intervention, not presented as part of the routing novelty.

The central thesis is therefore:

> Dynamic semantic–spectral routing can improve the accuracy–compute Pareto frontier of tiny agricultural object detection across variable target densities; scale-adaptive regression can independently improve the localization of the selected small objects.

This is not a proposal for one universal model trained on merged agriculture labels. Each dataset retains its own label space, official split, detector head, and evaluation protocol.

## 2. Dataset roles

| Dataset | Task and scale | Role in the paper | Required protocol |
|---|---|---|---|
| **AgriPest** | Field pest detection; 49,700 images, 264,700 boxes, 14 species on four crops; severe density, illumination, and background variation | Primary mechanism dataset. Tests whether routing adapts between sparse and dense field scenes | Preserve the official split and the dataset's dense/sparse, illumination, and clutter analyses where available |
| **Pest24** | High-resolution light-trap pest detection; about 25,000 images and 24 visually similar classes; crowded, adhesive, very small targets | Dense high-resolution stress test. Tests the failure boundary of sparse routing | Verify the redistribution licence and exact official/reported split before conversion; never infer a split from filenames |
| **GWHD 2021** | Wheat-head detection across global acquisition domains; more than 6,000 1024×1024 images and over 300,000 heads | Cross-task external validation and boundary condition: insects → plant organs, sparse → potentially dense | Use a published official split, preferably GlobalWheat-WILDS for domain-shift claims; do not create a private “small-image” benchmark |

For GWHD, `AP_small` is computed from native-resolution instance area (`area < 32^2` pixels under the COCO convention). Non-small ground truths must be ignored—not converted to background—during small-object evaluation. A predeclared sparse-image diagnostic may be reported separately, but it is never called the official GWHD benchmark.

## 3. Research questions and hypotheses

### RQ1 — Does dynamic routing beat fixed computation?

At comparable detector and input information, compare full/dense processing, uniform fixed-K patching, semantic routing, and the proposed reliability-aware semantic–spectral gated routing (§4.2).

**H1:** the proposed router provides a better AP–latency or AP–processed-pixel Pareto point than uniform fixed-K and semantic-only routing on AgriPest and Pest24.

### RQ2 — Does the benefit depend on target density?

**H2:** the selected-patch distribution responds to scene density, while recall at a fixed compute budget remains higher than uniform allocation. Gains should be largest in sparse-to-moderate scenes and may narrow on extremely dense Pest24 images.

### RQ3 — Is scale-adaptive localization complementary to routing?

Use the implemented `--box-loss sabl` option in a 2×2 factorial design: selector type × box loss.

**H3:** SABL improves `AP`, `AP75`, or `AP_small` without materially changing inference cost. An improvement only at AP50 is insufficient evidence of better localization.

### RQ4 — Does the routing mechanism transfer beyond insect detection?

**H4:** on GWHD 2021, routing maintains competitive accuracy under a reduced pixel/latency budget, or clearly identifies a density regime in which routing should fall back to denser processing.

## 4. Proposed method and novelty boundary

### 4.1 Dynamic local processing

The image is partitioned into an 8×8 routing grid. Threshold routing may select 0–64 cells; exact Top-K is a separate budget-control mode. `K` is therefore the number of selected grid cells/local regions, not the number of ways the whole image is divided. Lower K usually reduces processed local area and detector work, but measured end-to-end latency—not K alone—is the systems result.

The fixed threshold is an inference hyperparameter, not merely an initialization value. Threshold and exact Top-K results must be reported separately because they answer different questions:

- **threshold mode:** how much computation the image requests;
- **Top-K mode:** how much accuracy is possible under an enforced budget.

### 4.2 Reliability-aware semantic–spectral selector

**2026-08-15 revision.** An earlier draft of this section described the proposed selector as channel-pooled semantic + spectral concatenation. That was a scoping error, not a deliberate simplification: `BCRS-Budget-Constrained-Recall-Safe-Selector-Proposal.md` §5.3C is explicit that concatenation *"只作为容量匹配 baseline，不作为默认主方法"* (only a capacity-matched baseline, not the default main method), because concat has no mechanism forcing the network to learn "trust spectral evidence specifically when semantic confidence is insufficient" — it can just as easily learn "spectral always dominates" or "spectral gets pulled toward texture-rich background," neither of which is the intended behavior. This revision replaces concat as the proposed method with BCRS's actual headline mechanism.

**Evidence.** Two branches feed the selector for region $i$:

- semantic/objectness $q_i = P(y_i=1 \mid f_i^{sem})$ from the learned heatmap branch, as before;
- spectral evidence, compressed across channels (`ChannelPooledSpectralFilter`, depthwise multi-band filters — not raw per-patch FFT), but now producing **two** outputs instead of one: a signed residual $z_i^{spec}$ (must be able to go negative, so it can suppress texture-heavy background as well as rescue low-objectness targets) and a reliability score $c_i^{spec} \in [0,1]$.

A third, lightweight head estimates texture-contamination risk $t_i^{bg} \in [0,1]$ from the same shared shallow features.

**Fusion.** Semantic uncertainty is computed as normalized binary entropy:

$$h_i = \frac{-q_i\log(q_i+\varepsilon) - (1-q_i)\log(1-q_i+\varepsilon)}{\log 2}.$$

A gate MLP combines all three signals plus the routing budget embedding $e_B$:

$$a_i = \sigma\big(\mathrm{MLP}[q_i, h_i, c_i^{spec}, t_i^{bg}, e_B]\big),$$

and the final routing priority is a **residual fusion in logit space**, not a weighted sum in probability space:

$$u_i = \mathrm{logit}(q_i) + \alpha \cdot a_i \cdot z_i^{spec}.$$

$u_i$ is a ranking score, not a probability, so it is not clamped to $[0,1]$; $z_i^{spec}$ being signed means the gate can move priority down (background suppression) as well as up (tiny-object rescue).

**Why not just $(1-q_i)^\gamma$.** A simpler low-objectness rescue gate — spectral weight rises purely as semantic confidence falls — is tempting, but BCRS's own analysis (§5.4) flags the failure mode directly: such a gate activates on *all* low-$q_i$ regions, including low-$q_i$ background, so it cannot distinguish "quiet real target" from "quiet background" and would be expected to inflate textured-background false routing exactly where Pest24 (§2, this document) is meant to stress-test it. It is kept in this study only as a **named diagnostic ablation** ($a_i^{low}$, §4.2.1), not the proposed mechanism.

### 4.2.1 Fusion ablation ladder

Because the gate's value proposition rests on doing *better than* simpler alternatives, not merely *differently*, the following ladder is run parameter/latency-matched (BCRS §7.4), primary validation on AgriPest before promoting to Pest24/GWHD (BCRS §11.2 Phase 2 gate):

| ID | Fusion | Definition | Role |
|---|---|---|---|
| F0 | Objectness-only | $u_i = \mathrm{logit}(q_i)$ | Semantic baseline (= R0/R1 in §5) |
| F1 | Concat → MLP | $u_i = f([q_i, s_i])$ | Capacity-matched control — the *previous* draft's proposed method, now a required control |
| F2 | Unconstrained learned gate | $a_i = \sigma(\mathrm{MLP}[q_i, s_i])$, no uncertainty/confidence/texture inputs | Tests whether gate *structure* alone (vs. concat) helps |
| F3 | Uncertainty-only gate | $a_i = h_i$ | Tests "intervene exactly when semantic is genuinely uncertain" |
| F4 | Low-score rescue gate | $a_i = (1-q_i)^\gamma$ | Tests low-objectness rescue in isolation; expected to raise textured-background false routing — see above |
| F5 | Reliability-aware residual gate | as derived above | **Proposed method** |
| F6 | F5 + rescue-ranking + conditional-gate regularization + coverage | full training objective, §4.2.2 | **Full proposed method** |

F5/F6 not beating parameter-matched F1 on the low-objectness-tiny-recall / textured-background-false-routing trade-off is a named falsification condition (§9).

### 4.2.2 Rescue-ranking and conditional-gate regularization losses

Two loss terms shape the gate beyond ranking/coverage supervision (BCRS §5.4), used only for F6:

**Rescue-ranking loss.** Let $\mathcal{P}_{rescue} = \{(i,j): y_i=1, q_i < \tau_{low}, y_j=0, j \in \mathcal{B}_{tex}\}$ pair real low-objectness positives against texture hard-negatives (from AgriPest/Pest24's dense-background regions). The gate is trained so real rescued targets outrank texture background in priority:

$$\mathcal{L}_{rescue} = \frac{1}{|\mathcal{P}_{rescue}|} \sum_{(i,j) \in \mathcal{P}_{rescue}} \log\big(1 + \exp(m - u_i + u_j)\big).$$

**Conditional-gate regularization.** Suppresses gate activation both where semantic is already confidently correct and on texture hard-negatives, as a small anti-degeneration term (must not overpower coverage/ranking supervision):

$$\mathcal{L}_{cond} = \frac{1}{|\mathcal{C}_{sem}|}\sum_{i \in \mathcal{C}_{sem}} a_i + \frac{1}{|\mathcal{B}_{tex}|}\sum_{i \in \mathcal{B}_{tex}} a_i.$$

Hyperparameters ($\tau_{low}$, margin $m$, $\lambda_{rescue}$, $\lambda_{cond}$, $\alpha$) are **not locked** the way SABL's are (§4.3) — BCRS does not specify fixed values for these, so they require a validation sweep before the primary operating point is chosen (mirrors the `hm_threshold` sweep already required in §8.1 of the companion Experiment Plan). Candidate starting points for that sweep: $\tau_{low}=0.3$, $m=1.0$, $\lambda_{rescue}=0.5$, $\lambda_{cond}=0.1$, $\alpha=1.0$ — these are unvalidated defaults, not conclusions.

Architecture claims (F5/F6 over F0/F1) require a matched semantic-only control, the same detector backbone/head, the same label target, and measured routing overhead — the overhead of the confidence and texture-risk heads must be included in that measurement (§5.7–5.8 of the BCRS source proposal), not just the gate MLP itself.

### 4.3 SABL as an orthogonal loss ablation

SABL is applied only to box regression during training. The current implementation uses scale-adaptive mixing based on ground-truth box size and keeps the objectness target on the upstream CIoU path. It adds no inference parameters or FLOPs.

SABL can be combined directly with the reliability-aware gate (or, in the F0–F6 ablation, with any other fusion variant) because the selector controls **where** to process and SABL controls **how selected boxes are regressed**. It must be retrained; it cannot be enabled only at test time.

The paper must distinguish clearly between:

- the proposed routing contribution;
- a borrowed/adapted loss component with proper citation; and
- their interaction in the factorial ablation.

## 5. Minimum experimental evidence

The irreducible matrix on every admitted dataset is:

| Arm | Routing | Selector training | Box loss | Purpose |
|---|---|---|---|---|
| A0 | Full image or exhaustive local processing | N/A | CIoU | Accuracy-oriented dense reference |
| A1 | Uniform fixed-K | N/A | CIoU | Compute-matched non-learned control |
| R0 | Dynamic/fixed-budget | Semantic only (F0) | CIoU | Reproduced HESOD baseline |
| R1 | Dynamic/fixed-budget | Semantic only (F0) | SABL | Loss-only effect |
| R2 | Dynamic/fixed-budget | Reliability-aware residual gate (F5/F6, §4.2) | CIoU | Proposed selector-only effect |
| R3 | Dynamic/fixed-budget | Reliability-aware residual gate (F5/F6, §4.2) | SABL | Combined model and interaction |
| O | Ground-truth oracle patches | Oracle | CIoU | Upper bound on routing; not deployable |

R2/R3 use the gate (F5, or F6 once its losses validate), not concat — concat is F1 in §4.2.1's fusion ablation ladder, run as a required control on AgriPest before R2/R3 are trusted, not as a substitute for them. If F5/F6 fails to beat parameter-matched F1 (§9's falsification conditions), R2/R3 in this table should be understood to have reverted to F1 by default, and that reversion must be stated explicitly in any reported result, not silently substituted.

### 5.1 Cross-dataset baseline strategy

No verified paper was found that reports AgriPest, Pest24, and GWHD 2021 together under one split and metric contract. This is unsurprising: the first two are multi-class pest benchmarks, whereas GWHD is a one-class cross-domain wheat-head benchmark. The paper must not imply that unrelated published tables form a unified leaderboard.

The audited coverage of the papers currently in `reference/` is:

| Paper | Datasets actually evaluated | Coverage of the three target datasets |
|---|---|---|
| AgriPest dataset paper | AgriPest | AgriPest only |
| Pest-PVT | Pest24 | Pest24 only |
| QueryDet | COCO, VisDrone | None |
| SSABNet | VisDrone, UAVDT | None |
| GWHD 2021 dataset paper | GWHD 2021 | GWHD only |

Consequently, existing values can pre-populate only dataset-specific original-protocol tables. The unified three-dataset table remains empty until A0/X1-X4 and HESOD are rerun under the common evaluator.

Instead, the core comparison uses the **same public implementations rerun by us on all three datasets**:

| ID | Baseline | Why it is included | Three-dataset policy |
|---|---|---|---|
| A0 | Dense YOLOv5m | Architecture-matched control for the HESOD detector | Train/evaluate on all three |
| X1 | Faster R-CNN R50-FPN | The closest common literature anchor: a Faster R-CNN family baseline appears in the source/official evaluations of all three datasets | Rerun on all three; additionally reproduce each dataset's original metric where exact protocol is available |
| X2a | YOLOv5m + SAHI inference | Exhaustive uniform slicing control; directly tests whether learned patch selection is preferable to processing regular overlapping tiles | Mandatory on all three using the A0 checkpoint |
| X2b | YOLOv5m + SAHI sliced fine-tuning + inference | Stronger uniform-tile baseline | Run on all three if training budget permits; keep separate from X2a |
| X3 | QueryDet R50-FPN | Closest external learned sparse high-resolution detector | Run on all three if the official implementation can be ported without semantic changes; AgriPest is the minimum admission dataset |
| X4 | RTMDet-m | Strong modern dense detector and accuracy/efficiency control | Train/evaluate on all three with the common evaluator |

This design gives us a publishable three-dataset table even though the original papers did not create one. Direct comparison comes from rerunning their public code with our frozen data manifests and evaluator, not from copying incompatible numbers.

Dataset-specific agriculture methods remain secondary anchors. Pest-PVT and Pest-YOLO may be rerun on Pest24, but they do not replace X1–X4 because they have no reported AgriPest/GWHD results. Likewise, the AgriPest and GWHD official baselines remain protocol sanity checks rather than rows in a fabricated cross-dataset leaderboard.

Every external baseline must use a documented official split and a resolution that does not erase the tiny targets. Published values are called **directly comparable** only when dataset version, image IDs, class map, resizing, metric definition, and test-time policy match. Otherwise they appear in a separately labelled “published under original protocol” table.

## 6. Evaluation contract

### 6.1 Detection

- `AP` means COCO-style `AP@[0.50:0.95]`.
- `AP50` and `AP75` are reported separately and never mapped to `AP`.
- Report `AP_small`, per-class AP, recall at IoU 0.50/0.75, and the number of evaluated images/ground truths.
- Matching for custom recall audits is class-aware and one-to-one. Multiple predictions cannot recall the same ground truth more than once.
- AgriPest additionally reports density, illumination, and clutter strata where the source annotations/protocol permit.
- GWHD reports domain-level results and WDA only when the exact published split and definition are used.

### 6.2 Routing and recall

- `BPR_box`: fraction of ground-truth boxes whose overlap fraction with the selected region is greater than 0.5.
- `BPR_ctr`: fraction of ground-truth centers covered by the selected routing mask.
- Report BPR for all objects and native-resolution small objects.
- Report selected K distribution, empty-selection frequency, positive-cell occupancy, and oracle BPR at K = 8, 16, and 32.
- Every metric artifact stores image IDs, class mapping, split identity, checkpoint hash, routing mode, threshold/K, and exact denominator.

### 6.3 System cost

- end-to-end latency at batch 1 and one declared throughput batch;
- detector-only and router-only time where possible;
- peak GPU memory, processed local pixels/area, parameter count, and reliable MACs/FLOPs;
- hardware, precision, warm-up, synchronization, dataloader inclusion, and sample count.

The main systems claim is based on measured end-to-end cost. Theoretical patch count is supporting evidence only.

### 6.4 Statistical reporting

- use at least three seeds for claims about method superiority;
- report mean, standard deviation, and paired seed deltas where feasible;
- predeclare the primary metric and budget point before examining test results;
- do not select a threshold independently on the test set.

## 7. Data and label strategy

Each dataset is converted to a canonical internal representation while preserving raw annotations and an auditable mapping manifest. Dataset conversion is preprocessing; it must not silently alter boxes, classes, or empty images.

Initial experiments use deterministic box-derived routing targets. A Gaussian/coverage heatmap can be generated from boxes, but SAM/pseudo-mask hybrid labels are excluded from the core study because they would confound the selector and loss comparison. Hybrid labels may be a later extension only after the core matrix is complete.

No label space is shared across AgriPest, Pest24, and GWHD. The study shares method and protocol, not category semantics.

## 8. GWHD admission gate

Before committing GPU time, compute native-size distributions, objects/image, 8×8 positive-cell occupancy, and oracle BPR@K.

- If oracle BPR@32 ≥ 95% and median positive occupancy is well below 32/64, run the full routing matrix.
- If only a predeclared sparse diagnostic is routable, retain GWHD as a boundary/negative-control result and label the slice explicitly.
- If even the sparse stratum needs near-dense coverage, use GWHD only for localization-loss validation or omit routing claims.

This gate is a feasibility decision, not a licence to cherry-pick favorable images.

## 9. Expected contributions and falsification criteria

### Contributions worth claiming if supported

1. A budget-adaptive semantic–spectral router for tiny agricultural objects.
2. A cross-density evaluation showing when learned routing is preferable to uniform or dense computation.
3. A clean selector × localization-loss factorial separating routing recall from box-regression quality.
4. An auditable evaluation protocol coupling detection AP, routing recall, and actual system cost.

### Results that would falsify or narrow the story

- R2 does not improve the Pareto frontier over R0 or A1.
- Routing recall improves but detector AP does not, indicating downstream localization/classification is limiting.
- SABL gains disappear across seeds or occur only at AP50.
- Pest24/GWHD require almost all 64 cells, making routing overhead unjustified.
- Claimed speedup exists only in theoretical FLOPs, not end-to-end measurements.
- F5/F6 does not beat parameter/latency-matched F1 (concat) on low-objectness-tiny-recall or on textured-background false-routing rate (§4.2.1) — the gate structure would then not be earning its added complexity over the simpler control.
- The gate saturates ($a_i \to 1$ almost everywhere) or collapses ($a_i \to 0$ almost everywhere) after training with $\mathcal{L}_{rescue}$/$\mathcal{L}_{cond}$ — indicates conditional fusion failed to learn selectivity, not just a weak effect size.
- On Pest24 specifically, F5/F6's textured-background false-routing rate is not lower than F1's — this is Pest24's specific admission role (§2) and a result here is not an optional nicety.

Negative outcomes remain useful if reported as density-dependent operating limits rather than hidden.

## 10. Publication strategy: JCR Q1–Q2 candidates

Journal quartiles change annually and sometimes differ by category. The bands below are a targeting guide based on publicly available publisher metrics and recent public JCR listings; the current institutional Clarivate JCR entry must be checked immediately before submission.

| Priority | Journal | Likely recent JCR band | Best manuscript framing | Required evidence / fit risk |
|---|---|---|---|---|
| 1 | **Computers and Electronics in Agriculture** | Q1 candidate | Novel budget-adaptive agricultural vision method with three datasets and a measured AP–compute frontier | Strongest fit. Must be methodologically novel; a simple application of existing YOLO modules is insufficient |
| 2 | **Biosystems Engineering** | Q1 candidate | Engineering of an adaptive sensing/inference system for biological production | Emphasize system design, real latency/memory/energy proxy, reproducibility, and operational implications |
| 3 | **Precision Agriculture** | Q1 candidate | Resource-aware crop/pest monitoring that can support site-specific decisions | Benchmark-only results are risky; connect density/counting outputs to monitoring decisions |
| 4 | **Plant Phenomics** | Q1 candidate | Cross-domain tiny-object phenotyping with GWHD as a central biological experiment | Conditional target. GWHD/domain-shift and biological phenotyping insight must be central, not a small appendix |
| 5 | **Pest Management Science** | Q1 candidate | Reliable pest population monitoring under field and light-trap conditions | Conditional target. Needs pest-management relevance, counting/monitoring analysis, and practical error consequences |
| 6 | **Crop Protection** | Q2 candidate; verify current category | Practical automated pest monitoring and crop-protection support | Suitable fallback if the method contribution is moderate but AgriPest/Pest24 evidence is operationally strong |

### Recommended submission order

1. Submit to *Computers and Electronics in Agriculture* if R2/R3 establish a statistically credible accuracy–compute Pareto improvement on both pest datasets and GWHD supplies external validation or an informative boundary.
2. Prefer *Biosystems Engineering* if the strongest result is deployment efficiency, adaptive computation, or system behavior rather than a large detector AP gain.
3. Prefer *Precision Agriculture* or *Pest Management Science* only after adding decision-oriented pest counts, density strata, or monitoring consequences.
4. Prefer *Plant Phenomics* only if GWHD/domain generalization becomes a co-equal biological contribution.
5. Use *Crop Protection* as the practical Q2 route after verifying the current JCR category and quartile.

## 11. Sources

### Datasets and protocols

- [AgriPest dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7956390/)
- [Pest24 dataset paper](https://www.sciencedirect.com/science/article/pii/S0168169919324123)
- [Pest-YOLO evaluation on the 24-class light-trap dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC9783619/)
- [Pest-PVT paper and public implementation](https://www.sciencedirect.com/science/article/pii/S0168169924012559)
- [AgriPest-YOLO evaluation on the 24-class light-trap dataset](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1079384/full)
- [GWHD 2021 dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8548052/)
- [GWHD 2021 official Zenodo record](https://zenodo.org/records/5092309)
- [Global Wheat Head Dataset project](https://www.global-wheat.com/gwhd.html)

### Baseline methods

- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [SAHI: Slicing Aided Hyper Inference and Fine-tuning](https://arxiv.org/abs/2202.06934)
- [QueryDet: Cascaded Sparse Query](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)
- [RTMDet](https://arxiv.org/abs/2212.07784)

### Journal scope and metrics

- [Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture)
- [Biosystems Engineering](https://www.sciencedirect.com/journal/biosystems-engineering)
- [Precision Agriculture](https://link.springer.com/journal/11119)
- [Plant Phenomics](https://www.sciencedirect.com/journal/plant-phenomics)
- [Pest Management Science aims and scope](https://scijournals.onlinelibrary.wiley.com/journal/15264998/aims-and-scope)
- [Crop Protection](https://www.sciencedirect.com/journal/crop-protection)
- [Clarivate Journal Citation Reports](https://clarivate.com/academia-government/scientific-and-academic-research/research-discovery-and-workflow-solutions/journal-citation-reports/)
