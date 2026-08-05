# BCRS Target Failure Audit (E2.3) — Concat Dual-Evidence Spectral Results

**Model Experiment:** `bcrs_dual_evidence_concat_visdrone_yolov5m_test`  
**Budget Constraint:** $K=16$ patches (25% compute budget, Latency = 12.5ms)  
**Architecture:** Concat Evidence Segmenter (`[F_semantic, F_spectral]` 1x1 Joint Conv)  
**Prediction File:** `work_dirs/bcrs_dual_evidence_concat_visdrone_yolov5m_test/best_predictions.json`  
**Evaluated Images:** 548 VisDrone validation images  
**Total Predictions:** 190,706 bounding boxes  

---

## 1. Size-Bin Recall Breakdown Comparison (@ $K=16$ Budget)

| Size Category | Area Range | GT Count | Unsupervised $K=16$ | Gated Spectral $K=16$ | Semantic Only $K=16$ | **Concat Dual-Evidence $K=16$** | Recall Rate (%) | Target Recall Delta vs Semantic | Recall % Delta |
|---|---|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 (17.63%) | 2,260 (18.90%) | 2,443 (20.43%) | **3,249** | **27.18%** | **+806 targets** | **+6.75%** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 (21.14%) | 3,299 (22.55%) | 3,608 (24.66%) | **4,267** | **29.16%** | **+659 targets** | **+4.50%** |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 (24.33%) | 2,791 (25.13%) | 2,988 (26.91%) | **3,469** | **31.24%** | **+481 targets** | **+4.33%** |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 (31.84%) | 344 (32.21%) | 374 (35.02%) | **415** | **38.86%** | **+41 targets** | **+3.84%** |
| **TOTAL** | — | **38,759** | **8,243 (21.27%)** | **8,694 (22.43%)** | **9,413 (24.29%)** | **11,400** | **29.41%** | **+1,987 targets** | **+5.12%** |

---

## 2. Class-Bin Recall Breakdown (Concat Dual-Evidence @ $K=16$)

| Class Name | GT Count | Concat Recalled | Recall Rate (%) | Class Recall Gain vs Semantic Only |
|---|---|---|---|---|
| `pedestrian` | 8,844 | **2,301** | **26.02%** | +342 (+3.87%) |
| `people` | 5,125 | **1,230** | **24.00%** | +194 (+3.79%) |
| `bicycle` | 1,287 | **314** | **24.40%** | +67 (+5.21%) |
| `car` | 14,064 | **4,817** | **34.25%** | +923 (+6.56%) |
| `van` | 1,975 | **697** | **35.29%** | +169 (+8.56%) |
| `truck` | 750 | **225** | **30.00%** | -2 (-0.27%) |
| `tricycle` | 1,045 | **285** | **27.27%** | +31 (+2.96%) |
| `awning-tricycle` | 532 | **135** | **25.38%** | +21 (+3.95%) |
| `bus` | 251 | **84** | **33.47%** | +13 (+5.18%) |
| `motor` | 4,886 | **1,312** | **26.85%** | +229 (+4.68%) |

---

## 3. Detection Precision & mAP Metrics (@ `conf_thresh=0.25`, NMS IoU 0.5)

| Metric | Concat Dual-Evidence ($K=16$) | Baseline ESOD | Target | Observation |
|---|---|---|---|---|
| **Overall Precision (P)** | **61.50%** | 62.04% | $\ge 60.0\%$ | High Precision Preserved |
| **`car` Precision** | **81.87%** | — | — | Top Precision rigid category |
| **`motor` Precision** | **72.92%** | — | — | High Precision dense category |
| **`bus` Precision** | **72.73%** | — | — | High Precision vehicle category |
| **`pedestrian` Precision** | **69.57%** | — | — | High Precision non-rigid category |

---

## 4. Decisive Insight for BCRS Proposal (E2.3 Ablation Outcome)

1. **Concat Superiority Over Gated Fusion**:
   Concat Feature Fusion significantly outperforms Sigmoid Gated Evidence Fusion (+6.98% overall recall, +8.28% Very Tiny recall) and Semantic Single-Evidence (+5.12% overall recall, +6.75% Very Tiny recall).
2. **Elimination of Zero-Sum Gate Constraint**:
   Sigmoid Gated Fusion ($P_{\text{fused}} = g \cdot P_{\text{semantic}} + (1-g) \cdot P_{\text{spectral}}$) enforces a zero-sum trade-off where elevating spectral evidence suppresses semantic evidence. Concat Fusion allows joint non-linear feature interaction, enabling the selector to activate whenever *either* semantic objectness or high-frequency spectral edge features respond.
3. **High Detection Precision Protection**:
   Despite recovering +1,987 additional GT targets (+806 Very Tiny targets), the model maintains an overall **61.50% BBox Precision**, proving that the Concat selector does not flood the downstream backbone with false positives.

