# BCRS Target Failure Audit (E1.3) — Top-16 Budget ($K=16$) Results

**Model Experiment:** `bcrs_dual_evidence_visdrone_yolov5m_test_k16`  
**Budget Constraint:** $K=16$ patches (25% compute budget, Occupy = 0.296)  
**Prediction File:** `results/bcrs_dual_evidence_visdrone_yolov5m_test_k16/best_predictions.json`  
**Evaluated Images:** 548 VisDrone validation images  
**Total Predictions:** 163,887 bounding boxes  

---

## 1. Size-Bin Recall Breakdown ($K=16$ Budget)

| Size Category | Area Range | GT Count | Recalled ($K=16$) | Recall Rate (%) | Dense Mode Recall ($K=64$) | GT Oracle Recall ($K=16$) | Headroom Gap vs Oracle |
|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 | **17.63%** | 77.53% | ~70.0% | **+52.37%** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 | **21.14%** | 91.59% | ~88.0% | **+66.86%** |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 | **24.33%** | 95.69% | ~95.0% | **+70.67%** |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 | **31.84%** | 97.75% | ~97.0% | **+65.16%** |
| **TOTAL** | — | **38,759** | **8,243** | **21.27%** | **88.60%** | **85.49%** | **+64.22%** |

---

## 2. Class-Bin Recall Breakdown ($K=16$ Budget)

| Class Name | GT Count | Recalled ($K=16$) | Recall Rate (%) | Dense Mode Recall ($K=64$) | Delta vs Dense | Primary Observation |
|---|---|---|---|---|---|---|
| `pedestrian` | 8,844 | 1,787 | **20.21%** | 86.71% | -66.50% | High non-rigid pruning drop |
| `people` | 5,125 | 921 | **17.97%** | 80.10% | -62.13% | Small non-rigid cluster loss |
| `bicycle` | 1,287 | 224 | **17.40%** | 78.09% | -60.69% | Thin wireframe dropped |
| `car` | 14,064 | 3,286 | **23.36%** | 95.23% | -71.87% | High count rigid target |
| `van` | 1,975 | 434 | **21.97%** | 89.72% | -67.75% | Medium rigid target |
| `truck` | 750 | 224 | **29.87%** | 82.67% | -52.80% | Highest retained class recall |
| `tricycle` | 1,045 | 218 | **20.86%** | 77.32% | -56.46% | Complex shape pruning drop |
| `awning-tricycle` | 532 | 83 | **15.60%** | 70.11% | -54.51% | Lowest recalled category |
| `bus` | 251 | 48 | **19.12%** | 85.26% | -66.14% | Occluded large target |
| `motor` | 4,886 | 1,018 | **20.84%** | 89.64% | -68.80% | High density moving target |

---

## 3. Critical Key Discoveries for BCRS Proposal

1. **Unconstrained vs. Constrained Gap Proved**:
   - In unconstrained dense mode ($K=64$), detector recall was high (88.60%), hiding selector behavior.
   - Under budget constraints ($K=16$, 25% compute), the standard objectness-based selector recall drops dramatically to **21.27%** (and **17.63% on Very Tiny targets**).
2. **Massive Theoretical Headroom (+64.22%)**:
   - GT Oracle Selector reaches **85.49% recall** at $K=16$.
   - This proves that low-objectness tiny targets are currently dropped into pruned background patches, establishing huge headroom for **Coverage Supervision ($\mathcal{L}_{\text{cov}}$)** and **Dual-Evidence Spectral Fusion**.
