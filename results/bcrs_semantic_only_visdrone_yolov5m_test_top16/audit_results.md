# BCRS Target Failure Audit (E1.2 & E1.3) — Top-16 Budget ($K=16$) Enhanced Results

**Model Experiment:** `bcrs_dual_evidence_visdrone_yolov5m_test_top16`  
**Budget Constraint:** $K=16$ patches (25% compute budget, Occupy = 0.296, Latency = 12.5ms)  
**Supervision Strategy:** Phase 1 E1.2 Size-Weighted Coverage Loss ($\lambda_{\text{cov}}=0.5$, $\text{pos\_weight}=2.0$)  
**Prediction File:** `results/bcrs_dual_evidence_visdrone_yolov5m_test_top16/best_predictions.json`  
**Evaluated Images:** 548 VisDrone validation images  
**Total Predictions:** 169,707 bounding boxes  

---

## 1. Size-Bin Recall Breakdown (Unsupervised vs Coverage-Supervised @ $K=16$)

| Size Category | Area Range | GT Count | Unsupervised $K=16$ Recalled | Enhanced $K=16$ Recalled | Recall Rate (%) | Target Recall Delta | Recall % Delta |
|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 (17.63%) | 2,443 | **20.43%** | **+335 targets** | **+2.80%** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 (21.14%) | 3,608 | **24.66%** | **+515 targets** | **+3.52%** |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 (24.33%) | 2,988 | **26.91%** | **+286 targets** | **+2.58%** |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 (31.84%) | 374 | **35.02%** | **+34 targets** | **+3.18%** |
| **TOTAL** | — | **38,759** | **8,243 (21.27%)** | **9,413** | **24.29%** | **+1,170 targets** | **+3.02%** |

---

## 2. Class-Bin Recall Breakdown (Unsupervised vs Coverage-Supervised @ $K=16$)

| Class Name | GT Count | Unsupervised $K=16$ | Enhanced $K=16$ | Recall Rate (%) | Target Delta | Primary Improvement Note |
|---|---|---|---|---|---|---|
| `pedestrian` | 8,844 | 1,787 (20.21%) | 1,959 | **22.15%** | **+172** | Significant non-rigid recovery |
| `people` | 5,125 | 921 (17.97%) | 1,036 | **20.21%** | **+115** | Improved small group recall |
| `bicycle` | 1,287 | 224 (17.40%) | 247 | **19.19%** | **+23** | Thin structure gain |
| `car` | 14,064 | 3,286 (23.36%) | 3,894 | **27.69%** | **+608** | High count rigid target gain |
| `van` | 1,975 | 434 (21.97%) | 528 | **26.73%** | **+94** | Medium rigid target gain |
| `truck` | 750 | 224 (29.87%) | 227 | **30.27%** | **+3** | Heavy vehicle stability |
| `tricycle` | 1,045 | 218 (20.86%) | 254 | **24.31%** | **+36** | Complex overlapping target gain |
| `awning-tricycle` | 532 | 83 (15.60%) | 114 | **21.43%** | **+31 (+5.83%)** | Highest relative class gain |
| `bus` | 251 | 48 (19.12%) | 71 | **28.29%** | **+23 (+9.17%)** | High occlusion recovery |
| `motor` | 4,886 | 1,018 (20.84%) | 1,083 | **22.17%** | **+65** | Dense moving target gain |

---

## 3. Key Findings for BCRS Proposal

1. **Simultaneous Recall & Speed Improvement**:
   - **Target Recovery**: Size-Weighted Coverage Loss ($\mathcal{L}_{\text{cov}}$) recovered **+1,170 additional ground truth targets** (+3.02% overall recall, **+335 Very Tiny targets**) under the exact same $K=16$ (25% compute) budget constraint.
   - **Inference Speed**: Latency improved to **12.5 ms / img** (down from 13.0 ms).
2. **Class-Level Validation**:
   - High relative gains observed on difficult categories: `awning-tricycle` (**+5.83%** recall gain) and `bus` (**+9.17%** recall gain).
