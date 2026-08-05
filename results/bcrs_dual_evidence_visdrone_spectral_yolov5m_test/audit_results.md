# BCRS Phase 2 Dual-Evidence Spectral Gated Fusion — Top-16 Budget ($K=16$) Results

**Model Experiment:** `bcrs_dual_evidence_visdrone_spectral_yolov5m_test`  
**Budget Constraint:** $K=16$ patches (25% compute budget, Occupy ~ 0.29, Latency ~ 12.5ms)  
**Architecture:** `DualEvidenceSegmenter` (Semantic Objectness + Multi-kernel Laplacian/Sobel Spectral Branch + Gated Evidence Fusion)  
**Prediction File:** `results/bcrs_dual_evidence_visdrone_spectral_yolov5m_test/best_predictions.json`  
**Evaluated Images:** 548 VisDrone validation images  
**Total Predictions:** 164,925 bounding boxes  

---

## 1. Size-Bin Recall Breakdown (Unsupervised vs. Dual-Evidence Spectral @ $K=16$)

| Size Category | Area Range | GT Count | Unsupervised $K=16$ Recalled | Dual-Evidence Spectral $K=16$ Recalled | Recall Rate (%) | Target Recall Delta | Recall % Delta |
|---|---|---|---|---|---|---|---|
| **Very Tiny** | $< 16 \times 16\text{ px}$ | 11,955 | 2,108 (17.63%) | **2,260** | **18.90%** | **+152 targets** | **+1.27%** |
| **Tiny** | $16 \times 16 \sim 32 \times 32\text{ px}$ | 14,631 | 3,093 (21.14%) | **3,299** | **22.55%** | **+206 targets** | **+1.41%** |
| **Small** | $32 \times 32 \sim 96 \times 96\text{ px}$ | 11,105 | 2,702 (24.33%) | **2,791** | **25.13%** | **+89 targets** | **+0.80%** |
| **Medium / Large** | $> 96 \times 96\text{ px}$ | 1,068 | 340 (31.84%) | **344** | **32.21%** | **+4 targets** | **+0.37%** |
| **TOTAL** | — | **38,759** | **8,243 (21.27%)** | **8,694** | **22.43%** | **+451 targets** | **+1.16%** |

---

## 2. Class-Bin Recall Breakdown (Unsupervised vs. Dual-Evidence Spectral @ $K=16$)

| Class Name | GT Count | Unsupervised $K=16$ | Dual-Evidence Spectral $K=16$ | Recall Rate (%) | Target Delta |
|---|---|---|---|---|---|
| `pedestrian` | 8,844 | 1,787 (20.21%) | **1,714** | **19.38%** | -73 |
| `people` | 5,125 | 921 (17.97%) | **939** | **18.32%** | **+18** |
| `bicycle` | 1,287 | 224 (17.40%) | **246** | **19.11%** | **+22** |
| `car` | 14,064 | 3,286 (23.36%) | **3,666** | **26.07%** | **+380** |
| `van` | 1,975 | 434 (21.97%) | **483** | **24.46%** | **+49** |
| `truck` | 750 | 224 (29.87%) | **213** | **28.40%** | -11 |
| `tricycle` | 1,045 | 218 (20.86%) | **229** | **21.91%** | **+11** |
| `awning-tricycle` | 532 | 83 (15.60%) | **95** | **17.86%** | **+12 (+2.26%)** |
| `bus` | 251 | 48 (19.12%) | **60** | **23.90%** | **+12 (+4.78%)** |
| `motor` | 4,886 | 1,018 (20.84%) | **1,049** | **21.47%** | **+31** |
