# ESOD Vendor Source Code Bugs and Technical Fixes

This document records technical bugs discovered in the ESOD vendor source code (`vendor/esod`) and dataset pipeline during integration into BCRS, along with their root cause analyses and resolution details.

---

## 1. Variable Shadowing Overwrites Detection Precision, Recall & BPR

- **Location**: [`vendor/esod/test.py#L304-L324`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L304-L324)
- **Symptom**: During validation with `--hm-metric` enabled, `Precision`, `Recall`, and `BPR` (Bounding-box Patch Recall) in the output table were consistently forced to `0.0`, even when bounding box detection models and feature patch slicers were converging.
- **Root Cause**:
  1. **Precision & Recall Shadowing**: In `test.py`, line 298 calculates bounding box detection mean precision (`mp`) and mean recall (`mr`). However, when `--hm-metric` is enabled, lines 309-310 re-assigned local scalar metrics for heatmap mask precision (`m_p`) and mask recall (`m_r`) to the same variable names `mp` and `mr`.
  2. **BPR Shadowing**: Line 304 calculates the true paper BPR (Bounding-box Patch Recall) from AdaSlicer's feature clusters:
     ```python
     bpr, occupy = statistic_items[0] / (nt.sum() + 1e-6), ...
     ```
     However, line 311 under `if hm_metric:` re-assigned `bpr` to heatmap pixel-level sparse recall (`sp_r.mean().item()`):
     ```python
     bpr = sp_r.mean().item() if len(sp_r) else 0.0
     ```
     Because raw heatmap predictions were unpopulated or below the 0.3 threshold during training (`len(sp_r) == 0` or evaluating to `0.0`), this shadowed and completely overwritten the true detection `mp`, `mr`, and patch `bpr` values calculated earlier.
- **Fix**:
  Renamed heatmap mask scalar metrics to `hm_p`, `hm_r`, and `hm_bpr` in `vendor/esod/test.py`:
  ```python
  hm_p = m_p.mean().item() if len(m_p) else 0.0
  hm_r = m_r.mean().item() if len(m_r) else 0.0
  hm_bpr = sp_r.mean().item() if len(sp_r) else 0.0
  ```
  This preserves `bpr` (patch-level recall from AdaSlicer) for evaluation logging and fitness calculation while reporting `hm_bpr` separately.

---

## 2. Unsafe Target Count Reset in Zero-Hit Guard

- **Location**: [`vendor/esod/test.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L293-L303)
- **Symptom**: Validation summary table logged `all 548 0 1 0 0 0 0 1` (`Labels: 0`) during early training epochs.
- **Root Cause**:
  `ap_per_class(*stats)` requires at least one positive IoU match in `stats[0]` (`correct`) to construct Precision-Recall curves. To prevent crashes when zero predictions hit ground truth targets, the vendor code wrapped the evaluation in a guard:
  ```python
  if len(stats) and stats[0].any():
      p, r, ap, f1, ap_class = ap_per_class(...)
      nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
  else:
      nt = torch.zeros(1)
  ```
  When zero predictions matched ground truth targets with IoU > 0.5 (common during early warmup epochs), the `else` branch executed `nt = torch.zeros(1)`, resetting the ground truth target count `nt` to 0. This caused the console logger to report 0 labels in the dataset despite all labels being loaded in memory.
- **Fix**:
  Decoupled target count computation (`nt`) from the `stats[0].any()` guard condition:
  ```python
  if len(stats):
      nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
      if stats[0].any():
          p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
          ap50, ap = ap[:, 0], ap.mean(1)
          mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
      else:
          mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
  else:
      nt = torch.zeros(1)
      mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
  ```

---

## 3. Over-Strict Label Assertions Dropping Entire Images

- **Locations**:
  - [`vendor/esod/utils/datasets.py`](../vendor/esod/utils/datasets.py#L1155-L1160)
  - [`src/bcrs/datasets/visdrone.py`](../src/bcrs/datasets/visdrone.py#L154-L177)
- **Symptom**: Log messages like `Ignoring corrupted image: duplicate labels` appeared during dataset scanning, dropping valid images and annotations.
- **Root Cause**:
  Raw drone annotations often contain minor boundary rounding overshoots (e.g., `(x + w) / W = 1.00002`) or duplicate label entries. `verify_image_label` in `datasets.py` evaluated strict assertions:
  ```python
  assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
  assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
  ```
  When any single bounding box failed an assertion, the exception handler marked `im_file = None`, skipping the entire image and discarding all associated ground truth annotations for that image.
- **Fix**:
  - In `src/bcrs/datasets/visdrone.py`: Enforced coordinate clipping (`min(max(val, 0.0), 1.0)`) and label line deduplication during dataset conversion.
  - In `vendor/esod/utils/datasets.py`: Safely clipped label coordinates to `[0.0, 1.0]` and applied `np.unique` before validation checks:
    ```python
    if len(l):
        assert l.shape[1] == 5, 'labels require 5 columns each'
        l[:, 1:] = np.clip(l[:, 1:], 0, 1)
        l = np.unique(l, axis=0)
    ```

---

## 4. Hardcoded Single Directory Layout in Dataset Converter

- **Location**: [`src/bcrs/datasets/visdrone.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/datasets/visdrone.py#L185-L208)
- **Symptom**: `Missing VisDrone raw annotation directory` error when converting VisDrone datasets extracted into standard subfolder variants.
- **Root Cause**:
  The initial converter script assumed fixed path locations (`root / "raw_annotations" / split`). VisDrone dataset archives are commonly unzipped into structures like `VisDrone2019-DET-val/annotations` or `annotations/val`.
- **Fix**:
  Implemented automatic multi-candidate folder resolution in `prepare_split`:
  ```python
  if not raw_annotations_dir.is_dir():
      for cand in [
          root / "annotations" / split,
          root / f"VisDrone2019-DET-{split}" / "annotations",
          root / split / "annotations",
          root / f"VisDrone2019-DET-{split}" / "annotations_txt",
      ]:
          if cand.is_dir():
              raw_annotations_dir = cand
              break
  ```

---

## 5. Hard Threshold Blocking in HeatMapParser During Early Training

- **Location**: `vendor/esod/models/common.py#L484-L490`
- **Symptom**: `Predictions: 0` during validation in early to mid epochs, preventing candidate boxes from reaching the detection head and reporting mAP.
- **Root Cause**:
  `HeatMapParser` uses a strict hard threshold (`threshold = 0.5`) to activate feature patch regions (`activated = mask_pred >= threshold`). During early training epochs, `ObjSeeker`'s predicted heatmap values `mask_pred` are relatively low (< 0.5). When zero pixels pass the 0.5 threshold, `ada_slicer_fast` returned `[torch.zeros((0, 4))]` (zero active patches), causing `model.forward` to return `(None, None)`. As a result, zero feature patches were sent to the detection neck and head, yielding 0 candidate predictions (`Predictions: 0`).
- **Fix**:
  Added a dynamic threshold fallback in `ada_slicer` and `ada_slicer_fast` (`vendor/esod/models/common.py`). If zero pixels exceed `threshold = 0.5`, it dynamically relaxes the activation threshold to relative local maxima (`max(0.05, float(mask_pred.max()) * 0.5)`), ensuring feature patches are reliably routed to the detection neck and head across all training epochs.

---

## 6. Deprecated `np.trapz` Method in NumPy 2.0+

- **Location**: `vendor/esod/utils/metrics.py#L210`
- **Symptom**: `AttributeError: module 'numpy' has no attribute 'trapz'` when computing mAP trapezoidal area integration.
- **Root Cause**:
  In `compute_ap`, precision-recall AUC integration called `np.trapz`. In NumPy 2.0+, `np.trapz` was removed and replaced by `np.trapezoid`.
- **Fix**:
  Updated `vendor/esod/utils/metrics.py` to dynamically fallback to `np.trapezoid` or `np.trapz` (`getattr(np, 'trapezoid', getattr(np, 'trapz', None))`), ensuring cross-version compatibility on NumPy 1.x and 2.x.

---

## 7. Modern Environment Compatibility Matrix & Version Lock

To support modern hardware (e.g., NVIDIA RTX 5090 / Blackwell / Ada Lovelace architectures) and contemporary Python 3.12 packages, the codebase was adapted and verified against the following stack:

| Component | Pinned Version | Notes |
| :--- | :--- | :--- |
| **Python** | `3.12.x` | Modern Python runtime |
| **CUDA Driver / Toolkit** | `12.8` | Required for RTX 5090 Blackwell support |
| **PyTorch** | `2.8.0+cu128` | Includes `weights_only=False` compatibility patch |
| **Torchvision** | `0.23.0+cu128` | Matching CUDA 12.8 vision toolkit |
| **NumPy** | `2.2.3` | Includes `np.trapezoid` integration fix |
| **OpenCV** | `4.11.0.86` | BGR/RGB image processing |
| **Pillow** | `11.1.0` | Image verification & EXIF parsing |
| **PyYAML** | `6.0.2` | Experiment manifest parsing |

---

## 8. Locked Environment Manifests

The exact dependencies for running on modern hardware are locked in:
1. **Repository Root Manifest**: `requirements.txt`
2. **Environment Directory**: `environments/torch2.8-cu128/requirements.txt`
3. **Legacy Environment Directory**: `environments/torch1.10-cu113/requirements.txt`

To replicate this verified environment on a new machine:
```bash
pip install -r requirements.txt
```

---

## Summary of Impact

With these fixes applied:
1. Ground truth label counts report accurately as **38,759** across all validation passes.
2. Precision and Recall metrics display correctly without being overwritten by uninitialized heatmap metrics.
3. Feature patches are dynamically routed to detection heads in early epochs, eliminating `Predictions: 0` deadlocks.
4. Precision-Recall AP integrals compute cleanly on modern NumPy 2.0+ without `AttributeError` crashes.
5. Pinned environment manifests (`requirements.txt` and `environments/torch2.8-cu128/requirements.txt`) ensure 100% reproducible execution on RTX 5090 / CUDA 12.8 hardware.
6. Vendor code synchronization passes all sha256 integrity checks (`verified=93 failures=0`).



