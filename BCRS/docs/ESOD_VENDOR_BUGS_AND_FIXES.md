# ESOD Vendor Source Code Bugs and Technical Fixes

This document records technical bugs discovered in the ESOD vendor source code (`vendor/esod`) and dataset pipeline during integration into BCRS, along with their root cause analyses and resolution details.

---

## 1. Variable Shadowing Overwrites Detection Precision & Recall

- **Location**: `vendor/esod/test.py#L305-L311`
- **Symptom**: During validation, `Precision` and `Recall` metrics in the output table were consistently forced to `0.0`, even when bounding box detection models were converging.
- **Root Cause**:
  In `test.py`, line 298 calculates bounding box detection mean precision (`mp`) and mean recall (`mr`):
  ```python
  mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
  ```
  However, when the `--hm-metric` flag is enabled, lines 309-310 re-assigned local scalar metrics for heatmap mask precision (`m_p`) and mask recall (`m_r`) to the same variable names `mp` and `mr`:
  ```python
  mp = m_p.mean().item() if len(m_p) else 0.0
  mr = m_r.mean().item() if len(m_r) else 0.0
  ```
  During early training epochs, heatmap predictions were unpopulated (`len(m_p) == 0`), evaluating to `0.0`. This shadowed and completely overwritten the bounding box detection `mp` and `mr` values previously calculated.
- **Fix**:
  Renamed heatmap mask scalar metrics to `hm_p` and `hm_r` in `vendor/esod/test.py`:
  ```python
  hm_p = m_p.mean().item() if len(m_p) else 0.0
  hm_r = m_r.mean().item() if len(m_r) else 0.0
  ```

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

## Summary of Impact

With these fixes applied:
1. Ground truth label counts report accurately as **38,759** across all validation passes.
2. Precision and Recall metrics display correctly without being overwritten by uninitialized heatmap metrics.
3. Feature patches are dynamically routed to detection heads in early epochs, eliminating `Predictions: 0` deadlocks.
4. Vendor code synchronization passes all sha256 integrity checks (`verified=93 failures=0`).

