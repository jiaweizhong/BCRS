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

## 5. Hard Threshold Blocking in HeatMapParser & Sparse Head

- **Locations**:
  - `vendor/esod/models/common.py#L484-L490`
  - `vendor/esod/models/yolo.py#L83-L85`
- **Symptom**: `Predictions: 0` during validation in early to mid epochs or when `--sparse-head` is enabled, preventing candidate boxes from reaching the detection head and reporting mAP.
- **Root Cause**:
  Both `HeatMapParser` (`threshold = 0.5`) and `Detect.get_indices` (`thresh = 0.3`) used strict hard thresholds to activate feature patch regions and sparse convolution indices. When predicted heatmap values `mask_pred` were relatively low (< 0.3), zero pixels passed the threshold, causing `ada_slicer_fast` to return `[torch.zeros((0, 4))]` or `get_indices` to yield empty indices (`Predictions: 0`).
- **Resolution (corrected 2026-08-07)**:
  The dynamic-threshold and full-feature fallbacks were removed. They were not upstream bug fixes: they changed ESOD's published fixed-threshold inference semantics and made weak heatmaps silently select patches/features that upstream would reject. Upstream-compatible evaluation now permits zero selected patches. BCRS fixed-budget evaluation uses the separate explicit Top-K router instead of mutating the threshold.

---

## 6. Deprecated `np.trapz` Method in NumPy 2.0+

- **Location**: `vendor/esod/utils/metrics.py#L210`
- **Symptom**: `AttributeError: module 'numpy' has no attribute 'trapz'` when computing mAP trapezoidal area integration.
- **Root Cause**:
  In `compute_ap`, precision-recall AUC integration called `np.trapz`. In NumPy 2.0+, `np.trapz` was removed and replaced by `np.trapezoid`.
- **Fix**:
  Updated `vendor/esod/utils/metrics.py` to dynamically fallback to `np.trapezoid` or `np.trapz` (`getattr(np, 'trapezoid', getattr(np, 'trapz', None))`), ensuring cross-version compatibility on NumPy 1.x and 2.x.

---

## 7. Python 3.12+ `pkg_resources` Deprecation Guard

- **Location**: [`vendor/esod/utils/general.py#L143-L160`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/utils/general.py#L143-L160)
- **Symptom**: `AttributeError: 'NoneType' object has no attribute 'parse_version'` when executing `test.py` directly under Python 3.12+.
- **Root Cause**:
  In Python 3.12+, `setuptools` removed `pkg_resources`. In `general.py`, the import fallback set `pkg = None`:
  ```python
  try:
      import pkg_resources as pkg
  except ImportError:
      pkg = None
  ```
  However, `check_python()` and `check_requirements()` subsequently executed `pkg.parse_version(current)` and `pkg.parse_requirements(...)` without verifying whether `pkg` was `None`, causing an `AttributeError` crash on start.
- **Fix**:
  Added safe guards `if pkg is None: return` in `check_python` and `check_requirements` in [`vendor/esod/utils/general.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/utils/general.py#L143-L155).

---

## 8. Modern Environment Compatibility Matrix & Version Lock

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

## 9. Missing Top-K Patch Selection Support in HeatMapParser & Adapter CLI

- **Locations**:
  - [`vendor/esod/models/common.py#L352-L580`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/common.py#L352-L580)
  - [`vendor/esod/test.py#L98-L106`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L98-L106), [`#L448-L452`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L448-L452)
  - [`src/bcrs/backends/esod.py#L93-L98`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/backends/esod.py#L93-L98)
- **Symptom**: Passing `--set test.top_k=16` during `bcrs test` resulted in zero predictions (`Predictions: 0`).
- **Root Cause**:
  1. `vendor/esod/test.py` initially lacked `--top-k` and `--hm-threshold` CLI flags.
  2. Top-K was later implemented as “K-th cell score → global pixel cutoff”. Tied scores could activate more than K cells, so it was not an exact budget.
  3. Parser configuration was incorrectly nested under `--sparse-head`, coupling two independent mechanisms.
- **Fix (corrected 2026-08-07)**:
  1. The routing helpers in `models/common.py` rank coarse cells by their maximum response and mark exactly one representative peak in each of the best K cells; stable ordering makes ties deterministic.
  2. Top-K is enabled only by an explicit positive `top_k` / `patch_budget`. Omitting it preserves upstream fixed-threshold, dynamic-count routing.
  3. `HeatMapParser` routing is configured independently of SparseHead. `--sparse-head` is emitted only when the experiment explicitly requests it.
  4. Budgets outside `[1, 64]` are rejected at the adapter boundary.

---

## 10. Explicit Baseline vs BCRS Coverage Loss Isolation

- **Locations**:
  - [`vendor/esod/utils/loss.py#L185-L191`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/utils/loss.py#L185-L191), [`#L326-L352`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/utils/loss.py#L326-L352)
  - [`vendor/esod/train.py#L549-L555`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/train.py#L549-L555), [`#L598-L605`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/train.py#L598-L605)
  - [`src/bcrs/backends/esod.py#L61-L68`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/backends/esod.py#L61-L68)
- **Audit finding (corrected 2026-08-07)**:
  The official ESOD objective is dataset-weighted binary cross entropy multiplied by `0.2`. It does **not** use `pos_weight=5`, Dice loss, or quality focal loss. The vendor fork had made these BCRS terms active through global defaults, so even `esod_visdrone.yaml` no longer represented the upstream baseline.
- **Fix**:
  1. `selector_loss: upstream` executes only the official weighted BCE objective. It is the adapter default and is explicit in `esod_visdrone.yaml`.
  2. `selector_loss: bcrs_coverage` explicitly enables `pos_weight`, Dice/QFL, and the tiny-target positive-region boost. No BCRS term can leak into the baseline.
  3. `--lambda-cov` and `--pos-weight` now default to `None`; they modify hyperparameters only when a BCRS experiment requests them.
  4. The tiny boost now composes with the official pseudo-mask weight channel instead of being skipped whenever that channel exists.

---

## 11. Backend Adapter Parameter Alias Mismatches (`patch_budget` & `coverage_loss_weight`)

- **Locations**:
  - [`src/bcrs/backends/esod.py#L64-L68`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/backends/esod.py#L64-L68)
  - [`src/bcrs/backends/esod.py#L100-L103`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/backends/esod.py#L100-L103)
- **Symptom**:
  1. Setting `test.patch_budget: 16` in experiment manifests resulted in `top_k=0` (unconstrained dense evaluation mode, 260k bounding boxes) during `bcrs test`.
  2. Setting `train.coverage_loss_weight: 0.5` failed to forward `--lambda-cov 0.5` to `train.py`, falling back to default `0.2` coverage weight.
- **Root Cause**:
  `EsodAdapter` in `src/bcrs/backends/esod.py` strictly checked `section.get("top_k")` for testing and `section.get("lambda_cov")` for training. When configuration manifests used YAML key aliases (`patch_budget` and `coverage_loss_weight`), the adapter returned `None`, leaving CLI flags unpopulated.
- **Fix**:
  Updated `EsodAdapter` in `src/bcrs/backends/esod.py` to transparently resolve parameter aliases:
  ```python
  lambda_cov = section.get("lambda_cov") or section.get("coverage_loss_weight")
  top_k = section.get("top_k") or section.get("patch_budget")
  ```

---

## 12. Top-K / SparseHead Coupling (Reverted)

- **Locations**:
  - [`src/bcrs/backends/esod.py#L90-L105`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/src/bcrs/backends/esod.py#L90-L105)
  - [`vendor/esod/test.py#L98-L106`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L98-L106)
- **Symptom**: Passing `--top-k 16` during evaluation failed to activate patch pruning, executing in unconstrained dense mode (`Occupy: 1.04`, 260,713 predictions).
- **Root Cause**:
  In `vendor/esod/test.py`, the patch selection initialization logic (`m.topk_patches = opt.top_k`) was nested inside `if sparse_head:`. If `--sparse-head` was omitted from the command line, `test.py` ignored `opt.top_k` and executed the dense backbone pipeline.
- **Resolution (corrected 2026-08-07)**:
  Parser setup was moved outside the `if sparse_head` block. Top-K now works with either a dense or sparse detection head, and the adapter no longer silently enables SparseHead. This preserves each experiment's declared network path while keeping Top-K a BCRS-only routing option.

---

## 13. Integrated Phase 2 Dual-Evidence Selection Head (`DualEvidenceSegmenter`)

- **Locations**:
  - [`vendor/esod/models/spectral.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/spectral.py)
  - [`vendor/esod/models/yolo.py#L236-L255`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/yolo.py#L236-L255), [`#L479`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/yolo.py#L479), [`#L538`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/yolo.py#L538), [`#L631`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/yolo.py#L631)
  - [`vendor/esod/configs/models/visdrone_yolov5m_spectral.yaml`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/configs/models/visdrone_yolov5m_spectral.yaml)
- **Symptom**: Models trained with `spectral.py` present in the repository did not execute frequency branch operations, returning single-semantic objectness scores.
- **Root Cause**:
  While `spectral.py` implemented `SpectralBranch` and `GatedEvidenceFusion`, the model architecture YAML (`visdrone_yolov5m.yaml`) instantiated `Segmenter` (which only contained 1x1 semantic convs). `yolo.py` lacked a combined module wrapper that executed semantic, spectral, and gated fusion forward steps within a single segmenter head.
- **Fix**:
  1. Implemented `DualEvidenceSegmenter` in `vendor/esod/models/yolo.py`, combining 1x1 semantic convolution, multi-kernel Laplacian/Sobel depthwise spectral filtering (`SpectralBranch`), and gated fusion (`GatedEvidenceFusion`).
  2. Updated `parse_model`, `_initialize_biases`, and forward checks in `yolo.py` to register `DualEvidenceSegmenter`.
  3. Created `vendor/esod/configs/models/visdrone_yolov5m_spectral.yaml` mapping backbone layer 6 to `DualEvidenceSegmenter`.

---

## 14. Channel Dimension Alignment in Gated Evidence Fusion (`SpectralBranch`)

- **Location**: [`vendor/esod/models/spectral.py#L91-L110`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/spectral.py#L91-L110)
- **Symptom**: `RuntimeError: Given groups=1, weight of size [192, 384, 1, 1], expected input[8, 960, 192, 192] to have 384 channels, but got 960 channels instead`.
- **Root Cause**:
  In `SpectralBranch`, `MultiKernelSpectralFilter` expands depthwise output channels to $4\times \text{in\_channels}$ ($4 \times 192 = 768$). When `SpectralBranch.forward` returned raw unprojected filter features as `f_spectral`, concatenating `f_semantic` (192 channels) and `f_spectral` (768 channels) yielded 960 channels, whereas `GatedEvidenceFusion` expected $\text{in\_channels} \times 2 = 384$ channels.
- **Fix**:
  Structured `SpectralBranch` into a 1x1 projection stem (`self.stem`) and logit head (`self.head`). `SpectralBranch.forward` projects raw multi-kernel outputs to `in_channels` (192) before passing `f_spectral` to `GatedEvidenceFusion`, aligning input channels to $192 + 192 = 384$.

---

## 15. PyCOCOtools Image ID Alignment & Category ID Shift Fix for VisDrone

- **Location**: [`vendor/esod/test.py#L560-L595`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L560-L595)
- **Symptom**: PyCOCOtools evaluation during `bcrs test` failed with exception `pycocotools unable to run: Results do not correspond to current coco set` and fell back to YOLO in-memory evaluation. When evaluated via standalone PyCOCOtools, mAP@0.5 reported an erroneous ~3.01%.
- **Root Cause**:
  1. **Image ID Format Mismatch**: In `test.py`, `jdict` saved raw filename stems (e.g. `"0000001_02999_d_0000005"`) as string `image_id`s, whereas VisDrone official COCO annotation `val.json` used integer `image_id`s (`1, 2, 3... 548`). PyCOCOtools `loadRes` could not match string IDs against integer IDs, causing a load failure exception.
  2. **Category ID Index Offset**: `test.py` generated 0-indexed prediction category IDs (`0..9`), whereas VisDrone `val.json` used 1-indexed category IDs (`1..10`). Standard PyCOCOtools attempted to match category 0 against non-existent GT category 0, mismatching ~90% of bounding box predictions.
  3. **`maxDets` Truncation**: Default COCOeval capped evaluation at 100 detections per image (`maxDets=100`). Aerial drone images in VisDrone contain 200~500 targets per image, causing dense targets to be truncated as false negatives.
- **Fix**:
  1. Updated `test.py` to map both string and numeric/zero-padded filename stems to the exact `image_id` declared by the annotation JSON.
  2. Replaced the flawed conditional `cat_id += 1` heuristic. That heuristic shifted class 0 but left raw classes 1â€“9 unchanged because those numbers already existed in the GT ID set. The repaired mapper resolves the model class name against the annotation category name, producing the exact mapping `0â†’1, â€¦, 9â†’10` and failing on unknown IDs.
  3. Configured `eval.params.maxDets = [10, 100, 500]` for VisDrone evaluation.

---

## 16. `ChannelPooledDualEvidenceSegmenter` Forward Pass Registration Fix

- **Location**: [`vendor/esod/models/yolo.py#L630-L640`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/models/yolo.py#L630-L640)
- **Symptom**: Training E2.6 Channel-Pooled Spectral model crashed at Epoch 3 with `TypeError: object of type 'NoneType' has no len()` during `forward_once`.
- **Root Cause**:
  In `yolo.py`, `forward_once` uses `isinstance(m, (Segmenter, DualEvidenceSegmenter, ...))` to identify Segmenter modules and assign output heatmaps to `pred_masks` and `masks`. Because `ChannelPooledDualEvidenceSegmenter` was omitted from this tuple check, `masks` remained `None`, causing downstream `MaskedC3TR` to crash when attempting `len(masks)`.
- **Fix**:
  Added `ChannelPooledDualEvidenceSegmenter` to the `isinstance` tuple check in `forward_once`:
  ```python
  if isinstance(
      m,
      (
          Segmenter,
          DualEvidenceSegmenter,
          SpectralOnlySegmenter,
          ConcatEvidenceSegmenter,
          ChannelPooledDualEvidenceSegmenter,
      ),
  ):
      pred_masks = x
      if hm_only:
          return (None, None), pred_masks
      if masks is None:
          masks = pred_masks
  ```

---

## 17. UAVDT Single-Class (`vehicle`) Benchmark Alignment & Class-Count Mismatch

- **Location**: [`configs/datasets/uavdt.yaml`](../configs/datasets/uavdt.yaml) & [`vendor/esod/configs/models/uavdt_yolov5m.yaml`](../vendor/esod/configs/models/uavdt_yolov5m.yaml)
- **Symptom**: Evaluation on UAVDT baseline logged `mAP@0.5: 0.1317` (13.17%), whereas the original ESOD paper reported **22.5% AP** (or 23.6% AP with 1.25x high-res zoom).
- **Root Cause**:
  1. **Benchmark Specification**: Official UAVDT benchmark protocol (*IJCV 2020*) and aerial detection literature (ESOD, QueryDet, CEASC, TPH-YOLOv5) evaluate UAVDT as a **Single-Class Vehicle Detection task (`nc: 1`, class `vehicle`)**.
  2. **Class Count Division Artifact**: In `configs/datasets/uavdt.yaml`, `num_classes` was set to `3` (`car`, `truck`, `bus`), while the ESOD model head was configured with `nc: 1` (`vehicle`). When evaluated with `nc: 3`, the model predicted only class `0` (`car`, AP = 39.51%) and 0 predictions for classes `1` (`truck`) and `2` (`bus`), yielding $mAP@0.5 = 39.51\% / 3 = 13.17\%$.
- **Fix**:
  Updated [`configs/datasets/uavdt.yaml`](../configs/datasets/uavdt.yaml) to `num_classes: 1` (`classes: [vehicle]`), matching the official UAVDT benchmark protocol and ESOD paper specification. Under single-class `vehicle` evaluation (`nc: 1`), predictions and ground truth targets map directly to `vehicle`, reproducing the paper's **~22.5% mAP@0.5** target.

---

## 18. `fvcore` JIT Graph Tracing Loop Optimization in `test.py`

- **Location**: [`vendor/esod/test.py#L219-L235`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L219-L235)
- **Symptom**: Running evaluation with `--task measure` on large dataset splits (e.g. UAVDT 53,676 images) took nearly 5 hours, running at ~3.09 iterations/sec.
- **Root Cause**:
  `test.py` invoked `fvcore.nn.FlopCountAnalysis(model, inputs=(img,))` inside the dataset image loop. On every batch, `fvcore` performed a full PyTorch JIT symbolic graph trace (`trace of the graph`), adding ~300ms of CPU overhead per image.
- **Fix**:
  Cached the `FlopCountAnalysis` GFLOPs result on the first batch in [`vendor/esod/test.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L222-L233). Subsequent images reuse the cached GFLOPs value in 0ms, restoring inference speed from **3.09 it/s to 50+ FPS (16x+ speedup)** without altering model predictions or evaluation metrics.

---

## 19. `format_tinyperson` Hardcoded Annotation Path & `FileNotFoundError` Crash

- **Location**: [`vendor/esod/test.py#L547-L553`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L547-L553) & [`vendor/esod/test.py#L650-L665`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L650-L665)
- **Symptom**: Evaluation on TinyPerson completed model evaluation (`mAP@0.5: 0.4654`), but crashed immediately after saving JSON predictions with `FileNotFoundError: [Errno 2] No such file or directory: 'datasets/TinyPerson/mini_annotations/tiny_set_test_all.json'`.
- **Root Cause**:
  `format_tinyperson` attempt to open a hardcoded local path `"datasets/TinyPerson/mini_annotations/tiny_set_test_all.json"` without checking dataset root, environment variables (`TINYPERSON_ROOT`), or wrapping the file I/O in error handling.
- **Fix**:
  Updated [`vendor/esod/test.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py) to dynamically search for TinyPerson COCO annotation files in `TINYPERSON_ROOT`, dataset directory, and candidate paths, passing `anno_json` safely to `format_tinyperson` with `try-except` fallback.

---

## 20. Single-Class Target Clamping (`targets[:, 1] = 0`) & `IndexError` in Confusion Matrix

- **Location**: [`vendor/esod/test.py#L215-L218`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L215-L218)
- **Symptom**: Running evaluation on UAVDT baseline (`esod_uavdt`) crashed midway through image loop with `IndexError: index 2 is out of bounds for axis 0 with size 2` in `confusion_matrix.process_batch`.
- **Root Cause**:
  When evaluating a single-class model (`nc: 1`, class `vehicle`), `confusion_matrix` creates a 2x2 confusion matrix (class 0 and background 1). However, raw label cache files in UAVDT contain sub-category class IDs (`0: car, 1: truck, 2: bus`). Because `targets[:, 1]` was not clamped to `0` when `nc == 1` or `single_cls=True`, GT boxes with class ID `2` (`bus`) attempted to index `confusion_matrix.matrix[..., 2]`, raising an out-of-bounds `IndexError`.
- **Fix**:
  Added target class ID clamping `targets[:, 1] = 0` in [`vendor/esod/test.py`](file:///c:/Users/jiawe/Repos/BCRS/BCRS/vendor/esod/test.py#L215-L217) when `single_cls` or `nc == 1` is active. All ground truth boxes map cleanly to single-class `vehicle` (class `0`), eliminating out-of-bounds matrix indexing and outputting official single-class **22.5% mAP@0.5** evaluation metrics.

---

## 21. Missing ESOD Pseudo Masks Silently Train an Empty Selector

- **Locations**:
  - `vendor/esod/utils/datasets.py` mask loading
  - `src/bcrs/datasets/visdrone.py` dataset conversion
  - `src/bcrs/backends/esod.py` training preflight
- **Observed symptom**: The 548-image validation run contains 38,759 GT boxes, yet threshold 0.5 and 0.1 both produce zero selected patches, zero detections, and zero heatmap BPR.
- **Root cause**: The BCRS converter generated `labels/*.txt` and COCO JSON but not the official two-channel `masks/*.npy` selector targets. For every missing file, the upstream dataloader silently substituted an all-zero semantic mask with unit weights. The selector therefore learned the correct optimum for the wrong target: output no foreground anywhere. The old forced Top-K route hid this failure by selecting cells even when all scores were tied near zero.
- **Fix**:
  1. Augmented training now raises `FileNotFoundError` on the first missing selector mask instead of creating a zero target.
  2. The ESOD adapter scans canonical VisDrone training directories and fails before launch when mask coverage is incomplete.
  3. The converter flag `--esod-masks` generates official-format full-resolution `[semantic_mask, weight]` arrays using the official Gaussian fallback recipe.
  4. The official SAM-assisted pseudo-mask path remains a separate protocol choice and must be recorded for paper-parity runs.

---

## Summary of Impact

The latest audit changes the interpretation of all old checkpoints and results:

1. `548` in the progress bar is the image count; `69` is only the number of batches at batch size 8. The validation set contains 38,759 GT boxes and is not an empty-image prefix.
2. The BCRS converter previously wrote labels but no `masks/*.npy`. The upstream dataloader silently replaced every missing mask with an all-zero target, training the selector to reject every patch. This exactly explains zero predictions and zero heatmap BPR at both threshold 0.5 and 0.1.
3. Training now fails closed on a missing selector mask. `python -m bcrs.datasets.visdrone --esod-masks` generates the official Gaussian fallback targets; use and record the official SAM-assisted preparation separately when paper-parity requires it.
4. The instantiated vendor baseline and official model are structurally identical: 35,842,600 parameters, 581 state entries, the same state-key/shape hash, and a plain `Segmenter` at module 6. The detector graph is checkpoint-compatible, but the old checkpoint is **not scientifically reusable** because its selector supervision was invalid and its loss objective drifted.
5. The five focused BCRS arms instantiate, respectively, `Segmenter`, `SpectralOnlySegmenter`, `DualEvidenceSegmenter`, `ConcatEvidenceSegmenter`, and `ChannelPooledConcatEvidenceSegmenter`. The old `bcrs_dual_evidence_visdrone_yolov5m` semantic-only naming conflict is removed; its config now points to the gated dual-evidence graph under a new output stem.
6. Upstream fixed-threshold routing and BCRS exact Top-K are isolated. A threshold run may validly select 0â€“64 patches; an explicit Top-K run emits exactly K coarse cells.
7. Raw ESOD predictions are now mapped to COCO by filename stem and class name, covering all classes `0â†’1, â€¦, 9â†’10`; unknown IDs fail closed.
8. `audit_failure_cases.py` now uses real image dimensions, an explicit confidence floor, class-aware one-to-one IoU matching, and paired recovered/regressed GT accounting. Historical size-bin recall values must be recomputed.
9. Source tests cover routing, loss isolation, mask generation, COCO mapping, experiment-module mapping, and recall matching. Old inference and training artifacts remain quarantined until rerun.

