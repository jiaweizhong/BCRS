# ESOD Baseline Patches — Local Modifications for Modern-Environment Reproduction

This document records every local change made to the `esod/` checkout at the repo root (a pristine clone of the official [alibaba/esod](https://github.com/alibaba/esod)) so that the paper's YOLOv5m baseline can be trained and evaluated end-to-end on a modern stack (Python 3.12, recent CUDA/PyTorch) for VisDrone / UAVDT / TinyPerson.

**Scope boundary:** `esod/` is deliberately kept as close to upstream as possible. Every entry below is either (a) a genuine upstream bug that crashes or silently misbehaves under a modern environment, or (b) a compatibility shim — never a change to ESOD's published algorithm, loss, routing, or architecture. Algorithmic modifications (the BCRS/HESOD dual-evidence selector variants) live entirely in the separate `BCRS/vendor/esod` fork and are out of scope for this document; see `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` for that fork's history instead.

**Provenance note:** `esod/` was originally its own git clone (its own `.git`, tracked separately from this repo) and was flattened into plain files under this repo on 2026-08-08 so a single `git clone`/`git pull` of this project reliably brings the full source along (see "Pre-flatten commit history" below for what that clone's own history contained before it was discarded).

---

## 1. `pkg_resources` crashes on Python 3.12

- **Location:** `esod/utils/general.py` (module-level import, `check_python()`, `check_requirements()`)
- **Symptom:** Any entrypoint that imports `utils.general` (`train.py`, `test.py`, `scripts/data_prepare.py`, ...) crashes immediately:
  ```
  File ".../pkg_resources/__init__.py", line 2172, in <module>
      register_finder(pkgutil.ImpImporter, find_on_path)
  AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
  ```
- **Root cause:** `general.py` did `import pkg_resources as pkg` unconditionally, then used it in `check_python()` / `check_requirements()`. Under Python 3.12 this fails two different ways depending on the installed `setuptools` version:
  - Newer `setuptools` has started removing `pkg_resources` outright (plain `ImportError`).
  - Older/pinned `setuptools` (e.g. the `59.5.0` the upstream README recommends) still ships `pkg_resources`, but that code calls `pkgutil.ImpImporter`, which Python 3.12 removed from the standard library — this raises `AttributeError` *from inside* `pkg_resources`'s own module-level init code, not an import failure.
  - Pinning to an old `setuptools` to match the README does **not** fix this on Python 3.12; it only reaches the second failure mode instead of the first.
- **Fix:**
  ```python
  try:
      import pkg_resources as pkg
  except (ImportError, AttributeError):
      pkg = None
  ```
  `check_python()` and `check_requirements()` both `return`/no-op immediately when `pkg is None`. Neither function is required for training/eval correctness — they only auto-check/auto-install Python version and pip requirements, which this project's own environment setup already handles.

## 2. `test.py --save-json` crashed before writing any predictions for UAVDT/TinyPerson

- **Location:** `esod/test.py`, "Save JSON" block (`if save_json and len(jdict): ...`)
- **Symptom:** `python test.py ... --save-json` on UAVDT raised `NotImplementedError: Ground-Truth file for uavdt.yaml not found.`; on TinyPerson it raised `FileNotFoundError: datasets/TinyPerson/mini_annotations/tiny_set_test_all.json`. In both cases **no prediction JSON was ever written**, even though model evaluation (native AP/AP50) had already completed successfully.
- **Root cause:** Upstream resolved a dataset-specific `anno_json` path (and, for TinyPerson, called `format_tinyperson(jdict)`) *before* writing `pred_json` to disk, and this resolution step was not part of the surrounding `try/except`:
  ```python
  if 'visdrone' in opt.data.lower():
      anno_json = './datasets/VisDrone/annotations/val.json'
  elif 'tinyperson' in opt.data.lower():
      anno_json = './datasets/TinyPerson/test.json'
      format_tinyperson(jdict)
  else:
      raise NotImplementedError(...)          # <- UAVDT always hits this
  pred_json = str(save_dir / f"{w}_predictions.json")
  with open(pred_json, 'w') as f:
      json.dump(jdict, f)
  try:                                          # pycocotools comparison; the only
      ...                                       #   part upstream treated as optional
  except Exception as e:
      print(f'pycocotools unable to run: \n{e}')
  ```
  Both hardcoded `anno_json` paths are also wrong for this project's dataset layout (`/root/autodl-tmp/...` rather than `./datasets/...`), so even on VisDrone the pycocotools comparison never actually ran — it just happened not to *crash* there, because that dataset's `else` branch wasn't hit and the exception landed inside the `try`.
- **Fix:** Write `pred_json` first, unconditionally, right after computing the weights stem. The entire dataset-specific `anno_json` resolution + `format_tinyperson()` + pycocotools comparison moved inside the pre-existing `try/except Exception`, so it's best-effort for all three datasets and can never prevent (or crash) the raw prediction dump:
  ```python
  pred_json = str(save_dir / f"{w}_predictions.json")
  with open(pred_json, 'w') as f:
      json.dump(jdict, f)
  try:
      from pycocotools.coco import COCO
      from pycocotools.cocoeval import COCOeval
      if 'visdrone' in opt.data.lower():
          anno_json = './datasets/VisDrone/annotations/val.json'
      elif 'tinyperson' in opt.data.lower():
          anno_json = './datasets/TinyPerson/test.json'
          format_tinyperson(jdict)
      else:
          raise NotImplementedError(...)
      ...
  except Exception as e:
      print(f'pycocotools unable to run: \n{e}')
  ```
  **Trade-off, stated explicitly:** `format_tinyperson(jdict)` mutates `jdict` in memory (remaps `image_id`, `category_id += 1`) for the *upstream* pycocotools comparison. Since it now runs after `pred_json` is already on disk, the saved file always stays in raw ESOD format (0-indexed `category_id`, filename-stem `image_id`) regardless of dataset — which is exactly the format downstream tooling (`scripts/esod_baseline/audit_buckets.py`) expects. The upstream pycocotools comparison for TinyPerson specifically now evaluates against a mutated in-memory copy while reading the unmutated file from disk via `anno.loadRes(pred_json)`, so that comparison is unreliable for TinyPerson — it was already effectively non-functional in this environment anyway (its `anno_json`/annotation path never matched this project's dataset layout), and it was always best-effort/non-fatal by design (wrapped in the same `try/except`).

## 3. `torch.load` rejects legacy checkpoints under PyTorch 2.6+

- **Location:** `esod/train.py`, `esod/test.py` (top-level, right after `import torch`)
- **Symptom:** `train.py` crashed on startup while loading the pretrained YOLOv5m init weights:
  ```
  File ".../torch/serialization.py", line 1529, in load
      raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
  _pickle.UnpicklingError: Weights only load failed. ...
  WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct
  was not an allowed global by default.
  ```
- **Root cause:** PyTorch 2.6 changed `torch.load`'s default from `weights_only=False` to `weights_only=True`. The restricted unpickler then refuses any checkpoint containing non-allowlisted globals — which includes the `ultralytics/yolov5` v5.0-era `yolov5m.pt` release (and this codebase's own `.pt` saves), since they pickle plain Python/numpy objects (e.g. `numpy.core.multiarray._reconstruct`), not just tensors. Both `train.py` (`torch.load(weights).get('wandb_id')`, then `torch.load(weights, map_location=device)`) and everything reachable from it (`models/experimental.py::attempt_load`, checkpoint resume, etc.) call `torch.load` without `weights_only=False`.
- **Fix:** Same pattern already used in `BCRS/vendor/esod/{train,test}.py` for this exact issue — monkey-patch `torch.load` once near the top of each entrypoint, right after `import torch`, so every call in that process defaults to `weights_only=False` unless the caller explicitly overrides it:
  ```python
  try:
      _orig_torch_load = torch.load

      def _compat_torch_load(*args, **kwargs):
          if "weights_only" not in kwargs:
              kwargs["weights_only"] = False
          return _orig_torch_load(*args, **kwargs)

      torch.load = _compat_torch_load
  except Exception:
      pass
  ```
  Applied to `train.py` and `test.py` only (the two entrypoints this project's pipeline actually runs); every other module that calls `torch.load` after either of those has started (`models/experimental.py`, `utils/datasets.py` cache loading, ...) shares the same patched `torch` module object via Python's module cache, so it needs no separate edit. `detect.py` is unpatched since it isn't part of the reproduction pipeline.
  - **Trust note:** this widens `torch.load` back to full unrestricted unpickling for every checkpoint/cache loaded in-process, which is only safe because every `.pt`/cache file this pipeline touches is either the official ultralytics/ESOD release or produced locally by this same pipeline. Do not point this environment at untrusted third-party checkpoints without reconsidering this patch.

## 4. `train.py` not idempotent under `--exist-ok` reruns

- **Location:** `esod/train.py`, run-settings snapshot block (`if rank in [-1, 0] and not opt.resume: ...`)
- **Symptom:** Re-running `train.py` against an output directory from a previous (even failed) attempt crashed immediately:
  ```
  File ".../esod/train.py", line 81, in train
      os.mkdir(save_dir / 'scripts')
  FileExistsError: [Errno 17] File exists: '.../scripts'
  ```
- **Root cause:** `train.py` copies its own source (`train.py`, `test.py`, `utils/loss.py`, `utils/general.py`, `utils/datasets.py`, `models/yolo.py`, `models/common.py`, the model cfg) into `<save_dir>/scripts/` for provenance, via a plain `os.mkdir(save_dir / 'scripts')`. `scripts/esod_baseline/run_baseline.sh` always passes `--project`/`--name`/`--exist-ok` so retries reuse the same directory (see `ESOD_VENDOR_BUGS_AND_FIXES.md`'s own precedent of the same `os.mkdir` vs `os.makedirs(exist_ok=True)` class of bug in `data_prepare.py`) — any second attempt into that directory (crash-and-retry, or intentionally continuing a `--resume`-less run) hit the non-idempotent `os.mkdir` and died before training started.
- **Fix:**
  ```python
  os.makedirs(save_dir / 'scripts', exist_ok=True)
  ```

## Pre-flatten commit history (esod's own git history, discarded)

Before `esod/.git` was removed, the clone carried 3 local commits on top of upstream `alibaba/esod`. Preserved here since that history is no longer retrievable locally:

1. **`update README.md`** — updated the citation block (`liu2024esod` → `liu2025esod`, added volume/pages matching the published TIP version) and added a "Pretrained Weights" section linking the Google Drive folder.
2. **`add weights & readme & data-processing`** —
   - Swapped the order of the two `pip install` lines in the README's install instructions (`requirements.txt` before the pinned `torch`/`torchvision` wheel).
   - `scripts/data_prepare.py::prepare_visdrone()`: `os.mkdir` → `os.makedirs(..., exist_ok=True)` (idempotent re-runs), and removed a duplicate label-file write (the function used to write the same YOLO label file to two different paths; now writes once to the canonical `labels/` path before calling `gen_mask()`).
   - `scripts/data_prepare.py::segment_image()`: fixed a real upstream bug — the recursive tiled-SAM-segmentation branch passed the *outer* image's `width, height` into the recursive call instead of the *tile's* `width_, height_`, corrupting box-to-tile coordinate normalization for images above the 1024px tiling threshold.
3. **`Update data_prepare.py`** — one-line fix, same idempotency pattern (`os.mkdir` → `os.makedirs(..., exist_ok=True)`) applied to a second call site.

None of these three touch `train.py`, `test.py`, `models/`, or `utils/loss.py` — they are data-prep-only fixes and do not affect training/eval semantics.

## Related tooling (not a patch to `esod/` itself)

`scripts/esod_baseline/` drives baseline reproduction against this `esod/` checkout without modifying its own orchestration:

- `gen_masks.py` — generates/verifies the official `masks/<split>/*.npy` pseudo-labels (reusing `esod/scripts/data_prepare.py::gen_mask()`) for an already YOLO-formatted `images/`+`labels/` tree, and fails closed if any labeled image is missing a mask. This exists because the official dataloader (`esod/utils/datasets.py`) silently falls back to an all-zero selector target when a mask file is missing — the same failure mode `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` (#21) documents for the separate vendored fork.
- `audit_buckets.py` — standalone size-bin/class recall audit over `test.py --save-json` output, independent of the BCRS package.
- `run_baseline.sh` — end-to-end driver: mask gen/verify → train.py → test.py (native AP/AP50 + `--save-json`) → test.py `--task measure` (GFLOPs/FPS) → `audit_buckets.py`.
