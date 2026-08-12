# ESOD Baseline Patches — Local Modifications for Modern-Environment Reproduction

This document records every local change made to the `esod/` checkout at the repo root (a pristine clone of the official [alibaba/esod](https://github.com/alibaba/esod)) so that the paper's YOLOv5m baseline can be trained and evaluated end-to-end on a modern stack (Python 3.12, recent CUDA/PyTorch) for VisDrone / UAVDT / TinyPerson.

**Dual-tree coverage:** this doc is the single authority for environment-compat patches on *both* `esod/` (this reference/baseline-reproduction copy) and `hesod/backends/esod/` (the active HESOD development copy, started as a byte-identical copy of `esod/`). The two trees are expected to track the same Python/dependency version matrix indefinitely, not deliberately diverge — so any future environment issue found in either tree must be fixed and logged once here, then applied to the corresponding file in both trees.

**Scope boundary:** `esod/` is deliberately kept as close to upstream as possible. Every entry below is either (a) a genuine upstream bug that crashes or silently misbehaves under a modern environment, or (b) a compatibility shim — never a change to ESOD's published algorithm, loss, routing, or architecture. HESOD's actual algorithmic work (the dual-evidence selector, per `HESOD-Proposal.md`) lives in `hesod/backends/esod/` and is out of scope for this document once it starts diverging architecturally from upstream ESOD. `BCRS/vendor/esod` is an earlier, unofficial draft of related ideas that predates `hesod/`; it is untouched and unrelated to this doc — see `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` for that fork's own history if needed.

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

## 5. `np.trapz` removed in NumPy 2.0+

- **Location:** `esod/utils/metrics.py::compute_ap()`
- **Symptom:** `AttributeError: module 'numpy' has no attribute 'trapz'` during AP computation — hit on essentially every `test()` call (end of each training epoch, plus standalone `test.py` runs), since NumPy 2.0 renamed `trapz` to `trapezoid` and this project's stack pins NumPy 2.2.x.
- **Root cause:** `compute_ap()` called `np.trapz(...)` unconditionally. Same class of break already documented and fixed in `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` (#6) for the separate vendored fork; applied here to the pristine `esod/` copy too.
- **Fix:** module-level fallback, used in place of the direct call:
  ```python
  _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
  ...
  ap = _trapz(np.interp(x, mrec, mpre), x)
  ```

## 6. UAVDT class count (`nc`) mismatch — resolution reversed after finding stronger evidence

- **Location:** `esod/models/cfg/esod/uavdt_yolov5m.yaml`
- **Symptom:** Shipped as `nc: 1`, which does not match this project's `/root/autodl-tmp/UAVDT_processed/uavdt.yaml` (`nc: 3`, `names: [car, truck, bus]`), nor the real label content (`labels/train/*.txt` contains class ids `0`, `1`, `2`, confirmed by direct inspection).
- **Original investigation (superseded, see below):** `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` (#17) documents the same `nc` mismatch in the separate vendored fork and resolved it by forcing single-class (`nc: 1`, "vehicle") to match what that doc calls "the official UAVDT benchmark protocol." That characterization looked wrong against the ESOD paper's own prose at the time: §4.B's dataset description states "the UAVDT dataset consists of 50 video sequences with **3 categories to detect**." Based on that text, the original resolution here kept 3-class and changed `models/cfg/esod/uavdt_yolov5m.yaml` from `nc: 1` to `nc: 3` instead.
- **New evidence (stronger, and contradicts the above):** `esod/scripts/data_prepare.py::prepare_uavdt()` — the *executable* official label-conversion code, not the paper's prose — hardcodes every label's class id to `0` regardless of the raw annotation's category (`assert 1 <= cls <= 3` immediately followed by `f.write('%d ...' % (0, ...))`, see the function body). Code is more authoritative than prose for reproduction purposes: whatever the paper's dataset-description text says, the actual training pipeline behind the paper's reported UAVDT numbers appears to have trained and evaluated single-class ("vehicle"/"car"), not 3-class. This is also consistent with UAVDT's severe car-dominant class imbalance (this project's own `UAVDT_processed`: car 361055 / truck 7595 / bus 7234 GT boxes) and with single-class treatment being common practice for UAVDT in detection-method papers generally (as opposed to fine-grained vehicle-classification papers).
- **Resolved (round 2): re-fetched UAVDT from the official source, flipped to `nc: 1` matching the code.** `/root/autodl-tmp/UAVDT_processed` (third-party Kaggle repackaging, the "suspect data" referenced above) is superseded — official `UAV-benchmark-M.zip`/`M_attr.zip`/`UAV-benchmark-MOTD_v1.0.zip` converted via `prepare_uavdt()` into `UAVDT_v2`. `nc` flipped `3 -> 1` (`models/cfg/esod/uavdt_yolov5m.yaml`, `data/uavdt.yaml`, all three trees) to match what `prepare_uavdt()` actually writes, closing the "partial fix" gap flagged above. Baseline trained and evaluated: AP@[.5:.95]=0.406/AP50=0.781 (`test.py`, COCO-style average), far above the paper's reported 22.5/40.7.
- **Round 3, resolved: the paper's number is 3-class, not single-class.** The nc=1 gap above (~1.8-2.3x the paper) was too large to be a genuine improvement. Investigated whether UAVDT's vendored MATLAB toolkit (`evaluation/UAV-benchmark-MOTD_v1.0/utils/evalRes.m`, default `thr=.7`, VOC-style `VOCap.m` AP integration, not COCO 0.5:0.95 averaging) explains it — confirmed real (our own AP at IoU=0.70 specifically, via `scripts/esod_baseline/eval_at_iou_thresholds.py`, is 52.2%) but does not fully close the gap (52.2% vs. 22.5% is still ~2.3x), and `VOCap.m`'s all-point interpolation is not meaningfully different from pycocotools' 101-point method, so that's not it either. Tested the leading hypothesis directly: Table I's other baselines in the same row (FasterRCNN 11.0/23.4, ClusDet 13.7/26.5, DMNet 14.7/24.6, CDMNet 16.8/29.1, CEASC 17.1/30.9) are widely-cited numbers from the standard **3-class** UAVDT-DET literature. Added `--keep-classes` to `scripts/data_prepare.py` (all three trees, preserves real 0-indexed categories instead of forcing class 0) and trained `models/cfg/esod/uavdt_yolov5m_nc3.yaml` (nc=3, kept alongside the nc=1 default, not replacing it) with SAM's checkpoint temporarily renamed during data prep so masks stayed Gaussian-only, matching the nc=1 baseline (single-variable comparison). **Result: nc=3 AP@[.5:.95]=0.201/AP50=0.370 — 9-11% relative *below* the paper's 0.225/0.407, the same direction and magnitude as every other dataset in this project, vs. nc=1's 80-130% *above*.** Confirms `prepare_uavdt()`'s single-class collapse in the public code does not match the protocol that produced the paper's actual Table I number.
- **Reporting caveat — resolved:** `nc=3` (`UAVDT_v3`, `uavdt_yolov5m_nc3.yaml`) is the correct, paper-comparable protocol going forward — `run_baseline.sh`'s `uavdt` case now defaults to it. The nc=1 numbers (`UAVDT_v2`) are kept only as the historical record of how this mismatch was diagnosed, not as a result to report against the paper.

## 7. `test.py` displayed `Labels: 0` / `BPR: nan` on validation passes with zero IoU>0.5 hits

- **Location:** `esod/test.py`, statistics block (`stats = [np.concatenate(x, 0) for x in zip(*stats)]` onward)
- **Symptom:** Any validation pass with zero correct IoU>0.5 matches (routine in early training epochs, and observed on the first 3 epochs of a UAVDT run whose 53k-image validation set took longer to produce a first lucky match than VisDrone's 548-image one) printed `Labels: 0` and `BPR: nan`, even though the real label count was never zero (later epochs on the same run correctly showed 375883 labels):
  ```
  all      53676          0          0          0          0          0        nan          1
  ```
- **Root cause:**
  ```python
  if len(stats) and stats[0].any():
      p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
      ap50, ap = ap[:, 0], ap.mean(1)
      mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
      nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
  else:
      nt = torch.zeros(1)
  ```
  `nt` (labels-per-class, used both for the printed `Labels` column and as `bpr`'s denominator via `nt.sum()`) was gated behind `stats[0].any()` — "did any prediction hit a GT at IoU>0.5" — instead of being computed whenever `stats` has any rows at all. Same bug class already documented and fixed in `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` (#2) for the separate vendored fork; this pristine copy had the identical unfixed upstream code. `mp`/`mr`/`map50`/`map` were not affected — they already default to `0.0` earlier in the function (line ~149), so the `else` branch never raised `NameError`, it only silently mis-reported `nt`.
- **Fix:** decouple `nt` from the has-any-hit check:
  ```python
  if len(stats):
      nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
      if stats[0].any():
          p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
          ap50, ap = ap[:, 0], ap.mean(1)
          mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
  else:
      nt = torch.zeros(1)
  ```
  Applied to `esod/test.py`, `hesod/backends/esod/test.py`, and `hesod/backends/hesod/test.py` (the latter is HESOD's active dev copy, out of this doc's normal dual-tree scope per the Scope Boundary note above, but this specific fix is a pure upstream bug fix with no algorithmic content, so all three copies were kept in sync rather than letting a third silent fork of the same bug develop).
  - **Display-only, no effect on any reported metric to date:** neither the VisDrone E1.0 run (`HESOD-Experiment-Plan.md`) nor UAVDT's in-progress run had this branch active by their final epoch — VisDrone's very first validation pass already had real matches, so this bug never fired for VisDrone at all.

## 8. `gen_mask()` crashed on first real SAM invocation — CUDA tensor passed directly to `.numpy()`

- **Location:** `scripts/data_prepare.py::gen_mask()`, SAM-hybrid mask branch (`if predictor is not None:`)
- **Symptom:** the very first time SAM was actually installed and loaded successfully in this project (previously always silently fell back to Gaussian-only, see `HESOD-Experiment-Plan.md`'s "no SAM installed" caveat on every prior run), `gen_masks.py`/`data_prepare.py` crashed immediately on the first image:
  ```
  TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
  ```
- **Root cause:** `sam_res = (sam_res > 0.5).half().numpy()` — `sam_res` comes from `segment_image()`, which runs SAM inference on `device = "cuda"` (hardcoded in this file's own SAM setup, see #`predictor` init near the top) and returns a CUDA tensor. `torch.Tensor.numpy()` refuses to convert a CUDA tensor directly, no matter the dtype. This code path had apparently never been exercised end-to-end in this project before (every prior mask-generation run used the Gaussian-only fallback), so the bug was latent, not previously hit.
- **Fix:** insert `.cpu()` before `.numpy()`:
  ```python
  sam_res = (sam_res > 0.5).half().cpu().numpy()
  ```
  Applied to `esod/scripts/data_prepare.py`, `hesod/backends/esod/scripts/data_prepare.py`, and `hesod/backends/hesod/scripts/data_prepare.py` (all three had the identical unfixed line).
- **Separately, not a code bug but worth recording:** getting SAM importable at all required bypassing a broken `pip install -e third_party/segment-anything` (fails with a `setuptools`/`distutils` incompatibility under Python 3.12 — `pip install --upgrade setuptools` alone doesn't reliably fix it in a conda base env). Since `segment-anything` is pure Python with no compiled extensions, the working fix was a direct symlink into site-packages instead of fighting the packaging toolchain: `ln -sf <repo>/third_party/segment-anything/segment_anything <conda-env>/lib/python3.12/site-packages/segment_anything`.

## 9. `HeatMapParser`'s grid cache keyed by a product, not a shape tuple — silent stale-grid reuse, `"index out of bounds"` CUDA assertion at `ratio=16`

- **Location:** `models/common.py::HeatMapParser.ada_slicer()` (`self.grid`) and `HeatMapParser.ada_slicer_fast()` (`self.grid_vtx` and `self.grid`)
- **Symptom:** TinyPerson trained fine at `ratio=8` (every prior arm), but a new `ratio=16` arm (testing the paper's own "1/16 may be a more suitable choice [for TinyPerson]" patch-size guidance) crashed reliably on the first validation pass after epoch 0, with a wall of asynchronous CUDA errors and no usable Python traceback even under `CUDA_LAUNCH_BLOCKING=1`:
  ```
  /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [...], thread: [...]
  Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
  ```
- **Root cause, confirmed via targeted debug prints (not inference):** all three cache checks compared a **product** of two dimensions instead of the dimensions themselves:
  ```python
  if ... or self.grid[0].shape[-1] != cluster_h*cluster_w: ...          # ada_slicer / ada_slicer_fast
  if ... or self.grid_vtx.size(0) != ratio_x*ratio_y*bs: ...            # ada_slicer_fast
  ```
  Consecutive validation batches can have transposed `(cluster_h, cluster_w)` (equivalently `(ratio_y, ratio_x)`) if the images in one batch happen to be portrait and the next landscape (or vice versa) after letterboxing. At `ratio=16` on TinyPerson this genuinely happened: one batch computed `cluster_w=12, cluster_h=16` (product 192), the next `cluster_w=16, cluster_h=12` (also 192). The product-based check saw `192 == 192` and treated the cache as still valid, reusing a grid (`gy`/`gx`) built for the *previous* orientation. Debug output at the crash site: `cluster_w=16 cluster_h=12 ... activated.shape=(8, 144, 256) ... act_y[min,max]=(12,147)` — 147 exceeds the valid height range `[0,143]` by exactly the old cache's stale `cluster_h=16` contribution (`11*12 + 15 = 147` instead of the correct `11*12 + 11 = 143`). At `ratio=8`, `cluster_w`/`cluster_h` are larger numbers where an exact product collision after transposition is far less likely, which is why no dataset/arm at `ratio=8` had ever hit this in this project.
- **Fix:** cache the actual `(cluster_h, cluster_w)` / `(bs, ratio_y, ratio_x)` tuples and compare those, not their products:
  ```python
  if not hasattr(self, 'grid') or self.grid is None or getattr(self, '_grid_hw', None) != (cluster_h, cluster_w):
      ...
      self._grid_hw = (cluster_h, cluster_w)
  ```
  (and analogously `self._grid_vtx_shape = (bs, ratio_y, ratio_x)` for `self.grid_vtx`). Applied to all three sites in `esod/models/common.py`, `hesod/backends/esod/models/common.py`, and `hesod/backends/hesod/models/common.py` (identical unfixed code in all three — this is an upstream bug, not something HESOD introduced).
- **Re-verified end-to-end:** both `ratio=16` TinyPerson arms (plain baseline and full-spectral concat) subsequently completed clean 50-epoch runs with the fix in place, confirming the crash is fully resolved. Both arms turned out to be negative results (moved every metric further from the paper vs. the standard `ratio=8`, contradicting the paper's own guidance) — see `HESOD-Experiment-Plan.md` SS4.2 for the one-line conclusion; detailed run artifacts were intentionally not kept once the result was confirmed negative.

## 10. `prepare_uavdt()` / TinyPerson prep — non-idempotent `os.mkdir(split_dir)` crashes on any rerun of already-converted raw data

- **Location:** `scripts/data_prepare.py::prepare_uavdt()` and the TinyPerson prep function, both `os.mkdir(split_dir)`
- **Symptom:** re-running `python scripts/data_prepare.py --dataset UAVDT_raw --keep-classes` against a `UAVDT_raw` that had already been converted once before (for the nc=1 baseline) crashed immediately:
  ```
  File ".../scripts/data_prepare.py", line 292, in prepare_uavdt
      os.mkdir(split_dir)
  FileExistsError: [Errno 17] File exists: '/root/autodl-tmp/UAVDT_raw/split'
  ```
  Crashed before writing anything for that invocation (the crash is the first line of the function after the harmless, idempotent ignore-region re-masking loop), so no partial/corrupted output — safe to fix and simply rerun.
- **Root cause:** plain `os.mkdir()`, which raises if the directory already exists — expected the very first time `prepare_uavdt()` runs against a fresh raw checkout, but this project now legitimately reruns data prep against the *same* raw directory with different flags (e.g. `--keep-classes` for the nc=3 test, §6 above). `prepare_visdrone()` already received the identical `os.mkdir` → `os.makedirs(exist_ok=True)` fix in an earlier commit (see "Pre-flatten commit history" below) — `prepare_uavdt()` and the TinyPerson prep function were added/touched separately and never got the same treatment until this was hit.
- **Fix:** `os.mkdir(split_dir)` → `os.makedirs(split_dir, exist_ok=True)` in both functions. Applied to `esod/scripts/data_prepare.py`, `hesod/backends/esod/scripts/data_prepare.py`, `hesod/backends/hesod/scripts/data_prepare.py`.

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

- `gen_masks.py` — generates/verifies the official `masks/<split>/*.npy` pseudo-labels (reusing `esod/scripts/data_prepare.py::gen_mask()`) for an already YOLO-formatted `images/`+`labels/` tree, and fails closed if any labeled image is missing a mask. This exists because the official dataloader (`esod/utils/datasets.py`) silently falls back to an all-zero selector target when a mask file is missing — the same failure mode `BCRS/docs/ESOD_VENDOR_BUGS_AND_FIXES.md` (#21) documents for the separate vendored fork. Carries the same `torch.load(weights_only=False)` compat patch as #3 above, applied right before importing `data_prepare` — needed once `segment-anything` is installed, since `data_prepare.py` loads the SAM checkpoint via `torch.load` at module-import time (not lazily), so the crash happens even for callers that never touch SAM's predict path.
- `audit_buckets.py` — standalone size-bin/class recall audit over `test.py --save-json` output, independent of the BCRS package.
- `run_baseline.sh` — end-to-end driver: mask gen/verify → train.py → test.py (native AP/AP50 + `--save-json`) → test.py `--task measure` (GFLOPs/FPS) → `audit_buckets.py`.
