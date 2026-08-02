# BCRS experiment workspace

This directory is a self-contained experiment workspace for BCRS. It contains the minimum executable source closure for ESOD, QueryDet, and CEASC, one validated experiment schema, and runner adapters for training and testing them on shared dataset definitions. The three original source repositories are not runtime dependencies.

## Why the repository is structured this way

| Backend | Upstream stack | Custom surface | Runtime boundary |
|---|---|---|---|
| ESOD | YOLOv5-style runtime | `Segmenter`, `HeatMapParser`, sparse head, required training/data utilities | Original executable closure under `vendor/esod` |
| QueryDet | Detectron2 + spconv | Query meta-architecture, heads, CSQ inference, custom loaders | Original executable closure under `vendor/querydet` |
| CEASC | MMDetection 2.24.1 + MMCV 1.5.1 + CUDA extension | Dynamic heads, masks, CE-GN sparse convolution, UAV datasets | Original plugins and entrypoints under `vendor/ceasc`; a BCRS loader registers them without editing the model files |

This keeps BCRS orchestration independent of Torch/MMCV/Detectron2 imports. Backend modules are loaded only inside a child process. All supplied experiments default to the same `BCRS_PYTHON` interpreter and the reference environment in `environments/torch1.10-cu113`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the source-repository scan and extension rules. Vendored files retain their upstream licenses and are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Install the orchestration CLI

```powershell
cd BCRS
python -m pip install -e ".[test]"
bcrs list-backends
```

Install the unified detector stack on Linux with a CUDA 11.3 development toolkit, then compile CEASC's copied extension:

```bash
python -m pip install -r environments/torch1.10-cu113/requirements.txt
python -m pip install -v ./vendor/ceasc/Sparse_conv
```

See [the environment notes](environments/torch1.10-cu113/README.md). A different interpreter can be selected with `BCRS_PYTHON`.

## Configure datasets

Dataset metadata is centralized under `configs/datasets`. One logical dataset can describe the different layouts expected by each backend. Set the relevant root before running:

```powershell
$env:VISDRONE_ROOT = "D:\datasets\VisDrone"
$env:UAVDT_ROOT = "D:\datasets\UAVDT"
$env:AITOD_ROOT = "D:\datasets\AI-TOD"
```

AI-TOD data is distributed under CC BY-NC-SA 4.0; confirm that the intended use is non-commercial or otherwise permitted before downloading or training on it.

The default VisDrone layout includes:

- ESOD split lists under `split/`;
- QueryDet COCO conversion under `coco_format/`;
- CEASC COCO conversion under `annotations/` and `images/`.

If your conversion uses another layout, edit only the dataset YAML; backend code does not need to change.

## Validate and run

Always validate first. `doctor` checks the selected interpreter and required Python modules, vendored backend, entrypoint, model config, dataset paths, checkpoint, and backend-specific extension source.

```powershell
bcrs show configs/experiments/esod_visdrone.yaml
bcrs doctor configs/experiments/esod_visdrone.yaml --stage train
bcrs train configs/experiments/esod_visdrone.yaml --dry-run
bcrs train configs/experiments/esod_visdrone.yaml
bcrs test configs/experiments/esod_visdrone.yaml
```

Override any experiment field without duplicating a config:

```powershell
bcrs train configs/experiments/esod_visdrone.yaml `
  --set train.batch_size=4 `
  --set runtime.devices=1 `
  --dry-run
```

The CLI never uses a shell to execute a detector command. Arguments and paths are passed as an argv list, and `CUDA_VISIBLE_DEVICES` is scoped to the child process.

## Add a dataset

1. Copy one file in `configs/datasets/`.
2. Set canonical `name`, `root`, `num_classes`, and `classes`.
3. Add `train`, `val`, and `test` split paths under each supported adapter.
4. Reference it from an experiment config.
5. Run `bcrs doctor` for every intended backend.

No Python registration is needed.
Start from `configs/datasets/custom_coco.yaml.example`. A shared custom dataset
uses COCO JSON for CEASC/QueryDet and YOLO text labels for ESOD; only paths and
class metadata belong in the dataset config.

## Add a backend or network

- A new network on an existing backend usually requires only a model config and an experiment YAML.
- A new backend implements `BackendAdapter.build`, registers one lazy loader in `registry.py`, and adds command-construction tests.
- Framework code must not be imported from the BCRS core package. Keep it inside the backend process.
- Refresh copied code with `python tools/sync_vendor.py --write` while the source checkouts exist; review the generated manifest and preserve its licenses.

## Current boundaries

- The adapters validate and launch copied upstream training/testing entrypoints; they do not normalize framework-specific metrics yet.
- CEASC distributed training is intentionally not hidden behind the common CLI yet. Its adapter accepts one visible GPU until a tested launcher contract is added.
- AI-TOD metadata is provided, but each upstream backend still needs the expected annotation/category conversion to be verified before claim-bearing runs.
- Checkpoints and datasets are intentionally external artifacts. Place the ESOD initialization checkpoint at `artifacts/pretrained/yolov5m.pt`, or override `model.weights`.

## Verify before deleting the source repositories

```bash
python tools/sync_vendor.py --against-source
python tools/sync_vendor.py
pytest -q
```

The first command proves the copy still matches the source repositories. The second verifies only `vendor/manifest.json`, so it continues to work after `CEASC/`, `esod/`, and `QueryDet-PyTorch/` are removed.
