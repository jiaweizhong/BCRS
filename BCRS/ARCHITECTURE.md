# Architecture and upstream repository scan

## Design goals

1. One experiment schema for detector, dataset, seed, output, and runtime selection.
2. No import-time dependency on any detector framework.
3. One reference PyTorch/CUDA ABI, while framework imports remain subprocess-isolated.
4. Dataset paths and category metadata defined once.
5. Exact, inspectable argv construction instead of ad-hoc shell scripts.
6. Small allow-listed vendor snapshots with preserved provenance.

## Source repository findings

### ESOD

- **Entry points:** `train.py` and `test.py`; `scripts/train.sh` supplies the paper defaults.
- **Architecture:** YOLO-style YAML parser in `models/yolo.py`, common blocks in `models/common.py`, loss/data/runtime utilities under `utils/`.
- **Selection path:** `Segmenter` predicts the heatmap; `HeatMapParser` implements adaptive/uniform slicing; `Detect.set_sparse` and `models/spconv.py` implement the sparse head path.
- **Configuration:** model YAML selects `Segmenter` and `HeatMapParser`; dataset YAML uses Darknet/YOLO image lists; hyperparameters are separate YAML files.
- **Coupling:** `models/yolo.py` wildcard-imports the common model library and training depends on the repository's loader, loss, anchor, plotting, and evaluation utilities.
- **Adapter decision:** preserve the statically reachable executable closure byte-for-byte under `vendor/esod`; exclude unrelated evaluators, deployment tools, and model families.

### QueryDet

- **Entry points:** `train_visdrone.py`, `infer_visdrone.py`, `train_coco.py`, and `infer_coco.py`.
- **Architecture:** Detectron2 meta-architecture `RetinaNetQueryDet`, custom dense/query heads, and `QueryInfer` using spconv.
- **Data path:** the VisDrone loader reads COCO-format JSON and image roots from `cfg.VISDRONE.*`; these values are safely overridden by the BCRS adapter.
- **Configuration:** Detectron2 YAML plus custom `MODEL.QUERY`, `MODEL.CUSTOM`, and `VISDRONE` nodes.
- **Coupling:** the custom surface is compact and depends on Detectron2/fvcore/spconv plus a small set of local utilities.
- **Adapter decision:** copy the complete custom dependency closure, excluding evaluation binaries, cached bytecode, assets, and conversion-only scripts.

### CEASC

- **Entry points:** MMDetection `tools/train.py` and `tools/test.py`.
- **Architecture:** a full MMDetection 2.24.1 fork. `GFLDYHead`/`RetinaDYHead` use `DyConv2D`, Gumbel masks, CE-GN, and a custom CUDA sparse-convolution extension.
- **Configuration:** Python configs under `configs/UAV`; VisDrone and UAVDT are COCO-format datasets.
- **Coupling:** custom head files use relative imports into the fork's registries, base heads, builders, and test mixins. The CUDA extension must match the selected Torch/CUDA ABI.
- **Adapter decision:** use installed MMDetection 2.24.1 as the framework. A small BCRS loader installs the unmodified copied heads and datasets under their original module names, then runs copied `tools/train.py` or `tools/test.py`.

## Runtime flow

```text
experiment YAML
    -> environment and dotted override resolution
    -> canonical dataset spec + backend-specific layout
    -> lazy backend adapter
    -> path diagnostics and deterministic argv
    -> backend subprocess in its own cwd/interpreter
    -> BCRS work directory
```

ESOD additionally receives a generated YOLO dataset YAML under `.bcrs/generated/<experiment>/`. The file is derived only from canonical dataset metadata and is not committed.

## Configuration ownership

| Concern | Owner |
|---|---|
| Dataset root, categories, split files | `configs/datasets/*.yaml` |
| Vendored backend root and shared interpreter | experiment `backend` section |
| Network architecture and initial weights | experiment `model` section + `vendor/*/configs` |
| Devices, workers, output path | experiment `runtime` section |
| Epochs, batch size, seed, framework overrides | `train` / `test` sections |
| Framework registry and kernels | isolated backend runtime |

## Maintenance rules

- Keep core modules standard-library-only except for PyYAML.
- Do not add Torch, MMDetection, Detectron2, or spconv imports to `src/bcrs`.
- Prefer config-only network additions.
- Keep command construction pure and covered by tests.
- Treat backend-specific dataset conversions as data preparation, not model code.
- Do not copy an entire upstream framework. Expand a vendor allow-list only when a copied file is part of the custom dependency closure or required to reproduce a claim-bearing config.
- Record upstream commit and license for every snapshot update.
- Keep copied source byte-for-byte identical to `vendor/manifest.json`; put compatibility glue outside the managed file set.
- Default manifest verification must not depend on the sibling source repositories.
