# Third-party notices and provenance

The files listed in `vendor/manifest.json` are unmodified source/config snapshots from the sibling repositories present when this integration was created. Their original licenses remain in each vendor directory. Files not listed in the manifest, such as CEASC's small BCRS loader, are integration code.

| Project | Upstream commit | License | Copied scope |
|---|---|---|---|
| ESOD | `bde3571bd7db697e441eef0278cd425e888ea026` | GNU GPL v3 | Train/test entrypoints, selector/model parser, sparse head, required YOLO data/loss/runtime utilities, and experiment configs |
| CEASC | `2abfd1a99f1b0fe1ed3d51588b64549e1584da50` | Apache-2.0 | Original train/test entrypoints, UAV datasets, GFL/RetinaNet dynamic heads, helpers, CUDA extension, and configs |
| QueryDet-PyTorch | `feebf218d53d59ba054132dfa6ef84159f793967` | MIT | QueryDet/RetinaNet models, required utilities/loaders/trainers, entrypoints, and configs |

The BCRS orchestration code is newly written. Because the repository includes GPL-licensed ESOD snapshots, do not assign or redistribute the combined tree under terms that conflict with GPL v3. Review all three license files before redistribution.

Exact per-file hashes and upstream paths are recorded in `vendor/manifest.json`. The BCRS-owned `vendor/ceasc/bcrs_entry.py` loader is deliberately excluded from that manifest.

Files intentionally excluded include checkpoints, datasets, paper assets, cached bytecode, full framework forks, broad evaluation toolkits, deployment demos, and unrelated model configurations.
