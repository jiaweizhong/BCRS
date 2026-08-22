# baseline: torchvision competitor detectors (Faster R-CNN, RetinaNet)

Fills the gap in `HESOD-Experiment-Plan.md` SS9.3: Faster R-CNN and
RetinaNet are cited as required baselines (`HESOD-Proposal.md` SS7.3) but,
unlike QueryDet, both ship as ready-to-use pretrained detectors in
`torchvision.models.detection` -- no custom framework integration needed,
so they get their own lightweight backend here instead of the full
YOLOv5-fork treatment `backends/esod/`/`backends/hesod/` get.

**Scope (2026-08-22): UAVDT + SeaPerson only**, confirmed with the user.
VisDrone/TinyPerson are deferred, not blocked -- every script here takes
explicit `--images-dir`/`--labels-dir`/`--classes`/`--img-size` flags with
no dataset hardcoding, so adding them is two more `run_arm()`-style blocks
in `scripts/esod_baseline/run_torchvision_baseline.sh`, not a code change.
QueryDet remains out of scope here -- no off-the-shelf package exists for
it; it needs a real implementation and its own separate scoping decision
per SS9.3.

## Protocol notes (read before trusting a number against the rest of the doc)

- **Init**: COCO-pretrained torchvision weights
  (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` /
  `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`), head replaced for this
  dataset's classes and fine-tuned -- matches this project's own convention
  of fine-tuning from `weights/pretrained/yolov5m.pt`, not training from
  scratch.
- **Resolution**: `models.py::build_model`'s `min_size`/`max_size` are set
  equal to each dataset's existing img-size (UAVDT 1280, SeaPerson 2048),
  which resizes so the image's LONGER side equals that value, aspect-ratio
  preserved. This is **not pixel-identical** to the YOLOv5 letterbox-to-
  square resize every other arm uses -- same order of magnitude resolution,
  different resize geometry. An honest protocol note, not a bug.
- **Accuracy metric**: standard COCO AP/AP50/AP75 via pycocotools
  `COCOeval` (`coco_utils.py`), against a GT json built once from the
  existing YOLO labels and cached per run as `coco_gt.json`. This is a
  **different code path** than the custom `ap_per_class()` powering every
  other arm's mAP column in `HESOD-Experiment-Plan.md` -- report both
  numbers, don't present them as the same computation.
- **Efficiency**: `fvcore.nn.FlopCountAnalysis` for GFLOPs, matching
  `hesod/backends/hesod/test.py`'s own measure task (proven working in this
  project's environment). **Not** `thop`, despite it being this repo's other
  tracked FLOP-counting dependency -- `thop==0.1.1` imports the stdlib
  `distutils` module, removed in Python 3.12, and is confirmed broken
  (`ModuleNotFoundError`) under the Python 3.12 this project runs on. `thop`
  is tried as a second attempt, then param-count-only, if `fvcore` isn't
  installed or profiling throws. FPS is separately measured wall-clock.
  Two-stage/dense anchor-based architectures aren't mechanistically
  comparable to HESOD's patch-routing compute-savings story regardless.
  `fvcore` is not in `hesod/backends/hesod/requirements.txt` either (also
  installed ad hoc) -- if it's missing, `pip install fvcore`.
- **Recall-bucket comparability**: `scripts/esod_baseline/audit_buckets.py`
  and `vt_diagnose.py` run completely unmodified against
  `{name}_predictions.json` -- this is the one axis that's truly
  apples-to-apples against every existing arm in the doc.
- **Validation split during training**: SeaPerson has a genuine held-out
  `valid` split, used here. UAVDT does not (`reorganize_uavdt.py` only
  produces `train`/`test`) -- `train.py` validates against `images/test` +
  `labels/test` during training for UAVDT, same as the existing YOLOv5
  arms' own `val:` convention for this dataset (inherited, not introduced
  here).

## Files

- `datasets.py` -- `YoloDetectionDataset`: reads `images/{split}/` +
  `labels/{split}/` (same directories every other arm uses), returns
  1-indexed torchvision targets (label 0 is background). `parse_yolo_labels`
  stays 0-indexed for reuse by `coco_utils.py`.
- `coco_utils.py` -- builds/caches a 0-indexed COCO-format GT json (matching
  `{name}_predictions.json`'s own convention, no id translation needed) and
  wraps `pycocotools.COCOeval`.
- `models.py` -- `build_model("fasterrcnn" | "retinanet", num_classes, ...)`.
- `train.py` -- fine-tuning loop; `--resume last.pt` alone restores the full
  original config from the checkpoint (matches how
  `run_uavdt.sh`/`run_seaperson.sh` invoke `--resume`).
- `test.py` -- `--task test` writes the ESOD-schema predictions json + runs
  COCOeval; `--task measure` reports GFLOPs/FPS at batch=1.

## Running

Not yet run (GPU busy finishing SeaPerson's `concat+SABL+ISPPHead`, per
`HESOD-Experiment-Plan.md` SS8). Once free:

```bash
cd /root/BCRS
RUN_ROOT=/root/esod_baseline_runs \
  bash scripts/esod_baseline/run_torchvision_baseline.sh 0
```

Or target one arm (same `ARMS` convention as `run_seaperson.sh`):

```bash
ARMS="uavdt_fasterrcnn" bash scripts/esod_baseline/run_torchvision_baseline.sh 0
```
