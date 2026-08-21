# ESOD Baseline Patch Ledger

This is the complete current delta from the released ESOD baseline. It records
active behavior only; rejected experiments and execution order live in
`HESOD-Experiment-Plan.md`.

## Provenance and coverage contract

- Upstream: [`alibaba/esod`](https://github.com/alibaba/esod), pinned by the
  original gitlink at commit `bde3571bd7db697e441eef0278cd425e888ea026`.
- `hesod/backends/esod/` is the standalone baseline. A root-level `esod/`
  checkout previously existed as a byte-identical mirror; it was retired
  (2026-08-21) once `diff -rq` and `tests/test_esod_audit_and_inference.py`
  confirmed no source difference, and nothing outside `hesod/backends/esod/`
  itself depended on the root copy.
- Against the pinned upstream, the baseline has exactly **11 path deltas**:
  8 modified, 2 deleted, and 1 added. The other 433 upstream files are
  unchanged after line-ending normalization.
- `hesod/backends/hesod/` starts from this patched baseline and adds method-only
  routing, selector, and loss options. Those additions are listed separately.

Any future baseline change must add or update a ledger row here and include a
focused regression test. A path difference without a ledger entry is protocol
drift.

## Complete baseline source delta

| ID | Relative path(s) | Patch | Result impact |
|---|---|---|---|
| P01 | `utils/general.py` | Guard missing/broken `pkg_resources`; dependency checks no-op when unavailable | Runtime compatibility only |
| P02 | `train.py`, `test.py` | Default legacy checkpoint loads to `weights_only=False` on PyTorch 2.6+; explicit caller values win | Restores loading; no math change |
| P03 | `utils/metrics.py` | Use `np.trapezoid`, falling back to `np.trapz` | Same 101-point AP integration |
| P04 | `train.py`, `scripts/data_prepare.py` | Replace one-shot `os.mkdir` with recursive `exist_ok` creation | Idempotent reruns only |
| P05 | `scripts/data_prepare.py` | Move SAM tensors to CPU; make `paper-hybrid`, `released-hybrid`, and `gaussian` explicit; reject silent SAM fallback; pin TinyPerson to no-dense erased data and write a protocol manifest | Makes pseudo-mask and TinyPerson data provenance reproducible |
| P06 | `models/common.py` | Key HeatMapParser caches by full spatial tuple, not equal-area products | Prevents stale rectangular grids and CUDA index faults |
| P07 | `data/hyps/hyp.tinyperson.yaml`; delete alternate `hyp.tinyperson.{scratch,finetune,released}.yaml` profiles | Pin the retained official TinyPerson experiment to the paper-text setting (`lr0=0.01`) | Prevents stale protocol variants from being selected |
| P08 | `scripts/data_prepare.py`, `data/uavdt.yaml`, `models/cfg/esod/uavdt_yolov5m.yaml` | Preserve car/truck/bus as classes 0/1/2 and set `nc: 3` | Paper-comparable task; nc=1 checkpoints are incompatible |
| P09 | `test.py` | Save raw prediction JSON before optional dataset-specific COCO formatting/evaluation | Prevents missing artifacts; disk JSON stays zero-indexed |
| P10 | `test.py` | Compute target counts even when there are no true-positive hits | Correct labels/BPR display; AP math unchanged |
| P11 | `test.py` | Pad trailing zero-patch images to the true dataloader batch size | Prevents silent image/GT loss from AP and recall |
| P12 | `test.py` | Write `buckets.json` only for `--task measure` | Prevents empty validation artifacts from masquerading as compute evidence |
| P13 | `utils/metrics.py` | Change BPRbox boundary from `>= 0.5` to the paper Eq. (7) strict `> 0.5`; denominator remains GT area via `box_ios` | Removes boundary-only selector-recall drift; no AP effect |

P07 accounts for four paths, so P01-P13 map to the exact 12-path inventory:

```text
M data/uavdt.yaml
D data/hyps/hyp.tinyperson.finetune.yaml
D data/hyps/hyp.tinyperson.scratch.yaml
A data/hyps/hyp.tinyperson.yaml
M models/cfg/esod/uavdt_yolov5m.yaml
M models/common.py
M scripts/data_prepare.py
M test.py
M train.py
M utils/general.py
M utils/metrics.py
```

No baseline detector/selector architecture file other than the cache-key fix
in `models/common.py` differs from the pinned source. In particular, the AP
definition, AP/AP50 mapping, anchors, backbone, neck, and VisDrone/TinyPerson
detection heads are unchanged. UAVDT's output head differs only because the
paper-comparable task has three classes.

## Patch behavior details

### Evaluation invariants

`test.py --save-json` first writes the raw ESOD schema:

- filename-stem `image_id`;
- zero-indexed `category_id`;
- finite `[x, y, width, height]` and score values.

Optional VisDrone/TinyPerson annotation lookup and pycocotools evaluation run
afterward and remain best-effort. TinyPerson headline metrics come from
`scripts/esod_baseline/tinyperson_eval/eval_tinyperson_official.py`.

The patch detector derives an output batch dimension from observed patch batch
IDs. If the final images select no patches, upstream returns a short list.
`pad_trailing_empty_predictions()` appends `(0, 6)` tensors so every image and
its GT enter statistics. TinyPerson validation must therefore report exactly
786 images and 13,687 labels.

The metric implementation still computes `ap[:, 0]` as AP50 and the mean over
IoU 0.50:0.95 as AP. P03 changes only the NumPy API spelling. P10 changes
zero-hit accounting; P13 aligns the BPRbox comparison with Eq. (7). BPRbox
uses intersection divided by GT area, not IoU.

### Data and training invariants

Paper-comparable VisDrone uses the released `prepare_visdrone()` conversion,
including graying ignored/`others` regions. Gaussian and released-hybrid SAM
masks are selected explicitly by the runner; installed packages cannot choose
the mask protocol implicitly.

TinyPerson retains only the paper-text hyp file. UAVDT has one three-class
dataset/model config and one converter behavior (`raw_class - 1`); no nc=1 or
`_nc3` alternate remains.

The HeatMapParser cache fix changes neither weights nor the intended grid. It
only prevents two different rectangular shapes with the same product from
sharing cached indices.

## HESOD-only deltas

These options exist under `hesod/backends/hesod/` and do not change plain ESOD
defaults:

- `--selector-loss upstream`: released weighted BCE;
- `--selector-loss paper`: focal:dice 20:1 paper-text control;
- `--selector-loss coverage`: HESOD coverage objective;
- `--top-k`: exact patch budget;
- `--sparse-head`: sparse detector execution;
- `--box-loss size_weighted`: research switch, off by default.

For exact Top-K plus SparseHead, every location inside each selected patch is
evaluated; the old second hard-coded 0.3 heatmap threshold is bypassed.
Top-K ranks uniform coarse cells and emits their fixed boundaries; it does not
apply Algorithm 2's offset adjustment/overlap removal. Threshold routing keeps
released Algorithm 2 behavior. Spectral filters are trainable Sobel/Laplacian
initializations, not fixed filters.

## Supported experiment tooling

| Tool | Audited responsibility |
|---|---|
| `run_baseline.sh` | Canonical three-dataset reproduction; explicit Gaussian or released-hybrid masks; measurement and fatal audits |
| `run_visdrone_sam.sh` | Explicit Gaussian versus released-hybrid SAM control |
| `run_visdrone_roster.sh` | Paper-loss control and locked equal-budget HESOD matrix |
| `gen_masks.py` | Explicit `gaussian|released-hybrid` generation and completeness checks |
| `audit_buckets.py` | Validated image-ID mapping and confidence-ordered class-aware one-to-one recall |
| `dump_selected_patches.py` | Replay threshold/Top-K exactly; record routing, patches, and paper-0.5 local-maxima cells for every image |
| `audit_selector_coverage.py` | Strict paper BPRbox, paper BPRctr, patch-center diagnostic, and detector-recall decomposition; reject incomplete/mismatched artifacts |
| `reorganize_{visdrone,tinyperson,uavdt}.py` | Canonical dataset layouts |
| `tinyperson_eval/` | Official TinyPerson metric implementation |

Obsolete Top-K sweeps, nc=1 UAVDT runners, TinyPerson multi-hyp runners, and
ad-hoc IoU evaluators were removed in the 2026-08-12 audit. They are not part
of the supported protocol.
