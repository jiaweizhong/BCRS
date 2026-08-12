"""Dump HeatMapParser's selected-patch boxes (original-image pixel coords) for
every image in a validation split, using the model's own predicted mask to
drive selection -- not ground-truth masks.

Why this exists: `test.py --save-json` only ever writes the final, post-NMS
predictions. Which P3/stride-8 patches HeatMapParser actually selected for a
given image is a transient forward-pass detail that gets discarded once
inference completes -- nothing in a completed run's artifacts records it. To
test whether a size-bin's low recall (see audit_buckets.py) is caused by the
selector never routing a patch to the object at all (a hard, structural miss:
`hesod/backends/esod/models/yolo.py`'s P3 neck branch consumes only
`HeatMapParser`'s sliced patches, so anything outside every selected patch
contributes nothing to the one detection level with small enough anchors to
catch it) versus the detection head missing it despite being given the right
patch, we need the selector's own patch list alongside the GT boxes.

Mirrors `test.py`'s model-loading, dataloader, and letterbox/scale_coords
handling exactly, and -- like the existing baseline eval in run_baseline.sh --
never passes GT masks to the model (no `--use-gt` equivalent here), so the
patches recorded here are driven by the Segmenter's own learned prediction.
Use ``--top-k`` or ``--hm-threshold`` to replay the exact routing override used
by the associated test run.  The output embeds that routing metadata so a
threshold dump cannot later be mistaken for a Top-K dump.

Necessarily imports from the ESOD repo itself (models/utils), unlike
audit_buckets.py, since it needs the actual trained model and dataloader.

Usage:
  python dump_selected_patches.py \
    --esod-repo /path/to/hesod/backends/esod \
    --data /root/autodl-tmp/VisDrone.yaml \
    --weights /path/to/runs/train/visdrone_yolov5m_baseline/weights/best.pt \
    --img-size 1536 --batch-size 8 --device 0 \
    --out selected_patches.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esod-repo", required=True, help="path to hesod/backends/esod (or esod/)")
    parser.add_argument("--data", required=True, help="dataset yaml, e.g. VisDrone.yaml")
    parser.add_argument("--weights", required=True, help="trained checkpoint, e.g. .../best.pt")
    parser.add_argument("--img-size", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--task", default="val", help="train, val, or test (which data[] split to load)")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--top-k", type=int, help="retain exactly K coarse cells (HESOD method only)")
    route.add_argument("--hm-threshold", type=float, help="override HeatMapParser's fixed threshold")
    parser.add_argument("--out", default="selected_patches.json")
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.hm_threshold is not None and not 0.0 <= args.hm_threshold <= 1.0:
        parser.error("--hm-threshold must be in [0, 1]")

    esod_repo = str(Path(args.esod_repo).resolve())
    sys.path.insert(0, esod_repo)

    import torch
    import torch.nn.functional as F

    # Same PyTorch 2.6+ weights_only=True compat patch as train.py/test.py.
    try:
        _orig_torch_load = torch.load

        def _compat_torch_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)

        torch.load = _compat_torch_load
    except Exception:
        pass

    import yaml
    from models.common import HeatMapParser
    from models.experimental import attempt_load
    from utils.datasets import create_dataloader, norm_imgs
    from utils.general import check_dataset, check_img_size, colorstr, scale_coords
    from utils.torch_utils import select_device

    device = select_device(args.device, batch_size=args.batch_size)
    model = attempt_load([args.weights], map_location=device).float()
    model.eval()
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.img_size, s=gs)
    stride = int(model.stride.min().item())  # P3/8 is always the finest (smallest-stride) level

    with open(args.data) as f:
        data = yaml.safe_load(f)
    check_dataset(data)

    class _Opt:
        single_cls = False

    dataloader = create_dataloader(
        data[args.task], imgsz, args.batch_size, gs, _Opt(), pad=0.0, rect=True,
        prefix=colorstr(f"{args.task}: "),
    )[0]

    heatmap_parsers = [m for m in model.model if isinstance(m, HeatMapParser)]
    if len(heatmap_parsers) != 1:
        raise SystemExit(f"expected exactly one HeatMapParser in model.model, found {len(heatmap_parsers)}")
    heatmap_parser = heatmap_parsers[0]
    if args.top_k is not None:
        if not hasattr(heatmap_parser, "top_k"):
            raise SystemExit("--top-k requires the HESOD HeatMapParser implementation")
        heatmap_parser.top_k = args.top_k
    elif args.hm_threshold is not None:
        heatmap_parser.threshold = args.hm_threshold
        if hasattr(heatmap_parser, "top_k"):
            heatmap_parser.top_k = None

    active_top_k = getattr(heatmap_parser, "top_k", None)
    routing = (
        {"mode": "top_k", "top_k": int(active_top_k), "ratio": int(heatmap_parser.ratio)}
        if active_top_k is not None
        else {
            "mode": "fixed_threshold",
            "threshold": float(heatmap_parser.threshold),
            "ratio": int(heatmap_parser.ratio),
        }
    )

    captured: dict[str, torch.Tensor | None] = {}

    def _hook(_module, _inputs, output):
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[1], torch.Tensor)
            and output[1].ndim == 2
            and output[1].shape[-1] == 5
        ):
            captured["offsets"] = output[1].detach().cpu()
        else:
            captured["offsets"] = None
        try:
            heatmap = _inputs[0][1][0].detach()
            if torch.max(heatmap) > 1.0 or torch.min(heatmap) < 0.0:
                heatmap = heatmap.sigmoid()
            heatmap = heatmap[:, :1]
            captured["local_maxima"] = (
                (F.max_pool2d(heatmap, 3, stride=1, padding=1) == heatmap)
                # Paper Algorithms 1/2 define C from A = M >= 0.5. Keep this
                # metric cutoff fixed even when a released dataset config
                # (notably UAVDT) routes patches at another threshold.
                & (heatmap >= 0.5)
            ).detach().cpu()
        except (IndexError, TypeError, AttributeError):
            captured["local_maxima"] = None

    handle = heatmap_parser.register_forward_hook(_hook)

    results: dict[str, list[list[float]]] = {}
    maxima_results: dict[str, list[list[float]]] = {}
    try:
        with torch.no_grad():
            for img, _targets, _masks, _m_weights, paths, shapes in dataloader:
                img = img.to(device).float()
                img = norm_imgs(img, model)
                captured.clear()
                model(img)  # no GT masks -- matches run_baseline.sh's test.py invocation (no --use-gt)
                offsets = captured.get("offsets")
                local_maxima = captured.get("local_maxima")
                if local_maxima is None:
                    raise RuntimeError("HeatMapParser hook did not expose an objectness heatmap")
                for bi, path in enumerate(paths):
                    image_key = Path(path).stem
                    if image_key in results:
                        raise RuntimeError(f"duplicate image stem in dataloader: {image_key!r}")
                    if offsets is None or offsets.shape[0] == 0:
                        results[image_key] = []
                    else:
                        bi_mask = offsets[:, 0].long() == bi
                        boxes = offsets[bi_mask, 1:5].clone().float()
                        boxes *= stride  # P3-grid cells -> pixels in the letterboxed network-input image
                        scale_coords(img.shape[2:], boxes, shapes[bi][0], shapes[bi][1])
                        results[image_key] = boxes.tolist()

                    maxima_results[image_key] = []
                    if local_maxima is not None:
                        points = local_maxima[bi, 0].nonzero(as_tuple=False)
                        if points.numel():
                            heat_h, heat_w = local_maxima.shape[-2:]
                            cell_w = img.shape[3] / heat_w
                            cell_h = img.shape[2] / heat_h
                            maxima_boxes = torch.stack(
                                (
                                    points[:, 1] * cell_w,
                                    points[:, 0] * cell_h,
                                    (points[:, 1] + 1) * cell_w,
                                    (points[:, 0] + 1) * cell_h,
                                ),
                                dim=1,
                            ).float()
                            scale_coords(
                                img.shape[2:], maxima_boxes, shapes[bi][0], shapes[bi][1]
                            )
                            valid = (maxima_boxes[:, 2] > maxima_boxes[:, 0]) & (
                                maxima_boxes[:, 3] > maxima_boxes[:, 1]
                            )
                            maxima_results[image_key] = maxima_boxes[valid].tolist()
    finally:
        handle.remove()

    artifact = {
        "schema_version": 2,
        "metadata": {
            "routing": routing,
            "task": args.task,
            "data": str(Path(args.data).resolve()),
            "weights": str(Path(args.weights).resolve()),
            "img_size": int(imgsz),
            "batch_size": args.batch_size,
            "feature_stride": stride,
            "local_maxima_threshold": 0.5,
        },
        "images": results,
        "local_maxima_cells": maxima_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(artifact, f)

    n_with_patches = sum(1 for v in results.values() if v)
    n_patches = sum(len(v) for v in results.values())
    n_maxima = sum(len(v) for v in maxima_results.values())
    print(
        f"Wrote {args.out}: routing={routing}, {len(results)} images, "
        f"{n_with_patches} with >=1 selected patch "
        f"({n_with_patches / len(results) * 100:.1f}%), {n_patches} total selected patches, "
        f"{n_maxima} thresholded local maxima."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
