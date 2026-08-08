"""Decompose per-size-bin recall loss into a selector failure vs. a detection-
head failure, using the patch boxes from `dump_selected_patches.py` alongside
the same GT loading and match logic `audit_buckets.py` already uses.

For every GT box, checks whether its center fell inside any patch HeatMapParser
selected for that image (`hesod/backends/esod/models/yolo.py`'s P3 neck branch
consumes only those sliced patches, so "not covered" is a structural miss at
the one detection level with anchors small enough to catch a Very Tiny/Tiny
object -- not a probabilistic one), then cross-tabulates against
`audit_buckets.py`'s existing IoU-matched recall:

  selector_dropped     GT center not inside any selected patch -- invisible
                        to the P3/8 head regardless of its quality.
  covered_but_missed   Inside a selected patch, but no prediction matched at
                        IoU>=iou-thres -- a detection-head failure.
  covered_and_recalled Inside a selected patch and recalled.

If `selector_dropped` dominates a bin's shortfall, the coverage-loss work in
hesod/backends/hesod is targeting the right mechanism. If `covered_but_missed`
dominates instead, the selector is fine and the sparse head itself needs the
attention.

Usage:
  python audit_selector_coverage.py \
    --patches selected_patches.json \
    --pred best_predictions.json \
    --labels /root/autodl-tmp/VisDrone/labels/val \
    --images /root/autodl-tmp/VisDrone/images/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_buckets import (
    DEFAULT_CLASSES,
    SIZE_BINS,
    _canonical_key,
    load_ground_truth,
    load_predictions,
    match_predictions,
)


def load_selected_patches(path: str | Path) -> dict[str, list[tuple[float, float, float, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Patch JSON must contain an object: {path}")
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for key, boxes in payload.items():
        if not isinstance(boxes, list):
            raise ValueError(f"Patch list for image {key!r} is not a list")
        result[_canonical_key(key)] = [tuple(float(v) for v in box) for box in boxes]
    return result


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_in_any_patch(
    center: tuple[float, float], patches: list[tuple[float, float, float, float]]
) -> bool:
    cx, cy = center
    for x1, y1, x2, y2 in patches:
        if x1 <= cx < x2 and y1 <= cy < y2:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", required=True, help="output of dump_selected_patches.py")
    parser.add_argument("--pred", required=True, help="ESOD raw prediction JSON (test.py --save-json)")
    parser.add_argument("--labels", required=True, help="YOLO labels directory, e.g. .../labels/val")
    parser.add_argument("--images", required=True, help="native images directory, e.g. .../images/val")
    parser.add_argument("--classes", help="comma-separated class names; defaults to the 10-class VisDrone list")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    args = parser.parse_args(argv)

    class_names = (
        tuple(name.strip() for name in args.classes.split(",") if name.strip())
        if args.classes
        else DEFAULT_CLASSES
    )

    targets = load_ground_truth(Path(args.labels), Path(args.images), class_names)
    predictions = load_predictions(
        Path(args.pred), confidence_threshold=args.conf_thres, class_names=class_names
    )
    recalled = match_predictions(targets, predictions, args.iou_thres)
    patches_by_image = load_selected_patches(args.patches)

    missing_images = {t.image_key for t in targets} - set(patches_by_image)
    if missing_images:
        print(
            f"WARNING: {len(missing_images)} images have GT but no entry in {args.patches} "
            "(treated as zero selected patches -- check the two runs used the same split/image set)"
        )

    bins = {
        name: {"total": 0, "selector_dropped": 0, "covered_but_missed": 0, "covered_and_recalled": 0}
        for name, _, _ in SIZE_BINS
    }

    for target in targets:
        patches = patches_by_image.get(target.image_key, [])
        covered = _center_in_any_patch(_box_center(target.box), patches)
        is_recalled = target.key in recalled
        for name, lower, upper in SIZE_BINS:
            if lower <= target.area < upper:
                b = bins[name]
                b["total"] += 1
                if not covered:
                    b["selector_dropped"] += 1
                elif is_recalled:
                    b["covered_and_recalled"] += 1
                else:
                    b["covered_but_missed"] += 1
                break

    print(
        f"Selector-coverage audit (conf>={args.conf_thres:g}, IoU>={args.iou_thres:g}, "
        "GT-box-center-in-any-selected-patch test)"
    )
    print("=" * 100)
    header = f"{'Size bin':<25} | {'Total':<7} | {'Selector-dropped':<20} | {'Covered,missed':<18} | {'Covered,recalled':<18}"
    print(header)
    print("-" * 100)
    for name, v in bins.items():
        total = v["total"]
        denom = total or 1
        dropped, missed, ok = v["selector_dropped"], v["covered_but_missed"], v["covered_and_recalled"]
        print(
            f"{name:<25} | {total:<7} | "
            f"{dropped:>6} ({dropped / denom * 100:5.1f}%)     | "
            f"{missed:>6} ({missed / denom * 100:5.1f}%)     | "
            f"{ok:>6} ({ok / denom * 100:5.1f}%)"
        )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
