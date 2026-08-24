"""List specific GT boxes matching a (class, size-bin, status) filter, with
their image path and any nearby predictions of ANY class -- for visual/manual
root-cause triage of a specific recall anomaly (e.g. "why is recall for large
cars so much lower than large buses in the same UAVDT run").

Reuses audit_buckets.py's GT/prediction loading and matching so results are
directly comparable to that tool's numbers. For each sampled GT box, also
reports every prediction (any class, any confidence >= --near-conf) whose box
overlaps it at all, so the failure mode can usually be told apart without even
opening the image:
  - no prediction anywhere near it -> genuine total miss, no signal at all
  - a nearby prediction of the WRONG class -> class confusion, not geometry
  - a nearby prediction of the RIGHT class but IoU < --iou-thres -> localization
    imprecision, not a missing detection
  - a nearby, correctly-classified, well-localized prediction should not be
    possible to print for a "missed" example (match_predictions would have
    counted it) -- if it ever does, that itself points at an audit-tool bug,
    not a model issue.
The image path is still printed so the actual pixels can be inspected too
(e.g. to check for GT labeling noise: wrong class, bad box, motion blur).

Usage:
  python list_examples.py \
    --pred best_predictions.json --labels .../labels/test --images .../images/test \
    --classes car,truck,bus \
    --class car --size-bin "Medium/Large" --status missed --count 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from audit_buckets import (
    DEFAULT_CLASSES,
    SIZE_BINS,
    _image_index,
    box_iou_xyxy,
    load_ground_truth,
    load_predictions,
    match_predictions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pred", required=True, help="ESOD raw prediction JSON (test.py --save-json)"
    )
    parser.add_argument(
        "--labels", required=True, help="YOLO labels directory, e.g. .../labels/val"
    )
    parser.add_argument(
        "--images", required=True, help="native images directory, e.g. .../images/val"
    )
    parser.add_argument(
        "--classes",
        help="comma-separated class names; defaults to the 10-class VisDrone list",
    )
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument(
        "--near-conf",
        type=float,
        default=0.05,
        help="min score for a prediction to be listed as 'nearby' even if it didn't count as a match",
    )
    parser.add_argument(
        "--class",
        dest="target_class",
        required=True,
        help="GT class to sample, e.g. car",
    )
    parser.add_argument(
        "--size-bin",
        required=True,
        help="substring match against a size-bin name, e.g. 'Medium/Large', 'Very Tiny'",
    )
    parser.add_argument("--status", choices=["missed", "recalled"], default="missed")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    class_names = (
        tuple(c.strip() for c in args.classes.split(",") if c.strip())
        if args.classes
        else DEFAULT_CLASSES
    )
    if args.target_class not in class_names:
        raise SystemExit(f"--class {args.target_class!r} not in {class_names}")
    target_class_id = class_names.index(args.target_class)

    bin_matches = [
        name for name, _, _ in SIZE_BINS if args.size_bin.lower() in name.lower()
    ]
    if len(bin_matches) != 1:
        raise SystemExit(
            f"--size-bin {args.size_bin!r} matched {len(bin_matches)} bins (need exactly 1): {bin_matches}"
        )
    bin_name = bin_matches[0]
    bin_lower, bin_upper = next(
        (lo, hi) for name, lo, hi in SIZE_BINS if name == bin_name
    )

    images_dir = Path(args.images)
    image_paths = _image_index(images_dir)

    targets = load_ground_truth(Path(args.labels), images_dir, class_names)
    predictions = load_predictions(
        Path(args.pred), confidence_threshold=args.conf_thres, class_names=class_names
    )
    recalled = match_predictions(targets, predictions, args.iou_thres)

    preds_by_image: dict[str, list] = {}
    for p in predictions:
        preds_by_image.setdefault(p.image_key, []).append(p)

    want_recalled = args.status == "recalled"
    candidates = [
        t
        for t in targets
        if t.class_id == target_class_id
        and bin_lower <= t.area < bin_upper
        and (t.key in recalled) == want_recalled
    ]

    random.seed(args.seed)
    random.shuffle(candidates)
    candidates = candidates[: args.count]

    print(
        f"{len(candidates)} example(s): class={args.target_class}, bin={bin_name}, status={args.status}\n"
    )
    for t in candidates:
        path = image_paths.get(t.image_key, "<image not found>")
        x1, y1, x2, y2 = t.box
        print(
            f"image={path}  gt_box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})  area={t.area:.0f}px^2"
        )

        nearby = []
        for p in preds_by_image.get(t.image_key, []):
            if p.score < args.near_conf:
                continue
            iou = box_iou_xyxy(t.box, p.box)
            if iou <= 0:
                continue
            nearby.append((iou, p))
        nearby.sort(key=lambda item: item[0], reverse=True)

        if not nearby:
            print(
                f"  -> NO overlapping prediction of any class at score>={args.near_conf:g} (genuine total miss, no signal)"
            )
        else:
            for iou, p in nearby[:5]:
                cls = class_names[p.class_id]
                flag = (
                    "SAME CLASS" if p.class_id == target_class_id else "DIFFERENT CLASS"
                )
                print(
                    f"  -> nearby pred: class={cls} ({flag}) score={p.score:.3f} iou={iou:.3f}"
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
