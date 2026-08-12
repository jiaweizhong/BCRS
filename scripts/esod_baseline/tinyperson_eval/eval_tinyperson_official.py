"""Reproduce the paper's own TinyPerson evaluation protocol (APt50/APs50) against
a run's raw `best_predictions.json` (from ESOD `test.py --save-json`).

ESOD's own test.py already has a `format_tinyperson()` step that remaps our
sequential image_id / 0-indexed category_id into TinyPerson's official image
ids and 1-indexed category id -- but the pycocotools comparison right after it
uses *plain* pycocotools COCOeval (generic COCO-style AP, not TinyPerson's own
size bins or ignore/uncertain handling) and a hardcoded, wrong-for-this-project
relative path ('datasets/TinyPerson/mini_annotations/tiny_set_test_all.json'),
so it has never actually produced a result here (see ESOD-Baseline-Patches.md
#2 for the same class of hardcoded-path issue on other datasets). This script
redoes the same image_id/category_id remapping against the real path, then
evaluates with the *official* TinyPerson-modified COCOeval (vendored in
tinyperson_cocoeval.py, from ucas-vg/PointTinyBenchmark) instead of plain
pycocotools -- that modified evaluator is what actually produces APt50/APs50
in the paper's own convention (area bins tiny/tiny1/tiny2/tiny3/small/
reasonable at IoU 0.25/0.5/0.75, plus ignore/uncertain-region handling). This
is a *different* size convention from this project's own audit_buckets.py
(Very Tiny/Tiny/Small/Medium-Large) -- the two are not directly comparable.

--ignore-uncertain / --use-iod-for-ignore default to on (uncertain boxes
shouldn't count as false negatives if missed; ignore regions are matched by
intersection-over-detection rather than IoU since they are often much larger
than a single detection). Confirmed correct, not just assumed: matches
`evaluate_tiny.py`'s own hardcoded call in the vendored tiny_benchmark package
exactly. img-size=2048 is likewise confirmed against the paper, and the only
supported training profile is `data/hyps/hyp.tinyperson.yaml`.

Usage:
  python eval_tinyperson_official.py \
    --pred /path/to/best_predictions.json \
    --gt /root/autodl-tmp/tiny_set/mini_annotations/tiny_set_test_all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinyperson_cocoeval import COCOeval, Params  # noqa: E402


def remap_to_official(pred_path: Path, gt_path: Path) -> Path:
    with open(gt_path, encoding="utf-8") as f:
        anno = json.load(f)
    image_id_dict = {item["file_name"].split("/")[1][:-4]: item["id"] for item in anno["images"]}

    with open(pred_path, encoding="utf-8") as f:
        jdict = json.load(f)

    missing = 0
    remapped = []
    for item in jdict:
        key = str(item["image_id"])
        if key not in image_id_dict:
            missing += 1
            continue
        item = dict(item)
        item["image_id"] = image_id_dict[key]
        item["category_id"] = int(item["category_id"]) + 1
        remapped.append(item)

    if missing:
        print(f"WARNING: {missing} prediction(s) referenced an image_id not found in {gt_path} -- skipped")

    out_path = pred_path.with_name(pred_path.stem + "_official_remap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(remapped, f)
    print(f"Remapped {len(remapped)} predictions -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, help="raw best_predictions.json from test.py --save-json")
    parser.add_argument("--gt", required=True, help="official tiny_set_test_all.json")
    parser.add_argument(
        "--ignore-uncertain", dest="ignore_uncertain", action="store_true", default=True,
        help="treat 'uncertain' GT boxes as ignorable rather than hard negatives (default: on)",
    )
    parser.add_argument("--no-ignore-uncertain", dest="ignore_uncertain", action="store_false")
    parser.add_argument(
        "--use-iod-for-ignore", dest="use_iod_for_ignore", action="store_true", default=True,
        help="match against ignore regions by intersection-over-detection rather than IoU (default: on)",
    )
    parser.add_argument("--no-use-iod-for-ignore", dest="use_iod_for_ignore", action="store_false")
    args = parser.parse_args(argv)

    from pycocotools.coco import COCO

    pred_path = Path(args.pred).resolve()
    gt_path = Path(args.gt).resolve()
    if not pred_path.is_file():
        raise SystemExit(f"--pred not found: {pred_path}")
    if not gt_path.is_file():
        raise SystemExit(f"--gt not found: {gt_path}")

    remapped_path = remap_to_official(pred_path, gt_path)

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(remapped_path))

    Params.EVAL_STRANDARD = "tiny"
    coco_eval = COCOeval(
        coco_gt, coco_dt, "bbox",
        ignore_uncertain=args.ignore_uncertain,
        use_iod_for_ignore=args.use_iod_for_ignore,
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(
        "\nRead 'IoU=0.50 | area=tiny' for APt50 and 'IoU=0.50 | area=small' for APs50 "
        "from the table above (paper Table II convention)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
