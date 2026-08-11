"""Full per-IoU-threshold AP breakdown for ESOD `test.py --save-json` output,
via real pycocotools COCOeval (not the YOLOv5-style `ap_per_class` `test.py`
itself uses) -- so a specific single-IoU-threshold number (e.g. IoU=0.70) can
be read off directly, instead of only the COCO-convention AP@[.5:.95]/AP50/AP75
`test.py` prints.

Motivation: UAVDT's official MATLAB evaluation toolkit
(`evaluation/UAV-benchmark-MOTD_v1.0/utils/evalRes.m`) matches detections to
GT at a single IoU threshold that **defaults to 0.70**
(`if(nargin<3||isempty(thr)), thr=.7; end`), not COCO's 0.5:0.95 ten-threshold
average, and computes AP via VOC-style PR-curve integration
(`CalculateDetectionPR_overall.m` -> `VOCap.m`), not this script's/pycocotools'
101-point interpolation. This script gets close (same IoU-matching logic,
same class-agnostic single-class pooling for a 1-class problem) without
requiring MATLAB/Octave -- close enough to sanity-check whether an
unexpectedly large or small gap vs. a paper's reported AP is an evaluation-
protocol mismatch rather than a genuine model-quality difference. It is NOT
a byte-exact reproduction of VOCap's specific interpolation, so treat this as
a strong directional check, not a final number to publish as-is.

Reuses audit_buckets.py's YOLO-label/prediction loaders so results are
guaranteed consistent with this project's other audit tooling.

Usage:
  python eval_at_iou_thresholds.py \
    --pred  <run>/best_predictions.json \
    --labels /root/autodl-tmp/UAVDT_v2/labels/test \
    --images /root/autodl-tmp/UAVDT_v2/images/test \
    --classes vehicle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_buckets import Prediction, load_ground_truth, load_predictions  # noqa: E402


def build_coco_gt(labels_dir: Path, images_dir: Path, class_names: tuple[str, ...]) -> tuple[dict, dict[str, int]]:
    from PIL import Image

    gts = load_ground_truth(labels_dir, images_dir, class_names)
    image_keys = sorted({gt.image_key for gt in gts})
    image_id_of = {key: i + 1 for i, key in enumerate(image_keys)}

    images = []
    for key in image_keys:
        # Re-derive width/height the same way load_ground_truth did (by
        # matching filename stem against images_dir); cheap relative to the
        # PIL opens load_ground_truth already did per-annotation-file.
        candidates = list(images_dir.glob(f"{key}.*"))
        with Image.open(candidates[0]) as im:
            w, h = im.size
        images.append({"id": image_id_of[key], "file_name": key, "width": w, "height": h})

    annotations = []
    for i, gt in enumerate(gts, start=1):
        x1, y1, x2, y2 = gt.box
        w, h = x2 - x1, y2 - y1
        annotations.append({
            "id": i,
            "image_id": image_id_of[gt.image_key],
            "category_id": gt.class_id + 1,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, image_id_of


def build_coco_dt(preds: list[Prediction], image_id_of: dict[str, int]) -> list[dict]:
    results = []
    skipped = 0
    for p in preds:
        if p.image_key not in image_id_of:
            skipped += 1
            continue
        x1, y1, x2, y2 = p.box
        results.append({
            "image_id": image_id_of[p.image_key],
            "category_id": p.class_id + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": p.score,
        })
    if skipped:
        print(f"WARNING: {skipped} prediction(s) referenced an image not in the GT set -- skipped")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--classes", required=True, help="comma-separated, matches this dataset's 0-indexed label ids")
    ap.add_argument("--conf-thres", type=float, default=0.001)
    args = ap.parse_args()

    class_names = tuple(c.strip() for c in args.classes.split(","))

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print("Building COCO-format ground truth from YOLO labels ...")
    coco_gt_dict, image_id_of = build_coco_gt(args.labels, args.images, class_names)

    print("Loading predictions ...")
    preds = load_predictions(args.pred, confidence_threshold=args.conf_thres, class_names=class_names)
    dt_list = build_coco_dt(preds, image_id_of)

    gt_path = Path(__file__).resolve().parent / ".tmp_coco_gt.json"
    dt_path = Path(__file__).resolve().parent / ".tmp_coco_dt.json"
    gt_path.write_text(json.dumps(coco_gt_dict))
    dt_path.write_text(json.dumps(dt_list))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(dt_path))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\nPer-IoU-threshold AP (area=all, maxDets=100, averaged over classes/recall -- pycocotools' own array):")
    iou_thrs = coco_eval.params.iouThrs
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    a_idx = coco_eval.params.areaRngLbl.index("all")
    m_idx = len(coco_eval.params.maxDets) - 1
    for t_idx, thr in enumerate(iou_thrs):
        pr = precision[t_idx, :, :, a_idx, m_idx]
        pr = pr[pr > -1]
        ap = pr.mean() if pr.size else float("nan")
        marker = "  <-- UAVDT official MATLAB toolkit's default (evalRes.m thr=.7)" if abs(thr - 0.70) < 1e-6 else ""
        print(f"  IoU={thr:.2f}: AP={ap:.4f}{marker}")

    gt_path.unlink(missing_ok=True)
    dt_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
