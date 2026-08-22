"""Builds a COCO-format ground-truth JSON from a YoloDetectionDataset split,
cached to disk, so pycocotools.COCOeval can compute standard AP/AP50/AP75 for
the torchvision baselines.

No dataset in this project has this wiring for UAVDT/SeaPerson today --
`hesod/backends/hesod/test.py`'s COCOeval branch has no GT-json path for
either (see HESOD-Experiment-Plan.md SS9.3). `category_id` here is
deliberately 0-indexed to exactly match this project's own
`{name}_predictions.json` convention (audit_buckets.py/vt_diagnose.py), so
predictions can be evaluated against this GT with no id translation --
NOT torchvision's 1-indexed (background=0) training-target convention used
in datasets.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import YoloDetectionDataset


def build_coco_gt(
    dataset: YoloDetectionDataset, cache_path: str | Path, *, force: bool = False
) -> Path:
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        return cache_path

    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(dataset.class_names)]
    ann_id = 1
    for index in range(len(dataset)):
        width, height = dataset.native_size(index)
        image_id = dataset.image_id(index)
        images.append({"id": image_id, "file_name": image_id, "width": width, "height": height})
        boxes, class_ids = dataset.raw_targets(index)
        for (x1, y1, x2, y2), class_id in zip(boxes, class_ids):
            w, h = x2 - x1, y2 - y1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {"images": images, "annotations": annotations, "categories": categories}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return cache_path


def evaluate_coco(gt_json_path: str | Path, pred_json_path: str | Path) -> dict[str, float]:
    """Runs pycocotools COCOeval; returns the 12 standard summary stats
    keyed by name. Predictions must already be in COCO-result format
    (list of {"image_id","category_id","bbox":[x,y,w,h],"score"}) with
    category_id matching the GT json's (0-indexed, see module docstring).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    keys = (
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    )
    return dict(zip(keys, (float(x) for x in coco_eval.stats)))
