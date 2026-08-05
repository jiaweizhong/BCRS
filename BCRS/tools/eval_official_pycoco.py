import json
import os
import argparse
from pathlib import Path


def run_official_pycoco_eval(pred_json_path, anno_json_path):
    print("=" * 78)
    print(" OFFICIAL PYCOCOTOOLS BBOX DETECTION EVALUATION")
    print("=" * 78)
    print(f"Loading predictions from: {pred_json_path}")
    print(f"Loading ground truth COCO annotations from: {anno_json_path}\n")

    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    with open(anno_json_path, "r") as f:
        coco_anno = json.load(f)

    # Build mapping from image stem/filename to integer image_id in val.json
    stem_to_id = {}
    name_to_id = {}
    for img_info in coco_anno["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        stem_to_id[stem] = img_id
        name_to_id[file_name] = img_id

    # Align prediction image_id to integer image_id
    aligned_preds = []
    unmapped = 0
    for p in preds:
        raw_id = p["image_id"]
        matched_id = None
        if isinstance(raw_id, int) and raw_id in [
            img["id"] for img in coco_anno["images"]
        ]:
            matched_id = raw_id
        else:
            raw_str = str(raw_id)
            if raw_str in stem_to_id:
                matched_id = stem_to_id[raw_str]
            elif raw_str in name_to_id:
                matched_id = name_to_id[raw_str]
            elif raw_str.isnumeric() and int(raw_str) in stem_to_id:
                matched_id = stem_to_id[int(raw_str)]

        if matched_id is not None:
            p_copy = dict(p)
            p_copy["image_id"] = matched_id
            aligned_preds.append(p_copy)
        else:
            unmapped += 1

    print(
        f"Aligned {len(aligned_preds)} prediction boxes across {len(coco_anno['images'])} images. (Unmapped: {unmapped})\n"
    )

    # Temp file for aligned predictions
    temp_json = os.path.join(
        os.path.dirname(pred_json_path), "_aligned_preds_pycoco.json"
    )
    with open(temp_json, "w") as f:
        json.dump(aligned_preds, f)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        cocoGt = COCO(anno_json_path)
        cocoDt = cocoGt.loadRes(temp_json)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        stats = cocoEval.stats
        print("\n" + "=" * 78)
        print(" OFFICIAL COCOEVAL SUMMARY RESULTS")
        print("=" * 78)
        print(
            f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.4f} ({stats[0]*100:.2f}%)"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.4f} ({stats[1]*100:.2f}%)"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.4f} ({stats[2]*100:.2f}%)"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.4f} ({stats[3]*100:.2f}%)"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.4f} ({stats[4]*100:.2f}%)"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.4f} ({stats[5]*100:.2f}%)"
        )
        print("=" * 78 + "\n")

    finally:
        if os.path.exists(temp_json):
            os.remove(temp_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official PyCOCOtools Evaluator")
    parser.add_argument(
        "pred",
        nargs="?",
        default="work_dirs/bcrs_dual_evidence_concat_visdrone_yolov5m_test/best_predictions.json",
    )
    parser.add_argument(
        "anno",
        nargs="?",
        default="/root/autodl-tmp/VisDrone/annotations/val.json",
    )
    args = parser.parse_args()

    pred_path = args.pred
    anno_path = args.anno

    if not os.path.exists(pred_path):
        folder_name = Path(pred_path).parent.name
        for cand in [
            os.path.join("work_dirs", folder_name, "best_predictions.json"),
            os.path.join("results", folder_name, "best_predictions.json"),
            os.path.join(
                "/root/BCRS/BCRS/work_dirs", folder_name, "best_predictions.json"
            ),
            os.path.join(
                "/root/BCRS/BCRS/results", folder_name, "best_predictions.json"
            ),
        ]:
            if os.path.exists(cand):
                pred_path = cand
                break

    if not os.path.exists(anno_path):
        for cand_anno in [
            "/root/autodl-tmp/VisDrone/annotations/val.json",
            "../../data/VisDrone/annotations/val.json",
            "data/VisDrone/annotations/val.json",
        ]:
            if os.path.exists(cand_anno):
                anno_path = cand_anno
                break

    if os.path.exists(pred_path) and os.path.exists(anno_path):
        run_official_pycoco_eval(pred_path, anno_path)
    else:
        print(f"Error: pred_path={pred_path} or anno_path={anno_path} does not exist.")
