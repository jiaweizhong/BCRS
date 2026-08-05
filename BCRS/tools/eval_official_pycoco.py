import json
import os
import argparse
import numpy as np
from pathlib import Path


def box_iou_xywh(box1, box2):
    # box1: (N, 4) [x, y, w, h], box2: (M, 4) [x, y, w, h]
    b1_x1, b1_y1, b1_x2, b1_y2 = (
        box1[:, 0],
        box1[:, 1],
        box1[:, 0] + box1[:, 2],
        box1[:, 1] + box1[:, 3],
    )
    b2_x1, b2_y1, b2_x2, b2_y2 = (
        box2[:, 0],
        box2[:, 1],
        box2[:, 0] + box2[:, 2],
        box2[:, 1] + box2[:, 3],
    )

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    b1_area = b1_x2 - b1_x1
    b1_area = b1_area * (b1_y2 - b1_y1)
    b2_area = b2_x2 - b2_x1
    b2_area = b2_area * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area[None, :] - inter_area
    return inter_area / (union_area + 1e-6)


def compute_coco_ap(recall, precision):
    # 101-point COCO AP interpolation
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0 + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ap = trapz_fn(np.interp(x, mrec, mpre), x)
    return ap


def standalone_cocoeval(aligned_preds, coco_anno):
    print("Running Standalone Native COCOeval Engine...")
    img_ids = set(img["id"] for img in coco_anno["images"])
    cats = sorted(set(cat["id"] for cat in coco_anno["categories"]))

    # Group GTs by (img_id, category_id)
    gt_by_img_cat = {}
    for ann in coco_anno["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        key = (ann["image_id"], ann["category_id"])
        if key not in gt_by_img_cat:
            gt_by_img_cat[key] = []
        gt_by_img_cat[key].append(ann["bbox"])

    # Group Preds by category_id
    preds_by_cat = {c: [] for c in cats}
    for p in aligned_preds:
        if p["category_id"] in preds_by_cat:
            preds_by_cat[p["category_id"]].append(p)

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap_table = np.zeros((len(cats), len(iou_thresholds)))

    for c_idx, cat_id in enumerate(cats):
        cat_preds = preds_by_cat[cat_id]
        if not cat_preds:
            continue

        # Sort all predictions of this category globally by score descending
        cat_preds.sort(key=lambda x: x["score"], reverse=True)

        # Total GT count for this category
        total_gt = sum(
            len(boxes)
            for (img_id, c_id), boxes in gt_by_img_cat.items()
            if c_id == cat_id
        )
        if total_gt == 0:
            continue

        # Group predictions by image_id for fast matching
        preds_by_img = {}
        for p_idx, p in enumerate(cat_preds):
            i_id = p["image_id"]
            if i_id not in preds_by_img:
                preds_by_img[i_id] = []
            preds_by_img[i_id].append((p_idx, p["bbox"]))

        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            tp = np.zeros(len(cat_preds))
            fp = np.zeros(len(cat_preds))

            # Match per image
            for img_id, p_list in preds_by_img.items():
                gt_boxes = gt_by_img_cat.get((img_id, cat_id), [])
                if not gt_boxes:
                    for global_p_idx, _ in p_list:
                        fp[global_p_idx] = 1
                    continue

                p_boxes_arr = np.array([box for _, box in p_list])
                gt_boxes_arr = np.array(gt_boxes)

                ious = box_iou_xywh(p_boxes_arr, gt_boxes_arr)
                gt_detected = set()

                for local_p_idx, (global_p_idx, _) in enumerate(p_list):
                    iou_row = ious[local_p_idx]
                    best_gt = np.argmax(iou_row)
                    best_iou = iou_row[best_gt]

                    if best_iou >= iou_thresh and best_gt not in gt_detected:
                        tp[global_p_idx] = 1
                        gt_detected.add(best_gt)
                    else:
                        fp[global_p_idx] = 1

            tpc = tp.cumsum()
            fpc = fp.cumsum()
            recall = tpc / total_gt
            precision = tpc / (tpc + fpc)

            ap_table[c_idx, iou_idx] = compute_coco_ap(recall, precision)

    map_50 = ap_table[:, 0].mean()
    map_50_95 = ap_table.mean()

    print("\n" + "=" * 78)
    print(" OFFICIAL COCOEVAL NATIVE SUMMARY RESULTS")
    print("=" * 78)
    print(
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {map_50_95:.4f} ({map_50_95*100:.2f}%)"
    )
    print(
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {map_50:.4f} ({map_50*100:.2f}%)"
    )
    print("=" * 78 + "\n")


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

    # Try pycocotools first, fallback to native standalone COCOeval
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        temp_json = os.path.join(
            os.path.dirname(pred_json_path), "_aligned_preds_pycoco.json"
        )
        with open(temp_json, "w") as f:
            json.dump(aligned_preds, f)

        try:
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

    except ModuleNotFoundError:
        standalone_cocoeval(aligned_preds, coco_anno)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official PyCOCOtools Evaluator")
    parser.add_argument(
        "pred",
        nargs="?",
        default="results/bcrs_dual_evidence_concat_visdrone_yolov5m_test/best_predictions.json",
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
            os.path.join("results", folder_name, "best_predictions.json"),
            os.path.join("work_dirs", folder_name, "best_predictions.json"),
            os.path.join(
                "/root/BCRS/BCRS/results", folder_name, "best_predictions.json"
            ),
            os.path.join(
                "/root/BCRS/BCRS/work_dirs", folder_name, "best_predictions.json"
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
