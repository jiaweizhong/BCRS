import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# VisDrone class names
CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def box_iou(box1, box2):
    # box1: (N, 4), box2: (M, 4) in [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area[None, :] - inter_area
    return inter_area / (union_area + 1e-6)


def compute_ap(recall, precision):
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def nms_np(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def evaluate_predictions(
    pred_json, labels_dir, images_dir=None, conf_thresh=0.25, nms_thresh=0.5
):
    print(f"Loading predictions from: {pred_json}")
    print(
        f"Applying Confidence Threshold: {conf_thresh:.2f} | NMS IoU Threshold: {nms_thresh:.2f}"
    )
    with open(pred_json, "r") as f:
        preds = json.load(f)

    preds_by_img = {}
    for p in preds:
        if p["score"] >= conf_thresh:
            img_id = str(p["image_id"])
            if img_id not in preds_by_img:
                preds_by_img[img_id] = []
            preds_by_img[img_id].append(p)

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    total_raw_preds = sum(len(v) for v in preds_by_img.values())
    print(
        f"Filtered to {total_raw_preds} high-confidence boxes (conf >= {conf_thresh}) across {len(preds_by_img)} images."
    )
    print(f"Found {len(label_files)} label files in {labels_dir}.\n")

    if images_dir is None or not os.path.exists(images_dir):
        cand = os.path.join(os.path.dirname(labels_dir), "images")
        if os.path.exists(cand):
            images_dir = cand
        else:
            cand_val = os.path.join(os.path.dirname(labels_dir), "images", "val")
            if os.path.exists(cand_val):
                images_dir = cand_val

    iouv = np.linspace(0.5, 0.95, 10)
    stats = []

    for l_path in label_files:
        stem = Path(l_path).stem
        img_preds = preds_by_img.get(stem, [])
        if not img_preds and stem.isnumeric():
            img_preds = preds_by_img.get(str(int(stem)), [])

        # Get native image size
        orig_w, orig_h = 1920, 1080
        if images_dir:
            for ext in [".jpg", ".png", ".jpeg"]:
                img_p = os.path.join(images_dir, stem + ext)
                if os.path.exists(img_p):
                    try:
                        with Image.open(img_p) as im:
                            orig_w, orig_h = im.size
                    except Exception:
                        pass
                    break

        # Parse GT labels
        gt_boxes = []
        gt_classes = []
        with open(l_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                if "," in line_str:
                    parts = line_str.split(",")
                    if len(parts) >= 6:
                        x, y, w, h = (
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                        cls_id = int(parts[5]) - 1
                        if 0 <= cls_id < 10 and w > 0 and h > 0:
                            gt_boxes.append([x, y, x + w, y + h])
                            gt_classes.append(cls_id)
                else:
                    parts = line_str.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        px_w = w * orig_w
                        px_h = h * orig_h
                        px_x1 = (xc - w / 2) * orig_w
                        px_y1 = (yc - h / 2) * orig_h
                        px_x2 = px_x1 + px_w
                        px_y2 = px_y1 + px_h
                        if 0 <= cls_id < 10 and px_w > 0 and px_h > 0:
                            gt_boxes.append([px_x1, px_y1, px_x2, px_y2])
                            gt_classes.append(cls_id)

        gt_boxes = np.array(gt_boxes) if len(gt_boxes) else np.zeros((0, 4))
        gt_classes = (
            np.array(gt_classes, dtype=int)
            if len(gt_classes)
            else np.zeros(0, dtype=int)
        )

        if len(img_preds) == 0:
            if len(gt_classes):
                stats.append(
                    (
                        np.zeros((0, 10), dtype=bool),
                        np.array([]),
                        np.array([]),
                        gt_classes,
                    )
                )
            continue

        raw_p_boxes = np.array(
            [
                [
                    p["bbox"][0],
                    p["bbox"][1],
                    p["bbox"][0] + p["bbox"][2],
                    p["bbox"][1] + p["bbox"][3],
                ]
                for p in img_preds
            ]
        )
        raw_p_scores = np.array([p["score"] for p in img_preds])
        raw_p_classes = np.array([p["category_id"] for p in img_preds])

        # Per-class NMS
        p_boxes_list, p_scores_list, p_classes_list = [], [], []
        for cls in np.unique(raw_p_classes):
            c_mask = raw_p_classes == cls
            c_boxes = raw_p_boxes[c_mask]
            c_scores = raw_p_scores[c_mask]
            keep = nms_np(c_boxes, c_scores, iou_thresh=nms_thresh)
            if len(keep):
                p_boxes_list.append(c_boxes[keep])
                p_scores_list.append(c_scores[keep])
                p_classes_list.append(np.full(len(keep), cls, dtype=int))

        if len(p_boxes_list) == 0:
            if len(gt_classes):
                stats.append(
                    (
                        np.zeros((0, 10), dtype=bool),
                        np.array([]),
                        np.array([]),
                        gt_classes,
                    )
                )
            continue

        p_boxes = np.concatenate(p_boxes_list, axis=0)
        p_scores = np.concatenate(p_scores_list, axis=0)
        p_classes = np.concatenate(p_classes_list, axis=0)

        # Sort descending by score
        sort_ind = np.argsort(-p_scores)
        p_boxes = p_boxes[sort_ind]
        p_scores = p_scores[sort_ind]
        p_classes = p_classes[sort_ind]

        correct = np.zeros((len(p_boxes), 10), dtype=bool)
        if len(gt_classes):
            detected = []
            for cls in np.unique(gt_classes):
                gt_idx = np.where(gt_classes == cls)[0]
                p_idx = np.where(p_classes == cls)[0]

                if len(p_idx) and len(gt_idx):
                    ious = box_iou(p_boxes[p_idx], gt_boxes[gt_idx])
                    best_gt = np.argmax(ious, axis=1)
                    best_iou = np.max(ious, axis=1)

                    for i_idx, (g_idx, iou_val) in enumerate(zip(best_gt, best_iou)):
                        g_global = gt_idx[g_idx]
                        if iou_val >= 0.5 and g_global not in detected:
                            detected.append(g_global)
                            for k, iou_thresh in enumerate(iouv):
                                if iou_val >= iou_thresh:
                                    correct[p_idx[i_idx], k] = True

        stats.append((correct, p_scores, p_classes, gt_classes))

    # Aggregate stats safely
    all_correct = [s[0] for s in stats if len(s[0])]
    all_conf = [s[1] for s in stats if len(s[1])]
    all_pred_cls = [s[2] for s in stats if len(s[2])]
    all_gt_cls = [s[3] for s in stats if len(s[3])]

    if len(all_gt_cls) == 0:
        print("Error: Could not parse ground truth targets from label files.")
        return

    correct = (
        np.concatenate(all_correct, axis=0)
        if len(all_correct)
        else np.zeros((0, 10), dtype=bool)
    )
    conf = np.concatenate(all_conf, axis=0) if len(all_conf) else np.array([])
    pred_cls = (
        np.concatenate(all_pred_cls, axis=0) if len(all_pred_cls) else np.array([])
    )
    gt_cls = np.concatenate(all_gt_cls, axis=0)

    unique_cls = np.unique(gt_cls)
    ap_50 = np.zeros(len(CLASS_NAMES))
    ap_95 = np.zeros(len(CLASS_NAMES))
    precisions = np.zeros(len(CLASS_NAMES))
    recalls = np.zeros(len(CLASS_NAMES))
    gt_counts = np.zeros(len(CLASS_NAMES), dtype=int)

    for c in range(len(CLASS_NAMES)):
        n_gt = (gt_cls == c).sum()
        gt_counts[c] = n_gt
        idx = pred_cls == c
        n_p = idx.sum()

        if n_p == 0 or n_gt == 0:
            continue

        fpc = (1 - correct[idx, 0]).cumsum()
        tpc = correct[idx, 0].cumsum()

        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)

        precisions[c] = precision[-1] if len(precision) else 0
        recalls[c] = recall[-1] if len(recall) else 0

        # AP@0.5
        ap_50[c] = compute_ap(recall, precision)

        # AP@0.5:0.95
        aps = []
        for k in range(10):
            tpc_k = correct[idx, k].cumsum()
            fpc_k = (1 - correct[idx, k]).cumsum()
            rec_k = tpc_k / (n_gt + 1e-16)
            prec_k = tpc_k / (tpc_k + fpc_k)
            aps.append(compute_ap(rec_k, prec_k))
        ap_95[c] = np.mean(aps)

    mp = np.mean([precisions[c] for c in unique_cls])
    mr = np.mean([recalls[c] for c in unique_cls])
    map50 = np.mean([ap_50[c] for c in unique_cls])
    map95 = np.mean([ap_95[c] for c in unique_cls])

    print("=" * 78)
    print(" BCRS EXPERIMENT EVALUATION — DETAILED DETECTION METRICS")
    print("=" * 78)
    print(
        f"{'Class Name':<18} | {'GT Count':<8} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@.5:.95':<10}"
    )
    print("-" * 78)
    for c in range(len(CLASS_NAMES)):
        c_name = CLASS_NAMES[c]
        print(
            f"{c_name:<18} | {gt_counts[c]:<8} | {precisions[c]:<10.4f} | {recalls[c]:<10.4f} | {ap_50[c]:<10.4f} | {ap_95[c]:<10.4f}"
        )
    print("=" * 78)
    print(
        f"{'OVERALL (MEAN)':<18} | {len(gt_cls):<8} | {mp:<10.4f} | {mr:<10.4f} | {map50:<10.4f} | {map95:<10.4f}"
    )
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCRS Detection mAP Evaluator")
    parser.add_argument(
        "pred",
        nargs="?",
        default="work_dirs/bcrs_dual_evidence_concat_visdrone_yolov5m_test/best_predictions.json",
    )
    parser.add_argument(
        "labels", nargs="?", default="/root/autodl-tmp/VisDrone/labels/val"
    )
    parser.add_argument(
        "images", nargs="?", default="/root/autodl-tmp/VisDrone/images/val"
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Confidence score threshold for prediction filtering (default: 0.25)",
    )
    parser.add_argument(
        "--nms",
        "-n",
        type=float,
        default=0.5,
        help="NMS IoU threshold for overlapping box suppression (default: 0.5)",
    )
    args = parser.parse_args()

    pred_path = args.pred
    labels_path = args.labels
    images_path = args.images

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

    if not os.path.exists(labels_path):
        for cand_lbl in [
            "/root/autodl-tmp/VisDrone/labels/val",
            "../../data/VisDrone/labels/val",
            "data/VisDrone/labels/val",
        ]:
            if os.path.exists(cand_lbl):
                labels_path = cand_lbl
                break

    if os.path.exists(pred_path) and os.path.exists(labels_path):
        evaluate_predictions(
            pred_path,
            labels_path,
            images_path,
            conf_thresh=args.conf,
            nms_thresh=args.nms,
        )
    else:
        print(
            f"Error: pred_path={pred_path} or labels_path={labels_path} does not exist."
        )
