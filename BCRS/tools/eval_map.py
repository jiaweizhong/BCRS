import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path

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
    # box: [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

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


def evaluate_predictions(pred_json, labels_dir, images_dir=None):
    print(f"Evaluating predictions from: {pred_json}")
    with open(pred_json, "r") as f:
        preds = json.load(f)

    preds_by_img = {}
    for p in preds:
        img_id = str(p["image_id"])
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(p)

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Loaded {len(preds)} prediction boxes across {len(preds_by_img)} images.")
    print(f"Found {len(label_files)} ground truth label files in {labels_dir}.\n")

    iouv = np.linspace(0.5, 0.95, 10)
    stats = []

    for l_path in label_files:
        stem = Path(l_path).stem
        img_preds = preds_by_img.get(stem, [])
        if not img_preds and stem.isnumeric():
            img_preds = preds_by_img.get(str(int(stem)), [])

        # Parse GT boxes
        gt_boxes = []
        gt_classes = []
        orig_w, orig_h = 1920, 1080
        with open(l_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
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

        p_boxes = np.array(
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
        p_scores = np.array([p["score"] for p in img_preds])
        p_classes = np.array([p["category_id"] for p in img_preds])

        # Sort by score descending
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

    # Aggregate stats
    correct = (
        np.concatenate([s[0] for s in stats if len(s[0])], axis=0)
        if len(stats)
        else np.zeros((0, 10), dtype=bool)
    )
    conf = (
        np.concatenate([s[1] for s in stats if len(s[1])], axis=0)
        if len(stats)
        else np.array([])
    )
    pred_cls = (
        np.concatenate([s[2] for s in stats if len(s[2])], axis=0)
        if len(stats)
        else np.array([])
    )
    gt_cls = (
        np.concatenate([s[3] for s in stats if len(s[3])], axis=0)
        if len(stats)
        else np.array([])
    )

    if len(correct) == 0 or len(gt_cls) == 0:
        print("No predictions or targets to evaluate.")
        return

    # Compute metrics per class
    unique_cls = np.unique(gt_cls)
    ap_50 = np.zeros(len(unique_cls))
    ap_95 = np.zeros(len(unique_cls))
    precisions = np.zeros(len(unique_cls))
    recalls = np.zeros(len(unique_cls))

    for i, c in enumerate(unique_cls):
        idx = pred_cls == c
        n_gt = (gt_cls == c).sum()
        n_p = idx.sum()

        if n_p == 0 or n_gt == 0:
            continue

        fpc = (1 - correct[idx, 0]).cumsum()
        tpc = correct[idx, 0].cumsum()

        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)

        precisions[i] = precision[-1] if len(precision) else 0
        recalls[i] = recall[-1] if len(recall) else 0

        # AP@0.5
        ap_50[i] = compute_ap(recall, precision)

        # AP@0.5:0.95
        aps = []
        for k in range(10):
            tpc_k = correct[idx, k].cumsum()
            fpc_k = (1 - correct[idx, k]).cumsum()
            rec_k = tpc_k / (n_gt + 1e-16)
            prec_k = tpc_k / (tpc_k + fpc_k)
            aps.append(compute_ap(rec_k, prec_k))
        ap_95[i] = np.mean(aps)

    mp = np.mean(precisions)
    mr = np.mean(recalls)
    map50 = np.mean(ap_50)
    map95 = np.mean(ap_95)

    print("=" * 75)
    print(" BCRS EXPERIMENT EVALUATION — DETAILED DETECTION METRICS")
    print("=" * 75)
    print(
        f"{'Class Name':<18} | {'GT Count':<8} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@.5:.95':<10}"
    )
    print("-" * 75)
    for i, c in enumerate(unique_cls):
        c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        n_gt = (gt_cls == c).sum()
        print(
            f"{c_name:<18} | {n_gt:<8} | {precisions[i]:<10.4f} | {recalls[i]:<10.4f} | {ap_50[i]:<10.4f} | {ap_95[i]:<10.4f}"
        )
    print("=" * 75)
    print(
        f"{'OVERALL (ALL)':<18} | {len(gt_cls):<8} | {mp:<10.4f} | {mr:<10.4f} | {map50:<10.4f} | {map95:<10.4f}"
    )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCRS Detection mAP Evaluator")
    parser.add_argument(
        "pred",
        nargs="?",
        default="results/bcrs_dual_evidence_concat_visdrone_yolov5m_test/best_predictions.json",
    )
    parser.add_argument(
        "labels", nargs="?", default="/root/autodl-tmp/VisDrone/labels/val"
    )
    args = parser.parse_args()

    if os.path.exists(args.pred) and os.path.exists(args.labels):
        evaluate_predictions(args.pred, args.labels)
    else:
        print(f"Error: {args.pred} or {args.labels} does not exist.")
