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
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101-point interpolation (COCO standard)
    x = np.linspace(0, 1, 101)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ap = trapz_fn(np.interp(x, mrec, mpre), x)
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    # Sort globally by confidence descending
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    nc = len(CLASS_NAMES)
    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros(nc)
    r = np.zeros(nc)

    for ci in range(nc):
        i = pred_cls == ci
        n_l = (target_cls == ci).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)

        # Best F1 point for P and R
        f1 = (
            2
            * precision[:, 0]
            * recall[:, 0]
            / (precision[:, 0] + recall[:, 0] + 1e-16)
        )
        best_i = f1.argmax()
        p[ci] = precision[best_i, 0]
        r[ci] = recall[best_i, 0]

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    return p, r, ap, unique_classes


def evaluate_predictions(
    pred_json, labels_dir, images_dir=None, conf_thresh=0.001, nms_thresh=0.5
):
    print(f"Loading predictions from: {pred_json}")
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
    print(f"Loaded {len(preds)} total raw prediction boxes.")
    print(
        f"Filtered to {total_raw_preds} prediction boxes (conf >= {conf_thresh:.4f}) across {len(preds_by_img)} images."
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

        # Parse GT labels (YOLO format: class xc yc w h normalized or CSV format: x,y,w,h,score,cls)
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

                    # Sort by score descending for matching inside image
                    p_sort = np.argsort(-p_scores[p_idx])
                    for i_idx in p_sort:
                        iou_val = best_iou[i_idx]
                        g_idx = best_gt[i_idx]
                        g_global = gt_idx[g_idx]
                        if iou_val >= 0.5 and g_global not in detected:
                            detected.append(g_global)
                            for k, iou_thresh in enumerate(iouv):
                                if iou_val >= iou_thresh:
                                    correct[p_idx[i_idx], k] = True

        stats.append((correct, p_scores, p_classes, gt_classes))

    # Aggregate statistics globally across entire dataset (YOLO / ESOD standard)
    tp = np.concatenate([s[0] for s in stats if len(s[0])], axis=0)
    conf = np.concatenate([s[1] for s in stats if len(s[1])], axis=0)
    pred_cls = np.concatenate([s[2] for s in stats if len(s[2])], axis=0)
    gt_cls = np.concatenate([s[3] for s in stats if len(s[3])], axis=0)

    p, r, ap, _ = ap_per_class(tp, conf, pred_cls, gt_cls)
    ap_50 = ap[:, 0]
    ap_95 = ap.mean(1)

    mp = np.mean(p)
    mr = np.mean(r)
    map50 = np.mean(ap_50)
    map95 = np.mean(ap_95)

    print("=" * 78)
    print(" BCRS OFFICIAL YOLO/ESOD STANDARD DETECTION EVALUATION METRICS")
    print("=" * 78)
    print(
        f"{'Class Name':<18} | {'GT Count':<8} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@.5:.95':<10}"
    )
    print("-" * 78)
    for c in range(len(CLASS_NAMES)):
        c_name = CLASS_NAMES[c]
        n_gt = (gt_cls == c).sum()
        print(
            f"{c_name:<18} | {n_gt:<8} | {p[c]:<10.4f} | {r[c]:<10.4f} | {ap_50[c]:<10.4f} | {ap_95[c]:<10.4f}"
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
        default=0.001,
        help="Confidence score threshold for prediction filtering (default: 0.001)",
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
        evaluate_predictions(pred_path, labels_path, images_path, conf_thresh=args.conf)
    else:
        print(
            f"Error: pred_path={pred_path} or labels_path={labels_path} does not exist."
        )
