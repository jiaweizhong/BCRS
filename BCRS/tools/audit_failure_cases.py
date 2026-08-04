import json
import os
import glob
import numpy as np
from pathlib import Path


def box_iou_xywh(box1, box2):
    # box: [x, y, w, h] (top-left x, y)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]

    union_area = b1_area + b2_area - inter_area
    return inter_area / (union_area + 1e-6)


def run_audit(pred_json_path, labels_dir, imgsz=(1536, 1536)):
    print(f"Loading predictions from: {pred_json_path}")
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    # Group predictions by image_id
    preds_by_img = {}
    for p in preds:
        img_id = str(p["image_id"])
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(p)

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Found {len(label_files)} label files in {labels_dir}")

    # Bins definition (area in pixels at native image resolution 1536x1536)
    size_bins = {
        "Very Tiny (<16x16)": {"range": (0, 16 * 16), "total": 0, "recalled": 0},
        "Tiny (16x16 - 32x32)": {
            "range": (16 * 16, 32 * 32),
            "total": 0,
            "recalled": 0,
        },
        "Small (32x32 - 96x96)": {
            "range": (32 * 32, 96 * 96),
            "total": 0,
            "recalled": 0,
        },
        "Medium/Large (>96x96)": {"range": (96 * 96, 1e9), "total": 0, "recalled": 0},
    }

    class_names = [
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
    class_stats = {c: {"total": 0, "recalled": 0} for c in class_names}

    total_gt = 0
    total_recalled_gt = 0

    for l_path in label_files:
        stem = Path(l_path).stem
        img_id = stem.split("_")[-1] if "_" in stem else stem
        img_preds = preds_by_img.get(img_id, [])

        with open(l_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            # Convert normalized xywh (0-1) to pixel xywh for 1536x1536
            px_w = w * imgsz[0]
            px_h = h * imgsz[1]
            px_x = (xc - w / 2) * imgsz[0]
            px_y = (yc - h / 2) * imgsz[1]
            gt_box = [px_x, px_y, px_w, px_h]
            gt_area = px_w * px_h

            total_gt += 1

            # Match against predictions
            matched = False
            for pred in img_preds:
                iou = box_iou_xywh(gt_box, pred["bbox"])
                if iou >= 0.5:
                    matched = True
                    break

            if matched:
                total_recalled_gt += 1

            # Update size bin stats
            for b_name, b_data in size_bins.items():
                low, high = b_data["range"]
                if low <= gt_area < high:
                    b_data["total"] += 1
                    if matched:
                        b_data["recalled"] += 1
                    break

            # Update class stats
            if 0 <= cls_id < len(class_names):
                c_name = class_names[cls_id]
                class_stats[c_name]["total"] += 1
                if matched:
                    class_stats[c_name]["recalled"] += 1

    print("\n" + "=" * 70)
    print(" BCRS TARGET FAILURE AUDIT (E0.3) - SIZE BIN BREAKDOWN")
    print("=" * 70)
    print(
        f"{'Size Category':<25} | {'GT Count':<10} | {'Recalled':<10} | {'Recall Rate (%)':<15}"
    )
    print("-" * 70)
    for b_name, b_data in size_bins.items():
        tot = b_data["total"]
        rec = b_data["recalled"]
        rate = (rec / tot * 100) if tot > 0 else 0.0
        print(f"{b_name:<25} | {tot:<10} | {rec:<10} | {rate:<14.2f}%")
    print("-" * 70)
    overall_rate = (total_recalled_gt / total_gt * 100) if total_gt > 0 else 0.0
    print(
        f"{'TOTAL GT TARGETS':<25} | {total_gt:<10} | {total_recalled_gt:<10} | {overall_rate:<14.2f}%"
    )
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" BCRS TARGET FAILURE AUDIT (E0.3) - CLASS BREAKDOWN")
    print("=" * 70)
    print(
        f"{'Class Name':<20} | {'GT Count':<10} | {'Recalled':<10} | {'Recall Rate (%)':<15}"
    )
    print("-" * 70)
    for c_name, c_data in class_stats.items():
        tot = c_data["total"]
        rec = c_data["recalled"]
        rate = (rec / tot * 100) if tot > 0 else 0.0
        print(f"{c_name:<20} | {tot:<10} | {rec:<10} | {rate:<14.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    pred_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/root/BCRS/BCRS/work_dirs/esod_visdrone_yolov5m_test/best_predictions.json"
    )
    labels_path = (
        sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/VisDrone/labels/val"
    )
    if os.path.exists(pred_file) and os.path.exists(labels_path):
        run_audit(pred_file, labels_path)
    else:
        print(
            f"Paths check failed: pred_file={pred_file} (exists: {os.path.exists(pred_file)}), labels_path={labels_path} (exists: {os.path.exists(labels_path)})"
        )
