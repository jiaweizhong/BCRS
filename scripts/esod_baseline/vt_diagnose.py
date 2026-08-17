"""Very Tiny selector-dropped vs head-missed diagnostic for any Pest24 arm.
Run on the GPU box: python3 vt_diagnose.py <run_name>
  e.g. python3 vt_diagnose.py pest24_yolov5m_baseline          (R0)
       python3 vt_diagnose.py pest24_yolov5m_sabl              (R1)
       python3 vt_diagnose.py pest24_yolov5m_channel_pooled_concat        (R2)
       python3 vt_diagnose.py pest24_yolov5m_channel_pooled_concat_sabl   (R3)
       python3 vt_diagnose.py pest24_yolov5m_reliability_gate   (F5)
Defaults to pest24_yolov5m_baseline (R0) if no argument given.
"""
import json, os, sys
from collections import defaultdict

RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "pest24_yolov5m_baseline"
PRED_PATH = os.path.expanduser(f"~/esod_baseline_runs/test/{RUN_NAME}/best_predictions.json")
LABELS_DIR = "/root/autodl-tmp/Pest24_v1/labels/test"
print(f"=== {RUN_NAME} ===")

CLASS_NAMES = ['Bollworm', 'Meadow borer', 'Gryllotalpa orientalis', 'Little Gecko',
    'Agriotes fuscicollis Miwa', 'Nematode trench', 'Athetis lepigone',
    'Scotogramma trifolii Rottemberg', 'Armyworm', 'Spodoptera cabbage',
    'Anomala corpulenta', 'Spodoptera exigua', 'Plutella xylostella',
    'holotrichia parallela', 'Rice planthopper', 'Yellow tiger',
    'Land tiger', 'eight-character tiger', 'holotrichia oblita',
    'Stem borer', 'Striped rice bore', 'Rice Leaf Roller',
    'Spodoptera litura', 'Melahotus']

W, H = 800, 600  # confirmed uniform native resolution across Pest24

def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

with open(PRED_PATH, encoding="utf-8") as f:
    preds_raw = json.load(f)
print(f"predictions.json: {len(preds_raw)} rows")

preds_by_image = defaultdict(list)
for row in preds_raw:
    key = str(int(row["image_id"]))
    preds_by_image[key].append({
        "class_id": int(row["category_id"]),
        "bbox_xyxy": (row["bbox"][0], row["bbox"][1], row["bbox"][0] + row["bbox"][2], row["bbox"][1] + row["bbox"][3]),
        "score": row.get("score", 1.0),
    })

label_files = sorted(f for f in os.listdir(LABELS_DIR) if f.endswith(".txt"))
print(f"test label files: {len(label_files)}")

very_tiny_gt = []
for fn in label_files:
    stem = fn[:-4]
    key = str(int(stem))
    with open(os.path.join(LABELS_DIR, fn)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            pw, ph = bw * W, bh * H
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = x1 + pw, y1 + ph
            area = pw * ph
            if area < 256.0:
                very_tiny_gt.append((key, cls, (x1, y1, x2, y2)))

print(f"Very Tiny GT boxes (area<16x16): {len(very_tiny_gt)}")

targets_by_group = defaultdict(list)
for i, (key, cls, box) in enumerate(very_tiny_gt):
    targets_by_group[(key, cls)].append((i, box))

preds_by_group = defaultdict(list)
for key, plist in preds_by_image.items():
    for p in plist:
        preds_by_group[(key, p["class_id"])].append(p)

recalled = set()
for (key, cls), gtlist in targets_by_group.items():
    plist = preds_by_group.get((key, cls), [])
    pairs = []
    for gi, (idx, gbox) in enumerate(gtlist):
        for pi, p in enumerate(plist):
            iouv = iou_xyxy(gbox, p["bbox_xyxy"])
            if iouv >= 0.5:
                pairs.append((iouv, gi, pi))
    pairs.sort(reverse=True)
    matched_gi, matched_pi = set(), set()
    for iouv, gi, pi in pairs:
        if gi in matched_gi or pi in matched_pi:
            continue
        matched_gi.add(gi)
        matched_pi.add(pi)
        recalled.add(gtlist[gi][0])

print(f"Very Tiny recalled: {len(recalled)} / {len(very_tiny_gt)} = {len(recalled)/len(very_tiny_gt)*100:.2f}%")

outcome_counts = defaultdict(int)
examples = defaultdict(list)
for i, (key, cls, gbox) in enumerate(very_tiny_gt):
    if i in recalled:
        continue
    candidates = preds_by_image.get(key, [])
    best = None
    for p in candidates:
        iouv = iou_xyxy(gbox, p["bbox_xyxy"])
        if iouv > 0.0:
            if best is None or iouv > best[0]:
                best = (iouv, p["class_id"] == cls, p["score"], p["class_id"])
    if best is None:
        outcome_counts["no_nearby_prediction (selector likely dropped it)"] += 1
    else:
        iouv, class_match, score, pred_cls = best
        if class_match and iouv >= 0.5:
            outcome_counts["matched_but_stolen_by_other_gt"] += 1
        elif class_match:
            outcome_counts["right_class_low_iou (localization failure)"] += 1
            if len(examples["right_class_low_iou"]) < 5:
                examples["right_class_low_iou"].append((key, CLASS_NAMES[cls], round(iouv,3), round(score,3)))
        else:
            outcome_counts["wrong_class_nearby (confusion)"] += 1
            if len(examples["wrong_class_nearby"]) < 5:
                examples["wrong_class_nearby"].append((key, CLASS_NAMES[cls], CLASS_NAMES[pred_cls], round(iouv,3), round(score,3)))

n_missed = len(very_tiny_gt) - len(recalled)
print(f"\nMissed Very Tiny GT: {n_missed}")
for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/n_missed*100:.1f}%)")

print("\nExamples (right class, low IoU -- localization failure):")
for ex in examples["right_class_low_iou"]:
    print(" ", ex)
print("\nExamples (wrong class nearby -- confusion):")
for ex in examples["wrong_class_nearby"]:
    print(" ", ex)
