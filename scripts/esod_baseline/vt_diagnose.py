"""Very Tiny selector-dropped vs head-missed diagnostic for any dataset/arm.

Run on the GPU box: python3 vt_diagnose.py <run_name> [options]
  e.g. python3 vt_diagnose.py pest24_yolov5m_baseline          (Pest24 R0, all defaults --
           Pest24 is confirmed uniform 800x600 native, so --native-w/--native-h alone is exact)
       python3 vt_diagnose.py visdrone_yolov5m_channel_pooled_concat_sabl \\
           --labels-dir /root/autodl-tmp/VisDrone_v2/labels/val \\
           --images-dir /root/autodl-tmp/VisDrone_v2/images/val \\
           --classes pedestrian,people,bicycle,car,van,truck,tricycle,awning-tricycle,bus,motor

VisDrone/TinyPerson have genuinely variable per-image native resolution
(confirmed by direct sampling -- not a single fixed size like Pest24), so a
single global --native-w/--native-h would silently mis-bucket box sizes for
every image that doesn't match it. Pass --images-dir to read each image's
real (width, height) via PIL instead (only the header is read, not the full
pixel data, so this stays fast even at thousands of images) -- --native-w/
--native-h are then ignored except as a fallback for label files whose
matching image can't be found. Defaults reproduce the original Pest24-only
script's exact behavior when --images-dir is omitted -- no change for any
existing Pest24 invocation.
"""
import argparse
import json
import os
from collections import defaultdict

from PIL import Image

# Tried in order for every label file when --images-dir is given: the
# explicit --image-ext first (preserves exact prior behavior/priority for
# single-extension datasets like Pest24/VisDrone), then every other common
# suffix. Datasets with mixed image formats (SeaPerson: .bmp in some
# per-video subfolders, .jpg elsewhere) would otherwise silently fall back to
# --native-w/--native-h for every non-matching extension, corrupting their
# size-bucket classification without any visible error -- confirmed 2026-08-21
# (1000/5752 SeaPerson test label files fell back to an assumed 800x600 when
# only .jpg was tried, despite the actual images existing as .bmp).
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

PEST24_CLASS_NAMES = ['Bollworm', 'Meadow borer', 'Gryllotalpa orientalis', 'Little Gecko',
    'Agriotes fuscicollis Miwa', 'Nematode trench', 'Athetis lepigone',
    'Scotogramma trifolii Rottemberg', 'Armyworm', 'Spodoptera cabbage',
    'Anomala corpulenta', 'Spodoptera exigua', 'Plutella xylostella',
    'holotrichia parallela', 'Rice planthopper', 'Yellow tiger',
    'Land tiger', 'eight-character tiger', 'holotrichia oblita',
    'Stem borer', 'Striped rice bore', 'Rice Leaf Roller',
    'Spodoptera litura', 'Melahotus']


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name", nargs="?", default="pest24_yolov5m_baseline",
                         help="run name under ~/esod_baseline_runs/test/<run_name>/")
    parser.add_argument("--labels-dir", default="/root/autodl-tmp/Pest24_v1/labels/test",
                         help="YOLO-format labels directory for the split actually evaluated (test.py's split)")
    parser.add_argument("--images-dir", default=None,
                         help="native images directory, same split as --labels-dir -- if given, each image's "
                              "real (width, height) is read via PIL instead of assuming --native-w/--native-h "
                              "for every image. Required for datasets with variable native resolution "
                              "(VisDrone/TinyPerson); optional for Pest24 (confirmed uniform 800x600)")
    parser.add_argument("--image-ext", default=".jpg",
                         help="image file extension tried first, used with --images-dir; "
                              f"every other suffix in {IMAGE_SUFFIXES} is tried as a fallback "
                              "before giving up on a label file (mixed-format datasets)")
    parser.add_argument("--native-w", type=int, default=800,
                         help="fallback native image width (pixels) when --images-dir is not given, "
                              "or when a label file's matching image can't be found")
    parser.add_argument("--native-h", type=int, default=600, help="fallback native image height (pixels), see --native-w")
    parser.add_argument("--classes", default=",".join(PEST24_CLASS_NAMES),
                         help="comma-separated class names, index-matched to label file class ids")
    parser.add_argument("--very-tiny-area", type=float, default=256.0,
                         help="native-pixel area threshold for the 'Very Tiny' bucket (default 256 = 16x16)")
    opt = parser.parse_args()

    run_name = opt.run_name
    labels_dir, images_dir, image_ext = opt.labels_dir, opt.images_dir, opt.image_ext
    fallback_w, fallback_h = opt.native_w, opt.native_h
    class_names = opt.classes.split(",")
    very_tiny_area = opt.very_tiny_area

    pred_path = os.path.expanduser(f"~/esod_baseline_runs/test/{run_name}/best_predictions.json")
    print(f"=== {run_name} ===")
    if images_dir:
        print(f"per-image native size from: {images_dir}  labels: {labels_dir}  very-tiny area threshold: {very_tiny_area}")
    else:
        print(f"native size (global, no --images-dir given): {fallback_w}x{fallback_h}  "
              f"labels: {labels_dir}  very-tiny area threshold: {very_tiny_area}")

    with open(pred_path, encoding="utf-8") as f:
        preds_raw = json.load(f)
    print(f"predictions.json: {len(preds_raw)} rows")

    preds_by_image = defaultdict(list)
    for row in preds_raw:
        image_id = row["image_id"]
        key = str(int(image_id)) if str(image_id).isdigit() else str(image_id)
        preds_by_image[key].append({
            "class_id": int(row["category_id"]),
            "bbox_xyxy": (row["bbox"][0], row["bbox"][1], row["bbox"][0] + row["bbox"][2], row["bbox"][1] + row["bbox"][3]),
            "score": row.get("score", 1.0),
        })

    # Recursive: some datasets (TinyPerson's raw with_dense tree, UAVDT's
    # per-video UAV-benchmark-M/ folders) nest labels under category/video
    # subdirectories rather than one flat split directory.
    label_files = []
    for dirpath, _, filenames in os.walk(labels_dir):
        for f in filenames:
            if f.endswith(".txt"):
                label_files.append(os.path.relpath(os.path.join(dirpath, f), labels_dir))
    label_files.sort()
    print(f"label files: {len(label_files)}")

    n_size_fallback = 0
    very_tiny_gt = []
    for fn in label_files:
        # basename only for the match key -- test.py's own image_id is
        # `Path(path).stem`, which discards any subdirectory prefix, so a
        # nested label path's key must match on basename alone.
        stem = os.path.splitext(os.path.basename(fn))[0]
        key = str(int(stem)) if stem.isdigit() else stem

        w, h = fallback_w, fallback_h
        if images_dir:
            stem_path = os.path.join(images_dir, os.path.splitext(fn)[0])
            candidates = [stem_path + image_ext] + [
                stem_path + ext for ext in IMAGE_SUFFIXES if ext != image_ext
            ]
            img_path = next((p for p in candidates if os.path.isfile(p)), None)
            if img_path is not None:
                w, h = Image.open(img_path).size  # header-only read, not a full decode
            else:
                n_size_fallback += 1

        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                pw, ph = bw * w, bh * h
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = x1 + pw, y1 + ph
                area = pw * ph
                if area < very_tiny_area:
                    very_tiny_gt.append((key, cls, (x1, y1, x2, y2)))

    if images_dir and n_size_fallback:
        print(f"WARNING: {n_size_fallback}/{len(label_files)} label files had no matching image under "
              f"{images_dir} -- fell back to {fallback_w}x{fallback_h} for those")
    print(f"Very Tiny GT boxes (area<{very_tiny_area}): {len(very_tiny_gt)}")

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

    if not very_tiny_gt:
        print("no Very Tiny GT boxes found -- nothing to report")
        return
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
                    examples["right_class_low_iou"].append((key, class_names[cls], round(iouv, 3), round(score, 3)))
            else:
                outcome_counts["wrong_class_nearby (confusion)"] += 1
                if len(examples["wrong_class_nearby"]) < 5:
                    examples["wrong_class_nearby"].append((key, class_names[cls], class_names[pred_cls], round(iouv, 3), round(score, 3)))

    n_missed = len(very_tiny_gt) - len(recalled)
    print(f"\nMissed Very Tiny GT: {n_missed}")
    for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/n_missed*100:.1f}%)" if n_missed else f"  {k}: {v}")

    print("\nExamples (right class, low IoU -- localization failure):")
    for ex in examples["right_class_low_iou"]:
        print(" ", ex)
    print("\nExamples (wrong class nearby -- confusion):")
    for ex in examples["wrong_class_nearby"]:
        print(" ", ex)


if __name__ == "__main__":
    main()
