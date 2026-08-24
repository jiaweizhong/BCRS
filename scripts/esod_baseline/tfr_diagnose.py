"""TFR (textured-background false-routing rate) audit for any dataset/arm.
HESOD-Agri-Experiment-Plan.md SS1.3.1:
  TFR(B) = |selected regions in textured-background regions| / max(1, |selected regions|)

Run on the GPU box: python3 tfr_diagnose.py <run_name> [options]
  e.g. python3 tfr_diagnose.py visdrone_yolov5m_channel_pooled_concat_sabl \\
           --images-dir /root/autodl-tmp/VisDrone_v2/images/val \\
           --labels-dir /root/autodl-tmp/VisDrone_v2/labels/val \\
           --native-w 2000 --native-h 1500

Requires `<run_name>/best_selected_regions.json`, produced by test.py's
`--save-regions` flag -- re-run test.py with that flag added before using
this script; it will not exist on older runs.

B_tex is defined purely from image content + GT labels, independent of any
model output (avoids the circularity of using the model's own routing/
detection decisions to judge the model): a P3-resolution (stride-8) pixel is
"background" if no GT box covers it, and "texture-heavy" if its Sobel edge
magnitude is at or above the 75th percentile of all background pixels'
scores across the whole evaluated split. B_tex = background AND texture-heavy.

A selected region (as actually emitted by ada_slicer_fast -- position can be
shifted off the nominal grid to center on the activated area, see
models/common.py's ada_slicer_fast) counts as "in B_tex" if more than half
of its P3-pixel area falls in the B_tex mask. This works directly in the
same P3 coordinate space test.py dumps regions in, so no grid-alignment
assumption is needed.

Defaults reproduce the original Pest24-only script's exact behavior (all
flags optional, Pest24's paths/native size as defaults) -- no change in
behavior for any existing Pest24 invocation.
"""

import argparse
import json
import os

import numpy as np
import cv2

STRIDE = 8
BTEX_PERCENTILE = 75.0
BTEX_REGION_FRACTION = 0.5  # a selected region counts as "in B_tex" if > this fraction of its area is B_tex


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default="pest24_yolov5m_baseline",
        help="run name under ~/esod_baseline_runs/test/<run_name>/",
    )
    parser.add_argument(
        "--images-dir",
        default="/root/autodl-tmp/Pest24_v1/images/test",
        help="native images directory for the split actually evaluated",
    )
    parser.add_argument(
        "--labels-dir",
        default="/root/autodl-tmp/Pest24_v1/labels/test",
        help="YOLO-format labels directory for the same split",
    )
    parser.add_argument(
        "--native-w", type=int, default=800, help="native image width (pixels)"
    )
    parser.add_argument(
        "--native-h", type=int, default=600, help="native image height (pixels)"
    )
    parser.add_argument("--image-ext", default=".jpg", help="image file extension")
    opt = parser.parse_args()

    run_name = opt.run_name
    images_dir, labels_dir = opt.images_dir, opt.labels_dir
    native_w, native_h = opt.native_w, opt.native_h
    image_ext = opt.image_ext

    regions_path = os.path.expanduser(
        f"~/esod_baseline_runs/test/{run_name}/best_selected_regions.json"
    )
    print(f"=== TFR: {run_name} ===")

    with open(regions_path) as f:
        data = json.load(f)
    img_size = data["img_size"]
    regions_by_image = data["regions"]
    assert (
        data["stride"] == STRIDE
    ), f"unexpected stride in {regions_path}: {data['stride']}"

    r = img_size / max(native_w, native_h)
    tw, th = round(native_w * r), round(native_h * r)
    p3_w, p3_h = tw // STRIDE, th // STRIDE
    print(
        f"tensor size: {tw}x{th}  (native {native_w}x{native_h} * {r:.4f}), P3: {p3_w}x{p3_h}"
    )

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    print(f"label files: {len(label_files)}")

    # ---- Pass 1: per-image P3-resolution texture score + GT-occupancy maps ----
    tex_maps, gt_maps = {}, {}
    bg_scores = []
    n_missing_images = 0

    for fn in label_files:
        stem = fn[:-4]
        image_id = int(stem) if stem.isdigit() else stem
        img_path = os.path.join(images_dir, f"{stem}{image_ext}")
        if not os.path.isfile(img_path):
            n_missing_images += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            n_missing_images += 1
            continue

        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # downsample full-res edge magnitude to P3 resolution via block-average
        tex_map = cv2.resize(mag, (p3_w, p3_h), interpolation=cv2.INTER_AREA)

        gt_map = np.zeros((p3_h, p3_w), dtype=bool)
        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cxn, cyn, bwn, bhn = map(float, parts[1:5])
                bx1 = (cxn - bwn / 2) * native_w * r / STRIDE
                by1 = (cyn - bhn / 2) * native_h * r / STRIDE
                bx2 = (cxn + bwn / 2) * native_w * r / STRIDE
                by2 = (cyn + bhn / 2) * native_h * r / STRIDE
                c1, c2 = max(0, int(np.floor(bx1))), min(p3_w, int(np.ceil(bx2)))
                r1, r2 = max(0, int(np.floor(by1))), min(p3_h, int(np.ceil(by2)))
                if c2 > c1 and r2 > r1:
                    gt_map[r1:r2, c1:c2] = True

        tex_maps[image_id] = tex_map
        gt_maps[image_id] = gt_map
        bg_scores.append(tex_map[~gt_map])

    print(f"images missing/unreadable: {n_missing_images}")
    if not bg_scores:
        print("no images processed -- check --images-dir/--labels-dir")
        return

    bg_scores = np.concatenate(bg_scores)
    threshold = np.percentile(bg_scores, BTEX_PERCENTILE)
    print(
        f"background P3-pixel edge score: median={np.median(bg_scores):.2f}, "
        f"{BTEX_PERCENTILE:.0f}th pct (B_tex threshold)={threshold:.2f}, n_bg_pixels={len(bg_scores)}"
    )

    btex_maps = {}
    btex_px, total_bg_px = 0, 0
    for image_id, tex_map in tex_maps.items():
        gt_map = gt_maps[image_id]
        btex = (~gt_map) & (tex_map >= threshold)
        btex_maps[image_id] = btex
        btex_px += int(btex.sum())
        total_bg_px += int((~gt_map).sum())
    print(
        f"B_tex coverage: {btex_px}/{total_bg_px} background P3-pixels "
        f"({btex_px / max(1, total_bg_px) * 100:.1f}%) across the evaluated split"
    )

    # ---- Pass 2: TFR for this run's actual selected regions ----
    n_selected, n_in_btex = 0, 0
    n_images_with_selections = 0
    per_image_fracs = []

    for image_id_str, boxes in regions_by_image.items():
        image_id = int(image_id_str) if image_id_str.isdigit() else image_id_str
        if image_id not in btex_maps or not boxes:
            continue
        btex = btex_maps[image_id]
        n_images_with_selections += 1
        for x1, y1, x2, y2 in boxes:
            x1c, x2c = max(0, int(x1)), min(p3_w, int(x2))
            y1c, y2c = max(0, int(y1)), min(p3_h, int(y2))
            if x2c <= x1c or y2c <= y1c:
                continue
            n_selected += 1
            region_frac = btex[y1c:y2c, x1c:x2c].mean()
            per_image_fracs.append(region_frac)
            if region_frac > BTEX_REGION_FRACTION:
                n_in_btex += 1

    tfr = n_in_btex / max(1, n_selected)
    print(f"\nimages with >=1 selected region: {n_images_with_selections}")
    print(f"total selected regions: {n_selected}")
    print(
        f"selected regions classified in B_tex (>{BTEX_REGION_FRACTION*100:.0f}% overlap): {n_in_btex}"
    )
    print(f"TFR = {tfr * 100:.2f}%")
    if per_image_fracs:
        print(
            f"mean region B_tex-overlap fraction (continuous, not thresholded): "
            f"{np.mean(per_image_fracs) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
