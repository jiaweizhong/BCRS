import os
import glob
import math
import numpy as np
from pathlib import Path
from PIL import Image


def box_ios(box1, box2):
    # box: [x1, y1, x2, y2]
    # Intersection over Self (box1 area)
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return inter_area / (b1_area + 1e-6)


def run_oracle_headroom(labels_dir, images_dir=None, grid_size=(8, 8)):
    print(
        f"Running E0.4 Oracle Headroom Analysis across {grid_size[0]}x{grid_size[1]} patch grid..."
    )
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Found {len(label_files)} label files in {labels_dir}")

    if images_dir is None or not os.path.exists(images_dir):
        cand = os.path.join(os.path.dirname(labels_dir), "images")
        images_dir = (
            cand
            if os.path.exists(cand)
            else os.path.join(os.path.dirname(labels_dir), "images", "val")
        )

    total_patches = grid_size[0] * grid_size[1]  # 64 patches
    budget_ratios = [0.125, 0.25, 0.375, 0.50, 0.75, 1.00]  # top-k ratios
    budget_ks = [
        max(1, int(round(r * total_patches))) for r in budget_ratios
    ]  # [8, 16, 24, 32, 48, 64]

    results = {
        k: {"gt_oracle_recalled": 0, "random_recalled": 0, "total_gt": 0}
        for k in budget_ks
    }

    np.random.seed(42)

    for l_path in label_files:
        stem = Path(l_path).stem
        orig_w, orig_h = 1920, 1080
        if images_dir:
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = os.path.join(images_dir, stem + ext)
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        orig_w, orig_h = img.size
                    break

        with open(l_path, "r") as f:
            lines = f.readlines()

        gt_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            px_w, px_h = w * orig_w, h * orig_h
            px_x1 = (xc - w / 2) * orig_w
            px_y1 = (yc - h / 2) * orig_h
            px_x2 = px_x1 + px_w
            px_y2 = px_y1 + px_h
            gt_boxes.append([px_x1, px_y1, px_x2, px_y2])

        if not gt_boxes:
            continue

        num_gt = len(gt_boxes)

        # Generate 8x8 patches
        patch_w = orig_w / grid_size[1]
        patch_h = orig_h / grid_size[0]
        patches = []
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                p_x1 = c * patch_w
                p_y1 = r * patch_h
                p_x2 = p_x1 + patch_w
                p_y2 = p_y1 + patch_h
                patches.append([p_x1, p_y1, p_x2, p_y2])

        # Compute GT coverage per patch
        # patch_gt_coverage[p_idx] = set of GT indices covered by patch
        patch_gt_map = []
        for p_box in patches:
            covered = set()
            for g_idx, g_box in enumerate(gt_boxes):
                if box_ios(g_box, p_box) >= 0.5:
                    covered.add(g_idx)
            patch_gt_map.append(covered)

        for k in budget_ks:
            # 1. GT-Oracle Greedy Selection (Max-Coverage Top-K)
            selected_patches_oracle = []
            uncovered_gt = set(range(num_gt))
            available_patches = set(range(total_patches))

            for _ in range(k):
                if not available_patches:
                    break
                # Find patch that covers the most remaining uncovered GTs
                best_p = max(
                    available_patches, key=lambda p: len(patch_gt_map[p] & uncovered_gt)
                )
                selected_patches_oracle.append(best_p)
                uncovered_gt -= patch_gt_map[best_p]
                available_patches.remove(best_p)

            oracle_recalled_count = num_gt - len(uncovered_gt)

            # 2. Random Top-K Selection
            rand_selected = np.random.choice(total_patches, k, replace=False)
            rand_covered_gt = set()
            for p_idx in rand_selected:
                rand_covered_gt.update(patch_gt_map[p_idx])

            results[k]["total_gt"] += num_gt
            results[k]["gt_oracle_recalled"] += oracle_recalled_count
            results[k]["random_recalled"] += len(rand_covered_gt)

    print("\n" + "=" * 80)
    print(" E0.4 ORACLE HEADROOM ANALYSIS — PATCH BUDGET vs GT RECALL UPPER BOUND")
    print("=" * 80)
    print(
        f"{'Budget K (Patches)':<20} | {'Patch Ratio':<12} | {'GT Oracle Recall':<18} | {'Random Top-K Recall':<20}"
    )
    print("-" * 80)

    for k in budget_ks:
        tot = results[k]["total_gt"]
        orc_rec = results[k]["gt_oracle_recalled"]
        rnd_rec = results[k]["random_recalled"]
        ratio = k / total_patches
        orc_pct = (orc_rec / tot * 100) if tot > 0 else 0.0
        rnd_pct = (rnd_rec / tot * 100) if tot > 0 else 0.0
        print(f"{k:<20} | {ratio:<12.1%} | {orc_pct:<17.2f}% | {rnd_pct:<19.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    labels_path = (
        sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/VisDrone/labels/val"
    )
    img_path = (
        sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/VisDrone/images/val"
    )
    if os.path.exists(labels_path):
        run_oracle_headroom(labels_path, img_path)
    else:
        print(f"Labels directory not found at: {labels_path}")
