"""Prepare TinyPerson dataset (Roboflow YOLOv8 export) for BCRS training.

TinyPerson Dataset: A Benchmark for Tiny Person Detection (WACV 2020)
  - Paper: https://arxiv.org/abs/1912.10664
  - This script handles the Roboflow YOLOv8 export format from Kaggle:
    https://www.kaggle.com/datasets/yunusemrear/tinyperson-visdrone-widerperson

Input layout (Roboflow YOLOv8 export)::

    {raw_dir}/
      tinyperson.v5i.yolov8/
        data.yaml
        train/
          images/   *.jpg
          labels/   *.txt   (YOLO format: cls cx cy w h, normalised)
        valid/
          images/
          labels/
        test/
          images/
          labels/

This script:
  1. Reads the existing YOLO labels as-is.
  2. Generates COCO JSON annotation files for PyCOCOtools evaluation.
  3. Writes a BCRS-compatible dataset YAML (path-portable, no hardcoded roots).

Usage::

    python -m bcrs.datasets.tinyperson \\
        --raw-dir /path/to/TinyPerson_raw/tinyperson.v5i.yolov8 \\
        --output-dir /path/to/TinyPerson

Or via environment variables::

    BCRS_TINYPERSON_RAW=/path/to/tinyperson.v5i.yolov8 \\
    BCRS_TINYPERSON_OUTPUT=/path/to/TinyPerson \\
    python -m bcrs.datasets.tinyperson
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

TINYPERSON_CLASSES = ("person",)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Roboflow data.yaml split keys -> BCRS canonical split names
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "TinyPerson preparation requires Pillow; run `pip install Pillow`."
        ) from exc
    with Image.open(path) as img:
        return img.size  # (width, height)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    split_name: str,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    """Convert YOLO-format labels to a COCO annotation dict.

    Args:
        images_dir: Directory containing image files for this split.
        labels_dir: Directory containing .txt YOLO label files for this split.
        split_name: e.g. 'train', 'val', 'test' (used in file_name field).
        classes:    Ordered tuple of class names.
    """
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 1
    ann_id = 1
    n_images = 0
    n_annotations = 0
    n_skipped = 0

    img_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
    )
    if not img_files:
        print(f"  WARNING: No images found in {images_dir}")
        return coco

    for img_path in img_files:
        try:
            width, height = _image_size(img_path)
        except Exception as exc:
            print(f"  WARNING: Skipping {img_path.name}: {exc}")
            n_skipped += 1
            continue

        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{split_name}/{img_path.name}",
                "width": width,
                "height": height,
            }
        )

        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, nw, nh = (float(p) for p in parts[1:])
                except ValueError:
                    continue

                # Denormalize to pixel coordinates
                x = (cx - nw / 2.0) * width
                y = (cy - nh / 2.0) * height
                box_w = nw * width
                box_h = nh * height

                if box_w <= 0 or box_h <= 0:
                    n_skipped += 1
                    continue

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,  # 1-indexed COCO
                        "bbox": [
                            round(x, 3),
                            round(y, 3),
                            round(box_w, 3),
                            round(box_h, 3),
                        ],
                        "area": round(box_w * box_h, 3),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                n_annotations += 1

        image_id += 1
        n_images += 1

    print(
        f"  [{split_name}] {n_images} images, "
        f"{n_annotations} annotations, {n_skipped} skipped."
    )
    return coco


def prepare_tinyperson_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Convert Roboflow YOLOv8 TinyPerson export to BCRS-ready dataset.

    Args:
        raw_dir:    Root of the Roboflow export (contains data.yaml + train/valid/test/).
        output_dir: Output directory for the BCRS-ready dataset.
    """
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_anns = output_dir / "annotations"
    out_anns.mkdir(parents=True, exist_ok=True)

    # Read nc and class names from Roboflow data.yaml if present
    classes = TINYPERSON_CLASSES
    data_yaml = raw_dir / "data.yaml"
    if data_yaml.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            with data_yaml.open() as f:
                meta = yaml.safe_load(f)
            if "names" in meta:
                classes = tuple(meta["names"])
            print(f"  Loaded {len(classes)} classes from {data_yaml}: {classes}")
        except ImportError:
            print("  WARNING: PyYAML not installed; using default class names.")

    for roboflow_split, bcrs_split in SPLIT_MAP.items():
        split_dir = raw_dir / roboflow_split
        if not split_dir.is_dir():
            print(f"  Skipping missing split: {split_dir}")
            continue

        src_images = split_dir / "images"
        src_labels = split_dir / "labels"
        if not src_images.is_dir():
            print(f"  WARNING: No images/ dir for split {roboflow_split}, skipping.")
            continue

        # Symlink/copy images and labels into canonical output layout
        dst_images = out_images / bcrs_split
        dst_labels = out_labels / bcrs_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img_path in src_images.iterdir():
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dst = dst_images / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        if src_labels.is_dir():
            for lbl_path in src_labels.iterdir():
                if lbl_path.suffix == ".txt":
                    dst = dst_labels / lbl_path.name
                    if not dst.exists():
                        shutil.copy2(lbl_path, dst)

        # Generate COCO JSON from YOLO labels
        coco = _yolo_to_coco(dst_images, dst_labels, bcrs_split, classes)
        json_out = out_anns / f"{bcrs_split}.json"
        _atomic_write_json(json_out, coco)
        print(f"  COCO JSON -> {json_out}")

    # Write BCRS-compatible dataset YAML (portable: path is absolute output_dir)
    yaml_content = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(classes)}\n"
        f"names: {list(classes)}\n"
    )
    yaml_path = output_dir / "tinyperson.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nDataset YAML -> {yaml_path}")
    print("TinyPerson preprocessing complete.")


def main(argv: Sequence[str] | None = None) -> None:
    default_raw = os.environ.get("BCRS_TINYPERSON_RAW", "")
    default_out = os.environ.get("BCRS_TINYPERSON_OUTPUT", "")

    parser = argparse.ArgumentParser(
        description="Prepare TinyPerson (Roboflow YOLOv8 export) for BCRS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=not bool(default_raw),
        default=Path(default_raw) if default_raw else None,
        help=(
            "Root of Roboflow YOLOv8 export (the dir containing data.yaml, "
            "train/, valid/, test/). Overrides env var BCRS_TINYPERSON_RAW."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=not bool(default_out),
        default=Path(default_out) if default_out else None,
        help=(
            "Destination for processed YOLO + COCO dataset. "
            "Overrides env var BCRS_TINYPERSON_OUTPUT."
        ),
    )
    args = parser.parse_args(argv)
    prepare_tinyperson_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
