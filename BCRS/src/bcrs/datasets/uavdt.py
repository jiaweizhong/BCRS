"""Convert UAVDT Supervisely-format annotations to YOLO and COCO formats.

UAVDT Dataset: The Unmanned Aerial Vehicle Benchmark (ECCV 2018)
  - Paper: https://arxiv.org/abs/1804.00518
  - Categories: car (0), truck (1), bus (2)

Kaggle dataset layout (shakaibkaggle/uavdt-dataset)::

    {raw_dir}/
      train/
        img/
          M0101_img000001.jpg
          M0101_img000002.jpg
          ...
        ann/
          M0101_img000001.jpg.json   # Supervisely rectangle format
          M0101_img000002.jpg.json
          ...
      test/
        img/  ...
        ann/  ...

Supervisely annotation JSON format::

    {
      "size": {"height": 540, "width": 1024},
      "objects": [
        {
          "classTitle": "bus",
          "points": {"exterior": [[x1, y1], [x2, y2]], "interior": []}
        }, ...
      ]
    }

Usage::

    python -m bcrs.datasets.uavdt \\
        --raw-dir /path/to/UAVDT \\
        --output-dir /path/to/UAVDT_processed

Or via environment variables::

    BCRS_UAVDT_RAW=/path/to/UAVDT \\
    BCRS_UAVDT_OUTPUT=/path/to/UAVDT_processed \\
    python -m bcrs.datasets.uavdt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

UAVDT_CLASSES = ("car", "truck", "bus")

# classTitle -> 0-indexed YOLO class id
_CLASS_MAP = {cls: i for i, cls in enumerate(UAVDT_CLASSES)}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def _parse_supervisely_ann(
    ann_path: Path,
) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    """Parse a Supervisely per-image JSON annotation file.

    Returns:
        (width, height, list of (cls_id, cx_norm, cy_norm, w_norm, h_norm))
    """
    with ann_path.open(encoding="utf-8") as f:
        data = json.load(f)

    width: int = data["size"]["width"]
    height: int = data["size"]["height"]
    boxes: list[tuple[int, float, float, float, float]] = []

    for obj in data.get("objects", []):
        cls_title: str = obj.get("classTitle", "")
        cls_id = _CLASS_MAP.get(cls_title)
        if cls_id is None:
            continue  # ignore classes not in our vocabulary

        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) < 2:
            continue

        x1, y1 = float(exterior[0][0]), float(exterior[0][1])
        x2, y2 = float(exterior[1][0]), float(exterior[1][1])
        # Normalise to [0, 1], ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        cx = min(max((x1 + bw / 2.0) / width, 0.0), 1.0)
        cy = min(max((y1 + bh / 2.0) / height, 0.0), 1.0)
        nw = min(max(bw / width, 0.0), 1.0)
        nh = min(max(bh / height, 0.0), 1.0)
        boxes.append((cls_id, cx, cy, nw, nh))

    return width, height, boxes


def _prepare_split(
    split_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    """Convert one UAVDT split (train/test) to YOLO labels + COCO dict.

    Args:
        split_dir:      e.g. /path/to/UAVDT/train
        out_images_dir: Root output images dir; images land in out_images_dir/split_name/
        out_labels_dir: Root output labels dir; labels land in out_labels_dir/split_name/
        split_name:     'train' or 'test'
    """
    img_root = split_dir / "img"
    ann_root = split_dir / "ann"

    if not img_root.is_dir():
        raise FileNotFoundError(
            f"Expected images directory at {img_root}. "
            "Ensure --raw-dir points to the UAVDT root with train/ and test/ subdirs."
        )
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Expected annotations directory at {ann_root}.")

    categories = [{"id": i + 1, "name": c} for i, c in enumerate(UAVDT_CLASSES)]
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

    dst_img_dir = out_images_dir / split_name
    dst_lbl_dir = out_labels_dir / split_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in img_root.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
    )
    if not img_files:
        raise FileNotFoundError(f"No images found inside {img_root}.")

    print(f"  [{split_name}] Processing {len(img_files)} images from {img_root} ...")

    for img_path in img_files:
        # Locate paired Supervisely JSON annotation
        ann_path = ann_root / (img_path.name + ".json")
        if not ann_path.exists():
            print(f"  WARNING: No annotation found for {img_path.name}, skipping.")
            n_skipped += 1
            continue

        try:
            width, height, boxes = _parse_supervisely_ann(ann_path)
        except Exception as exc:
            print(f"  WARNING: Failed to parse {ann_path.name}: {exc}")
            n_skipped += 1
            continue

        # Copy image
        dst_img = dst_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # COCO image record
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{split_name}/{img_path.name}",
                "width": width,
                "height": height,
            }
        )

        # YOLO label file + COCO annotations
        label_lines: list[str] = []
        for cls_id, cx, cy, nw, nh in boxes:
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            # Denormalize for COCO bbox [x, y, w, h]
            bx = (cx - nw / 2.0) * width
            by = (cy - nh / 2.0) * height
            bw = nw * width
            bh = nh * height
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,  # 1-indexed for COCO
                    "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                    "area": round(bw * bh, 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            n_annotations += 1

        lbl_path = dst_lbl_dir / (img_path.stem + ".txt")
        lbl_path.write_text("".join(label_lines), encoding="utf-8")

        image_id += 1
        n_images += 1

    print(
        f"  [{split_name}] Done: {n_images} images, "
        f"{n_annotations} annotations, {n_skipped} skipped."
    )
    return coco


def prepare_uavdt_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Convert UAVDT Supervisely-format dataset to YOLO + COCO format.

    Args:
        raw_dir:    Root of the extracted UAVDT Kaggle zip (contains train/ and test/).
        output_dir: Output directory for processed YOLO + COCO dataset.
    """
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_anns = output_dir / "annotations"
    out_anns.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        split_dir = raw_dir / split_name
        if not split_dir.is_dir():
            print(f"WARNING: Split not found, skipping: {split_dir}")
            continue
        coco = _prepare_split(split_dir, out_images, out_labels, split_name)
        json_path = out_anns / f"{split_name}.json"
        _atomic_write_json(json_path, coco)
        print(f"  COCO JSON -> {json_path}")

    # Portable dataset YAML (absolute path in 'path:' field, relative sub-paths)
    yaml_path = output_dir / "uavdt.yaml"
    yaml_path.write_text(
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/test\n"
        "test: images/test\n"
        f"nc: {len(UAVDT_CLASSES)}\n"
        f"names: {list(UAVDT_CLASSES)}\n",
        encoding="utf-8",
    )
    print(f"\nDataset YAML -> {yaml_path}")
    print("UAVDT preprocessing complete.")


def main(argv: Sequence[str] | None = None) -> None:
    default_raw = os.environ.get("BCRS_UAVDT_RAW", "")
    default_out = os.environ.get("BCRS_UAVDT_OUTPUT", "")

    parser = argparse.ArgumentParser(
        description="Prepare UAVDT dataset (Supervisely JSON) for BCRS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=not bool(default_raw),
        default=Path(default_raw) if default_raw else None,
        help=(
            "Root of extracted UAVDT directory (contains train/ and test/ with "
            "img/ and ann/ subdirs). Overrides env var BCRS_UAVDT_RAW."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=not bool(default_out),
        default=Path(default_out) if default_out else None,
        help=(
            "Destination for processed YOLO + COCO dataset. "
            "Overrides env var BCRS_UAVDT_OUTPUT."
        ),
    )
    args = parser.parse_args(argv)
    prepare_uavdt_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
