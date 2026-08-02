"""Convert canonical VisDrone source annotations to YOLO and COCO formats."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


VISDRONE_CLASSES = (
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
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitSummary:
    """Counts and output paths produced for one dataset split."""

    split: str
    images: int
    annotations: int
    skipped_rows: int
    labels_dir: Path
    coco_file: Path


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError(
            "VisDrone preparation requires Pillow; install the unified environment "
            "requirements or run `python -m pip install Pillow`."
        ) from exc

    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError as exc:
        raise ValueError(f"Cannot read image dimensions from {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions {width}x{height}: {path}")
    return width, height


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _as_coco_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _parse_annotation_file(
    path: Path, *, image_id: int, width: int, height: int, next_annotation_id: int
) -> tuple[list[str], list[dict[str, Any]], int, int]:
    labels: list[str] = []
    annotations: list[dict[str, Any]] = []
    skipped_rows = 0
    annotation_id = next_annotation_id

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError(
                    f"Malformed VisDrone annotation at {path}:{line_number}: "
                    f"expected at least 6 comma-separated fields, found {len(row)}"
                )
            try:
                x, y, box_width, box_height, score = (
                    float(value.strip()) for value in row[:5]
                )
                category_value = float(row[5].strip())
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric VisDrone annotation at {path}:{line_number}: {row[:6]}"
                ) from exc

            values = (x, y, box_width, box_height, score, category_value)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Non-finite VisDrone annotation at {path}:{line_number}: {row[:6]}"
                )
            if not category_value.is_integer():
                raise ValueError(
                    f"Non-integer category at {path}:{line_number}: {category_value}"
                )

            category_id = int(category_value)
            if score <= 0 or category_id not in range(1, len(VISDRONE_CLASSES) + 1):
                skipped_rows += 1
                continue
            if box_width <= 0 or box_height <= 0:
                raise ValueError(
                    f"Non-positive bounding box at {path}:{line_number}: "
                    f"{box_width}x{box_height}"
                )

            center_x = (x + box_width / 2.0) / width
            center_y = (y + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            labels.append(
                f"{category_id - 1} {center_x:.6f} {center_y:.6f} "
                f"{normalized_width:.6f} {normalized_height:.6f}\n"
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        _as_coco_number(x),
                        _as_coco_number(y),
                        _as_coco_number(box_width),
                        _as_coco_number(box_height),
                    ],
                    "area": _as_coco_number(box_width * box_height),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return labels, annotations, skipped_rows, annotation_id


def prepare_split(root: Path, split: str, *, dry_run: bool = False) -> SplitSummary:
    """Prepare one canonical VisDrone split below ``root``."""

    images_dir = root / "images" / split
    raw_annotations_dir = root / "raw_annotations" / split
    labels_dir = root / "labels" / split
    coco_file = root / "annotations" / f"{split}.json"

    if not images_dir.is_dir():
        raise ValueError(f"Missing VisDrone image directory: {images_dir}")
    if not raw_annotations_dir.is_dir():
        raise ValueError(f"Missing VisDrone raw annotation directory: {raw_annotations_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")

    image_stems: dict[str, Path] = {}
    for image in images:
        if image.stem in image_stems:
            raise ValueError(
                f"Multiple images share annotation stem {image.stem!r}: "
                f"{image_stems[image.stem]} and {image}"
            )
        image_stems[image.stem] = image

    raw_annotations = {path.stem: path for path in raw_annotations_dir.glob("*.txt")}
    missing = sorted(set(image_stems) - set(raw_annotations))
    extra = sorted(set(raw_annotations) - set(image_stems))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} raw annotation file(s) for split {split!r}: {preview}"
        )
    if extra:
        preview = ", ".join(extra[:5])
        raise ValueError(
            f"Found {len(extra)} raw annotation file(s) without images for split "
            f"{split!r}: {preview}"
        )

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    label_outputs: list[tuple[Path, str]] = []
    skipped_rows = 0
    next_annotation_id = 1

    for image_id, image_path in enumerate(images, start=1):
        width, height = _image_size(image_path)
        labels, annotations, skipped, next_annotation_id = _parse_annotation_file(
            raw_annotations[image_path.stem],
            image_id=image_id,
            width=width,
            height=height,
            next_annotation_id=next_annotation_id,
        )
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        coco_annotations.extend(annotations)
        label_outputs.append((labels_dir / f"{image_path.stem}.txt", "".join(labels)))
        skipped_rows += skipped

    payload = {
        "info": {"description": f"VisDrone2019-DET {split} converted by BCRS"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "object"}
            for category_id, name in enumerate(VISDRONE_CLASSES, start=1)
        ],
    }

    if not dry_run:
        for label_path, content in label_outputs:
            _atomic_write_text(label_path, content)
        _atomic_write_json(coco_file, payload)

    return SplitSummary(
        split=split,
        images=len(coco_images),
        annotations=len(coco_annotations),
        skipped_rows=skipped_rows,
        labels_dir=labels_dir,
        coco_file=coco_file,
    )


def prepare_visdrone(
    root: str | Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    dry_run: bool = False,
) -> tuple[SplitSummary, ...]:
    """Generate YOLO labels and COCO JSON below a unified VisDrone root."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"VisDrone root is not a directory: {dataset_root}")
    if not splits:
        raise ValueError("At least one split is required")
    unsupported = [split for split in splits if split not in {"train", "val", "test"}]
    if unsupported:
        raise ValueError(f"Unsupported VisDrone split(s): {', '.join(unsupported)}")
    if len(set(splits)) != len(splits):
        raise ValueError("VisDrone splits must not be repeated")

    return tuple(prepare_split(dataset_root, split, dry_run=dry_run) for split in splits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert unified VisDrone annotations to YOLO and COCO formats."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("VISDRONE_ROOT"),
        help="unified VisDrone root (defaults to VISDRONE_ROOT)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="splits to prepare",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count annotations without writing outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.root is None:
        parser.error("--root is required when VISDRONE_ROOT is not set")
    try:
        summaries = prepare_visdrone(args.root, splits=args.splits, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    for summary in summaries:
        action = "validated" if args.dry_run else "prepared"
        print(
            f"{summary.split}: {action} {summary.images} images, "
            f"{summary.annotations} annotations, skipped {summary.skipped_rows} ignored rows"
        )
        if not args.dry_run:
            print(f"  YOLO: {summary.labels_dir}")
            print(f"  COCO: {summary.coco_file}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
