#!/usr/bin/env python3
"""Convert the Roboflow TinyPerson-Aug COCO export for ESOD/HESOD training.

This is deliberately separate from ``data_prepare.py::prepare_tinyperson``.
The Roboflow v5 export is an augmented, three-category dataset and is not the
official TinyPerson benchmark protocol.  All source categories are merged into
the single ``person`` class expected by the TinyPerson YOLO model.

Expected source layout::

    tinyperson-aug/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

The output contains ``images/``, ``labels/`` and ``masks/`` for the same three
splits, plus a dataset YAML and an audit manifest.  The source splits are never
reshuffled because augmented variants must not leak across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2


SPLITS = ("train", "valid", "test")
ANNOTATION_NAME = "_annotations.coco.json"
PROTOCOL = "tinyperson-roboflow-aug-v5-nonpaper"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_file_name(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid COCO file_name: {value!r}")
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe COCO file_name: {value!r}")
    return Path(*posix.parts)


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy-fallback"


def load_mask_generator(backend_root: Path, mask_mode: str) -> Callable:
    module_path = backend_root / "scripts" / "data_prepare.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"HESOD mask generator not found: {module_path}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(backend_root))
    try:
        os.chdir(backend_root)
        spec = importlib.util.spec_from_file_location(
            "hesod_tinyperson_aug_mask_generator", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import mask generator: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)

    if mask_mode != "gaussian" and getattr(module, "predictor", None) is None:
        checkpoint = backend_root / "weights" / "sam_vit_h_4b8939.pth"
        raise RuntimeError(
            f"{mask_mode} requires Segment Anything and {checkpoint}; "
            "the converter will not silently fall back to Gaussian masks"
        )
    return module.gen_mask


def validate_source(source_root: Path) -> dict[str, dict]:
    datasets: dict[str, dict] = {}
    for split in SPLITS:
        split_root = source_root / split
        annotation_path = split_root / ANNOTATION_NAME
        if not annotation_path.is_file():
            raise FileNotFoundError(f"missing Roboflow annotation: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("images"), list):
            raise ValueError(f"{annotation_path}: missing COCO images list")
        if not isinstance(data.get("annotations"), list):
            raise ValueError(f"{annotation_path}: missing COCO annotations list")
        if not isinstance(data.get("categories"), list) or not data["categories"]:
            raise ValueError(f"{annotation_path}: missing COCO categories list")
        datasets[split] = data
    return datasets


def estimate_output_bytes(datasets: dict[str, dict], source_root: Path, image_mode: str) -> int:
    # gen_mask stores two float16 planes (heatmap and weight) at full resolution.
    mask_bytes = sum(
        int(image["width"]) * int(image["height"]) * 2 * 2
        for data in datasets.values()
        for image in data["images"]
    )
    if image_mode == "hardlink":
        return mask_bytes
    image_bytes = 0
    for split, data in datasets.items():
        for image in data["images"]:
            image_path = source_root / split / safe_relative_file_name(image["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(f"{split}: missing image {image_path}")
            image_bytes += image_path.stat().st_size
    return mask_bytes + image_bytes


def convert_split(
    split: str,
    data: dict,
    source_root: Path,
    stage_root: Path,
    mask_mode: str,
    image_mode: str,
    gen_mask: Callable,
) -> dict:
    categories = {}
    for category in data["categories"]:
        category_id = category.get("id")
        category_name = category.get("name")
        if category_id in categories:
            raise ValueError(f"{split}: duplicate category id {category_id!r}")
        if not isinstance(category_name, str) or not category_name:
            raise ValueError(f"{split}: invalid category {category!r}")
        categories[category_id] = category_name

    images = {}
    output_stems = set()
    for image in data["images"]:
        image_id = image.get("id")
        if image_id in images:
            raise ValueError(f"{split}: duplicate image id {image_id!r}")
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"{split}: invalid image dimensions for id {image_id!r}")
        relative = safe_relative_file_name(image.get("file_name"))
        label_relative = relative.with_suffix(".txt")
        stem_key = label_relative.as_posix().lower()
        if stem_key in output_stems:
            raise ValueError(f"{split}: colliding image stems at {label_relative}")
        output_stems.add(stem_key)
        images[image_id] = {
            "relative": relative,
            "label_relative": label_relative,
            "width": width,
            "height": height,
        }

    labels: dict[object, list[str]] = defaultdict(list)
    category_counts = Counter()
    clipped_boxes = 0
    skipped_crowd = 0
    for annotation in data["annotations"]:
        image_id = annotation.get("image_id")
        if image_id not in images:
            raise ValueError(f"{split}: annotation references unknown image {image_id!r}")
        category_id = annotation.get("category_id")
        if category_id not in categories:
            raise ValueError(f"{split}: annotation uses unknown category {category_id!r}")
        if annotation.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{split}: invalid bbox in annotation {annotation.get('id')!r}")
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{split}: non-numeric bbox {bbox!r}") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"{split}: non-finite bbox {bbox!r}")
        if width <= 0 or height <= 0:
            raise ValueError(f"{split}: non-positive bbox {bbox!r}")

        image = images[image_id]
        image_width, image_height = image["width"], image["height"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(image_width), x + width), min(float(image_height), y + height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{split}: bbox is outside image bounds: {bbox!r}")
        if (x1, y1, x2, y2) != (x, y, x + width, y + height):
            clipped_boxes += 1

        center_x = ((x1 + x2) / 2.0) / image_width
        center_y = ((y1 + y2) / 2.0) / image_height
        norm_width = (x2 - x1) / image_width
        norm_height = (y2 - y1) / image_height
        labels[image_id].append(
            f"0 {center_x:.8f} {center_y:.8f} {norm_width:.8f} {norm_height:.8f}\n"
        )
        category_counts[categories[category_id]] += 1

    materialization_counts = Counter()
    empty_images = 0
    for image_id, image in images.items():
        relative = image["relative"]
        source_image = source_root / split / relative
        if not source_image.is_file():
            raise FileNotFoundError(f"{split}: missing image {source_image}")
        output_image = stage_root / "images" / split / relative
        materialization_counts[materialize_image(source_image, output_image, image_mode)] += 1

        label_path = stage_root / "labels" / split / image["label_relative"]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_labels = labels.get(image_id, [])
        if not image_labels:
            empty_images += 1
        label_path.write_text("".join(image_labels), encoding="utf-8")

        pixels = cv2.imread(str(output_image))
        if pixels is None:
            raise ValueError(f"{split}: OpenCV cannot read {output_image}")
        actual_height, actual_width = pixels.shape[:2]
        if (actual_width, actual_height) != (image["width"], image["height"]):
            raise ValueError(
                f"{split}: COCO dimensions for {relative} are "
                f"{image['width']}x{image['height']}, actual image is "
                f"{actual_width}x{actual_height}"
            )
        # The shared generator derives masks/ from labels/ by path replacement.
        gen_mask(label_path.as_posix(), pixels, cls_ratio=False, sam_mode=mask_mode)

    return {
        "images": len(images),
        "source_annotations": len(data["annotations"]),
        "training_boxes": sum(len(items) for items in labels.values()),
        "empty_images": empty_images,
        "skipped_crowd": skipped_crowd,
        "clipped_boxes": clipped_boxes,
        "source_category_counts": dict(sorted(category_counts.items())),
        "source_categories": [
            {"id": category_id, "name": name}
            for category_id, name in sorted(categories.items(), key=lambda item: str(item[0]))
        ],
        "image_materialization": dict(sorted(materialization_counts.items())),
    }


def write_yaml(path: Path, output_root: Path) -> None:
    content = (
        f"# {PROTOCOL}; exploratory only, not the official TinyPerson benchmark.\n"
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names: [person]\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--yaml",
        type=Path,
        help="dataset YAML path (default: OUT_ROOT with .yaml suffix)",
    )
    parser.add_argument(
        "--mask-mode",
        required=True,
        choices=("paper-hybrid", "released-hybrid", "gaussian"),
        help="explicit pseudo-mask protocol; hybrid modes require SAM",
    )
    parser.add_argument(
        "--image-mode",
        default="hardlink",
        choices=("hardlink", "copy"),
        help="hardlink saves space and falls back to copying across filesystems",
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=repo_root / "hesod" / "backends" / "hesod",
        help="HESOD backend containing scripts/data_prepare.py and SAM weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.out_root.resolve()
    yaml_path = (args.yaml or output_root.with_suffix(".yaml")).resolve()
    backend_root = args.backend_root.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    if output_root.exists():
        raise SystemExit(
            f"output root already exists: {output_root}; use a new path or remove it explicitly"
        )
    if yaml_path.exists():
        raise SystemExit(
            f"dataset YAML already exists: {yaml_path}; use a new path or remove it explicitly"
        )

    if output_root == source_root or source_root in output_root.parents:
        raise SystemExit("out-root must not be the source dataset or one of its subdirectories")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    datasets = validate_source(source_root)
    estimated_bytes = estimate_output_bytes(datasets, source_root, args.image_mode)
    free_bytes = shutil.disk_usage(output_root.parent).free
    print(
        f"estimated output: {estimated_bytes / 1024 ** 3:.2f} GiB; "
        f"free at {output_root.parent}: {free_bytes / 1024 ** 3:.2f} GiB"
    )
    if free_bytes < estimated_bytes * 1.10:
        raise SystemExit(
            "insufficient free space: require estimated output plus a 10% safety margin"
        )
    gen_mask = load_mask_generator(backend_root, args.mask_mode)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent)
    )
    try:
        split_stats = {}
        for split in SPLITS:
            print(f"[{split}] converting COCO labels and generating {args.mask_mode} masks")
            split_stats[split] = convert_split(
                split,
                datasets[split],
                source_root,
                stage_root,
                args.mask_mode,
                args.image_mode,
                gen_mask,
            )

        manifest = {
            "protocol": PROTOCOL,
            "paper_comparable": False,
            "reason": (
                "Roboflow augmented export with non-official splits/categories; "
                "must not be evaluated as the TinyPerson paper reproduction"
            ),
            "source_root": source_root.as_posix(),
            "mask_mode": args.mask_mode,
            "category_mapping": "all source categories -> class 0 person",
            "annotations": {
                split: {
                    "path": (source_root / split / ANNOTATION_NAME).as_posix(),
                    "sha256": sha256(source_root / split / ANNOTATION_NAME),
                }
                for split in SPLITS
            },
            "splits": split_stats,
        }
        (stage_root / "tinyperson_aug_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage_root, output_root)
        write_yaml(yaml_path, output_root)
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

    print(f"dataset:  {output_root}")
    print(f"yaml:     {yaml_path}")
    print(f"manifest: {output_root / 'tinyperson_aug_manifest.json'}")
    print("NOTE: this dataset is exploratory and is not paper-comparable TinyPerson.")


if __name__ == "__main__":
    main()
