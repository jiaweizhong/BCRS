#!/usr/bin/env python3
"""Rebuild TinyPerson's erased no-dense image tree from the official archives.

The public TinyPerson protocol removes images in the dense subset and replaces
regions marked ``ignore``, ``uncertain``, or ``logo`` with the mean colour of
that region.  This utility reads images directly from ``train.tar.gz`` and
``test.tar.gz`` and writes only the images referenced by the audited mini
annotations.  It never treats the unmodified archive images as erased data.

The authors document the erasure semantics but do not publish the encoder and
rounding implementation.  Consequently this is a reproducible semantic
reconstruction, not a claim of byte identity with their pre-generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

SPLITS = {
    "train": {
        "archive": "train.tar.gz",
        "source_annotation": "tiny_set_train.json",
        "target_annotation": "tiny_set_train_all_erase.json",
    },
    "test": {
        "archive": "test.tar.gz",
        "source_annotation": "tiny_set_test.json",
        "target_annotation": "tiny_set_test_all.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_annotation(root: Path, filename: str) -> Path:
    candidates = [
        root / "annotations" / filename,
        root / "annotations" / "annotations" / filename,
    ]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"expected exactly one source annotation named {filename}; checked:\n{rendered}"
        )
    return hits[0]


def safe_relative_file_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"unsafe TinyPerson file_name: {value!r}")
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_index(data: dict) -> tuple[dict[str, int], dict[int, list[dict]]]:
    name_to_id: dict[str, int] = {}
    for image in data["images"]:
        name = safe_relative_file_name(image["file_name"]).as_posix()
        if name in name_to_id:
            raise ValueError(f"duplicate source annotation image: {name}")
        name_to_id[name] = image["id"]

    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    return name_to_id, annotations


def erase_regions(image: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, int]:
    height, width = image.shape[:2]
    original = image.copy()
    erased = 0

    for annotation in annotations:
        if not any(
            bool(annotation.get(flag, False))
            for flag in ("ignore", "uncertain", "logo")
        ):
            continue
        x, y, box_width, box_height = map(float, annotation["bbox"])
        x1 = max(0, min(width, math.floor(x)))
        y1 = max(0, min(height, math.floor(y)))
        x2 = max(0, min(width, math.ceil(x + box_width)))
        y2 = max(0, min(height, math.ceil(y + box_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        region = original[y1:y2, x1:x2]
        mean_colour = np.rint(region.mean(axis=(0, 1))).astype(np.uint8)
        image[y1:y2, x1:x2] = mean_colour
        erased += 1
    return image, erased


def encode_image(image: np.ndarray, suffix: str) -> bytes:
    suffix = suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"cv2 could not encode image with suffix {suffix}")
    return encoded.tobytes()


def prepare_split(root: Path, split: str) -> dict:
    config = SPLITS[split]
    archive_path = root / config["archive"]
    source_annotation = find_source_annotation(root, config["source_annotation"])
    target_annotation = root / "mini_annotations" / config["target_annotation"]
    output_root = root / "erase_with_uncertain_dataset" / split

    for path in (archive_path, target_annotation):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = load_json(source_annotation)
    target = load_json(target_annotation)
    name_to_id, annotations = source_index(source)

    target_names = []
    for image in target["images"]:
        name = safe_relative_file_name(image["file_name"]).as_posix()
        if name not in name_to_id:
            raise ValueError(
                f"{target_annotation} references image absent from {source_annotation}: {name}"
            )
        target_names.append(name)
    if len(target_names) != len(set(target_names)):
        raise ValueError(f"duplicate target image names in {target_annotation}")

    protocol_regions_total = sum(
        sum(
            any(
                bool(annotation.get(flag, False))
                for flag in ("ignore", "uncertain", "logo")
            )
            for annotation in annotations[name_to_id[name]]
        )
        for name in target_names
    )

    output_root.mkdir(parents=True, exist_ok=True)
    written = skipped = erased_regions_count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {
            member.name.lstrip("./"): member
            for member in archive.getmembers()
            if member.isfile()
        }
        for index, name in enumerate(target_names, 1):
            archive_name = f"{split}/{name}"
            member = members.get(archive_name)
            if member is None:
                raise FileNotFoundError(f"{archive_path} has no member {archive_name}")

            destination = output_root.joinpath(*PurePosixPath(name).parts)
            if destination.is_file():
                skipped += 1
                continue

            source_file = archive.extractfile(member)
            if source_file is None:
                raise RuntimeError(f"could not read {archive_name} from {archive_path}")
            payload = source_file.read()
            image = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"could not decode {archive_name}")

            image, erased = erase_regions(image, annotations[name_to_id[name]])
            output = payload if erased == 0 else encode_image(image, destination.suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(output)
            os.replace(temporary, destination)
            written += 1
            erased_regions_count += erased

            if index % 50 == 0 or index == len(target_names):
                print(f"[{split}] {index}/{len(target_names)} images")

    present = sum(
        (output_root.joinpath(*PurePosixPath(name).parts)).is_file()
        for name in target_names
    )
    if present != len(target_names):
        raise RuntimeError(
            f"{split}: only {present}/{len(target_names)} target images are present"
        )

    return {
        "images": len(target_names),
        "written": written,
        "resumed_existing": skipped,
        "protocol_regions_total": protocol_regions_total,
        "erased_regions_written_this_run": erased_regions_count,
        "source_archive": str(archive_path),
        "source_archive_sha256": sha256(archive_path),
        "source_annotation": str(source_annotation),
        "source_annotation_sha256": sha256(source_annotation),
        "target_annotation": str(target_annotation),
        "target_annotation_sha256": sha256(target_annotation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stats = {split: prepare_split(root, split) for split in SPLITS}
    manifest = {
        "schema_version": 1,
        "protocol": "tinyperson-official-erased-reconstruction",
        "byte_identical_to_official_pregenerated_images": False,
        "method": {
            "flags": ["ignore", "uncertain", "logo"],
            "bbox_rounding": "floor x1/y1, ceil x2/y2, clipped to image",
            "fill": "per-channel mean of the original bbox region",
            "modified_jpeg_quality": 95,
            "unmodified_images": "copied byte-for-byte from archive",
        },
        "splits": stats,
    }
    manifest_path = (
        root / "erase_with_uncertain_dataset" / "tinyperson_erasure_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
