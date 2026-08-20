#!/usr/bin/env python3
"""Fail-fast audit for ``prepare_tinyperson_aug.py`` output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = "tinyperson-roboflow-aug-v5-nonpaper"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_stems(root: Path, suffixes: set[str]) -> set[str]:
    return {
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def audit(data_root: Path, expected_mask_mode: str) -> dict[str, dict[str, int]]:
    manifest_path = data_root / "tinyperson_aug_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing TinyPerson-Aug manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected TinyPerson-Aug protocol: {manifest.get('protocol')!r}")
    if manifest.get("paper_comparable") is not False:
        raise ValueError("TinyPerson-Aug manifest must explicitly set paper_comparable=false")
    if manifest.get("mask_mode") != expected_mask_mode:
        raise ValueError(
            f"mask mode mismatch: manifest={manifest.get('mask_mode')!r}, "
            f"expected={expected_mask_mode!r}"
        )
    if manifest.get("category_mapping") != "all source categories -> class 0 person":
        raise ValueError("unexpected TinyPerson-Aug category mapping")

    annotations = manifest.get("annotations")
    split_manifest = manifest.get("splits")
    if not isinstance(annotations, dict) or not isinstance(split_manifest, dict):
        raise ValueError("manifest is missing annotations or split statistics")

    summary = {}
    for split in SPLITS:
        images_root = data_root / "images" / split
        labels_root = data_root / "labels" / split
        masks_root = data_root / "masks" / split
        for path in (images_root, labels_root, masks_root):
            if not path.is_dir():
                raise ValueError(f"missing TinyPerson-Aug directory: {path}")

        image_stems = relative_stems(images_root, IMAGE_SUFFIXES)
        label_stems = relative_stems(labels_root, {".txt"})
        mask_stems = relative_stems(masks_root, {".npy"})
        expected_images = split_manifest.get(split, {}).get("images")
        if expected_images != len(image_stems):
            raise ValueError(
                f"{split}: manifest says {expected_images} images, found {len(image_stems)}"
            )
        if image_stems != label_stems:
            raise ValueError(
                f"{split}: image/label stems differ "
                f"(missing labels={len(image_stems - label_stems)}, "
                f"orphan labels={len(label_stems - image_stems)})"
            )
        if image_stems != mask_stems:
            raise ValueError(
                f"{split}: image/mask stems differ "
                f"(missing masks={len(image_stems - mask_stems)}, "
                f"orphan masks={len(mask_stems - image_stems)})"
            )

        annotation_info = annotations.get(split, {})
        annotation_path = Path(annotation_info.get("path", ""))
        expected_hash = annotation_info.get("sha256")
        if annotation_path.is_file() and sha256(annotation_path) != expected_hash:
            raise ValueError(f"{split}: source annotation hash changed: {annotation_path}")
        summary[split] = {
            "images": len(image_stems),
            "labels": len(label_stems),
            "masks": len(mask_stems),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--expected-mask-mode",
        required=True,
        choices=("paper-hybrid", "released-hybrid", "gaussian"),
    )
    args = parser.parse_args()
    summary = audit(args.data_root.resolve(), args.expected_mask_mode)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("TinyPerson-Aug audit passed (exploratory, not paper-comparable).")


if __name__ == "__main__":
    main()
