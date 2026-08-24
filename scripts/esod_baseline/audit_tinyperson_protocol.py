"""Fail-fast validation for a paper/released-code TinyPerson experiment.

The TinyPerson benchmark excludes dense images and evaluates the erased-image
test split from ``tiny_set_test_all.json``.  This audit prevents the complete
794/816 ``with_dense`` dataset, raw images, or silently changed pseudo masks
from being presented as a paper-comparable run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROTOCOL = "tinyperson-official-no-dense-erased"


def files_with_suffixes(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument(
        "--expected-mask-mode",
        required=True,
        choices=("paper-hybrid", "released-hybrid", "gaussian"),
    )
    parser.add_argument("--expected-test-images", type=int, default=786)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    gt_path = args.gt.resolve()
    manifest_path = data_root / "tinyperson_protocol.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing protocol manifest: {manifest_path}")
    if not gt_path.is_file():
        raise SystemExit(f"missing official TinyPerson GT: {gt_path}")
    if gt_path.name != "tiny_set_test_all.json":
        raise SystemExit(
            f"wrong TinyPerson GT {gt_path.name!r}; paper-comparable evaluation requires "
            "tiny_set_test_all.json (not a with_dense JSON)"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "protocol": PROTOCOL,
        "dense_images": False,
        "erased_ignore_uncertain_pixels": True,
        "mask_mode": args.expected_mask_mode,
    }
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        if actual != expected:
            raise SystemExit(f"manifest {key}={actual!r}, expected {expected!r}")

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_images = gt.get("images", [])
    if len(gt_images) != args.expected_test_images:
        raise SystemExit(
            f"official test GT has {len(gt_images)} images, expected {args.expected_test_images}"
        )
    dense_annotations = [
        a for a in gt.get("annotations", []) if a.get("in_dense_image", False)
    ]
    if dense_annotations:
        raise SystemExit(
            f"GT unexpectedly contains {len(dense_annotations)} dense-image annotations"
        )

    expected_counts = {
        "train": int(manifest["splits"]["train"]["images"]),
        "val": int(manifest["splits"]["test"]["images"]),
    }
    observed: dict[str, dict[str, int]] = {}
    for split, expected in expected_counts.items():
        images = files_with_suffixes(data_root / "images" / split, IMAGE_SUFFIXES)
        labels = files_with_suffixes(data_root / "labels" / split, {".txt"})
        masks = files_with_suffixes(data_root / "masks" / split, {".npy"})
        if len(images) != expected or len(labels) != expected or len(masks) != expected:
            raise SystemExit(
                f"{split} manifest expects {expected} samples, observed "
                f"images={len(images)} labels={len(labels)} masks={len(masks)}"
            )
        image_stems = {p.stem for p in images}
        if image_stems != {p.stem for p in labels} or image_stems != {
            p.stem for p in masks
        }:
            raise SystemExit(f"{split} image/label/mask stem sets differ")
        observed[split] = {
            "images": len(images),
            "labels": len(labels),
            "masks": len(masks),
        }

    if observed["val"]["images"] != args.expected_test_images:
        raise SystemExit(
            f"converted val has {observed['val']['images']} images, expected {args.expected_test_images}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "protocol": PROTOCOL,
                "mask_mode": args.expected_mask_mode,
                "gt": str(gt_path),
                "splits": observed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
