"""Diagnostic script to inspect dataset label loading and cache contents."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


def inspect_dataset(val_path: str) -> None:
    path = Path(val_path).expanduser().resolve()
    print(f"=== Inspecting Dataset Path: {path} ===")

    if not path.exists():
        print(f"ERROR: Path {path} does not exist!")
        return

    # Check images dir
    images_dir = path / "images" / "val" if (path / "images" / "val").is_dir() else path
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(images)} images in {images_dir}")

    # Check labels dir
    labels_dir = (
        path / "labels" / "val"
        if (path / "labels" / "val").is_dir()
        else path.parent / "labels" / "val"
    )
    if not labels_dir.is_dir():
        print(f"ERROR: Labels directory {labels_dir} does not exist!")
        return
    labels = list(labels_dir.glob("*.txt"))
    print(f"Found {len(labels)} txt label files in {labels_dir}")

    if not labels:
        print("ERROR: No .txt label files found!")
        return

    # Inspect sample txt label file
    sample_file = labels[0]
    sample_content = sample_file.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nSample label file: {sample_file.name}")
    print(f"Total lines in sample label: {len(sample_content)}")
    print("First 3 lines:")
    for line in sample_content[:3]:
        print("  ", line)

    # Check cache file if present
    cache_file = labels_dir.parent / "val.cache"
    if not cache_file.is_file():
        cache_file = path / "val.cache"
    print(f"\nChecking cache file: {cache_file}")
    if cache_file.is_file():
        try:
            cache = torch.load(cache_file)
            print("Cache keys:", [k for k in cache.keys() if isinstance(k, str)][:5])
            results = cache.get("results")
            print(
                f"Cache results tuple (found, missing, empty, corrupt, total): {results}"
            )

            # Inspect first cached image labels
            items = [
                (k, v)
                for k, v in cache.items()
                if k not in ("hash", "version", "results")
            ]
            if items:
                im_path, (l, shape, segments) = items[0]
                print(f"Sample cached image: {im_path}")
                print(f"Sample cached label array shape: {l.shape}, dtype: {l.dtype}")
                if len(l) > 0:
                    print("Sample cached label first row:", l[0])
                else:
                    print("WARNING: Sample cached label is EMPTY (0 rows)!")
        except Exception as exc:
            print(f"Error loading cache file: {exc}")
    else:
        print("Cache file does not exist yet.")


if __name__ == "__main__":
    val_root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/VisDrone"
    inspect_dataset(val_root)
