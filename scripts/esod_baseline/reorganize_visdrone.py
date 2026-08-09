"""One-off reorganizer: take the official `scripts/data_prepare.py::prepare_visdrone()`
output -- per-split `VisDrone2019-DET-{train,val}/{images,labels,masks}` plus a
`split/{train,val}.txt` file listing the authoritative image path for each
entry (this is what correctly resolves the `_masked.jpg`/`_masked.txt` variant
that images with VisDrone "ignored regions"/"others" annotations get, vs. the
plain filename for images without any) -- and reorganize it into the flat
`images/<split>/`, `labels/<split>/`, `masks/<split>/` layout this project's
tooling (gen_masks.py, audit_buckets.py, the ESOD-format VisDrone.yaml,
LoadImagesAndLabels via a directory glob) expects.

Why not just copy `VisDrone2019-DET-train/images/*` wholesale: for every image
that had an ignored-region box, prepare_visdrone() writes BOTH the untouched
original `xxx.jpg` (never deleted) AND the pixel-masked `xxx_masked.jpg` side
by side in the same directory -- a directory glob would pick up both,
duplicating that image with two different pixel contents and two different
label files. `split/<split>.txt` is the one place that records which of the
two prepare_visdrone() actually intends for training/eval, so it's the
authoritative source for this reorganization, not a directory listing.

Symlinks images and masks (large/binary, no reason to duplicate); copies
labels (tiny text files).

Usage:
  python reorganize_visdrone.py \
    --raw-root /root/autodl-tmp/VisDrone_raw \
    --out-root /root/autodl-tmp/VisDrone_v2 \
    --splits train val
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def reorganize(raw_root: Path, out_root: Path, split: str) -> None:
    split_file = raw_root / "split" / f"{split}.txt"
    if not split_file.is_file():
        raise SystemExit(f"missing {split_file} -- did prepare_visdrone() run for this split?")

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    mask_out = out_root / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    n_images, n_masks = 0, 0
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        img_path = Path(line)
        if not img_path.is_file():
            raise SystemExit(f"{split_file} references missing image: {img_path}")

        label_path = Path(str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt"))
        if not label_path.is_file():
            raise SystemExit(f"missing label for {img_path}: {label_path}")

        _link(img_path, img_out / img_path.name)
        shutil.copyfile(label_path, lbl_out / label_path.name)
        n_images += 1

        mask_path = Path(str(label_path).replace("/labels/", "/masks/").replace(".txt", ".npy"))
        if mask_path.is_file():
            _link(mask_path, mask_out / mask_path.name)
            n_masks += 1

    print(f"[{split}] {n_images} image/label pairs, {n_masks} masks -> {out_root}/*/{split}")
    if n_masks != n_images:
        print(
            f"[{split}] NOTE: {n_images - n_masks} image(s) have no mask yet "
            "(only matters for images with GT boxes -- background-only images "
            "correctly have none; gen_masks.py --verify-only will confirm which)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, type=Path, help="dir containing VisDrone2019-DET-* and split/")
    ap.add_argument("--out-root", required=True, type=Path, help="target dataset root (images/labels/masks per split)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    for split in args.splits:
        reorganize(raw_root, out_root, split)


if __name__ == "__main__":
    main()
