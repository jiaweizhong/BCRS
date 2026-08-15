"""One-off reorganizer for Pest24 (VOCdevkit/voc2007 layout) into the flat
`images/<split>/`, `labels/<split>/` layout this project's tooling
(gen_masks.py, audit_buckets.py, LoadImagesAndLabels via a directory glob)
expects.

Pest24's raw layout ships `images/`, `Annotations/` (VOC XML, unused here),
`labels/` (already-converted YOLO-format .txt, one per image, all splits
flattened together), and `ImageSets/{train,val,test}.txt` (bare image-ID
lists, e.g. "0000014", one per line -- the authoritative split membership).
`yolo_path/{train,val,test}.txt` also exists but holds stale absolute paths
from whoever originally packaged the release (e.g. "D:/projects/Pest24/...")
and is not used.

The pre-existing `labels/*.txt` was independently verified against a 200-file
random sample of `Annotations/*.xml` (VOC's 1-indexed xmin/ymin convention,
0 mismatches across 1798 boxes) before trusting it here -- see
HESOD-Experiment-Plan.md's Pest24 section. No masks exist yet; run
gen_masks.py on the output of this script to generate them (same as any
other already-YOLO-formatted images/+labels/ tree in this project).

Symlinks images (large/binary, no reason to duplicate); copies labels (tiny
text files) -- same convention as reorganize_visdrone.py/reorganize_uavdt.py.

Usage:
  python reorganize_pest24.py \
    --raw-root /root/autodl-tmp/Pest24/VOCdevkit/voc2007 \
    --out-root /root/autodl-tmp/Pest24_v1 \
    --splits train val test
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def reorganize(raw_root: Path, out_root: Path, split: str) -> None:
    split_file = raw_root / "ImageSets" / f"{split}.txt"
    if not split_file.is_file():
        raise SystemExit(f"missing {split_file}")

    img_dir = raw_root / "images"
    lbl_dir = raw_root / "labels"

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_images, n_empty = 0, 0
    for line in split_file.read_text(encoding="utf-8").splitlines():
        stem = line.strip()
        if not stem:
            continue

        img_path = img_dir / f"{stem}.jpg"
        label_path = lbl_dir / f"{stem}.txt"
        if not img_path.is_file():
            raise SystemExit(f"{split_file} references missing image: {img_path}")
        if not label_path.is_file():
            raise SystemExit(f"missing label for {img_path}: {label_path}")

        _link(img_path, img_out / img_path.name)
        content = label_path.read_bytes()
        (lbl_out / label_path.name).write_bytes(content)
        if not content.strip():
            n_empty += 1
        n_images += 1

    print(f"[{split}] {n_images} image/label pairs ({n_empty} background-only) -> {out_root}/*/{split}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, type=Path, help="dir containing images/, labels/, ImageSets/")
    ap.add_argument("--out-root", required=True, type=Path, help="target dataset root (images/labels per split)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    for split in args.splits:
        reorganize(raw_root, out_root, split)


if __name__ == "__main__":
    main()
