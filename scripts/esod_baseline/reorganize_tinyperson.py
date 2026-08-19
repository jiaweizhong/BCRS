"""One-off reorganizer: take `scripts/data_prepare.py::prepare_tinyperson()`'s
output and reorganize it into the flat `images/<split>/`, `labels/<split>/`,
`masks/<split>/` layout this project's tooling expects.

Unlike VisDrone/UAVDT, TinyPerson's raw image paths (e.g.
`erase_with_uncertain_dataset/train/labeled_images/foo.jpg`) contain no
`/labels/` component, so `gen_mask()`'s `path.replace('/labels/', '/masks/')`
is a no-op there -- the `.npy` mask lands right next to the `.jpg`/`.txt` in
the same `labeled_images/`/`labeled_dense_images/`/`pure_bg_images/`
subdirectory, not in a separate tree. This script derives the mask path the
same way (swap `.jpg` -> `.npy`, same directory) rather than assuming a
`/masks/` swap works.

Uses `trainval.txt` (the full official train split) for "train" and
`test.txt` (the full official test split) for "val", not the extra
90/10 `train.txt`/`valid.txt` subsplit `prepare_tinyperson()` also writes --
that subsplit is this script's own convenience addition, not part of the
official TinyPerson protocol, and ESOD's paper-comparable numbers are almost
certainly train-on-full-trainval / eval-on-official-test (flagged as an
assumption, same status as this project's other TinyPerson hyp-file choice).

Since `labeled_images/`, `labeled_dense_images/`, and `pure_bg_images/` are
separate source subdirectories, a same-named file could in principle collide
once flattened -- this fails loudly on any collision rather than silently
overwriting.

Usage:
  python reorganize_tinyperson.py \
    --raw-root /root/autodl-tmp/tiny_set \
    --out-root /root/autodl-tmp/TinyPerson_v1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLIT_FILES = {"train": "trainval.txt", "val": "test.txt"}


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def reorganize(raw_root: Path, out_root: Path, split: str, split_file_name: str) -> None:
    split_file = raw_root / "split" / split_file_name
    if not split_file.is_file():
        raise SystemExit(f"missing {split_file} -- did prepare_tinyperson() run?")

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    mask_out = out_root / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    seen_names: dict[str, Path] = {}
    n_images, n_masks = 0, 0
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        img_path = Path(line)
        if not img_path.is_file():
            reanchored = raw_root / Path(*img_path.parts[1:])
            if not reanchored.is_file():
                raise SystemExit(
                    f"{split_file} references missing image: {img_path} "
                    f"(also tried re-anchored path {reanchored})"
                )
            img_path = reanchored

        if img_path.name in seen_names:
            raise SystemExit(
                f"filename collision in '{split}': {img_path} and {seen_names[img_path.name]} "
                "both map to the same output name -- pick a naming scheme that disambiguates "
                "source subdirectory (e.g. prefix with the parent dir) before proceeding"
            )
        seen_names[img_path.name] = img_path

        label_path = img_path.with_suffix(".txt")
        if not label_path.is_file():
            raise SystemExit(f"missing label for {img_path}: {label_path}")

        _link(img_path, img_out / img_path.name)
        shutil.copyfile(label_path, lbl_out / label_path.name)
        n_images += 1

        mask_path = img_path.with_suffix(".npy")
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
    ap.add_argument("--raw-root", required=True, type=Path, help="dir containing erase_with_uncertain_dataset/ and split/")
    ap.add_argument("--out-root", required=True, type=Path, help="target dataset root (images/labels/masks per split)")
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    manifest = raw_root / "split" / "tinyperson_protocol.json"
    if not manifest.is_file():
        raise SystemExit(
            f"missing {manifest} -- rerun the audited TinyPerson data_prepare.py; "
            "legacy/unmanifested data is intentionally rejected"
        )
    for split, split_file_name in SPLIT_FILES.items():
        reorganize(raw_root, out_root, split, split_file_name)
    shutil.copyfile(manifest, out_root / "tinyperson_protocol.json")
    print(f"[manifest] {manifest} -> {out_root / 'tinyperson_protocol.json'}")


if __name__ == "__main__":
    main()
