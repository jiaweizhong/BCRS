"""One-off reorganizer: take the official
`scripts/data_prepare.py::prepare_seaperson()` output -- images left in place
under `rgb/<video>/<file>(_erased).{jpg,bmp}` with a same-directory `.txt`
label and `.npy` mask per image, plus `split/{train,valid,test}.txt` (full
per-mode image-path lists) -- and reorganize it into the flat
`images/<split>/`, `labels/<split>/`, `masks/<split>/` layout this project's
tooling (audit_buckets.py, vt_diagnose.py, run_*.sh's `require_dir` checks)
expects.

Same situation as reorganize_uavdt.py: SeaPerson's raw layout is in-place per
per-video-sequence subfolder (~300 of them), not split into separate images/
and labels/ dirs the way prepare_visdrone()/prepare_seadronesseev2() do it.
Flattened names are prefixed with the immediate parent directory name to
guard against filename collisions across subfolders (not confirmed to
actually occur for SeaPerson the way it does for UAVDT's identical
img000001.jpg-per-video naming, but cheap to make safe unconditionally).

Unlike UAVDT (train/test only -- its own "valid" slice is unused by this
project's pipeline), SeaPerson ships a genuine official 3-way split and all
three are reorganized here. Unlike reorganize_uavdt.py (which only
regenerates a downsampled train_ds.txt against flattened paths, not full
per-split lists), this script also writes out_root/split/{train,valid,test}.txt
against the flattened paths, so out_root is immediately trainable on its own
-- data/seaperson.yaml should point at out_root/split/*.txt, not raw_root's.

Usage:
  python reorganize_seaperson.py \
    --raw-root /root/autodl-tmp/seaperson \
    --out-root /root/autodl-tmp/seaperson_v2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLIT_FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _flat_name(img_path: Path) -> str:
    # .../rgb/<video>/<file>.jpg -> <video>_<file>.jpg
    return f"{img_path.parent.name}_{img_path.name}"


def reorganize(raw_root: Path, out_root: Path, split: str, split_file_name: str) -> list[Path]:
    split_file = raw_root / "split" / split_file_name
    if not split_file.is_file():
        raise SystemExit(f"missing {split_file} -- did prepare_seaperson() run?")

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    mask_out = out_root / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    flat_paths: list[Path] = []
    n_images, n_masks = 0, 0
    seen_names: dict[str, Path] = {}
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

        flat = _flat_name(img_path)
        if flat in seen_names and seen_names[flat] != img_path:
            raise SystemExit(
                f"flattened name collision: {flat} maps to both "
                f"{seen_names[flat]} and {img_path} -- parent-dir prefix is "
                "not enough disambiguation for this dataset, extend _flat_name()"
            )
        seen_names[flat] = img_path

        label_path = img_path.with_suffix(".txt")
        if not label_path.is_file():
            raise SystemExit(f"missing label for {img_path}: {label_path}")

        img_dst = img_out / flat
        _link(img_path, img_dst)
        shutil.copyfile(label_path, lbl_out / Path(flat).with_suffix(".txt").name)
        n_images += 1
        flat_paths.append(img_dst)

        mask_path = img_path.with_suffix(".npy")
        if mask_path.is_file():
            _link(mask_path, mask_out / Path(flat).with_suffix(".npy").name)
            n_masks += 1

    print(f"[{split}] {n_images} image/label pairs, {n_masks} masks -> {out_root}/*/{split}")
    if n_masks != n_images:
        print(
            f"[{split}] NOTE: {n_images - n_masks} image(s) have no mask yet "
            "(only matters for images with GT boxes -- background-only images "
            "correctly have none; gen_masks.py --verify-only will confirm which)"
        )
    return flat_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, type=Path, help="dir containing rgb/, split/ (prepare_seaperson() output)")
    ap.add_argument("--out-root", required=True, type=Path, help="target dataset root (images/labels/masks per split)")
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()

    split_dir = out_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, split_file_name in SPLIT_FILES.items():
        flat_paths = reorganize(raw_root, out_root, split, split_file_name)
        (split_dir / split_file_name).write_text(
            "".join(str(p) + "\n" for p in flat_paths), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
