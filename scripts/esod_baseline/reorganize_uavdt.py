"""One-off reorganizer: take the official `scripts/data_prepare.py::prepare_uavdt()`
output -- images left in place under `UAV-benchmark-M/<video>/imgXXXXXX(_masked).jpg`
with a same-directory `.txt` label and `.npy` mask per frame, plus
`split/{train,valid,test}.txt` (full per-mode image-path lists) and
`split/train_ds.txt` (every 10th line of train.txt -- UAVDT is a video dataset,
consecutive frames are near-duplicates, so the official pipeline trains on a
10x-downsampled subset, not every frame) -- and reorganize it into the flat
`images/<split>/`, `labels/<split>/`, `masks/<split>/` layout this project's
tooling (gen_masks.py, audit_buckets.py, run_baseline.sh's `require_dir`
checks) expects.

Unlike VisDrone (already split into separate images/ and labels/ dirs by its
own prepare_visdrone()), UAVDT's raw layout is in-place per-video, same as
TinyPerson -- label/mask paths are derived by suffix swap on the SAME
directory, not a directory swap. Unlike TinyPerson, filenames collide across
videos here (every video has its own img000001.jpg, img000002.jpg, ...), so
flattened names are prefixed with the video directory name
(`<video>_imgXXXXXX(_masked).jpg`) to disambiguate -- this is not optional
here the way TinyPerson's fail-loudly check was, collisions are the normal
case.

Splits used: "train" (from split/train.txt, full) and "test" (from
split/test.txt) -- matches run_baseline.sh's existing uavdt convention
("val_split is 'test', not 'val'"). "valid" (prepare_uavdt()'s own held-out
slice of train) is not used by this project's single train/test pipeline and
is skipped. Also regenerates a downsampled `split/train_ds.txt` inside
--out-root, using the FLATTENED (renamed) paths at the same every-10th
stride as the original, since the raw one's paths point at files this script
just moved.

Usage:
  python reorganize_uavdt.py \
    --raw-root /root/autodl-tmp/UAVDT_raw \
    --out-root /root/autodl-tmp/UAVDT_v2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLIT_FILES = {"train": "train.txt", "test": "test.txt"}


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _flat_name(img_path: Path) -> str:
    # .../UAV-benchmark-M/<video>/imgXXXXXX(_masked).jpg -> <video>_imgXXXXXX(_masked).jpg
    return f"{img_path.parent.name}_{img_path.name}"


def reorganize(
    raw_root: Path, out_root: Path, split: str, split_file_name: str
) -> list[Path]:
    split_file = raw_root / "split" / split_file_name
    if not split_file.is_file():
        raise SystemExit(f"missing {split_file} -- did prepare_uavdt() run?")

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    mask_out = out_root / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    flat_paths: list[Path] = []
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

        flat = _flat_name(img_path)
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

    print(
        f"[{split}] {n_images} image/label pairs, {n_masks} masks -> {out_root}/*/{split}"
    )
    if n_masks != n_images:
        print(
            f"[{split}] NOTE: {n_images - n_masks} image(s) have no mask yet "
            "(only matters for images with GT boxes -- background-only images "
            "correctly have none; gen_masks.py --verify-only will confirm which)"
        )
    return flat_paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="dir containing UAV-benchmark-M/, M_attr/, split/",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="target dataset root (images/labels/masks per split)",
    )
    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()

    flat_by_split = {}
    for split, split_file_name in SPLIT_FILES.items():
        flat_by_split[split] = reorganize(raw_root, out_root, split, split_file_name)

    # Regenerate the downsampled training list against the FLATTENED paths,
    # same every-10th stride prepare_uavdt() used on the raw (pre-flatten) list.
    train_ds = flat_by_split["train"][::10]
    split_dir = out_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_ds.txt").write_text(
        "".join(str(p) + "\n" for p in train_ds), encoding="utf-8"
    )
    print(
        f"[train_ds] {len(train_ds)} images (every 10th of {len(flat_by_split['train'])}) -> {split_dir}/train_ds.txt"
    )


if __name__ == "__main__":
    main()
