"""Generate/verify official-style ESOD pseudo-masks for an already YOLO-formatted dataset.

The official ESOD dataloader (utils/datasets.py) silently falls back to an
all-zero objectness mask (weight=1) whenever `masks/<split>/<stem>.npy` is
missing for an image. For an image with zero GT boxes that fallback happens
to be correct. For an image WITH GT boxes it silently trains the selector on
a wrong target -- this is the exact bug documented in
`HESOD-Experiment-Plan.md`, and the official repo
has the identical code path.

This script reuses the released `gen_mask()` routine from
`<esod_repo>/scripts/data_prepare.py`, so masks are bit-for-bit consistent
with the public code's Gaussian/SAM-hybrid implementation. The paper's stated
hybrid equation is not identical to that implementation (see
`HESOD-Experiment-Plan.md`), so this is a released-code reproduction target,
not a claim of bit-exact paper-equation reproduction. It walks an
already-converted `images/<split>/` + `labels/<split>/` tree instead of the
raw per-dataset layout that `data_prepare.py` expects.

After writing masks it verifies: every image with a non-empty label file has
a corresponding mask; it exits non-zero if any are missing, so a training
run cannot silently start on zero targets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _mask_path_for(label_path: Path) -> Path:
    parts = list(label_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "labels":
            parts[i] = "masks"
            break
    else:
        raise ValueError(f"expected a '.../labels/...' path, got {label_path}")
    return Path(*parts).with_suffix(".npy")


def _label_has_boxes(label_path: Path) -> bool:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    with open(label_path, "r") as f:
        return any(line.strip() for line in f)


def generate(
    esod_repo: Path,
    dataset_root: Path,
    splits: list[str],
    cls_ratio: bool,
    overwrite: bool,
    mask_mode: str,
) -> None:
    sys.path.insert(0, str(esod_repo))
    cwd = os.getcwd()
    os.chdir(
        esod_repo
    )  # data_prepare.py resolves `utils.general` relative to repo root
    try:
        # HESOD local patch: same PyTorch 2.6+ weights_only=True default issue
        # documented in ESOD-Baseline-Patches.md, hit here because data_prepare.py
        # loads the SAM checkpoint via torch.load at import time (module-level
        # code, not lazy) the moment `segment-anything` is importable.
        import torch

        _orig_torch_load = torch.load

        def _compat_torch_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)

        torch.load = _compat_torch_load

        from scripts import data_prepare  # noqa: E402

        if mask_mode == "gaussian":
            data_prepare.predictor = None
        elif data_prepare.predictor is None:
            raise RuntimeError(
                "--mask-mode released-hybrid requires segment-anything and "
                f"{esod_repo / 'weights' / 'sam_vit_h_4b8939.pth'}"
            )
        gen_mask = data_prepare.gen_mask
    finally:
        os.chdir(cwd)

    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.is_dir():
            print(f"[skip] {img_dir} does not exist")
            continue
        if not lbl_dir.is_dir():
            print(
                f"[skip] {lbl_dir} does not exist (labels required to generate masks)"
            )
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        written, skipped_existing, missing_labels, failed_reads = 0, 0, 0, 0
        for img_path in tqdm(images, desc=f"masks:{split}"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            mask_path = _mask_path_for(label_path)
            if mask_path.exists() and not overwrite:
                skipped_existing += 1
                continue
            if not label_path.exists():
                missing_labels += 1
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[warn] failed to read {img_path}")
                failed_reads += 1
                continue
            gen_mask(str(label_path), image, cls_ratio=cls_ratio)
            written += 1

        print(
            f"[{split}] wrote={written} already_present={skipped_existing} "
            f"no_label_file={missing_labels} unreadable_image={failed_reads}"
        )


def verify(dataset_root: Path, splits: list[str]) -> bool:
    ok = True
    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.is_dir():
            continue
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        missing = []
        for img_path in images:
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not _label_has_boxes(label_path):
                continue  # background-only image: the all-zero fallback IS correct here
            mask_path = _mask_path_for(label_path)
            if not mask_path.exists():
                missing.append(img_path.name)
        if missing:
            ok = False
            preview = ", ".join(missing[:5])
            print(
                f"[FAIL] {split}: {len(missing)}/{len(images)} images have GT boxes but no "
                f"mask (e.g. {preview}). Training would silently use a zero selector target "
                f"for these images."
            )
        else:
            print(f"[PASS] {split}: every labeled image has a mask.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--esod-repo",
        required=True,
        type=Path,
        help="Path to the official alibaba/esod checkout",
    )
    ap.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Dataset root with images/<split>, labels/<split>",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument(
        "--cls-ratio",
        action="store_true",
        help="Per-class weight boosting (VisDrone only, matches prepare_visdrone())",
    )
    ap.add_argument(
        "--mask-mode",
        choices=("gaussian", "released-hybrid"),
        default="gaussian",
        help="Explicit pseudo-mask implementation; never infer the protocol from installed packages",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks even if the .npy already exists",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip generation, only check for missing masks",
    )
    args = ap.parse_args()

    esod_repo = args.esod_repo.resolve()
    dataset_root = args.dataset_root.resolve()
    if not (esod_repo / "scripts" / "data_prepare.py").is_file():
        raise SystemExit(
            f"--esod-repo does not look like the official ESOD checkout: {esod_repo}"
        )

    if not args.verify_only:
        generate(
            esod_repo,
            dataset_root,
            args.splits,
            args.cls_ratio,
            args.overwrite,
            args.mask_mode,
        )

    if not verify(dataset_root, args.splits):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
