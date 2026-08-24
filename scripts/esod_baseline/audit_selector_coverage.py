"""Audit selector coverage and detector recall with non-overlapping names.

This tool consumes the patch artifact produced by ``dump_selected_patches.py``
and the raw prediction JSON produced by ESOD ``test.py --save-json``.  It
reports three deliberately distinct quantities:

* paper BPRbox: a GT is covered when one selected patch contains strictly more
  than ``--bpr-thres`` (default 0.5) of the GT box area;
* paper BPRctr: the GT center's discretized objectness cell belongs to the
  thresholded 3x3-local-maxima collection dumped from HeatMapParser;
* patch-center coverage: a separate patch-routing diagnostic;
* final detector recall: class-aware one-to-one matching at ``--iou-thres``.

The artifact must contain an entry for every image in ``--images``.  Missing
or unknown image ids, malformed boxes, and boxes outside native image bounds
are fatal because silently treating an incomplete dump as zero coverage makes
the selector look worse than it was.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image

from audit_buckets import (
    DEFAULT_CLASSES,
    IMAGE_SUFFIXES,
    SIZE_BINS,
    GroundTruth,
    _canonical_key,
    load_ground_truth,
    load_predictions,
    match_predictions,
)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectedPatchArtifact:
    images: dict[str, list[Box]]
    local_maxima_cells: dict[str, list[Box]] | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class SelectorAuditResult:
    total_gt: int
    paper_bprbox_hits: int
    paper_bprctr_hits: int | None
    center_coverage_hits: int
    recalled_gt: int
    bins: dict[str, dict[str, int]]

    @property
    def paper_bprbox(self) -> float:
        return self.paper_bprbox_hits / self.total_gt if self.total_gt else 0.0

    @property
    def center_coverage(self) -> float:
        return self.center_coverage_hits / self.total_gt if self.total_gt else 0.0

    @property
    def paper_bprctr(self) -> float | None:
        if self.paper_bprctr_hits is None:
            return None
        return self.paper_bprctr_hits / self.total_gt if self.total_gt else 0.0

    @property
    def detector_recall(self) -> float:
        return self.recalled_gt / self.total_gt if self.total_gt else 0.0


def load_selected_patches(path: str | Path) -> SelectedPatchArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Patch JSON must contain an object: {path}")

    if "schema_version" in payload:
        if payload.get("schema_version") != 2:
            raise ValueError(
                f"Unsupported patch artifact schema: {payload.get('schema_version')!r}"
            )
        raw_images = payload.get("images")
        metadata = payload.get("metadata", {})
        if not isinstance(raw_images, dict) or not isinstance(metadata, dict):
            raise ValueError(
                "Schema-v2 patch artifact requires object-valued images and metadata"
            )
    else:
        # Backward-compatible reader for old dumps.  New dumps always carry
        # routing metadata so threshold and Top-K runs cannot be confused.
        raw_images = payload
        metadata = {"legacy_artifact": True}

    def parse_box_map(raw: object, label: str) -> dict[str, list[Box]]:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        result: dict[str, list[Box]] = {}
        for key, boxes in raw.items():
            canonical = _canonical_key(key)
            if canonical in result:
                raise ValueError(f"Duplicate canonical image id in {label}: {key!r}")
            if not isinstance(boxes, list):
                raise ValueError(f"{label} list for image {key!r} is not a list")
            parsed: list[Box] = []
            for index, box in enumerate(boxes):
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError(
                        f"{label} box {index} for image {key!r} must have four coordinates"
                    )
                values = tuple(float(v) for v in box)
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"{label} box {index} for image {key!r} contains a non-finite value"
                    )
                parsed.append(values)  # type: ignore[arg-type]
            result[canonical] = parsed
        return result

    result = parse_box_map(raw_images, "patch")
    raw_maxima = (
        payload.get("local_maxima_cells") if "schema_version" in payload else None
    )
    maxima = (
        parse_box_map(raw_maxima, "local-maxima cell")
        if raw_maxima is not None
        else None
    )
    return SelectedPatchArtifact(
        images=result, local_maxima_cells=maxima, metadata=dict(metadata)
    )


def _native_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise ValueError(f"Images directory is not a directory: {images_dir}")
    result: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = _canonical_key(path.stem)
        if key in result:
            raise ValueError(f"Duplicate canonical image id {key!r} in {images_dir}")
        result[key] = path
    if not result:
        raise ValueError(f"No supported images found in {images_dir}")
    return result


def validate_selected_patches(
    artifact: SelectedPatchArtifact, images_dir: Path
) -> None:
    native = _native_images(images_dir)
    dumped_ids, native_ids = set(artifact.images), set(native)
    missing = sorted(native_ids - dumped_ids)
    unknown = sorted(dumped_ids - native_ids)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} image ids (e.g. {missing[:3]})")
        if unknown:
            parts.append(f"unknown {len(unknown)} image ids (e.g. {unknown[:3]})")
        raise ValueError("Patch artifact and image split differ: " + "; ".join(parts))

    if (
        artifact.local_maxima_cells is not None
        and set(artifact.local_maxima_cells) != native_ids
    ):
        missing_maxima = sorted(native_ids - set(artifact.local_maxima_cells))
        unknown_maxima = sorted(set(artifact.local_maxima_cells) - native_ids)
        raise ValueError(
            "Local-maxima artifact and image split differ: "
            f"missing={missing_maxima[:3]}, unknown={unknown_maxima[:3]}"
        )

    tolerance = 1e-3
    for key, patches in artifact.images.items():
        with Image.open(native[key]) as image:
            width, height = image.size
        collections = [("patch", patches)]
        if artifact.local_maxima_cells is not None:
            collections.append(("local-maxima cell", artifact.local_maxima_cells[key]))
        for label, boxes in collections:
            for index, (x1, y1, x2, y2) in enumerate(boxes):
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        f"Degenerate {label} {index} for image {key!r}: {(x1, y1, x2, y2)}"
                    )
                if (
                    x1 < -tolerance
                    or y1 < -tolerance
                    or x2 > width + tolerance
                    or y2 > height + tolerance
                ):
                    raise ValueError(
                        f"{label.title()} {index} for image {key!r} lies outside {width}x{height}: "
                        f"{(x1, y1, x2, y2)}"
                    )


def _box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_in_any_patch(center: tuple[float, float], patches: list[Box]) -> bool:
    cx, cy = center
    return any(x1 <= cx < x2 and y1 <= cy < y2 for x1, y1, x2, y2 in patches)


def _intersection_over_gt(box: Box, patch: Box) -> float:
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = patch
    intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
        0.0, min(y2, py2) - max(y1, py1)
    )
    gt_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / gt_area if gt_area > 0 else 0.0


def _paper_bprbox_hit(box: Box, patches: list[Box], threshold: float = 0.5) -> bool:
    # ESOD Eq. (7) uses a strict greater-than comparison and divides by GT area,
    # not union area (so this is not IoU).
    return any(_intersection_over_gt(box, patch) > threshold for patch in patches)


def audit_selector(
    targets: list[GroundTruth],
    recalled: set[tuple[str, int]] | frozenset[tuple[str, int]],
    patches_by_image: dict[str, list[Box]],
    *,
    bpr_threshold: float = 0.5,
    local_maxima_by_image: dict[str, list[Box]] | None = None,
) -> SelectorAuditResult:
    bins = {
        name: {
            "total": 0,
            "bprbox_missed": 0,
            "bprbox_covered_but_detector_missed": 0,
            "bprbox_covered_and_recalled": 0,
            "center_covered": 0,
            "bprctr_hit": 0,
        }
        for name, _, _ in SIZE_BINS
    }
    bpr_hits = center_hits = recalled_hits = 0
    bprctr_hits = 0 if local_maxima_by_image is not None else None

    for target in targets:
        patches = patches_by_image[target.image_key]
        bpr_hit = _paper_bprbox_hit(target.box, patches, bpr_threshold)
        center_hit = _center_in_any_patch(_box_center(target.box), patches)
        bprctr_hit = (
            _center_in_any_patch(
                _box_center(target.box), local_maxima_by_image[target.image_key]
            )
            if local_maxima_by_image is not None
            else False
        )
        detector_hit = target.key in recalled
        bpr_hits += int(bpr_hit)
        center_hits += int(center_hit)
        recalled_hits += int(detector_hit)
        if bprctr_hits is not None:
            bprctr_hits += int(bprctr_hit)
        for name, lower, upper in SIZE_BINS:
            if lower <= target.area < upper:
                bucket = bins[name]
                bucket["total"] += 1
                bucket["center_covered"] += int(center_hit)
                bucket["bprctr_hit"] += int(bprctr_hit)
                if not bpr_hit:
                    bucket["bprbox_missed"] += 1
                elif detector_hit:
                    bucket["bprbox_covered_and_recalled"] += 1
                else:
                    bucket["bprbox_covered_but_detector_missed"] += 1
                break

    return SelectorAuditResult(
        total_gt=len(targets),
        paper_bprbox_hits=bpr_hits,
        paper_bprctr_hits=bprctr_hits,
        center_coverage_hits=center_hits,
        recalled_gt=recalled_hits,
        bins=bins,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patches", required=True, help="output of dump_selected_patches.py"
    )
    parser.add_argument(
        "--pred", required=True, help="ESOD raw prediction JSON (test.py --save-json)"
    )
    parser.add_argument("--labels", required=True, help="YOLO labels directory")
    parser.add_argument("--images", required=True, help="native images directory")
    parser.add_argument(
        "--classes", help="comma-separated class names; defaults to VisDrone"
    )
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--bpr-thres", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0.0 <= args.bpr_thres <= 1.0:
        parser.error("--bpr-thres must be in [0, 1]")

    class_names = (
        tuple(name.strip() for name in args.classes.split(",") if name.strip())
        if args.classes
        else DEFAULT_CLASSES
    )
    images_dir = Path(args.images)
    targets = load_ground_truth(Path(args.labels), images_dir, class_names)
    predictions = load_predictions(
        Path(args.pred), confidence_threshold=args.conf_thres, class_names=class_names
    )
    recalled = match_predictions(targets, predictions, args.iou_thres)
    artifact = load_selected_patches(args.patches)
    validate_selected_patches(artifact, images_dir)
    result = audit_selector(
        targets,
        recalled,
        artifact.images,
        bpr_threshold=args.bpr_thres,
        local_maxima_by_image=artifact.local_maxima_cells,
    )

    routing = artifact.metadata.get("routing", "unknown/legacy")
    print(f"Selector artifact routing: {routing}")
    bprctr_text = (
        f"paper BPRctr={result.paper_bprctr:.6f} "
        f"({result.paper_bprctr_hits}/{result.total_gt}, GT-center cell in local-maxima C); "
        if result.paper_bprctr is not None
        else "paper BPRctr=unavailable (legacy artifact has no local-maxima cells); "
    )
    print(
        f"Overall: paper BPRbox={result.paper_bprbox:.6f} "
        f"({result.paper_bprbox_hits}/{result.total_gt}, intersection/GT > {args.bpr_thres:g}); "
        f"{bprctr_text}"
        f"patch-center coverage={result.center_coverage:.6f} "
        f"({result.center_coverage_hits}/{result.total_gt}, routing diagnostic); "
        f"detector recall={result.detector_recall:.6f} "
        f"({result.recalled_gt}/{result.total_gt}, class-aware one-to-one IoU >= {args.iou_thres:g})"
    )
    print("=" * 118)
    print(
        f"{'Size bin':<25} | {'Total':>7} | {'BPRbox miss':>16} | "
        f"{'Covered,det miss':>18} | {'Covered,recalled':>18} | {'BPRctr':>12} | {'Center covered':>16}"
    )
    print("-" * 118)
    for name, values in result.bins.items():
        total = values["total"]
        denom = total or 1
        miss = values["bprbox_missed"]
        det_miss = values["bprbox_covered_but_detector_missed"]
        recalled_count = values["bprbox_covered_and_recalled"]
        center = values["center_covered"]
        bprctr = values["bprctr_hit"]
        print(
            f"{name:<25} | {total:>7} | {miss:>7} ({miss / denom:6.1%}) | "
            f"{det_miss:>8} ({det_miss / denom:6.1%}) | "
            f"{recalled_count:>8} ({recalled_count / denom:6.1%}) | "
            f"{bprctr:>5} ({bprctr / denom:6.1%}) | "
            f"{center:>7} ({center / denom:6.1%})"
        )
    print("=" * 118)
    print(
        "Patch-center coverage is a routing diagnostic and is never relabeled as paper BPRctr."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
