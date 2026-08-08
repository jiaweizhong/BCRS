"""Size-bin / class recall audit for official ESOD `test.py --save-json` output.

Standalone by design -- lives under scripts/, no dependency on the BCRS
package. Consumes the raw ESOD prediction JSON (0-indexed category_id, as
written by `<esod_repo>/test.py --save-json`) plus the YOLO labels/images
tree, and reports one-to-one class-aware recall broken down by GT box size
and by class, matching a detector's own recall semantics (predictions
sorted by score, greedily matched to still-unmatched GT of the same class
and image at IoU >= threshold).

Usage:
  python audit_buckets.py \
    --pred  <run>/best_predictions.json \
    --labels /root/autodl-tmp/VisDrone/labels/val \
    --images /root/autodl-tmp/VisDrone/images/val \
    --classes pedestrian,people,bicycle,car,van,truck,tricycle,awning-tricycle,bus,motor
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIZE_BINS = (
    ("Very Tiny (<16x16)", 0.0, float(16 * 16)),
    ("Tiny (16x16 - 32x32)", float(16 * 16), float(32 * 32)),
    ("Small (32x32 - 96x96)", float(32 * 32), float(96 * 96)),
    ("Medium/Large (>96x96)", float(96 * 96), math.inf),
)
DEFAULT_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


@dataclass(frozen=True)
class GroundTruth:
    key: tuple[str, int]
    image_key: str
    class_id: int
    box: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True)
class Prediction:
    image_key: str
    class_id: int
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class AuditResult:
    total_gt: int
    recalled_keys: frozenset[tuple[str, int]]
    size_stats: dict[str, dict[str, int]]
    class_stats: dict[str, dict[str, int]]
    prediction_count: int
    confidence_threshold: float
    iou_threshold: float

    @property
    def recalled_gt(self) -> int:
        return len(self.recalled_keys)

    @property
    def recall(self) -> float:
        return self.recalled_gt / self.total_gt if self.total_gt else 0.0


def box_iou_xyxy(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    inter_w = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    inter_h = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def _canonical_key(value: object) -> str:
    text = str(value)
    return str(int(text)) if text.isdecimal() else text


def _image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise ValueError(f"Images directory is not a directory: {images_dir}")
    result: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        for key in {path.stem, _canonical_key(path.stem)}:
            result[key] = path
    if not result:
        raise ValueError(f"No supported images found in {images_dir}")
    return result


def load_ground_truth(
    labels_dir: Path, images_dir: Path, class_names: tuple[str, ...]
) -> list[GroundTruth]:
    if not labels_dir.is_dir():
        raise ValueError(f"Labels directory is not a directory: {labels_dir}")
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise ValueError(f"No YOLO label files found in {labels_dir}")
    images = _image_index(images_dir)
    targets: list[GroundTruth] = []

    for label_path in label_files:
        image_key = _canonical_key(label_path.stem)
        image_path = images.get(label_path.stem) or images.get(image_key)
        if image_path is None:
            raise ValueError(
                f"No image in {images_dir} matches label {label_path.name}; "
                "native dimensions are required for a valid size audit"
            )
        with Image.open(image_path) as image:
            width, height = image.size

        for line_index, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO label at {label_path}:{line_index}")
            class_id = int(parts[0])
            if not 0 <= class_id < len(class_names):
                raise ValueError(
                    f"Class id {class_id} outside [0, {len(class_names) - 1}] at "
                    f"{label_path}:{line_index} -- pass --classes matching this "
                    "dataset's label ids"
                )
            cx, cy, bw, bh = map(float, parts[1:5])
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                raise ValueError(f"Non-finite YOLO label at {label_path}:{line_index}")
            if bw <= 0 or bh <= 0:
                raise ValueError(f"Non-positive box at {label_path}:{line_index}")
            pw, ph = bw * width, bh * height
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            targets.append(
                GroundTruth(
                    key=(image_key, line_index),
                    image_key=image_key,
                    class_id=class_id,
                    box=(x1, y1, x1 + pw, y1 + ph),
                    area=pw * ph,
                )
            )
    return targets


def load_predictions(
    prediction_path: Path, *, confidence_threshold: float, class_names: tuple[str, ...]
) -> list[Prediction]:
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction JSON must contain a list: {prediction_path}")

    predictions: list[Prediction] = []
    degenerate_count = 0
    for row_index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row {row_index} is not an object")
        score = float(row.get("score", 1.0))
        if not math.isfinite(score) or score < confidence_threshold:
            continue
        try:
            image_key = _canonical_key(row["image_id"])
            class_id = int(row["category_id"])
            x, y, width, height = map(float, row["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed prediction row {row_index}: {row}") from exc
        if not 0 <= class_id < len(class_names):
            raise ValueError(
                f"category_id must be 0-indexed in [0, {len(class_names) - 1}], got "
                f"{class_id} in row {row_index}"
            )
        if not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError(f"Non-finite bbox in prediction row {row_index}: {row['bbox']}")
        if width <= 0 or height <= 0:
            # A real (if rare) model output, not a parse error -- e.g. a near-zero
            # regressed height that rounds to 0.000 at the 3-decimal precision
            # test.py writes (`round(x, 3)`). Can never achieve IoU>0 with any GT
            # box, so it contributes nothing to recall either way; skip rather than
            # abort the whole audit over it.
            degenerate_count += 1
            continue
        predictions.append(
            Prediction(image_key=image_key, class_id=class_id, box=(x, y, x + width, y + height), score=score)
        )
    if degenerate_count:
        print(f"NOTE: skipped {degenerate_count} degenerate (zero/negative-area) prediction box(es)")
    return predictions


def match_predictions(
    targets: Iterable[GroundTruth], predictions: Iterable[Prediction], iou_threshold: float
) -> frozenset[tuple[str, int]]:
    targets_by_group: dict[tuple[str, int], list[GroundTruth]] = defaultdict(list)
    predictions_by_group: dict[tuple[str, int], list[Prediction]] = defaultdict(list)
    for target in targets:
        targets_by_group[(target.image_key, target.class_id)].append(target)
    for prediction in predictions:
        predictions_by_group[(prediction.image_key, prediction.class_id)].append(prediction)

    recalled: set[tuple[str, int]] = set()
    for group, group_predictions in predictions_by_group.items():
        group_targets = targets_by_group.get(group, [])
        unmatched = set(range(len(group_targets)))
        for prediction in sorted(group_predictions, key=lambda p: p.score, reverse=True):
            if not unmatched:
                break
            best_index, best_iou = max(
                ((i, box_iou_xyxy(prediction.box, group_targets[i].box)) for i in unmatched),
                key=lambda item: item[1],
            )
            if best_iou >= iou_threshold:
                unmatched.remove(best_index)
                recalled.add(group_targets[best_index].key)
    return frozenset(recalled)


def audit_predictions(
    pred_json_path: str | Path,
    labels_dir: str | Path,
    images_dir: str | Path,
    *,
    class_names: tuple[str, ...] = DEFAULT_CLASSES,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> AuditResult:
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    targets = load_ground_truth(Path(labels_dir), Path(images_dir), class_names)
    predictions = load_predictions(
        Path(pred_json_path), confidence_threshold=confidence_threshold, class_names=class_names
    )
    recalled = match_predictions(targets, predictions, iou_threshold)

    size_stats = {name: {"total": 0, "recalled": 0} for name, _, _ in SIZE_BINS}
    class_stats = {name: {"total": 0, "recalled": 0} for name in class_names}
    for target in targets:
        is_recalled = target.key in recalled
        for name, lower, upper in SIZE_BINS:
            if lower <= target.area < upper:
                size_stats[name]["total"] += 1
                size_stats[name]["recalled"] += int(is_recalled)
                break
        class_name = class_names[target.class_id]
        class_stats[class_name]["total"] += 1
        class_stats[class_name]["recalled"] += int(is_recalled)

    return AuditResult(
        total_gt=len(targets),
        recalled_keys=recalled,
        size_stats=size_stats,
        class_stats=class_stats,
        prediction_count=len(predictions),
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )


def _print_table(title: str, stats: dict[str, dict[str, int]]) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"{'Group':<25} | {'GT Count':<10} | {'Recalled':<10} | {'Recall Rate (%)':<15}")
    print("-" * 70)
    for name, values in stats.items():
        total, recalled = values["total"], values["recalled"]
        rate = recalled / total * 100.0 if total else 0.0
        print(f"{name:<25} | {total:<10} | {recalled:<10} | {rate:<14.2f}%")
    print("=" * 70)


def print_audit(result: AuditResult) -> None:
    print(
        f"Evaluated {result.prediction_count} predictions at "
        f"conf>={result.confidence_threshold:g}, IoU>={result.iou_threshold:g} "
        "with class-aware one-to-one matching."
    )
    _print_table("SIZE BIN RECALL", result.size_stats)
    overall_rate = result.recall * 100.0
    print(
        f"{'TOTAL GT TARGETS':<25} | {result.total_gt:<10} | "
        f"{result.recalled_gt:<10} | {overall_rate:<14.2f}%"
    )
    _print_table("CLASS RECALL", result.class_stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", "-p", required=True, help="ESOD raw prediction JSON (test.py --save-json)")
    parser.add_argument("--labels", "-l", required=True, help="YOLO labels directory, e.g. .../labels/val")
    parser.add_argument("--images", "-i", required=True, help="native images directory, e.g. .../images/val")
    parser.add_argument(
        "--classes",
        help=(
            "comma-separated class names in id order; defaults to the 10-class "
            "VisDrone list. Pass e.g. 'car,truck,bus' for UAVDT or 'person' for "
            "TinyPerson."
        ),
    )
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    args = parser.parse_args(argv)

    class_names = (
        tuple(name.strip() for name in args.classes.split(",") if name.strip())
        if args.classes
        else DEFAULT_CLASSES
    )

    try:
        result = audit_predictions(
            args.pred,
            args.labels,
            args.images,
            class_names=class_names,
            confidence_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse.error() already exits

    print_audit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
