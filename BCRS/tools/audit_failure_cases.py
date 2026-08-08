"""Audit object recall by size/class with detector-compatible one-to-one matching."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


CLASS_NAMES = (
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIZE_BINS = (
    ("Very Tiny (<16x16)", 0.0, float(16 * 16)),
    ("Tiny (16x16 - 32x32)", float(16 * 16), float(32 * 32)),
    ("Small (32x32 - 96x96)", float(32 * 32), float(96 * 96)),
    ("Medium/Large (>96x96)", float(96 * 96), math.inf),
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


def box_iou_xyxy(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Return IoU for two native-pixel ``xyxy`` boxes."""

    inter_width = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    inter_height = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = inter_width * inter_height
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


# Backward-compatible name retained for external imports; the inputs have always
# been xyxy despite the old function name.
box_iou_xywh = box_iou_xyxy


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
            previous = result.get(key)
            if previous is not None and previous != path:
                raise ValueError(f"Ambiguous image stem key {key!r}: {previous}, {path}")
            result[key] = path
    if not result:
        raise ValueError(f"No supported images found in {images_dir}")
    return result


def _load_ground_truth(labels_dir: Path, images_dir: Path) -> list[GroundTruth]:
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
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"Class id {class_id} outside [0, {len(CLASS_NAMES) - 1}] at "
                    f"{label_path}:{line_index}"
                )
            center_x, center_y, box_width, box_height = map(float, parts[1:5])
            values = (center_x, center_y, box_width, box_height)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite YOLO label at {label_path}:{line_index}")
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Non-positive box at {label_path}:{line_index}")
            pixel_width, pixel_height = box_width * width, box_height * height
            x1 = (center_x - box_width / 2.0) * width
            y1 = (center_y - box_height / 2.0) * height
            targets.append(
                GroundTruth(
                    key=(image_key, line_index),
                    image_key=image_key,
                    class_id=class_id,
                    box=(x1, y1, x1 + pixel_width, y1 + pixel_height),
                    area=pixel_width * pixel_height,
                )
            )
    return targets


def _load_coco_mappings(
    annotations_path: Path,
) -> tuple[dict[str, str], dict[int, int]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_keys = {
        str(image["id"]): _canonical_key(Path(image["file_name"]).stem)
        for image in payload.get("images", [])
    }
    class_by_name = {name: index for index, name in enumerate(CLASS_NAMES)}
    category_ids: dict[int, int] = {}
    for category in payload.get("categories", []):
        name = str(category.get("name", ""))
        if name in class_by_name:
            category_ids[int(category["id"])] = class_by_name[name]
    if not image_keys or not category_ids:
        raise ValueError(
            f"COCO annotations must provide images and matching VisDrone categories: "
            f"{annotations_path}"
        )
    return image_keys, category_ids


def _load_predictions(
    prediction_path: Path,
    *,
    confidence_threshold: float,
    prediction_format: str,
    annotations_path: Path | None,
) -> list[Prediction]:
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction JSON must contain a list: {prediction_path}")

    image_ids: dict[str, str] = {}
    category_ids: dict[int, int] = {}
    if prediction_format == "coco":
        if annotations_path is None:
            raise ValueError("COCO predictions require --annotations for ID mapping")
        image_ids, category_ids = _load_coco_mappings(annotations_path)

    predictions: list[Prediction] = []
    for row_index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row {row_index} is not an object")
        score = float(row.get("score", 1.0))
        if not math.isfinite(score) or score < confidence_threshold:
            continue
        try:
            raw_image_id = str(row["image_id"])
            raw_class_id = int(row["category_id"])
            x, y, width, height = map(float, row["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed prediction row {row_index}: {row}") from exc
        if prediction_format == "coco":
            if raw_image_id not in image_ids:
                raise ValueError(f"Unknown COCO image_id {raw_image_id!r} in row {row_index}")
            if raw_class_id not in category_ids:
                raise ValueError(
                    f"Unknown COCO category_id {raw_class_id!r} in row {row_index}"
                )
            image_key = image_ids[raw_image_id]
            class_id = category_ids[raw_class_id]
        else:
            image_key = _canonical_key(raw_image_id)
            class_id = raw_class_id
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"ESOD category_id must be 0-indexed in [0, 9], got {class_id} "
                    f"in row {row_index}; use --prediction-format coco for aligned JSON"
                )
        if width <= 0 or height <= 0 or not all(
            math.isfinite(value) for value in (x, y, width, height)
        ):
            raise ValueError(f"Invalid bbox in prediction row {row_index}: {row['bbox']}")
        predictions.append(
            Prediction(
                image_key=image_key,
                class_id=class_id,
                box=(x, y, x + width, y + height),
                score=score,
            )
        )
    return predictions


def _match_predictions(
    targets: Iterable[GroundTruth],
    predictions: Iterable[Prediction],
    iou_threshold: float,
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
        for prediction in sorted(group_predictions, key=lambda item: item.score, reverse=True):
            if not unmatched:
                break
            best_index, best_iou = max(
                (
                    (index, box_iou_xyxy(prediction.box, group_targets[index].box))
                    for index in unmatched
                ),
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
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    prediction_format: str = "esod",
    annotations_path: str | Path | None = None,
) -> AuditResult:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if prediction_format not in {"esod", "coco"}:
        raise ValueError("prediction_format must be 'esod' or 'coco'")

    targets = _load_ground_truth(Path(labels_dir), Path(images_dir))
    predictions = _load_predictions(
        Path(pred_json_path),
        confidence_threshold=confidence_threshold,
        prediction_format=prediction_format,
        annotations_path=Path(annotations_path) if annotations_path else None,
    )
    recalled = _match_predictions(targets, predictions, iou_threshold)
    size_stats = {name: {"total": 0, "recalled": 0} for name, _, _ in SIZE_BINS}
    class_stats = {name: {"total": 0, "recalled": 0} for name in CLASS_NAMES}
    for target in targets:
        is_recalled = target.key in recalled
        for name, lower, upper in SIZE_BINS:
            if lower <= target.area < upper:
                size_stats[name]["total"] += 1
                size_stats[name]["recalled"] += int(is_recalled)
                break
        class_name = CLASS_NAMES[target.class_id]
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
    _print_table("BCRS TARGET FAILURE AUDIT - SIZE BIN BREAKDOWN", result.size_stats)
    overall_rate = result.recall * 100.0
    print(
        f"{'TOTAL GT TARGETS':<25} | {result.total_gt:<10} | "
        f"{result.recalled_gt:<10} | {overall_rate:<14.2f}%"
    )
    _print_table("BCRS TARGET FAILURE AUDIT - CLASS BREAKDOWN", result.class_stats)


def run_audit(
    pred_json_path: str | Path,
    labels_dir: str | Path,
    images_dir: str | Path,
    **kwargs: Any,
) -> AuditResult:
    print(f"Loading predictions from: {pred_json_path}")
    result = audit_predictions(pred_json_path, labels_dir, images_dir, **kwargs)
    print_audit(result)
    return result


def print_comparison(baseline: AuditResult, candidate: AuditResult) -> None:
    if baseline.total_gt != candidate.total_gt:
        raise ValueError("Baseline and candidate audits contain different GT counts")
    recovered = candidate.recalled_keys - baseline.recalled_keys
    regressed = baseline.recalled_keys - candidate.recalled_keys
    delta = candidate.recall - baseline.recall
    print("\n" + "=" * 70)
    print(" PAIRED RECALL CHANGE (CANDIDATE - BASELINE)")
    print("=" * 70)
    print(f"Baseline recalled: {baseline.recalled_gt}/{baseline.total_gt} ({baseline.recall * 100:.2f}%)")
    print(f"Candidate recalled: {candidate.recalled_gt}/{candidate.total_gt} ({candidate.recall * 100:.2f}%)")
    print(f"Recovered GT: {len(recovered)}")
    print(f"Regressed GT: {len(regressed)}")
    print(f"Net recalled GT: {len(recovered) - len(regressed):+d}")
    print(f"Recall delta: {delta * 100:+.2f} percentage points")
    print("=" * 70)


def _infer_images_dir(labels_dir: Path) -> Path | None:
    candidates = (
        labels_dir.parent.parent / "images" / labels_dir.name,
        labels_dir.parent / "images" / labels_dir.name,
        labels_dir.parent / "images",
    )
    return next((path for path in candidates if path.is_dir()), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pos_pred", nargs="?", help="ESOD prediction JSON")
    parser.add_argument("pos_labels", nargs="?", help="YOLO labels directory")
    parser.add_argument("pos_images", nargs="?", help="native images directory")
    parser.add_argument("--pred", "-p", help="candidate prediction JSON")
    parser.add_argument("--labels", "-l", help="YOLO labels directory")
    parser.add_argument("--images", "-i", help="native images directory")
    parser.add_argument(
        "--prediction-format",
        choices=("esod", "coco"),
        default="esod",
        help="ESOD raw JSON is 0-indexed; COCO JSON requires --annotations",
    )
    parser.add_argument("--annotations", help="COCO ground-truth JSON for COCO ID mapping")
    parser.add_argument("--baseline-pred", help="optional baseline JSON for paired recall delta")
    parser.add_argument(
        "--baseline-format",
        choices=("esod", "coco"),
        help="defaults to --prediction-format",
    )
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pred_path = args.pred or args.pos_pred
    labels_path = args.labels or args.pos_labels
    images_path = args.images or args.pos_images
    if pred_path is None or labels_path is None:
        parser.error("prediction JSON and labels directory are required")
    if images_path is None:
        inferred = _infer_images_dir(Path(labels_path))
        if inferred is None:
            parser.error("--images is required; no canonical images directory was found")
        images_path = str(inferred)

    common = {
        "confidence_threshold": args.conf_thres,
        "iou_threshold": args.iou_thres,
        "annotations_path": args.annotations,
    }
    try:
        candidate = run_audit(
            pred_path,
            labels_path,
            images_path,
            prediction_format=args.prediction_format,
            **common,
        )
        if args.baseline_pred:
            baseline = run_audit(
                args.baseline_pred,
                labels_path,
                images_path,
                prediction_format=args.baseline_format or args.prediction_format,
                **common,
            )
            print_comparison(baseline, candidate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
