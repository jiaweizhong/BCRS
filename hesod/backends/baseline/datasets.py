"""YOLO-format dataset adapter for torchvision detection models.

Reads the SAME `images/{split}/` + `labels/{split}/` directories every other
arm in this project already uses (produced by `reorganize_uavdt.py` /
`reorganize_seaperson.py`) -- no new data prep. YOLO labels are
`class cx cy bw bh` (normalized), one .txt per image.

torchvision's detection models (Faster R-CNN, RetinaNet) reserve label 0 for
background, so `YoloDetectionDataset.__getitem__` returns 1-indexed labels.
`parse_yolo_labels` itself stays 0-indexed (matching this project's own
`audit_buckets.py`/`predictions.json` convention) so `coco_utils.py` can
build a GT json that lines up with predictions without any id translation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_image(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No image for stem {stem!r} under {images_dir} (tried {IMAGE_SUFFIXES})"
    )


def parse_yolo_labels(
    label_path: Path, width: int, height: int, num_classes: int
) -> tuple[list[list[float]], list[int]]:
    """Returns (boxes_xyxy_abs_px, class_ids_0_indexed). Degenerate boxes
    (zero/negative area once clamped to the image frame) are skipped, same
    discipline as data_prepare.py::prepare_seaperson()'s gen_mask() guard.
    """
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    for line_index, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO label at {label_path}:{line_index}")
        class_id = int(parts[0])
        if not 0 <= class_id < num_classes:
            raise ValueError(
                f"Class id {class_id} outside [0, {num_classes - 1}] at "
                f"{label_path}:{line_index}"
            )
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = max((cx - bw / 2.0) * width, 0.0)
        y1 = max((cy - bh / 2.0) * height, 0.0)
        x2 = min((cx + bw / 2.0) * width, float(width))
        y2 = min((cy + bh / 2.0) * height, float(height))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        boxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
    return boxes, class_ids


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: Sequence[str],
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = tuple(class_names)
        label_files = sorted(self.labels_dir.glob("**/*.txt"))
        if not label_files:
            raise ValueError(f"No YOLO label files found under {self.labels_dir}")
        self.samples: list[tuple[Path, Path]] = [
            (find_image(self.images_dir, label_path.stem), label_path)
            for label_path in label_files
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def image_id(self, index: int) -> str:
        return self.samples[index][0].stem

    def native_size(self, index: int) -> tuple[int, int]:
        image_path, _ = self.samples[index]
        with Image.open(image_path) as image:
            return image.size  # (width, height)

    def raw_targets(self, index: int) -> tuple[list[list[float]], list[int]]:
        """0-indexed (boxes_xyxy_abs_px, class_ids), no background offset."""
        image_path, label_path = self.samples[index]
        width, height = self.native_size(index)
        return parse_yolo_labels(label_path, width, height, len(self.class_names))

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        boxes, class_ids = self.raw_targets(index)

        image_tensor = TF.to_tensor(image)
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor([c + 1 for c in class_ids], dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": index}
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
