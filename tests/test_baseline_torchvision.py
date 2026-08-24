"""Unit tests for hesod/backends/baseline/ -- the torchvision Faster R-CNN /
RetinaNet competitor-baseline backend (HESOD-Experiment-Plan.md SS9.3).

No GPU, no network, no real dataset needed: covers the YOLO-label -> 1-
indexed torchvision target conversion (the exact class-index-off-by-one
class of bug this project has hit before, e.g. the UAVDT image_id
collision), the degenerate-box clamp/skip discipline, and the
predictions.json schema round-tripping through audit_buckets.py's own
loader -- the thing that lets audit_buckets.py/vt_diagnose.py run
unmodified against this new backend.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "hesod" / "backends" / "baseline"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def datasets_mod():
    # Registered under the literal name "datasets" so coco_utils.py's own
    # `from datasets import YoloDetectionDataset` resolves via sys.modules
    # without needing a sys.path trick in that file too.
    return _load_module(BASELINE_DIR / "datasets.py", "datasets")


@pytest.fixture(scope="module")
def coco_utils_mod(datasets_mod):
    return _load_module(BASELINE_DIR / "coco_utils.py", "baseline_coco_utils")


@pytest.fixture(scope="module")
def audit_buckets_mod():
    return _load_module(
        ROOT / "scripts" / "esod_baseline" / "audit_buckets.py",
        "baseline_test_audit_buckets",
    )


@pytest.fixture(scope="module")
def test_mod(datasets_mod, coco_utils_mod):
    return _load_module(BASELINE_DIR / "test.py", "baseline_test")


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


class TestParseYoloLabels:
    def test_basic_conversion(self, datasets_mod):
        boxes, class_ids = datasets_mod.parse_yolo_labels(
            _fake_label_path("0 0.5 0.5 0.2 0.25"), width=100, height=80, num_classes=1
        )
        assert class_ids == [0]
        assert boxes == [[40.0, 30.0, 60.0, 50.0]]

    def test_skips_degenerate_box(self, datasets_mod):
        boxes, class_ids = datasets_mod.parse_yolo_labels(
            _fake_label_path("0 0.005 0.005 0.001 0.001"),
            width=100,
            height=80,
            num_classes=1,
        )
        assert boxes == []
        assert class_ids == []

    def test_mixed_valid_and_degenerate(self, datasets_mod):
        boxes, class_ids = datasets_mod.parse_yolo_labels(
            _fake_label_path("0 0.5 0.5 0.2 0.25\n0 0.005 0.005 0.001 0.001"),
            width=100,
            height=80,
            num_classes=1,
        )
        assert class_ids == [0]
        assert len(boxes) == 1

    def test_rejects_out_of_range_class(self, datasets_mod):
        with pytest.raises(ValueError):
            datasets_mod.parse_yolo_labels(
                _fake_label_path("5 0.5 0.5 0.2 0.25"),
                width=100,
                height=80,
                num_classes=1,
            )

    def test_rejects_malformed_line(self, datasets_mod):
        with pytest.raises(ValueError):
            datasets_mod.parse_yolo_labels(
                _fake_label_path("0 0.5 0.5"), width=100, height=80, num_classes=1
            )


def _fake_label_path(content: str, tmp_path: Path | None = None) -> Path:
    import tempfile

    directory = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    path = directory / "label.txt"
    path.write_text(content + "\n", encoding="utf-8")
    return path


class TestYoloDetectionDataset:
    def _build(self, datasets_mod, tmp_path: Path, class_names=("person",)):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _write_image(images_dir / "img0001.jpg", size=(100, 80))
        _write_label(labels_dir / "img0001.txt", ["0 0.5 0.5 0.2 0.25"])
        return datasets_mod.YoloDetectionDataset(images_dir, labels_dir, class_names)

    def test_len_and_image_id(self, datasets_mod, tmp_path):
        ds = self._build(datasets_mod, tmp_path)
        assert len(ds) == 1
        assert ds.image_id(0) == "img0001"

    def test_getitem_labels_are_one_indexed(self, datasets_mod, tmp_path):
        ds = self._build(datasets_mod, tmp_path)
        image_tensor, target = ds[0]
        assert image_tensor.shape == (3, 80, 100)
        assert target["labels"].tolist() == [
            1
        ]  # class 0 in file -> torchvision label 1
        assert target["boxes"].tolist() == [[40.0, 30.0, 60.0, 50.0]]

    def test_raw_targets_stay_zero_indexed(self, datasets_mod, tmp_path):
        ds = self._build(datasets_mod, tmp_path)
        boxes, class_ids = ds.raw_targets(0)
        assert class_ids == [0]
        assert boxes == [[40.0, 30.0, 60.0, 50.0]]

    def test_mixed_image_suffixes_resolve(self, datasets_mod, tmp_path):
        # SeaPerson ships mixed .jpg/.bmp under one split -- same discipline
        # as vt_diagnose.py's IMAGE_SUFFIXES fallback (HESOD-Experiment-Plan.md SS1).
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _write_image(images_dir / "img0002.bmp", size=(100, 80))
        _write_label(labels_dir / "img0002.txt", ["0 0.5 0.5 0.2 0.25"])
        ds = datasets_mod.YoloDetectionDataset(images_dir, labels_dir, ("person",))
        assert len(ds) == 1
        assert ds.image_id(0) == "img0002"


class TestPredictionSchema:
    def test_prediction_dict_matches_audit_buckets_schema(
        self, test_mod, audit_buckets_mod, tmp_path
    ):
        rows = [
            test_mod.prediction_dict(
                "img0001", (40.0, 30.0, 60.0, 70.0), label=1, score=0.9
            ),
            test_mod.prediction_dict(
                "img0001", (0.0, 0.0, 10.0, 10.0), label=1, score=0.4
            ),
        ]
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps(rows), encoding="utf-8")

        predictions = audit_buckets_mod.load_predictions(
            pred_path, confidence_threshold=0.0, class_names=("person",)
        )
        assert len(predictions) == 2
        first = predictions[0]
        assert first.class_id == 0  # torchvision label 1 -> 0-indexed category_id
        assert first.box == pytest.approx((40.0, 30.0, 60.0, 70.0))
        assert first.score == pytest.approx(0.9)

    def test_category_id_stays_in_range_for_multiclass(
        self, test_mod, audit_buckets_mod, tmp_path
    ):
        rows = [
            test_mod.prediction_dict(
                "img0001", (0.0, 0.0, 10.0, 10.0), label=3, score=0.5
            )
        ]
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps(rows), encoding="utf-8")

        predictions = audit_buckets_mod.load_predictions(
            pred_path, confidence_threshold=0.0, class_names=("car", "truck", "bus")
        )
        assert (
            predictions[0].class_id == 2
        )  # label 3 (1-indexed) -> category_id 2 (0-indexed)


class TestCocoGt:
    def test_build_coco_gt_uses_zero_indexed_category_id(
        self, datasets_mod, coco_utils_mod, tmp_path
    ):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _write_image(images_dir / "img0001.jpg", size=(100, 80))
        _write_label(labels_dir / "img0001.txt", ["0 0.5 0.5 0.2 0.25"])
        ds = datasets_mod.YoloDetectionDataset(images_dir, labels_dir, ("person",))

        gt_path = coco_utils_mod.build_coco_gt(ds, tmp_path / "gt.json")
        payload = json.loads(gt_path.read_text(encoding="utf-8"))

        assert payload["categories"] == [{"id": 0, "name": "person"}]
        assert payload["annotations"][0]["category_id"] == 0
        assert payload["annotations"][0]["image_id"] == "img0001"
        assert payload["images"][0]["id"] == "img0001"
