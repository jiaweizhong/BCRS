from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from bcrs.datasets.visdrone import VISDRONE_CLASSES, prepare_visdrone


def make_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def write_raw_annotation(root: Path, split: str, stem: str, content: str) -> None:
    path = root / "raw_annotations" / split / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_visdrone_writes_yolo_and_coco(tmp_path: Path) -> None:
    root = tmp_path / "VisDrone"
    make_image(root / "images/train/000001.jpg", (100, 50))
    make_image(root / "images/train/000002.jpg", (200, 100))
    write_raw_annotation(
        root,
        "train",
        "000001",
        "10,5,20,10,1,4,0,0\n"
        "0,0,5,5,0,0,0,0\n"
        "0,0,5,5,1,11,0,0\n"
        "10,5,4,0,1,4,0,0\n",
    )
    write_raw_annotation(root, "train", "000002", "20,10,40,20,1,1,0,0\n")

    summaries = prepare_visdrone(root, splits=("train",))

    assert len(summaries) == 1
    assert summaries[0].images == 2
    assert summaries[0].annotations == 2
    assert summaries[0].skipped_rows == 3
    assert (root / "labels/train/000001.txt").read_text(encoding="utf-8") == (
        "3 0.200000 0.200000 0.200000 0.200000\n"
    )
    assert (root / "labels/train/000002.txt").read_text(encoding="utf-8") == (
        "0 0.200000 0.200000 0.200000 0.200000\n"
    )

    coco = json.loads((root / "annotations/train.json").read_text(encoding="utf-8"))
    assert [image["file_name"] for image in coco["images"]] == [
        "000001.jpg",
        "000002.jpg",
    ]
    assert coco["annotations"][0]["bbox"] == [10, 5, 20, 10]
    assert coco["annotations"][0]["category_id"] == 4
    assert [category["id"] for category in coco["categories"]] == list(range(1, 11))
    assert [category["name"] for category in coco["categories"]] == list(
        VISDRONE_CLASSES
    )


def test_prepare_visdrone_dry_run_does_not_write_outputs(tmp_path: Path) -> None:
    root = tmp_path / "VisDrone"
    make_image(root / "images/val/example.jpg", (32, 16))
    write_raw_annotation(root, "val", "example", "0,0,4,4,1,2,0,0\n")

    summaries = prepare_visdrone(root, splits=("val",), dry_run=True)

    assert summaries[0].annotations == 1
    assert not (root / "labels").exists()
    assert not (root / "annotations").exists()


def test_prepare_visdrone_rejects_missing_source_annotation(tmp_path: Path) -> None:
    root = tmp_path / "VisDrone"
    make_image(root / "images/test/missing.jpg", (32, 16))
    (root / "raw_annotations/test").mkdir(parents=True)

    with pytest.raises(ValueError, match="Missing 1 raw annotation"):
        prepare_visdrone(root, splits=("test",))


def test_visdrone_config_uses_canonical_shared_layout() -> None:
    config_path = Path(__file__).parents[1] / "configs/datasets/visdrone.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["splits"]["train"] == {
        "images": "images/train",
        "annotations": "annotations/train.json",
    }
    assert config["adapters"]["esod"] == {}
    assert config["adapters"]["querydet"]["runner"] == "visdrone"
    assert config["adapters"]["ceasc"]["dataset_type"] == "CocoDataset"
