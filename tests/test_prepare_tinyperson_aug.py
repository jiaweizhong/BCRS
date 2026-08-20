import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "esod_baseline" / "prepare_tinyperson_aug.py"
AUDIT_SCRIPT = ROOT / "scripts" / "esod_baseline" / "audit_tinyperson_aug.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_tinyperson_aug", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_tinyperson_aug", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_yaml_uses_absolute_paths_for_bundled_yolov5(tmp_path):
    module = load_module()
    output_root = tmp_path / "TinyPerson_aug"
    yaml_path = tmp_path / "TinyPerson_aug.yaml"

    module.write_yaml(yaml_path, output_root)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert data["train"] == (output_root / "images" / "train").resolve().as_posix()
    assert data["val"] == (output_root / "images" / "valid").resolve().as_posix()
    assert data["test"] == (output_root / "images" / "test").resolve().as_posix()
    assert "path" not in data


def test_convert_roboflow_coco_merges_categories_and_writes_masks(tmp_path):
    module = load_module()
    source = tmp_path / "tinyperson-aug"
    stage = tmp_path / "stage"
    stage.mkdir()

    categories = [
        {"id": 0, "name": "tiny-people", "supercategory": "none"},
        {"id": 1, "name": "dry-person", "supercategory": "tiny-people"},
        {"id": 2, "name": "wet-swimmer", "supercategory": "tiny-people"},
    ]
    for split in module.SPLITS:
        split_root = source / split
        split_root.mkdir(parents=True)
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        assert cv2.imwrite(str(split_root / "sample.jpg"), image)
        data = {
            "images": [{"id": 7, "file_name": "sample.jpg", "width": 40, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 1, "bbox": [4, 2, 8, 6]},
                {"id": 2, "image_id": 7, "category_id": 2, "bbox": [20, 10, 10, 5]},
            ],
            "categories": categories,
        }
        (split_root / module.ANNOTATION_NAME).write_text(json.dumps(data), encoding="utf-8")

    datasets = module.validate_source(source)

    def fake_mask(label_path, pixels, cls_ratio, sam_mode):
        assert cls_ratio is False
        assert sam_mode == "gaussian"
        mask_path = Path(label_path.replace("/labels/", "/masks/").replace(".txt", ".npy"))
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, np.zeros((*pixels.shape[:2], 2), dtype=np.float16))

    for split in module.SPLITS:
        stats = module.convert_split(
            split,
            datasets[split],
            source,
            stage,
            "gaussian",
            "copy",
            fake_mask,
        )
        assert stats["images"] == 1
        assert stats["training_boxes"] == 2
        assert stats["source_category_counts"] == {"dry-person": 1, "wet-swimmer": 1}

        labels = (stage / "labels" / split / "sample.txt").read_text().splitlines()
        assert len(labels) == 2
        assert all(line.startswith("0 ") for line in labels)
        mask = np.load(stage / "masks" / split / "sample.npy")
        assert mask.shape == (20, 40, 2)


def test_rejects_path_traversal():
    module = load_module()
    try:
        module.safe_relative_file_name("../outside.jpg")
    except ValueError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("path traversal was accepted")


def test_estimates_two_float16_mask_planes(tmp_path):
    module = load_module()
    datasets = {
        split: {"images": [{"file_name": "sample.jpg", "width": 40, "height": 20}]}
        for split in module.SPLITS
    }
    assert module.estimate_output_bytes(datasets, tmp_path, "hardlink") == 3 * 40 * 20 * 4


def test_aug_audit_checks_manifest_and_matching_stems(tmp_path):
    audit_module = load_audit_module()
    data_root = tmp_path / "TinyPerson_aug_paper-hybrid"
    split_stats = {}
    for split in audit_module.SPLITS:
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".npy")):
            path = data_root / kind / split / f"sample{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        split_stats[split] = {"images": 1}
    manifest = {
        "protocol": audit_module.PROTOCOL,
        "paper_comparable": False,
        "mask_mode": "paper-hybrid",
        "category_mapping": "all source categories -> class 0 person",
        "annotations": {split: {"path": "", "sha256": "unused"} for split in audit_module.SPLITS},
        "splits": split_stats,
    }
    (data_root / "tinyperson_aug_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = audit_module.audit(data_root, "paper-hybrid")
    assert summary["train"] == {"images": 1, "labels": 1, "masks": 1}
