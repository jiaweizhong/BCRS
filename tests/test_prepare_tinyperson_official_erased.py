import importlib.util
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "esod_baseline"
    / "prepare_tinyperson_official_erased.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_tinyperson_official_erased", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_erase_regions_uses_original_region_mean_and_all_protocol_flags():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    annotations = [
        {"bbox": [1.2, 1.2, 2.2, 2.2], "ignore": True},
        {"bbox": [4, 0, 2, 2], "uncertain": True},
        {"bbox": [0, 5, 2, 2], "logo": True},
        {"bbox": [6, 6, 2, 2]},
    ]

    erased, count = MODULE.erase_regions(image.copy(), annotations)

    assert count == 3
    expected = np.rint(image[1:4, 1:4].mean(axis=(0, 1))).astype(np.uint8)
    assert np.all(erased[1:4, 1:4] == expected)
    assert np.array_equal(erased[6:8, 6:8], image[6:8, 6:8])


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _add_image(archive: tarfile.TarFile, name: str, image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = encoded.tobytes()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))
    return payload


def test_prepare_split_reads_archive_filters_target_and_is_resumable(tmp_path):
    root = tmp_path
    source = {
        "images": [
            {"id": 1, "file_name": "labeled_images/keep.jpg"},
            {"id": 2, "file_name": "labeled_dense_images/drop.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [1, 1, 3, 3], "ignore": True},
            {"image_id": 1, "bbox": [5, 5, 1, 1], "ignore": False},
        ],
    }
    target = {
        "images": [{"id": 11, "file_name": "labeled_images/keep.jpg"}],
        "annotations": [],
    }
    _write_json(root / "annotations" / "tiny_set_train.json", source)
    _write_json(root / "mini_annotations" / "tiny_set_train_all_erase.json", target)

    archive_path = root / "train.tar.gz"
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    image[1:4, 1:4] = (10, 20, 30)
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_image(archive, "train/labeled_images/keep.jpg", image)
        _add_image(archive, "train/labeled_dense_images/drop.jpg", image)

    stats = MODULE.prepare_split(root, "train")
    destination = (
        root / "erase_with_uncertain_dataset" / "train" / "labeled_images" / "keep.jpg"
    )
    assert stats["images"] == 1
    assert stats["written"] == 1
    assert stats["protocol_regions_total"] == 1
    assert stats["erased_regions_written_this_run"] == 1
    assert destination.is_file()
    assert not (
        root / "erase_with_uncertain_dataset" / "train" / "labeled_dense_images"
    ).exists()

    resumed = MODULE.prepare_split(root, "train")
    assert resumed["written"] == 0
    assert resumed["resumed_existing"] == 1
    assert resumed["protocol_regions_total"] == 1


def test_safe_relative_file_name_rejects_traversal():
    for value in ("../escape.jpg", "/absolute.jpg", "folder/../../escape.jpg"):
        try:
            MODULE.safe_relative_file_name(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {value}")
