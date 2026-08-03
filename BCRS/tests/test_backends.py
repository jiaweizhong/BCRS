from __future__ import annotations

from pathlib import Path

import pytest

from bcrs.config import ConfigError, load_experiment
from bcrs.registry import get_backend
from conftest import write_yaml


def create_files(root: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        path = root / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)


def build_experiment(tmp_path: Path, backend: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    create_files(
        tmp_path,
        "data/train",
        "data/val",
        "data/train.json",
        "data/val.json",
        "data/train.txt",
        "data/val.txt",
        "model.yaml",
        "checkpoint.pth",
    )
    dataset = {
        "name": "sample",
        "root": "../../data",
        "num_classes": 2,
        "classes": ["a", "b"],
        "splits": {"train": {}, "val": {}, "test": {}},
        "adapters": {
            "esod": {
                "splits": {
                    "train": {"list": "train.txt"},
                    "val": {"list": "val.txt"},
                    "test": {"list": "val.txt"},
                }
            },
            "querydet": {
                "runner": "visdrone",
                "splits": {
                    "train": {"images": "train", "annotations": "train.json"},
                    "val": {"images": "val", "annotations": "val.json"},
                },
            },
            "ceasc": {
                "splits": {
                    "train": {"images": "train", "annotations": "train.json"},
                    "val": {"images": "val", "annotations": "val.json"},
                }
            },
        },
    }
    write_yaml(tmp_path / "configs/datasets/sample.yaml", dataset)
    backend_root = tmp_path / "backend"
    if backend == "esod":
        create_files(backend_root, "train.py", "test.py")
    elif backend == "querydet":
        create_files(backend_root, "train_visdrone.py", "infer_visdrone.py")
    else:
        create_files(
            backend_root,
            "bcrs_entry.py",
            "tools/train.py",
            "tools/test.py",
            "Sparse_conv",
        )
    payload = {
        "schema_version": 1,
        "name": f"sample_{backend}",
        "backend": {"name": backend, "root": "backend", "python": "python"},
        "model": {
            "config": "model.yaml",
            "weights": (
                "detectron2://pretrained/model.pkl"
                if backend == "querydet"
                else "checkpoint.pth"
            ),
        },
        "dataset": {"config": "configs/datasets/sample.yaml"},
        "runtime": {"output_dir": f"work/{backend}", "devices": "0", "workers": 2},
        "train": {"epochs": 2, "batch_size": 1, "image_size": 64, "seed": 17},
        "test": {"checkpoint": "checkpoint.pth"},
    }
    return write_yaml(tmp_path / f"configs/experiments/{backend}.yaml", payload)


def test_esod_builds_generated_dataset_and_exact_top_level_command(
    tmp_path: Path,
) -> None:
    experiment = load_experiment(build_experiment(tmp_path, "esod"))
    command = get_backend("esod").build("train", experiment)
    assert command.cwd == tmp_path / "backend"
    assert "--cfg" in command.argv
    assert "--data" in command.argv
    assert command.env == {
        "SETUPTOOLS_USE_DISTUTILS": "stdlib",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert command.generated_files[0].is_file()
    generated = command.generated_files[0].read_text(encoding="utf-8")
    assert "nc: 2" in generated
    assert str((tmp_path / "data/train.txt").resolve()) in generated

    test_command = get_backend("esod").build("test", experiment)
    assert Path(test_command.argv[3]).name == "test.py"
    assert "--weights" in test_command.argv


def test_querydet_builds_path_overrides_and_preserves_weight_uri(
    tmp_path: Path,
) -> None:
    experiment = load_experiment(build_experiment(tmp_path, "querydet"))
    command = get_backend("querydet").build("train", experiment)
    assert command.argv[3].endswith("train_visdrone.py")
    assert "MODEL.RETINANET.NUM_CLASSES" in command.argv
    assert "detectron2://pretrained/model.pkl" in command.argv
    assert str((tmp_path / "data/train.json").resolve()) in command.argv

    test_command = get_backend("querydet").build("test", experiment)
    assert Path(test_command.argv[3]).name == "infer_visdrone.py"
    assert "--eval-only" in test_command.argv
    assert str((tmp_path / "checkpoint.pth").resolve()) in test_command.argv


def test_ceasc_builds_mmdet_config_overrides(tmp_path: Path) -> None:
    experiment = load_experiment(build_experiment(tmp_path, "ceasc"))
    command = get_backend("ceasc").build("train", experiment)
    assert Path(command.argv[3]).name == "bcrs_entry.py"
    assert command.argv[4] == "train"
    assert "--cfg-options" in command.argv
    assert "model.bbox_head.num_classes=2" in command.argv
    assert "data.train.dataset.type=CocoDataset" in command.argv
    assert "data.train.dataset.classes=('a', 'b')" in command.argv
    assert any(item.startswith("data.train.dataset.ann_file=") for item in command.argv)

    test_command = get_backend("ceasc").build("test", experiment)
    assert Path(test_command.argv[3]).name == "bcrs_entry.py"
    assert test_command.argv[4] == "test"
    assert "--eval" in test_command.argv
    assert "bbox" in test_command.argv


def test_ceasc_rejects_untested_multi_gpu_launcher(tmp_path: Path) -> None:
    path = build_experiment(tmp_path, "ceasc")
    experiment = load_experiment(path, ["runtime.devices=0,1"])
    with pytest.raises(ConfigError, match="one visible GPU"):
        get_backend("ceasc").build("train", experiment)
