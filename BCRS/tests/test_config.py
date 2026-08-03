from __future__ import annotations

from pathlib import Path

import pytest

from bcrs.config import ConfigError, load_experiment
from conftest import write_yaml


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    dataset = write_yaml(
        tmp_path / "configs" / "datasets" / "sample.yaml",
        {
            "name": "sample",
            "root": "../../data/sample",
            "num_classes": 2,
            "classes": ["a", "b"],
            "splits": {"train": {"images": "train"}, "val": {"images": "val"}},
            "adapters": {},
        },
    )
    experiment = write_yaml(
        tmp_path / "configs" / "experiments" / "sample.yaml",
        {
            "schema_version": 1,
            "name": "sample",
            "backend": {"name": "esod", "root": "backend"},
            "model": {"config": "model.yaml"},
            "dataset": {"config": "configs/datasets/sample.yaml"},
            "runtime": {"output_dir": "work/sample", "devices": "0"},
            "train": {"batch_size": 2},
            "test": {},
        },
    )
    return dataset, experiment


def test_loads_dataset_and_applies_dotted_override(tmp_path: Path) -> None:
    _, experiment_path = make_project(tmp_path)
    experiment = load_experiment(
        experiment_path, ["train.batch_size=4", "runtime.devices=1"]
    )
    assert experiment.name == "sample"
    assert experiment.dataset.classes == ("a", "b")
    assert experiment.train["batch_size"] == 4
    assert experiment.devices == "1"
    assert experiment.dataset.root == (tmp_path / "data" / "sample").resolve()


def test_rejects_class_count_mismatch(tmp_path: Path) -> None:
    dataset_path, experiment_path = make_project(tmp_path)
    dataset = {
        "name": "bad",
        "root": ".",
        "num_classes": 3,
        "classes": ["a", "b"],
        "splits": {},
        "adapters": {},
    }
    write_yaml(dataset_path, dataset)
    with pytest.raises(ConfigError, match="declares 3 classes"):
        load_experiment(experiment_path)
