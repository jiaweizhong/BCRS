from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tinyperson_hyp_profiles_are_explicit_and_distinct():
    for backend in (ROOT / "esod", ROOT / "hesod" / "backends" / "esod", ROOT / "hesod" / "backends" / "hesod"):
        paper = yaml.safe_load((backend / "data/hyps/hyp.tinyperson.yaml").read_text(encoding="utf-8"))
        released = yaml.safe_load(
            (backend / "data/hyps/hyp.tinyperson.released.yaml").read_text(encoding="utf-8")
        )
        assert paper["lr0"] == pytest.approx(0.01)
        assert released["lr0"] == pytest.approx(0.005)
        assert {**paper, "lr0": released["lr0"]} == released


def test_tinyperson_converter_pins_no_dense_erased_data_and_explicit_masks():
    for backend in (ROOT / "esod", ROOT / "hesod" / "backends" / "esod", ROOT / "hesod" / "backends" / "hesod"):
        source = (backend / "scripts/data_prepare.py").read_text(encoding="utf-8")
        start = source.index("def prepare_tinyperson")
        next_function = source.find("\ndef ", start + 1)
        end = next_function if next_function != -1 else source.index("\nif __name__", start + 1)
        function = source[start:end]
        assert "tiny_set_train_all_erase.json" in function
        assert "tiny_set_test_all.json" in function
        assert "erase_with_uncertain_dataset" in function
        assert "tiny_set_train_with_dense.json" not in function
        assert "tiny_set_test_with_dense.json" not in function
        assert "tinyperson_protocol.json" in function
        assert "sam_mode=mask_mode" in function


def test_tinyperson_r0_runner_separates_paper_and_released_protocols():
    source = (ROOT / "scripts/esod_baseline/run_tinyperson_fresh_r0.sh").read_text(encoding="utf-8")
    assert 'PROTOCOL="${PROTOCOL:-paper}"' in source
    assert "hyp.tinyperson.released.yaml" in source
    assert "TRAIN_FLAGS=(--selector-loss paper)" in source
    assert "TRAIN_FLAGS=(--selector-loss upstream)" in source
    assert "audit_tinyperson_protocol.py" in source
    assert "tiny_set_test_all.json" in source
    assert "tiny_set_test_with_dense.json" not in source
    assert "REUSE_CHECKPOINTS" in source


def test_tinyperson_official_evaluator_rejects_nonstandard_gt(tmp_path: Path):
    evaluator_path = ROOT / "scripts/esod_baseline/tinyperson_eval/eval_tinyperson_official.py"
    spec = importlib.util.spec_from_file_location("eval_tinyperson_official", evaluator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wrong_name = tmp_path / "tiny_set_test_with_dense.json"
    wrong_name.write_text(json.dumps({"images": [{}] * 816, "annotations": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="tiny_set_test_all.json"):
        module.validate_official_gt(wrong_name)

    wrong_count = tmp_path / "tiny_set_test_all.json"
    wrong_count.write_text(json.dumps({"images": [{}], "annotations": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="786"):
        module.validate_official_gt(wrong_count)


def test_reorganizer_writes_runner_ready_yaml(tmp_path: Path):
    module = load_module(ROOT / "scripts/esod_baseline/reorganize_tinyperson.py")
    out_root = tmp_path / "TinyPerson_official_paper"
    yaml_path = tmp_path / "TinyPerson_official_paper.yaml"
    module.write_dataset_yaml(yaml_path, out_root)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["train"] == (out_root / "images" / "train").resolve().as_posix()
    assert data["val"] == (out_root / "images" / "val").resolve().as_posix()
    assert data["test"] == (out_root / "images" / "val").resolve().as_posix()
    assert "path" not in data


def test_two_arm_runner_is_exactly_r0_and_concat_isphead():
    runner = (ROOT / "scripts/esod_baseline/run_tinyperson_two_arm.sh").read_text(encoding="utf-8")
    calls = [line for line in runner.splitlines() if line.startswith('run_arm "')]
    assert len(calls) == 2
    assert "_r0${SUFFIX}" in calls[0]
    assert "concat_isphead_coverage_sabl" in calls[1]
    assert "tinyperson_yolov5m_channel_pooled_concat_isphead.yaml" in runner
    assert "DATASET must be official or aug" in runner
