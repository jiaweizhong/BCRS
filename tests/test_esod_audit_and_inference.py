from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _isolated_function(path: Path, name: str, globals_: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    node = nodes[0]
    node.decorator_list = []
    namespace = dict(globals_)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "esod" / "test.py",
        ROOT / "hesod" / "backends" / "esod" / "test.py",
        ROOT / "hesod" / "backends" / "hesod" / "test.py",
    ],
)
def test_trailing_zero_patch_images_are_restored(path: Path):
    pad = _isolated_function(path, "pad_trailing_empty_predictions", {"torch": torch})
    existing = [torch.ones((2, 6))]
    outputs = pad(existing, 3, torch.device("cpu"))
    assert len(outputs) == 3
    assert outputs[0] is existing[0]
    assert outputs[1].shape == outputs[2].shape == (0, 6)
    with pytest.raises(RuntimeError):
        pad(existing * 2, 1, torch.device("cpu"))


def test_metric_columns_map_ap50_then_coco_map():
    for path in (
        ROOT / "esod" / "test.py",
        ROOT / "hesod" / "backends" / "esod" / "test.py",
        ROOT / "hesod" / "backends" / "hesod" / "test.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "ap50, ap = ap[:, 0], ap.mean(1)" in source
        assert "mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()" in source
        assert "('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'" in source


def test_topk_router_is_exact_and_stable():
    path = ROOT / "hesod" / "backends" / "hesod" / "models" / "common.py"
    topk = _isolated_function(path, "_top_k_slicer", {"torch": torch, "F": F})

    class Router:
        top_k = 2

    # Four 4x4 cells. Cells 1 and 2 tie; stable ordering keeps cell 1 first.
    mask = torch.zeros((1, 8, 8))
    mask[:, :4, 4:] = 0.9
    mask[:, 4:, :4] = 0.9
    mask[:, 4:, 4:] = 0.2
    first = topk(Router(), mask, 2, 2, 4, 4)[0]
    second = topk(Router(), mask, 2, 2, 4, 4)[0]
    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    assert first.tolist() == [[4, 0, 8, 4], [0, 4, 4, 8]]


def test_sparse_head_does_not_rethreshold_exact_topk_patches():
    path = ROOT / "hesod" / "backends" / "hesod" / "models" / "yolo.py"
    get_indices = _isolated_function(path, "get_indices", {"torch": torch, "F": F})

    class Detect:
        nl = 3
        sparse_gird = None
        sparse_all_selected = True

    offsets = torch.tensor([[0, 0, 0, 4, 4], [0, 4, 0, 8, 4]])
    # A blank heatmap would be discarded by the old secondary 0.3 threshold,
    # even though both patches were already selected by Top-K.
    mask = torch.zeros((1, 1, 8, 8))
    indices = get_indices(Detect(), offsets, mask)
    assert len(indices) == 3
    assert indices[0].shape[0] == 2 * 4 * 4


def test_paper_selector_loss_is_focal_to_dice_20_to_1():
    path = ROOT / "hesod" / "backends" / "hesod" / "utils" / "loss.py"
    globals_ = {"torch": torch, "F": F}
    compute = _isolated_function(path, "compute_loss_seg", globals_)
    focal = _isolated_function(path, "sigmoid_focal_loss", globals_)
    dice = _isolated_function(path, "dice_loss", globals_)

    class Loss:
        selector_loss = "paper"
        mask_pos_weight = None
        lambda_cov = 0.0
        sigmoid_focal_loss = staticmethod(focal)
        dice_loss = staticmethod(dice)

    logits = torch.tensor([[[[0.2, -0.4], [0.7, -1.0]]]])
    masks = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    targets = torch.zeros((0, 6))
    lpixl, larea, ldist = compute(Loss(), logits, masks, targets)
    assert lpixl.item() == 0.0
    assert torch.allclose(larea, dice(logits, masks))
    assert torch.allclose(ldist, focal(logits, masks) * 20.0)


def test_recall_audit_is_one_to_one_and_rejects_unknown_image_ids(tmp_path: Path):
    audit = _load_module(
        ROOT / "scripts" / "esod_baseline" / "audit_buckets.py", "audit_buckets_test"
    )
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 100)).save(images / "0001.jpg")
    # Two same-class GT boxes.
    (labels / "0001.txt").write_text(
        "0 0.25 0.25 0.20 0.20\n0 0.75 0.75 0.20 0.20\n", encoding="utf-8"
    )

    pred = tmp_path / "pred.json"
    pred.write_text(
        '[{"image_id":1,"category_id":0,"bbox":[15,15,20,20],"score":0.9},'
        '{"image_id":"0001","category_id":0,"bbox":[65,65,20,20],"score":0.8}]',
        encoding="utf-8",
    )
    result = audit.audit_predictions(pred, labels, images, class_names=("object",))
    assert result.total_gt == 2
    assert result.recalled_gt == 2
    assert result.recall == 1.0

    pred.write_text(
        '[{"image_id":999,"category_id":0,"bbox":[15,15,20,20],"score":0.9}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not present"):
        audit.audit_predictions(pred, labels, images, class_names=("object",))


def test_selector_audit_uses_strict_paper_bprbox_and_validates_full_split(tmp_path: Path):
    scripts = ROOT / "scripts" / "esod_baseline"
    _load_module(scripts / "audit_buckets.py", "audit_buckets")
    selector = _load_module(
        scripts / "audit_selector_coverage.py", "audit_selector_coverage_test"
    )

    gt = (0.0, 0.0, 10.0, 10.0)
    assert not selector._paper_bprbox_hit(gt, [(0.0, 0.0, 5.0, 10.0)])
    assert selector._paper_bprbox_hit(gt, [(0.0, 0.0, 5.01, 10.0)])
    center_only = [(4.9, 4.9, 5.1, 5.1)]
    assert selector._center_in_any_patch(selector._box_center(gt), center_only)
    assert not selector._paper_bprbox_hit(gt, center_only)

    target = sys.modules["audit_buckets"].GroundTruth(
        key=("1", 1), image_key="1", class_id=0, box=gt, area=100.0
    )
    result = selector.audit_selector(
        [target],
        {target.key},
        {"1": [(0.0, 0.0, 5.01, 10.0)]},
        local_maxima_by_image={"1": center_only},
    )
    assert result.paper_bprbox == 1.0
    assert result.paper_bprctr == 1.0
    assert result.detector_recall == 1.0

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 100)).save(images / "0001.jpg")
    artifact_path = tmp_path / "patches.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "metadata": {"routing": {"mode": "top_k", "top_k": 1}},
                "images": {"0001": [[0, 0, 50, 50]]},
                "local_maxima_cells": {"0001": [[48, 48, 52, 52]]},
            }
        ),
        encoding="utf-8",
    )
    artifact = selector.load_selected_patches(artifact_path)
    assert artifact.metadata["routing"]["mode"] == "top_k"
    selector.validate_selected_patches(artifact, images)

    artifact_path.write_text(
        json.dumps({"schema_version": 2, "metadata": {}, "images": {}}),
        encoding="utf-8",
    )
    missing = selector.load_selected_patches(artifact_path)
    with pytest.raises(ValueError, match="missing 1 image"):
        selector.validate_selected_patches(missing, images)


def test_native_bprbox_uses_paper_strict_greater_than():
    for path in (
        ROOT / "esod" / "utils" / "metrics.py",
        ROOT / "hesod" / "backends" / "esod" / "utils" / "metrics.py",
        ROOT / "hesod" / "backends" / "hesod" / "utils" / "metrics.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "tp += (ios > 0.5).any(dim=1).sum()" in source
        assert "tp += (ios >= 0.5).any(dim=1).sum()" not in source


def test_mask_protocol_is_explicit_and_runners_cannot_infer_it_from_environment():
    generator = (ROOT / "scripts" / "esod_baseline" / "gen_masks.py").read_text(
        encoding="utf-8"
    )
    assert 'choices=("gaussian", "released-hybrid")' in generator
    assert 'if mask_mode == "gaussian"' in generator
    assert 'elif data_prepare.predictor is None' in generator

    expected = {
        "run_baseline.sh": '--mask-mode "$MASK_MODE"',
        "run_visdrone_roster.sh": "--mask-mode gaussian",
        "run_visdrone_sam.sh": "--mask-mode released-hybrid",
    }
    for name, flag in expected.items():
        source = (ROOT / "scripts" / "esod_baseline" / name).read_text(encoding="utf-8")
        assert flag in source
        assert "--overwrite" in source


def test_every_runner_records_exact_route_and_audits_paper_bprbox():
    scripts = ROOT / "scripts" / "esod_baseline"
    dump = (scripts / "dump_selected_patches.py").read_text(encoding="utf-8")
    assert 'route.add_argument("--top-k"' in dump
    assert 'route.add_argument("--hm-threshold"' in dump
    assert '"schema_version": 2' in dump

    for name in ("run_baseline.sh", "run_visdrone_roster.sh", "run_visdrone_sam.sh"):
        source = (scripts / name).read_text(encoding="utf-8")
        assert "dump_selected_patches.py" in source
        assert "audit_selector_coverage.py" in source
    roster = (scripts / "run_visdrone_roster.sh").read_text(encoding="utf-8")
    assert 'audit_eval "$topk_name" "$ckpt" "$topk_dir" --top-k "$TOP_K"' in roster


def test_only_explicit_tinyperson_protocol_configs_remain():
    for root in (
        ROOT / "esod",
        ROOT / "hesod" / "backends" / "esod",
        ROOT / "hesod" / "backends" / "hesod",
    ):
        assert sorted(p.name for p in root.rglob("hyp.tinyperson*.yaml")) == [
            "hyp.tinyperson.yaml",
        ]


def test_plain_baseline_mirror_and_patch_ledger_stay_in_sync():
    standalone = ROOT / "esod"
    mirror = ROOT / "hesod" / "backends" / "esod"

    def source_tree(root):
        return {
            path.relative_to(root).as_posix(): path.read_bytes().replace(b"\r\n", b"\n")
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        }

    assert source_tree(standalone) == source_tree(mirror)

    ledger = (ROOT / "ESOD-Baseline-Patches.md").read_text(encoding="utf-8")
    expected_delta = {
        "data/uavdt.yaml",
        "data/hyps/hyp.tinyperson.finetune.yaml",
        "data/hyps/hyp.tinyperson.scratch.yaml",
        "data/hyps/hyp.tinyperson.yaml",
        "models/cfg/esod/uavdt_yolov5m.yaml",
        "models/common.py",
        "scripts/data_prepare.py",
        "test.py",
        "train.py",
        "utils/general.py",
        "utils/metrics.py",
    }
    assert "exactly **11 path deltas**" in ledger
    for path in expected_delta:
        assert path in ledger
