from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LOSS_PATH = ROOT / "hesod" / "backends" / "hesod" / "utils" / "loss.py"
GENERAL_PATH = ROOT / "hesod" / "backends" / "hesod" / "utils" / "general.py"
TRAIN_PATH = ROOT / "hesod" / "backends" / "hesod" / "train.py"
ROSTER_PATH = ROOT / "scripts" / "esod_baseline" / "run_visdrone_roster.sh"


def _isolated_function(path: Path, name: str, globals_: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(nodes) == 1
    node = nodes[0]
    node.decorator_list = []
    namespace = dict(globals_)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _sabl_loss():
    return _isolated_function(LOSS_PATH, "sabl_loss", {"torch": torch, "math": math})


def _bbox_iou():
    return _isolated_function(GENERAL_PATH, "bbox_iou", {"torch": torch, "math": math})


def test_sabl_is_zero_at_an_exact_match_and_has_finite_zero_overlap_gradient():
    sabl_loss = _sabl_loss()

    matched = torch.tensor([[2.0, 3.0, 1.5, 2.0]], requires_grad=True)
    exact = sabl_loss(matched, matched.detach().clone(), stride=8)
    assert exact.item() < 1e-6

    predicted = torch.tensor([[0.0, 0.0, 2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[8.0, 8.0, 2.0, 2.0]])
    loss = sabl_loss(predicted, target, stride=8).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, :2].abs().sum() > 0


def test_sabl_uses_input_pixel_units_not_detection_grid_units():
    sabl_loss = _sabl_loss()

    predicted_pixels = torch.tensor([[18.0, 20.0, 8.0, 10.0]])
    target_pixels = torch.tensor([[16.0, 18.0, 6.0, 8.0]])
    stride8 = sabl_loss(predicted_pixels / 8, target_pixels / 8, stride=8)
    stride16 = sabl_loss(predicted_pixels / 16, target_pixels / 16, stride=16)
    assert torch.allclose(stride8, stride16, atol=1e-6, rtol=1e-6)


def test_sabl_converges_to_vendor_ciou_for_large_targets():
    sabl_loss = _sabl_loss()
    bbox_iou = _bbox_iou()

    predicted = torch.tensor([[10.25, 9.75, 15.0, 17.0]])
    target = torch.tensor([[10.0, 10.0, 16.0, 16.0]])  # sqrt(area) * stride = 128 px
    sabl = sabl_loss(predicted, target, stride=8)
    ciou = bbox_iou(predicted.T, target, x1y1x2y2=False, CIoU=True)
    assert torch.allclose(sabl, 1.0 - ciou, atol=1e-6, rtol=1e-6)


def test_sabl_changes_lbox_only_and_keeps_ciou_objectness_target():
    sabl_loss = _sabl_loss()
    bbox_iou = _bbox_iou()
    compute = _isolated_function(
        LOSS_PATH,
        "__call__",
        {"torch": torch, "sabl_loss": sabl_loss, "bbox_iou": bbox_iou},
    )

    class CaptureBCE:
        def __init__(self):
            self.targets = []

        def __call__(self, prediction, target):
            self.targets.append(target.detach().clone())
            return prediction.sum() * 0

    class Loss:
        box_loss = "sabl"
        box_weight_ref_area = 4.0
        box_weight_max = 5.0
        stride = (torch.tensor(8.0), torch.tensor(16.0))
        nc = 1
        gr = 1.0
        balance = [1.0, 1.0]
        autobalance = False
        lambda_rescue = 0.0
        lambda_cond = 0.0
        hyp = {"box": 1.0, "obj": 1.0, "cls": 1.0}
        BCEcls = CaptureBCE()
        BCEobj = CaptureBCE()

        @staticmethod
        def build_targets(predictions, targets):
            positive = torch.tensor([0], dtype=torch.long)
            empty = torch.empty(0, dtype=torch.long)
            indices = [
                (positive, positive, positive, positive),
                (empty, empty, empty, empty),
            ]
            boxes = [torch.tensor([[0.15, 0.20, 0.8, 0.9]]), torch.empty((0, 4))]
            anchors = [torch.tensor([[1.0, 1.0]]), torch.empty((0, 2))]
            classes = [positive, empty]
            return classes, boxes, indices, anchors

    raw = torch.zeros((1, 1, 1, 1, 6))
    predictions = [raw, raw.clone()]
    segmentation = [torch.zeros((1, 1, 1, 1))]
    compute(Loss(), (predictions, segmentation), torch.zeros((1, 6)))

    pbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    tbox = torch.tensor([[0.15, 0.20, 0.8, 0.9]])
    expected_quality = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True).clamp(min=0)
    assert torch.allclose(Loss.BCEobj.targets[0][0, 0, 0, 0], expected_quality[0])


def test_train_cli_exposes_sabl_without_changing_the_default():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    box_loss_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if (
            isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--box-loss"
        ):
            box_loss_calls.append(node)

    assert len(box_loss_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in box_loss_calls[0].keywords}
    assert ast.literal_eval(keywords["default"]) == "upstream"
    assert ast.literal_eval(keywords["choices"]) == (
        "upstream",
        "size_weighted",
        "sabl",
    )


def test_roster_runner_keeps_sabl_opt_in_and_defines_both_factorial_arms():
    source = ROSTER_PATH.read_text(encoding="utf-8")
    assert 'INCLUDE_SABL="${INCLUDE_SABL:-0}"' in source
    assert '--selector-loss "$selector_loss" --box-loss "$box_loss"' in source
    assert "visdrone_r1_semantic_sabl" in source
    assert "visdrone_r3_channel_pooled_concat_sabl" in source
