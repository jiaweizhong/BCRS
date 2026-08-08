from __future__ import annotations

import sys
from pathlib import Path
import types

import pytest

torch = pytest.importorskip("torch")

ESOD_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "esod"
sys.path.insert(0, str(ESOD_ROOT))
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
sys.modules.setdefault("pkg_resources", types.ModuleType("pkg_resources"))

from models.common import HeatMapParser  # noqa: E402
from models.routing import (  # noqa: E402
    expanded_threshold_centers,
    fixed_threshold_centers,
    topk_cell_centers,
)


def test_fixed_threshold_can_select_zero_patches() -> None:
    mask = torch.full((2, 64, 64), 0.49)

    activated, centers = fixed_threshold_centers(mask, 0.5)

    assert not activated.any()
    assert not centers.any()
    assert not expanded_threshold_centers(mask.unsqueeze(1), 0.5).any()


def test_fixed_threshold_patch_count_is_dynamic() -> None:
    mask = torch.zeros((1, 64, 64))
    mask[0, 4, 4] = 0.9
    mask[0, 7, 7] = 0.8  # same 8x8 coarse cell as the first peak
    mask[0, 40, 40] = 0.7

    _, centers = fixed_threshold_centers(mask, 0.5)
    selected_cells = torch.nn.functional.max_pool2d(
        centers.float(), 8, stride=8
    )

    assert int(centers.sum()) == 3
    assert int(selected_cells.sum()) == 2


@pytest.mark.parametrize("budget", [1, 16, 32, 64])
def test_top_k_selects_exact_cell_budget_even_when_scores_tie(budget: int) -> None:
    mask = torch.zeros((2, 64, 64))

    centers = topk_cell_centers(
        mask, cluster_height=8, cluster_width=8, top_k=budget
    )
    selected_cells = torch.nn.functional.max_pool2d(
        centers.float(), 8, stride=8
    )

    assert centers.sum(dim=(1, 2)).tolist() == [budget, budget]
    assert selected_cells.sum(dim=(1, 2)).tolist() == [budget, budget]


def test_top_k_clamps_to_available_cells_and_zero_disables_it() -> None:
    mask = torch.rand((1, 16, 16))

    assert not topk_cell_centers(
        mask, cluster_height=8, cluster_width=8, top_k=0
    ).any()
    assert int(
        topk_cell_centers(
            mask, cluster_height=8, cluster_width=8, top_k=64
        ).sum()
    ) == 4


def test_heatmap_parser_preserves_zero_patch_threshold_result() -> None:
    parser = HeatMapParser(c=1, ratio=8, threshold=0.5)

    outputs = parser.ada_slicer_fast(torch.zeros((2, 64, 64)))

    assert [len(output) for output in outputs] == [0, 0]


@pytest.mark.parametrize("budget", [1, 16, 32, 64])
def test_heatmap_parser_emits_exact_top_k_patches(budget: int) -> None:
    parser = HeatMapParser(c=1, ratio=8, threshold=0.5)

    outputs = parser.ada_slicer_fast(
        torch.zeros((2, 64, 64)), topk=budget
    )

    assert [len(output) for output in outputs] == [budget, budget]
