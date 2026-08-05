"""Load the vendored CEASC plugins, then run an original MMDetection entrypoint.

The model and dataset files are byte-for-byte copies from CEASC.  They retain
their original relative imports (for example ``.anchor_dy_head``), so this
loader installs them under their original MMDetection module names.  The
installed MMDetection package continues to provide the framework itself.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def _load(module_name: str, source: Path) -> None:
    """Execute one copied source file under its original package name."""
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def load_plugins() -> None:
    """Register the original CEASC heads and UAV datasets with MMDetection."""
    importlib.import_module("mmdet.models.dense_heads")
    importlib.import_module("mmdet.datasets")

    dense_head = "mmdet.models.dense_heads"
    for name in (
        "sparse_conv_net",
        "cuda_dynamic_conv_module",
        "sparseconv_utils",
        "anchor_dy_head",
        "gfl_dy_head",
        "retina_dy_head",
    ):
        _load(f"{dense_head}.{name}", ROOT / "models" / f"{name}.py")

    for name in ("UAV", "UAVDT"):
        _load(f"mmdet.datasets.{name}", ROOT / "datasets" / f"{name}.py")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "test"}:
        raise SystemExit(
            "usage: bcrs_entry.py {train|test} [MMDetection arguments ...]"
        )
    stage = sys.argv.pop(1)
    load_plugins()
    entrypoint = ROOT / "tools" / f"{stage}.py"
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
