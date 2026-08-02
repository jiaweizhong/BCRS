"""Run the BCRS VisDrone converter from a source checkout."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bcrs.datasets.visdrone import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
