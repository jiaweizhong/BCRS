from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vendor_manifest_is_complete_and_self_verifying() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "vendor/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    repositories = manifest["repositories"]
    assert set(repositories) == {"ceasc", "esod", "querydet"}
    assert all(
        repository["commit"] != "unknown" for repository in repositories.values()
    )

    files = manifest["files"]
    assert len(files) >= 90
    for entry in files:
        target = PROJECT_ROOT / entry["target"]
        assert target.is_file(), entry["target"]
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == entry["sha256"], entry["target"]


def test_experiments_only_launch_vendored_backends() -> None:
    for path in (PROJECT_ROOT / "configs/experiments").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = str(payload["backend"]["root"]).replace("\\", "/")
        assert root.startswith("vendor/"), f"{path.name}: {root}"
        assert payload["backend"]["python"] == "${BCRS_PYTHON:-python}"


def test_ceasc_glue_is_not_mixed_with_upstream_checksums() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "vendor/manifest.json").read_text(encoding="utf-8")
    )
    targets = {entry["target"] for entry in manifest["files"]}
    assert "vendor/ceasc/bcrs_entry.py" not in targets
    assert (PROJECT_ROOT / "vendor/ceasc/bcrs_entry.py").is_file()
