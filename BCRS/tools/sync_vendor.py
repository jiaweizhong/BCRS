"""Manage byte-for-byte snapshots of the three research implementations.

The default command verifies vendored files against ``vendor/manifest.json``
and therefore keeps working after the sibling source repositories are deleted.
Use ``--against-source`` while the source repositories still exist, or
``--write`` to intentionally refresh the snapshots and manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MANIFEST = PROJECT_ROOT / "vendor" / "manifest.json"


@dataclass(frozen=True)
class CopyRule:
    name: str
    source_root: Path
    target_root: Path
    upstream: str
    commit: str
    license: str
    patterns: tuple[str, ...]


RULES = (
    CopyRule(
        "esod",
        WORKSPACE_ROOT / "esod",
        PROJECT_ROOT / "vendor" / "esod",
        "https://github.com/alibaba/esod",
        "bde3571bd7db697e441eef0278cd425e888ea026",
        "GPL-3.0",
        (
            "LICENSE",
            "requirements.txt",
            "train.py",
            "test.py",
            "models/__init__.py",
            "models/common.py",
            "models/experimental.py",
            "models/replknet.py",
            "models/spconv.py",
            "models/yolo.py",
            "utils/__init__.py",
            "utils/autoanchor.py",
            "utils/datasets.py",
            "utils/general.py",
            "utils/google_utils.py",
            "utils/loss.py",
            "utils/metrics.py",
            "utils/plots.py",
            "utils/torch_utils.py",
            "utils/wandb_logging/__init__.py",
            "utils/wandb_logging/wandb_utils.py",
            "models/cfg/esod/*yolov5*.yaml",
            "data/visdrone.yaml",
            "data/uavdt.yaml",
            "data/tinyperson.yaml",
            "data/hyps/hyp.visdrone.yaml",
            "data/hyps/hyp.uavdt.yaml",
            "data/hyps/hyp.tinyperson.scratch.yaml",
        ),
    ),
    CopyRule(
        "ceasc",
        WORKSPACE_ROOT / "CEASC",
        PROJECT_ROOT / "vendor" / "ceasc",
        "https://github.com/Cuogeihong/CEASC",
        "2abfd1a99f1b0fe1ed3d51588b64549e1584da50",
        "Apache-2.0",
        (
            "LICENSE",
            "tools/train.py",
            "tools/test.py",
            "mmdet/datasets/UAV.py",
            "mmdet/datasets/UAVDT.py",
            "mmdet/models/dense_heads/anchor_dy_head.py",
            "mmdet/models/dense_heads/gfl_dy_head.py",
            "mmdet/models/dense_heads/retina_dy_head.py",
            "mmdet/models/dense_heads/cuda_dynamic_conv_module.py",
            "mmdet/models/dense_heads/sparse_conv_net.py",
            "mmdet/models/dense_heads/sparseconv_utils.py",
            "configs/UAV/baseline_gfl_res18_visdrone.py",
            "configs/UAV/dynamic_gfl_res18_visdrone.py",
            "configs/UAV/baseline_gfl_res18_uavdt.py",
            "configs/UAV/dynamic_gfl_res18_uavdt.py",
            "configs/UAV/baseline_retinanet_res18_visdrone.py",
            "configs/UAV/dynamic_retinanet_res18_visdrone.py",
            "configs/_base_/default_runtime.py",
            "configs/_base_/schedules/schedule_15e.py",
            "configs/_base_/schedules/schedule_6e.py",
            "Sparse_conv/*",
        ),
    ),
    CopyRule(
        "querydet",
        WORKSPACE_ROOT / "QueryDet-PyTorch",
        PROJECT_ROOT / "vendor" / "querydet",
        "https://github.com/ChenhongyiYang/QueryDet-PyTorch",
        "feebf218d53d59ba054132dfa6ef84159f793967",
        "MIT",
        (
            "LICENSE",
            "train_visdrone.py",
            "infer_visdrone.py",
            "train_coco.py",
            "infer_coco.py",
            "models/querydet/*.py",
            "models/retinanet/retinanet.py",
            "utils/*.py",
            "visdrone/dataloader.py",
            "visdrone/mapper.py",
            "visdrone/utils.py",
            "train_tools/*.py",
            "configs/BaseRetina.yaml",
            "configs/custom_config.py",
            "configs/visdrone/*.yaml",
            "configs/coco/*.yaml",
        ),
    ),
)


def destination(rule: CopyRule, source: Path) -> Path:
    relative = source.relative_to(rule.source_root)
    parts = list(relative.parts)
    if parts[:3] == ["models", "cfg", "esod"]:
        relative = Path("configs", "models", *parts[3:])
    elif parts[:2] == ["data", "hyps"]:
        relative = Path("configs", "hyps", *parts[2:])
    elif parts[:1] == ["data"]:
        relative = Path("configs", "data", *parts[1:])
    elif parts[:3] == ["mmdet", "models", "dense_heads"]:
        relative = Path("models", *parts[3:])
    elif parts[:2] == ["mmdet", "datasets"]:
        relative = Path("datasets", *parts[2:])
    return rule.target_root / relative


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_files(rule: CopyRule) -> list[Path]:
    selected: set[Path] = set()
    for pattern in rule.patterns:
        selected.update(
            path for path in rule.source_root.glob(pattern) if path.is_file()
        )
    return sorted(selected)


def source_commit(rule: CopyRule) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=rule.source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else rule.commit


def write_snapshot() -> int:
    entries: list[dict[str, str]] = []
    repositories: dict[str, dict[str, str]] = {}
    for rule in RULES:
        if not rule.source_root.is_dir():
            print(f"MISSING SOURCE {rule.source_root}")
            return 1
        repositories[rule.name] = {
            "commit": source_commit(rule),
            "license": rule.license,
            "upstream": rule.upstream,
        }
        for source in selected_files(rule):
            target = destination(rule, source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entries.append(
                {
                    "repository": rule.name,
                    "source": source.relative_to(rule.source_root).as_posix(),
                    "target": target.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": digest(target),
                }
            )
    payload = {
        "schema_version": 1,
        "repositories": repositories,
        "files": sorted(entries, key=lambda item: item["target"]),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"copied={len(entries)} manifest={MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


def load_manifest() -> dict[str, object]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST}; run with --write while sources exist"
        )
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError(f"Unsupported vendor manifest: {MANIFEST}")
    return payload


def verify_snapshot() -> int:
    payload = load_manifest()
    failures = 0
    entries = payload["files"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        target = (PROJECT_ROOT / str(entry["target"])).resolve()
        if PROJECT_ROOT.resolve() not in target.parents:
            print(f"INVALID TARGET {entry['target']}")
            failures += 1
        elif not target.is_file():
            print(f"MISSING {entry['target']}")
            failures += 1
        elif digest(target) != entry["sha256"]:
            print(f"MODIFIED {entry['target']}")
            failures += 1
    print(f"verified={len(entries)} failures={failures}")
    return 0 if failures == 0 else 1


def compare_sources() -> int:
    payload = load_manifest()
    rules = {rule.name: rule for rule in RULES}
    failures = 0
    entries = payload["files"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        rule = rules[str(entry["repository"])]
        source = rule.source_root / str(entry["source"])
        if not source.is_file():
            print(f"MISSING SOURCE {source}")
            failures += 1
        elif digest(source) != entry["sha256"]:
            print(f"SOURCE CHANGED {entry['repository']}:{entry['source']}")
            failures += 1
    print(f"source_checked={len(entries)} failures={failures}")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write", action="store_true", help="refresh copies and manifest"
    )
    action.add_argument(
        "--against-source",
        action="store_true",
        help="compare the manifest with sibling source repositories",
    )
    args = parser.parse_args(argv)
    try:
        if args.write:
            return write_snapshot()
        if args.against_source:
            return compare_sources()
        return verify_snapshot()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
