"""Configuration loading and validation for backend-neutral experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment or dataset configuration is invalid."""


_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            if name in os.environ:
                return os.environ[name]
            default = match.group("default")
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name!r} is required")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return content


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _apply_override(config: MutableMapping[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"Override must use dotted.path=value syntax: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = [key for key in dotted_key.split(".") if key]
    if not keys:
        raise ConfigError(f"Override key is empty: {expression!r}")
    node: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, MutableMapping):
            raise ConfigError(f"Override traverses a non-mapping field: {dotted_key!r}")
        node = child
    node[keys[-1]] = yaml.safe_load(raw_value)


@dataclass(frozen=True)
class DatasetSpec:
    """Canonical dataset metadata with optional backend-specific layouts."""

    name: str
    root: Path
    num_classes: int
    classes: tuple[str, ...]
    splits: Mapping[str, Mapping[str, Any]]
    adapters: Mapping[str, Mapping[str, Any]]
    source: Path

    def adapter_options(self, backend: str) -> Mapping[str, Any]:
        options = self.adapters.get(backend, {})
        return options if isinstance(options, Mapping) else {}

    def split(self, role: str, backend: str) -> dict[str, Any]:
        merged: dict[str, Any] = dict(self.splits.get(role, {}))
        adapter_splits = self.adapter_options(backend).get("splits", {})
        if isinstance(adapter_splits, Mapping):
            backend_split = adapter_splits.get(role, {})
            if isinstance(backend_split, Mapping):
                merged.update(backend_split)
        if not merged:
            raise ConfigError(
                f"Dataset {self.name!r} has no {role!r} split for backend {backend!r}"
            )
        return merged

    def split_path(self, role: str, key: str, backend: str) -> Path:
        split = self.split(role, backend)
        if key not in split or split[key] in (None, ""):
            raise ConfigError(
                f"Dataset {self.name!r} split {role!r} has no {key!r} path "
                f"for backend {backend!r}"
            )
        path = Path(str(split[key])).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully resolved experiment configuration."""

    name: str
    source: Path
    project_root: Path
    backend: Mapping[str, Any]
    model: Mapping[str, Any]
    dataset: DatasetSpec
    runtime: Mapping[str, Any]
    train: Mapping[str, Any]
    test: Mapping[str, Any]

    @property
    def backend_name(self) -> str:
        return str(self.backend["name"])

    @property
    def backend_root(self) -> Path:
        return _project_path(self.project_root, str(self.backend["root"]))

    @property
    def python(self) -> str:
        return str(self.backend.get("python", "python"))

    @property
    def model_config(self) -> Path:
        return _project_path(self.project_root, str(self.model["config"]))

    def model_config_for(self, stage: str) -> Path:
        """Return a stage-specific model config when one is configured."""
        section = self.test if stage == "test" else self.train
        value = section.get("model_config", self.model["config"])
        return _project_path(self.project_root, str(value))

    @property
    def output_dir(self) -> Path:
        value = self.runtime.get("output_dir", f"work_dirs/{self.name}")
        return _project_path(self.project_root, str(value))

    @property
    def devices(self) -> str:
        return str(self.runtime.get("devices", "0"))

    def optional_project_path(self, value: Any) -> Path | None:
        if value in (None, ""):
            return None
        return _project_path(self.project_root, str(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": str(self.source),
            "project_root": str(self.project_root),
            "backend": dict(self.backend),
            "model": dict(self.model),
            "dataset": {
                "name": self.dataset.name,
                "root": str(self.dataset.root),
                "num_classes": self.dataset.num_classes,
                "classes": list(self.dataset.classes),
            },
            "runtime": dict(self.runtime),
            "train": dict(self.train),
            "test": dict(self.test),
        }


def _load_dataset(path: Path, root_override: Any = None) -> DatasetSpec:
    raw = _expand_environment(_load_yaml(path))
    name = str(raw.get("name", "")).strip()
    classes = raw.get("classes", [])
    if not name:
        raise ConfigError(f"Dataset name is required: {path}")
    if not isinstance(classes, list) or not all(
        isinstance(item, str) for item in classes
    ):
        raise ConfigError(f"Dataset classes must be a list of strings: {path}")
    num_classes = int(raw.get("num_classes", len(classes)))
    if classes and len(classes) != num_classes:
        raise ConfigError(
            f"Dataset {name!r} declares {num_classes} classes but lists {len(classes)} names"
        )
    root_value = root_override if root_override not in (None, "") else raw.get("root")
    if root_value in (None, ""):
        raise ConfigError(f"Dataset root is required: {path}")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    splits = raw.get("splits", {})
    adapters = raw.get("adapters", {})
    if not isinstance(splits, Mapping) or not isinstance(adapters, Mapping):
        raise ConfigError(f"Dataset splits and adapters must be mappings: {path}")
    return DatasetSpec(
        name=name,
        root=root,
        num_classes=num_classes,
        classes=tuple(classes),
        splits=splits,
        adapters=adapters,
        source=path,
    )


def load_experiment(
    path: str | Path, overrides: Iterable[str] = ()
) -> ExperimentConfig:
    """Load an experiment, resolve environment variables, and validate its schema."""

    source = Path(path).expanduser().resolve()
    project_root = _find_project_root(source)
    raw = _load_yaml(source)
    for expression in overrides:
        _apply_override(raw, expression)
    raw = _expand_environment(raw)

    schema_version = int(raw.get("schema_version", 0))
    if schema_version != 1:
        raise ConfigError(f"Unsupported schema_version {schema_version}; expected 1")
    name = str(raw.get("name", "")).strip()
    backend = raw.get("backend", {})
    model = raw.get("model", {})
    dataset_ref = raw.get("dataset", {})
    if not name:
        raise ConfigError("Experiment name is required")
    if (
        not isinstance(backend, Mapping)
        or not backend.get("name")
        or not backend.get("root")
    ):
        raise ConfigError("backend.name and backend.root are required")
    if not isinstance(model, Mapping) or not model.get("config"):
        raise ConfigError("model.config is required")
    if not isinstance(dataset_ref, Mapping) or not dataset_ref.get("config"):
        raise ConfigError("dataset.config is required")

    dataset_path = _project_path(project_root, str(dataset_ref["config"]))
    dataset = _load_dataset(dataset_path, dataset_ref.get("root"))
    runtime = raw.get("runtime", {})
    train = raw.get("train", {})
    test = raw.get("test", {})
    for label, value in (("runtime", runtime), ("train", train), ("test", test)):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{label} must be a mapping")

    return ExperimentConfig(
        name=name,
        source=source,
        project_root=project_root,
        backend=backend,
        model=model,
        dataset=dataset,
        runtime=runtime,
        train=train,
        test=test,
    )
