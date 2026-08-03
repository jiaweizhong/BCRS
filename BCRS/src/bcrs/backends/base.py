"""Shared command and diagnostic contracts for detector backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from bcrs.config import ExperimentConfig


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)
    generated_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Diagnostic:
    ok: bool
    label: str
    detail: str


class BackendAdapter(ABC):
    name: str
    description: str
    required_modules: tuple[str, ...] = ()

    @abstractmethod
    def build(self, stage: str, experiment: ExperimentConfig) -> CommandSpec:
        """Build, but do not execute, the backend command."""

    def diagnostics(self, stage: str, experiment: ExperimentConfig) -> list[Diagnostic]:
        command = self.build(stage, experiment)
        checks = [
            self._path_diagnostic("backend root", command.cwd, directory=True),
            self._path_diagnostic("model config", experiment.model_config_for(stage)),
        ]
        executable = experiment.python
        resolved = (
            str(Path(executable).resolve())
            if Path(executable).is_file()
            else shutil.which(executable)
        )
        checks.append(
            Diagnostic(bool(resolved), "python", resolved or f"not found: {executable}")
        )
        if self.required_modules:
            checks.append(self._module_diagnostic(executable, self.required_modules))
        return checks

    @staticmethod
    def _module_diagnostic(executable: str, modules: tuple[str, ...]) -> Diagnostic:
        script = (
            "import importlib.util; "
            f"modules={modules!r}; "
            "missing=[name for name in modules if importlib.util.find_spec(name) is None]; "
            "print(','.join(missing)); "
            "raise SystemExit(bool(missing))"
        )
        try:
            completed = subprocess.run(
                [executable, "-c", script],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Diagnostic(False, "runtime modules", str(exc))
        missing = completed.stdout.strip()
        detail = (
            "available"
            if completed.returncode == 0
            else f"missing: {missing or 'unknown'}"
        )
        return Diagnostic(completed.returncode == 0, "runtime modules", detail)

    @staticmethod
    def _path_diagnostic(label: str, path: Path, directory: bool = False) -> Diagnostic:
        ok = path.is_dir() if directory else path.is_file()
        expected = "directory" if directory else "file"
        return Diagnostic(ok, label, f"{path} ({expected})")

    @staticmethod
    def _device_environment(experiment: ExperimentConfig) -> dict[str, str]:
        devices = experiment.devices.strip()
        env = {"SETUPTOOLS_USE_DISTUTILS": "stdlib"}
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices
        return env

    @staticmethod
    def _extra_args(section: Mapping[str, object]) -> list[str]:
        value = section.get("extra_args", [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("extra_args must be a list")
        return [str(item) for item in value]
