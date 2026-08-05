"""Adapter for the YOLOv5-style ESOD repository."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from bcrs.config import ConfigError, ExperimentConfig
from .base import BackendAdapter, CommandSpec, Diagnostic


class EsodAdapter(BackendAdapter):
    name = "esod"
    description = "ESOD patch selection on the upstream YOLOv5-style runtime"
    required_modules = ("torch", "torchvision", "cv2", "numpy", "yaml")

    def build(self, stage: str, experiment: ExperimentConfig) -> CommandSpec:
        if stage not in {"train", "test"}:
            raise ConfigError(f"ESOD does not support stage {stage!r}")
        data_config = self._materialize_dataset(experiment)
        section = experiment.train if stage == "train" else experiment.test
        output_dir = experiment.output_dir
        entrypoint = experiment.backend_root / (
            "train.py" if stage == "train" else "test.py"
        )
        argv = self._python_cmd(experiment, entrypoint)

        if stage == "train":
            argv += [
                "--data",
                str(data_config),
                "--cfg",
                str(experiment.model_config_for(stage)),
                "--epochs",
                str(section.get("epochs", 50)),
                "--batch-size",
                str(section.get("batch_size", 8)),
                "--img-size",
                str(section.get("image_size", 1536)),
                "--workers",
                str(experiment.runtime.get("workers", 8)),
                "--device",
                experiment.devices,
                "--project",
                str(output_dir.parent),
                "--name",
                output_dir.name,
                "--exist-ok",
            ]
            weights = experiment.optional_project_path(experiment.model.get("weights"))
            if weights is not None:
                argv += ["--weights", str(weights)]
            hyp = experiment.optional_project_path(section.get("hyp"))
            if hyp is not None:
                argv += ["--hyp", str(hyp)]
            for flag, option in (
                ("hm_only", "--hm-only"),
                ("hm_metric", "--hm-metric"),
            ):
                if bool(section.get(flag, False)):
                    argv.append(option)
            if section.get("lambda_cov") is not None:
                argv.extend(["--lambda-cov", str(section["lambda_cov"])])
            if section.get("pos_weight") is not None:
                argv.extend(["--pos-weight", str(section["pos_weight"])])
        else:
            checkpoint = experiment.optional_project_path(
                section.get("checkpoint", experiment.model.get("weights"))
            )
            if checkpoint is None:
                raise ConfigError("ESOD test requires test.checkpoint or model.weights")
            argv += [
                "--data",
                str(data_config),
                "--weights",
                str(checkpoint),
                "--batch-size",
                str(section.get("batch_size", 1)),
                "--img-size",
                str(section.get("image_size", 1536)),
                "--device",
                experiment.devices,
                "--task",
                str(section.get("task", "val")),
                "--project",
                str(output_dir.parent),
                "--name",
                output_dir.name + "_test",
                "--exist-ok",
            ]
            top_k = section.get("top_k") or section.get("patch_budget")
            is_sparse = bool(section.get("sparse_head", False)) or (
                top_k is not None and int(top_k) > 0
            )
            for flag, option in (
                ("hm_metric", "--hm-metric"),
                ("save_json", "--save-json"),
            ):
                if bool(section.get(flag, False)):
                    argv.append(option)
            if is_sparse:
                argv.append("--sparse-head")
            if top_k is not None:
                argv.extend(["--top-k", str(top_k)])
            if section.get("hm_threshold"):
                argv.extend(["--hm-threshold", str(section["hm_threshold"])])
        argv.extend(self._extra_args(section))
        return CommandSpec(
            argv=tuple(argv),
            cwd=experiment.backend_root,
            env=self._device_environment(experiment),
            generated_files=(data_config,),
        )

    def diagnostics(self, stage: str, experiment: ExperimentConfig) -> list[Diagnostic]:
        checks = super().diagnostics(stage, experiment)
        entrypoint = experiment.backend_root / (
            "train.py" if stage == "train" else "test.py"
        )
        checks.append(self._path_diagnostic(f"{stage} entrypoint", entrypoint))
        section = experiment.train if stage == "train" else experiment.test
        for role in (("train", "val") if stage == "train" else ("val",)):
            split = experiment.dataset.split(role, "esod")
            is_dir = not bool(split.get("list"))
            checks.append(
                self._path_diagnostic(
                    f"dataset {role}",
                    self._split_input(experiment, role),
                    directory=is_dir,
                )
            )
        if stage == "train" and section.get("hyp"):
            checks.append(
                self._path_diagnostic(
                    "hyperparameters", experiment.optional_project_path(section["hyp"])  # type: ignore[arg-type]
                )
            )
        if stage == "test":
            checkpoint = experiment.optional_project_path(
                section.get("checkpoint", experiment.model.get("weights"))
            )
            if checkpoint is not None:
                checks.append(self._path_diagnostic("checkpoint", checkpoint))
        return checks

    @staticmethod
    def _split_input(experiment: ExperimentConfig, role: str) -> Path:
        split = experiment.dataset.split(role, "esod")
        key = "list" if split.get("list") else "images"
        return experiment.dataset.split_path(role, key, "esod")

    def _materialize_dataset(self, experiment: ExperimentConfig) -> Path:
        output = (
            experiment.project_root
            / ".bcrs"
            / "generated"
            / experiment.name
            / "dataset.esod.yaml"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        test_role = (
            "test"
            if "test" in experiment.dataset.splits
            or "test" in experiment.dataset.adapter_options("esod").get("splits", {})
            else "val"
        )
        payload = {
            "train": str(self._split_input(experiment, "train")),
            "val": str(self._split_input(experiment, "val")),
            "test": str(self._split_input(experiment, test_role)),
            "nc": experiment.dataset.num_classes,
            "names": list(experiment.dataset.classes),
        }
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return output
