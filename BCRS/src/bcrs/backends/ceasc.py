"""Adapter for the self-contained, MMDetection-based CEASC snapshot."""

from __future__ import annotations

from bcrs.config import ConfigError, ExperimentConfig
from .base import BackendAdapter, CommandSpec, Diagnostic


class CeascAdapter(BackendAdapter):
    name = "ceasc"
    description = (
        "CEASC activation masks loaded into an installed MMDetection 2.x runtime"
    )
    required_modules = ("torch", "mmcv", "mmdet", "sparse_conv")

    def build(self, stage: str, experiment: ExperimentConfig) -> CommandSpec:
        if stage not in {"train", "test"}:
            raise ConfigError(f"CEASC does not support stage {stage!r}")
        devices = [item for item in experiment.devices.split(",") if item.strip()]
        if len(devices) > 1:
            raise ConfigError(
                "The CEASC adapter currently supports one visible GPU; use a per-backend "
                "launcher override when distributed training is added"
            )
        section = experiment.train if stage == "train" else experiment.test
        entrypoint = experiment.backend_root / "bcrs_entry.py"
        argv = self._python_cmd(experiment, entrypoint) + [
            stage,
            str(experiment.model_config_for(stage)),
        ]
        if stage == "train":
            argv += ["--work-dir", str(experiment.output_dir)]
            if "seed" in section:
                argv += ["--seed", str(section["seed"])]
            if bool(section.get("deterministic", False)):
                argv.append("--deterministic")
        else:
            checkpoint = experiment.optional_project_path(
                section.get("checkpoint", experiment.model.get("weights"))
            )
            if checkpoint is None:
                raise ConfigError(
                    "CEASC test requires test.checkpoint or model.weights"
                )
            argv += [
                str(checkpoint),
                "--work-dir",
                str(experiment.output_dir / "test"),
                "--eval",
                "bbox",
            ]
        cfg_options = self._config_options(experiment, stage)
        if cfg_options:
            argv.append("--cfg-options")
            argv.extend(cfg_options)
        argv.extend(self._extra_args(section))
        return CommandSpec(
            argv=tuple(argv),
            cwd=experiment.backend_root,
            env=self._device_environment(experiment),
        )

    def diagnostics(self, stage: str, experiment: ExperimentConfig) -> list[Diagnostic]:
        checks = super().diagnostics(stage, experiment)
        checks.append(
            self._path_diagnostic(
                "BCRS plugin entrypoint", experiment.backend_root / "bcrs_entry.py"
            )
        )
        checks.append(
            self._path_diagnostic(
                f"original {stage} entrypoint",
                experiment.backend_root / "tools" / f"{stage}.py",
            )
        )
        roles = ("train", "val") if stage == "train" else ("val",)
        for role in roles:
            checks += [
                self._path_diagnostic(
                    f"dataset {role} annotations",
                    experiment.dataset.split_path(role, "annotations", "ceasc"),
                ),
                self._path_diagnostic(
                    f"dataset {role} images",
                    experiment.dataset.split_path(role, "images", "ceasc"),
                    directory=True,
                ),
            ]
        if stage == "test":
            checkpoint = experiment.optional_project_path(
                experiment.test.get("checkpoint", experiment.model.get("weights"))
            )
            if checkpoint is not None:
                checks.append(self._path_diagnostic("checkpoint", checkpoint))
        sparse_extension = experiment.backend_root / "Sparse_conv"
        checks.append(
            self._path_diagnostic(
                "sparse extension source", sparse_extension, directory=True
            )
        )
        return checks

    def _config_options(self, experiment: ExperimentConfig, stage: str) -> list[str]:
        adapter_options = experiment.dataset.adapter_options("ceasc")
        dataset_type = str(adapter_options.get("dataset_type", "CocoDataset"))
        classes = repr(tuple(experiment.dataset.classes))
        train_ann = experiment.dataset.split_path("train", "annotations", "ceasc")
        train_images = experiment.dataset.split_path("train", "images", "ceasc")
        val_ann = experiment.dataset.split_path("val", "annotations", "ceasc")
        val_images = experiment.dataset.split_path("val", "images", "ceasc")
        result = [
            f"model.bbox_head.num_classes={experiment.dataset.num_classes}",
            f"data.train.dataset.type={dataset_type}",
            f"data.train.dataset.classes={classes}",
            f"data.train.dataset.ann_file={train_ann}",
            f"data.train.dataset.img_prefix={train_images}",
            f"data.val.type={dataset_type}",
            f"data.val.classes={classes}",
            f"data.val.ann_file={val_ann}",
            f"data.val.img_prefix={val_images}",
            f"data.test.type={dataset_type}",
            f"data.test.classes={classes}",
            f"data.test.ann_file={val_ann}",
            f"data.test.img_prefix={val_images}",
        ]
        extra = (experiment.train if stage == "train" else experiment.test).get(
            "config_options", []
        )
        if not isinstance(extra, list):
            raise ConfigError(
                "CEASC config_options must be a list of key=value strings"
            )
        result.extend(str(item) for item in extra)
        return result
