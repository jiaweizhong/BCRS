"""Adapter for the Detectron2-based QueryDet implementation."""

from __future__ import annotations

from bcrs.config import ConfigError, ExperimentConfig
from .base import BackendAdapter, CommandSpec, Diagnostic


class QueryDetAdapter(BackendAdapter):
    name = "querydet"
    description = "QueryDet/CSQ query selection on Detectron2"
    required_modules = ("torch", "detectron2", "fvcore", "spconv")

    def build(self, stage: str, experiment: ExperimentConfig) -> CommandSpec:
        if stage not in {"train", "test"}:
            raise ConfigError(f"QueryDet does not support stage {stage!r}")
        options = experiment.dataset.adapter_options("querydet")
        runner = str(options.get("runner", "visdrone"))
        prefix = "train" if stage == "train" else "infer"
        entrypoint = experiment.backend_root / f"{prefix}_{runner}.py"
        section = experiment.train if stage == "train" else experiment.test
        devices = [item for item in experiment.devices.split(",") if item.strip()]
        argv = [
            experiment.python,
            str(entrypoint),
            "--config-file", str(experiment.model_config_for(stage)),
            "--num-gpus", str(max(1, len(devices))),
        ]
        if bool(section.get("resume", False)):
            argv.append("--resume")
        if stage == "test":
            argv.append("--eval-only")
        cfg_options = self._config_options(experiment, stage)
        argv.extend(cfg_options)
        argv.extend(self._extra_args(section))
        return CommandSpec(
            argv=tuple(argv),
            cwd=experiment.backend_root,
            env=self._device_environment(experiment),
        )

    def diagnostics(self, stage: str, experiment: ExperimentConfig) -> list[Diagnostic]:
        checks = super().diagnostics(stage, experiment)
        runner = str(experiment.dataset.adapter_options("querydet").get("runner", "visdrone"))
        prefix = "train" if stage == "train" else "infer"
        checks.append(
            self._path_diagnostic(
                f"{stage} entrypoint", experiment.backend_root / f"{prefix}_{runner}.py"
            )
        )
        if runner == "visdrone":
            roles = ("train", "val") if stage == "train" else ("val",)
            for role in roles:
                checks.append(
                    self._path_diagnostic(
                        f"dataset {role} annotations",
                        experiment.dataset.split_path(role, "annotations", "querydet"),
                    )
                )
                checks.append(
                    self._path_diagnostic(
                        f"dataset {role} images",
                        experiment.dataset.split_path(role, "images", "querydet"),
                        directory=True,
                    )
                )
        if stage == "test":
            checkpoint = experiment.optional_project_path(
                experiment.test.get("checkpoint", experiment.model.get("weights"))
            )
            if checkpoint is not None:
                checks.append(self._path_diagnostic("checkpoint", checkpoint))
        return checks

    def _config_options(self, experiment: ExperimentConfig, stage: str) -> list[str]:
        options = experiment.dataset.adapter_options("querydet")
        runner = str(options.get("runner", "visdrone"))
        result = [
            "OUTPUT_DIR", str(experiment.output_dir),
            "MODEL.RETINANET.NUM_CLASSES", str(experiment.dataset.num_classes),
        ]
        if runner == "visdrone":
            result += [
                "VISDRONE.TRAIN_JSON", str(experiment.dataset.split_path("train", "annotations", "querydet")),
                "VISDRONE.TRING_IMG_ROOT", str(experiment.dataset.split_path("train", "images", "querydet")),
                "VISDRONE.TEST_JSON", str(experiment.dataset.split_path("val", "annotations", "querydet")),
                "VISDRONE.TEST_IMG_ROOT", str(experiment.dataset.split_path("val", "images", "querydet")),
            ]
        if "short_length" in options:
            result += ["VISDRONE.SHORT_LENGTH", str(options["short_length"])]
        if "max_length" in options:
            result += ["VISDRONE.MAX_LENGTH", str(options["max_length"])]
        weights = experiment.test.get("checkpoint") if stage == "test" else experiment.model.get("weights")
        if weights not in (None, ""):
            weight_value = str(weights)
            if "://" not in weight_value:
                weight_value = str(experiment.optional_project_path(weight_value))
            result += ["MODEL.WEIGHTS", weight_value]
        extra = (experiment.train if stage == "train" else experiment.test).get("config_options", [])
        if not isinstance(extra, list):
            raise ConfigError("QueryDet config_options must be a list of alternating keys and values")
        result.extend(str(item) for item in extra)
        return result
