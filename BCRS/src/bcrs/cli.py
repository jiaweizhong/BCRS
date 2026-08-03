"""Command-line interface for planning, validating, training, and testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bcrs.config import ConfigError, load_experiment
from bcrs.registry import backend_names, get_backend
from bcrs.runner import format_command, run


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="experiment YAML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="PATH=VALUE",
        help="override a configuration value; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bcrs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-backends", help="list registered detector backends")

    show = subparsers.add_parser("show", help="print a resolved experiment")
    _add_config_arguments(show)

    doctor = subparsers.add_parser("doctor", help="validate paths and render a command")
    _add_config_arguments(doctor)
    doctor.add_argument("--stage", choices=["train", "test"], default="train")

    for stage in ("train", "test"):
        command = subparsers.add_parser(stage, help=f"run the {stage} adapter")
        _add_config_arguments(command)
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="render the command without executing it",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-backends":
            for name in backend_names():
                adapter = get_backend(name)
                print(f"{name:10} {adapter.description}")
            return 0

        experiment = load_experiment(args.config, args.overrides)
        adapter = get_backend(experiment.backend_name)
        if args.command == "show":
            print(json.dumps(experiment.as_dict(), indent=2, sort_keys=True))
            return 0

        stage = args.stage if args.command == "doctor" else args.command
        command = adapter.build(stage, experiment)
        if args.command == "doctor":
            diagnostics = adapter.diagnostics(stage, experiment)
            for diagnostic in diagnostics:
                marker = "OK" if diagnostic.ok else "MISSING"
                print(f"[{marker:7}] {diagnostic.label}: {diagnostic.detail}")
            print(f"[COMMAND] cwd={command.cwd}")
            print(format_command(command))
            return 0 if all(item.ok for item in diagnostics) else 1

        print(f"cwd: {command.cwd}")
        print(format_command(command))
        if args.dry_run:
            return 0
        return run(command)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"bcrs: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
