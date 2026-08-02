"""Command execution utilities."""

from __future__ import annotations

import os
import shlex
import subprocess

from bcrs.backends.base import CommandSpec


def format_command(command: CommandSpec) -> str:
    environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in command.env.items())
    argv = shlex.join(command.argv)
    return f"{environment} {argv}".strip()


def run(command: CommandSpec) -> int:
    environment = os.environ.copy()
    environment.update(command.env)
    completed = subprocess.run(
        list(command.argv),
        cwd=command.cwd,
        env=environment,
        check=False,
    )
    return int(completed.returncode)

