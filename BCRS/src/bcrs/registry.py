"""Lazy backend registry."""

from __future__ import annotations

from typing import Callable

from bcrs.config import ConfigError
from bcrs.backends.base import BackendAdapter


def _load_esod() -> BackendAdapter:
    from bcrs.backends.esod import EsodAdapter

    return EsodAdapter()


def _load_querydet() -> BackendAdapter:
    from bcrs.backends.querydet import QueryDetAdapter

    return QueryDetAdapter()


def _load_ceasc() -> BackendAdapter:
    from bcrs.backends.ceasc import CeascAdapter

    return CeascAdapter()


_BACKENDS: dict[str, Callable[[], BackendAdapter]] = {
    "esod": _load_esod,
    "querydet": _load_querydet,
    "ceasc": _load_ceasc,
}


def backend_names() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_backend(name: str) -> BackendAdapter:
    try:
        return _BACKENDS[name.lower()]()
    except KeyError as exc:
        raise ConfigError(
            f"Unknown backend {name!r}; choose one of {', '.join(backend_names())}"
        ) from exc
