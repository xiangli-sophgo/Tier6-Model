"""Operator helper utilities."""

from __future__ import annotations

from typing import Any


def get_int(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_first_int(params: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in params and params[key] is not None:
            return get_int(params, key, default)
    return default


def bytes_to_elements(size_bytes: int, element_bytes: int) -> int:
    if element_bytes <= 0:
        return size_bytes
    return (size_bytes + element_bytes - 1) // element_bytes
