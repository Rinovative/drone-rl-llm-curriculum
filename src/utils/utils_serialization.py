"""
===============================================================================
utils_serialization.py
===============================================================================
Convert nested experiment payloads into deterministic JSON-compatible data.

Responsibilities:
  - Convert NumPy arrays and scalars before metrics and manifests are written
  - Preserve nested metric structure while replacing unsupported scalar types
  - Identify raw non-JSON values with stable diagnostic paths for tests

Design principles:
  - Keep JSON artifacts portable by avoiding NaN and Infinity literals
  - Preserve numeric arrays as numeric lists instead of stringifying them
  - Fail loudly for unsupported objects rather than silently dropping fields

Boundaries:
  - Artifact path layout belongs in utils_artifacts.py
  - Callers decide which payloads are small enough to write
===============================================================================

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

JSON_KEY_TYPES = (str, int, float, bool, type(None))
MAX_REPORTED_PATHS = 12


def to_jsonable(value: Any) -> Any:  # noqa: PLR0911
    """
    Return a deterministic JSON-compatible copy of a nested value.

    Parameters
    ----------
    value
        Arbitrary nested payload containing JSON primitives, mappings, sequences,
        dataclasses, paths, enums, or NumPy values.

    Returns
    -------
    Any
        A structure accepted by ``json.dumps(..., allow_nan=False)`` when all
        custom objects are among the supported conversion types. Non-finite
        floats are converted to ``None`` so artifacts contain portable JSON
        ``null`` values rather than NaN or Infinity literals.

    """
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {_jsonable_key(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=_stable_sort_key)]
    return value


def assert_json_serializable(value: Any, context: str = "") -> None:
    """
    Raise if a payload still contains values that strict JSON cannot represent.

    Parameters
    ----------
    value
        Payload to check. Call ``to_jsonable`` first when conversion is desired.
    context
        Optional human-readable payload name included in failures.

    Raises
    ------
    TypeError
        If unsupported objects or non-finite floats are present.

    """
    paths = find_non_jsonable_paths(value)
    if paths:
        label = context or "payload"
        shown = ", ".join(paths[:MAX_REPORTED_PATHS])
        suffix = "" if len(paths) <= MAX_REPORTED_PATHS else f", ... ({len(paths)} total)"
        message = f"{label} is not JSON serializable at: {shown}{suffix}"
        raise TypeError(message)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        label = context or "payload"
        message = f"{label} is not JSON serializable: {exc}"
        raise TypeError(message) from exc


def find_non_jsonable_paths(value: Any) -> list[str]:
    """
    Return diagnostic paths to raw values that strict JSON cannot serialize.

    Parameters
    ----------
    value
        Payload to inspect without applying conversions.

    Returns
    -------
    list[str]
        Stable paths such as ``$.metrics.initial_xyz`` or ``$.items[0]``.

    """
    paths: list[str] = []
    _find_non_jsonable_paths(value, "$", paths)
    return paths


def _find_non_jsonable_paths(value: Any, path: str, paths: list[str]) -> None:
    """Append non-JSON paths found under one nested value."""
    if isinstance(value, (np.ndarray, np.generic, Path, Enum)):
        paths.append(path)
        return
    if is_dataclass(value) and not isinstance(value, type):
        paths.append(path)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            paths.append(path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, JSON_KEY_TYPES):
                paths.append(f"{path}.{_key_label(key)}")
            _find_non_jsonable_paths(item, _child_path(path, key), paths)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _find_non_jsonable_paths(item, f"{path}[{index}]", paths)
        return
    paths.append(path)


def _jsonable_key(key: Any) -> str:
    """Return a deterministic string key for JSON object mappings."""
    if isinstance(key, np.generic):
        return str(key.item())
    if isinstance(key, Enum):
        return str(key.value)
    if isinstance(key, Path):
        return str(key)
    return str(key)


def _stable_sort_key(value: Any) -> str:
    """Return a deterministic sort key for unordered JSON array sources."""
    try:
        return json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _child_path(path: str, key: Any) -> str:
    """Return a readable child path for a mapping key."""
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{_key_label(key)}]"


def _key_label(key: Any) -> str:
    """Return a compact diagnostic label for a mapping key."""
    if isinstance(key, str):
        return repr(key)
    if isinstance(key, np.generic):
        return repr(key.item())
    if isinstance(key, Enum):
        return repr(key.value)
    if isinstance(key, Path):
        return repr(str(key))
    return repr(key)


__all__ = [
    "assert_json_serializable",
    "find_non_jsonable_paths",
    "to_jsonable",
]
