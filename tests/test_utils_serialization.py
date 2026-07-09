"""Tests for JSON-safe serialization helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import utils

EXPECTED_SCALAR_INT = 7


class _Mode(Enum):
    """Small enum used to verify value conversion."""

    PID = "pid_position"


@dataclass(frozen=True)
class _Payload:
    """Dataclass payload with supported nested conversion types."""

    path: Path
    values: np.ndarray


def test_to_jsonable_converts_numpy_paths_enums_dataclasses_and_nonfinite_values() -> None:
    """Verify nested experiment payloads become strict portable JSON."""
    payload: dict[str, Any] = {
        "array": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "scalar_float": np.float32(1.5),
        "scalar_int": np.int64(EXPECTED_SCALAR_INT),
        "scalar_bool": np.bool_(True),
        "path": Path("storage/runs/example"),
        "enum": _Mode.PID,
        "dataclass": _Payload(path=Path("artifact.json"), values=np.asarray([0.1, 0.2], dtype=np.float64)),
        "tuple": (np.int32(1), np.float64(2.0)),
        "set": {"b", "a"},
        "nan": np.float64(np.nan),
        "inf": float("inf"),
    }

    safe_payload = utils.serialization.to_jsonable(payload)

    assert safe_payload["array"] == [[1.0, 2.0], [3.0, 4.0]]
    assert safe_payload["scalar_float"] == pytest.approx(1.5)
    assert safe_payload["scalar_int"] == EXPECTED_SCALAR_INT
    assert safe_payload["scalar_bool"] is True
    assert safe_payload["path"] == "storage/runs/example"
    assert safe_payload["enum"] == "pid_position"
    assert safe_payload["dataclass"] == {"path": "artifact.json", "values": [0.1, 0.2]}
    assert safe_payload["tuple"] == [1, 2.0]
    assert safe_payload["set"] == ["a", "b"]
    assert safe_payload["nan"] is None
    assert safe_payload["inf"] is None
    assert utils.serialization.find_non_jsonable_paths(safe_payload) == []
    json.dumps(safe_payload, allow_nan=False)


def test_find_non_jsonable_paths_identifies_raw_numpy_and_nonfinite_fields() -> None:
    """Verify debugging paths point to raw values that would break strict JSON."""
    payload = {
        "metrics": {
            "initial_xyz": np.asarray([0.0, 0.0, 1.0]),
            "position_error": np.float32(0.1),
            "nonfinite": float("nan"),
        }
    }

    paths = utils.serialization.find_non_jsonable_paths(payload)

    assert "$.metrics.initial_xyz" in paths
    assert "$.metrics.position_error" in paths
    assert "$.metrics.nonfinite" in paths
    with pytest.raises(TypeError, match="unit payload is not JSON serializable"):
        utils.serialization.assert_json_serializable(payload, "unit payload")
