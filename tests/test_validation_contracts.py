"""Tests for shared validation task contract constants."""

# ruff: noqa: S101

from __future__ import annotations

from src import validation


def test_public_contract_alias_exposes_supported_trajectory_vocabulary() -> None:
    """Verify the validation package exposes the foundation trajectory vocabulary."""
    assert validation.contracts.TASK_TYPE_TRAJECTORY == "trajectory"
    assert validation.contracts.FIELD_SHAPE == "shape"
    assert validation.contracts.SHAPE_VERTICAL == "vertical"
    assert validation.contracts.SHAPE_POLYLINE == "polyline"
    assert validation.contracts.SUPPORTED_TRAJECTORY_SHAPES == ("hover", "circle", "line", "vertical", "polyline")


def test_field_constants_match_task_dictionary_keys() -> None:
    """Verify shared field constants match the intended task dictionary keys."""
    expected_fields = {
        "FIELD_TASK_TYPE": "task_type",
        "FIELD_SHAPE": "shape",
        "FIELD_DURATION_SEC": "duration_sec",
        "FIELD_SAMPLE_RATE_HZ": "sample_rate_hz",
        "FIELD_POSITION": "position",
        "FIELD_CENTER": "center",
        "FIELD_RADIUS": "radius",
        "FIELD_HEIGHT": "height",
        "FIELD_CLOCKWISE": "clockwise",
        "FIELD_START": "start",
        "FIELD_END": "end",
        "FIELD_XY": "xy",
        "FIELD_START_HEIGHT": "start_height",
        "FIELD_END_HEIGHT": "end_height",
        "FIELD_POINTS": "points",
    }

    for constant_name, expected_value in expected_fields.items():
        assert getattr(validation.contracts, constant_name) == expected_value
