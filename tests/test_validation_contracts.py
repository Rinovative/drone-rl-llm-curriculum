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
    assert validation.contracts.SHAPE_HOVER_STABILIZATION == "hover_stabilization"
    assert validation.contracts.SHAPE_NEARBY_TARGET_HOVER == "nearby_target_hover"
    assert validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE == "start_hold_then_short_line"
    assert validation.contracts.SHAPE_SHORT_SLOW_LINE == "short_slow_line"
    assert validation.contracts.SUPPORTED_TRAJECTORY_SHAPES == (
        "hover_stabilization",
        "nearby_target_hover",
        "start_hold_then_short_line",
        "short_slow_line",
        "hover",
        "circle",
        "ellipse",
        "figure_eight",
        "line",
        "vertical",
        "polyline",
    )


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
        "FIELD_RADIUS_X": "radius_x",
        "FIELD_RADIUS_Y": "radius_y",
        "FIELD_HEIGHT": "height",
        "FIELD_CLOCKWISE": "clockwise",
        "FIELD_START": "start",
        "FIELD_END": "end",
        "FIELD_XY": "xy",
        "FIELD_START_HEIGHT": "start_height",
        "FIELD_END_HEIGHT": "end_height",
        "FIELD_POINTS": "points",
        "FIELD_HOLD_DURATION_SEC": "hold_duration_sec",
        "FIELD_MOVE_DURATION_SEC": "move_duration_sec",
        "FIELD_START_HOLD_ENABLED": "start_hold_enabled",
        "FIELD_START_HOLD_SEC": "start_hold_sec",
        "FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS": "exclude_start_hold_from_tracking_metrics",
        "FIELD_FINAL_HOLD_ENABLED": "final_hold_enabled",
        "FIELD_FINAL_HOLD_SEC": "final_hold_sec",
        "FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS": "exclude_final_hold_from_tracking_metrics",
    }

    for constant_name, expected_value in expected_fields.items():
        assert getattr(validation.contracts, constant_name) == expected_value
