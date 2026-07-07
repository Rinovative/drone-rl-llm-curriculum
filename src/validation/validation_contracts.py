"""
===============================================================================
validation_contracts.py
===============================================================================
Define shared vocabulary for trajectory task dictionaries.

Responsibilities:
  - Centralize task type, shape, and field-name constants
  - Declare the trajectory shapes supported by foundation validation

Design principles:
  - Keep the contract lightweight and importable by validation, LLM, config, and environment code
  - Avoid schema or class hierarchy decisions until later phases need them

Boundaries:
  - Task feasibility checks belong in validation_tasks.py
  - Trajectory sampling belongs in trajectory modules
===============================================================================

"""

TASK_TYPE_TRAJECTORY = "trajectory"

SHAPE_HOVER = "hover"
SHAPE_CIRCLE = "circle"
SHAPE_LINE = "line"
SHAPE_VERTICAL = "vertical"
SHAPE_POLYLINE = "polyline"
SHAPE_HOVER_STABILIZATION = "hover_stabilization"
SHAPE_NEARBY_TARGET_HOVER = "nearby_target_hover"
SHAPE_START_HOLD_THEN_SHORT_LINE = "start_hold_then_short_line"
SHAPE_SHORT_SLOW_LINE = "short_slow_line"
SUPPORTED_TRAJECTORY_SHAPES: tuple[str, ...] = (
    SHAPE_HOVER_STABILIZATION,
    SHAPE_NEARBY_TARGET_HOVER,
    SHAPE_START_HOLD_THEN_SHORT_LINE,
    SHAPE_SHORT_SLOW_LINE,
    SHAPE_HOVER,
    SHAPE_CIRCLE,
    SHAPE_LINE,
    SHAPE_VERTICAL,
    SHAPE_POLYLINE,
)

FIELD_TASK_TYPE = "task_type"
FIELD_SHAPE = "shape"
FIELD_DURATION_SEC = "duration_sec"
FIELD_SAMPLE_RATE_HZ = "sample_rate_hz"
FIELD_POSITION = "position"
FIELD_CENTER = "center"
FIELD_RADIUS = "radius"
FIELD_HEIGHT = "height"
FIELD_CLOCKWISE = "clockwise"
FIELD_START = "start"
FIELD_END = "end"
FIELD_XY = "xy"
FIELD_START_HEIGHT = "start_height"
FIELD_END_HEIGHT = "end_height"
FIELD_POINTS = "points"
FIELD_HOLD_DURATION_SEC = "hold_duration_sec"
FIELD_MOVE_DURATION_SEC = "move_duration_sec"
FIELD_START_HOLD_ENABLED = "start_hold_enabled"
FIELD_START_HOLD_SEC = "start_hold_sec"
FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS = "exclude_start_hold_from_tracking_metrics"

__all__ = [
    "FIELD_CENTER",
    "FIELD_CLOCKWISE",
    "FIELD_DURATION_SEC",
    "FIELD_END",
    "FIELD_END_HEIGHT",
    "FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS",
    "FIELD_HEIGHT",
    "FIELD_HOLD_DURATION_SEC",
    "FIELD_MOVE_DURATION_SEC",
    "FIELD_POINTS",
    "FIELD_POSITION",
    "FIELD_RADIUS",
    "FIELD_SAMPLE_RATE_HZ",
    "FIELD_SHAPE",
    "FIELD_START",
    "FIELD_START_HEIGHT",
    "FIELD_START_HOLD_ENABLED",
    "FIELD_START_HOLD_SEC",
    "FIELD_TASK_TYPE",
    "FIELD_XY",
    "SHAPE_CIRCLE",
    "SHAPE_HOVER",
    "SHAPE_HOVER_STABILIZATION",
    "SHAPE_LINE",
    "SHAPE_NEARBY_TARGET_HOVER",
    "SHAPE_POLYLINE",
    "SHAPE_SHORT_SLOW_LINE",
    "SHAPE_START_HOLD_THEN_SHORT_LINE",
    "SHAPE_VERTICAL",
    "SUPPORTED_TRAJECTORY_SHAPES",
    "TASK_TYPE_TRAJECTORY",
]
