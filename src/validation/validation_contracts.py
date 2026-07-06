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
SUPPORTED_TRAJECTORY_SHAPES: tuple[str, ...] = (
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

__all__ = [
    "FIELD_CENTER",
    "FIELD_CLOCKWISE",
    "FIELD_DURATION_SEC",
    "FIELD_END",
    "FIELD_END_HEIGHT",
    "FIELD_HEIGHT",
    "FIELD_POINTS",
    "FIELD_POSITION",
    "FIELD_RADIUS",
    "FIELD_SAMPLE_RATE_HZ",
    "FIELD_SHAPE",
    "FIELD_START",
    "FIELD_START_HEIGHT",
    "FIELD_TASK_TYPE",
    "FIELD_XY",
    "SHAPE_CIRCLE",
    "SHAPE_HOVER",
    "SHAPE_LINE",
    "SHAPE_POLYLINE",
    "SHAPE_VERTICAL",
    "SUPPORTED_TRAJECTORY_SHAPES",
    "TASK_TYPE_TRAJECTORY",
]
