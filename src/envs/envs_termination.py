"""
===============================================================================
envs_termination.py
===============================================================================
Configure trajectory-tracking termination and diagnostic safety limits.

Responsibilities:
  - Define strict/default and relaxed safety limit presets for tracking rollouts
  - Parse user-provided termination and diagnostic limit configuration mappings
  - Classify state-vector limit violations without mutating simulator state

Design principles:
  - Preserve upstream HoverAviary termination behavior unless a config opts out
  - Keep hard episode control separate from diagnostic instability reporting
  - Make direct-RPM training more permissive while still bounding unrecoverable states

Boundaries:
  - Simulator stepping belongs in envs_tracking_env.py
  - Evaluation aggregation and failure reports belong in evaluation modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from src import envs

MODE_DEFAULT = "default"
MODE_RELAXED = "relaxed"
MODE_CUSTOM = "custom"
PROFILE_DEFAULT = "default"
PROFILE_PID_RELAXED = "pid_relaxed"
PROFILE_DIRECT_RPM_RELAXED = "direct_rpm_relaxed"
PROFILE_CUSTOM = "custom"

DEFAULT_MAX_ABS_XY_M = 1.5
DEFAULT_MAX_Z_M = 2.0
DEFAULT_MAX_ROLL_PITCH_RAD = 0.4
DEFAULT_MAX_SPEED_MPS = 5.0
DEFAULT_MAX_ANGULAR_VELOCITY_RADPS = 15.0
DEFAULT_MIN_Z_M = 0.0

PID_RELAXED_MAX_ABS_XY_M = 3.0
PID_RELAXED_MAX_Z_M = 3.0
PID_RELAXED_MIN_Z_M = -0.05
PID_RELAXED_MAX_ROLL_PITCH_RAD = 0.7
PID_RELAXED_MAX_SPEED_MPS = 8.0
PID_RELAXED_MAX_ANGULAR_VELOCITY_RADPS = 20.0
PID_RELAXED_RECOVERY_STEPS = 8

DIRECT_RPM_RELAXED_MAX_ABS_XY_M = 5.0
DIRECT_RPM_RELAXED_MAX_Z_M = 4.0
DIRECT_RPM_RELAXED_MIN_Z_M = -0.1
DIRECT_RPM_RELAXED_MAX_ROLL_PITCH_RAD = 1.2
DIRECT_RPM_RELAXED_MAX_SPEED_MPS = 12.0
DIRECT_RPM_RELAXED_MAX_ANGULAR_VELOCITY_RADPS = 30.0
DIRECT_RPM_RELAXED_RECOVERY_STEPS = 20

LIMIT_FIELD_NAMES = (
    "max_abs_xy_m",
    "max_roll_pitch_rad",
    "max_speed_mps",
    "max_angular_velocity_radps",
    "min_z_m",
    "max_z_m",
)


@dataclass(frozen=True)
class TerminationLimitConfig:
    """
    Hard episode-control limits for trajectory tracking environments.

    Parameters
    ----------
    mode
        User-facing configuration mode: ``default``, ``relaxed``, or ``custom``.
    profile
        Resolved preset profile after action-interface-aware parsing.
    max_abs_xy_m
        Optional symmetric absolute X/Y position limit in meters.
    max_roll_pitch_rad
        Optional absolute roll/pitch limit in radians.
    max_speed_mps
        Optional Euclidean linear speed limit in meters per second.
    max_angular_velocity_radps
        Optional Euclidean angular velocity limit in radians per second.
    min_z_m
        Optional lower altitude limit in meters.
    max_z_m
        Optional upper altitude limit in meters.
    allow_recovery_steps
        Number of consecutive hard-limit violation steps allowed before truncation.
    terminate_on_base_truncation
        Whether upstream HoverAviary truncation immediately truncates the wrapper episode.

    """

    mode: str
    profile: str
    max_abs_xy_m: float | None
    max_roll_pitch_rad: float | None
    max_speed_mps: float | None
    max_angular_velocity_radps: float | None
    min_z_m: float | None
    max_z_m: float | None
    allow_recovery_steps: int
    terminate_on_base_truncation: bool

    @property
    def base_truncation_policy(self) -> str:
        """Return the compact policy name for upstream truncation flags."""
        return "terminate" if self.terminate_on_base_truncation else "diagnose_only"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable termination limit metadata."""
        return {
            "mode": self.mode,
            "profile": self.profile,
            "max_abs_xy_m": self.max_abs_xy_m,
            "max_roll_pitch_rad": self.max_roll_pitch_rad,
            "max_speed_mps": self.max_speed_mps,
            "max_angular_velocity_radps": self.max_angular_velocity_radps,
            "min_z_m": self.min_z_m,
            "max_z_m": self.max_z_m,
            "allow_recovery_steps": self.allow_recovery_steps,
            "terminate_on_base_truncation": self.terminate_on_base_truncation,
            "base_truncation_policy": self.base_truncation_policy,
        }


@dataclass(frozen=True)
class DiagnosticLimitConfig:
    """
    Strict diagnostic thresholds reported independently of episode termination.

    Parameters
    ----------
    mode
        User-facing configuration mode: ``default`` or ``custom``.
    profile
        Resolved diagnostic profile name.
    max_abs_xy_m
        Optional symmetric absolute X/Y position diagnostic threshold in meters.
    max_roll_pitch_rad
        Optional absolute roll/pitch diagnostic threshold in radians.
    max_speed_mps
        Optional Euclidean linear speed diagnostic threshold in meters per second.
    max_angular_velocity_radps
        Optional Euclidean angular velocity diagnostic threshold in radians per second.
    min_z_m
        Optional lower altitude diagnostic threshold in meters.
    max_z_m
        Optional upper altitude diagnostic threshold in meters.

    """

    mode: str
    profile: str
    max_abs_xy_m: float | None
    max_roll_pitch_rad: float | None
    max_speed_mps: float | None
    max_angular_velocity_radps: float | None
    min_z_m: float | None
    max_z_m: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostic limit metadata."""
        return {
            "mode": self.mode,
            "profile": self.profile,
            "max_abs_xy_m": self.max_abs_xy_m,
            "max_roll_pitch_rad": self.max_roll_pitch_rad,
            "max_speed_mps": self.max_speed_mps,
            "max_angular_velocity_radps": self.max_angular_velocity_radps,
            "min_z_m": self.min_z_m,
            "max_z_m": self.max_z_m,
        }


def default_termination_limits() -> TerminationLimitConfig:
    """Return termination limits that preserve upstream HoverAviary truncation behavior."""
    return TerminationLimitConfig(
        mode=MODE_DEFAULT,
        profile=PROFILE_DEFAULT,
        max_abs_xy_m=DEFAULT_MAX_ABS_XY_M,
        max_roll_pitch_rad=DEFAULT_MAX_ROLL_PITCH_RAD,
        max_speed_mps=None,
        max_angular_velocity_radps=None,
        min_z_m=None,
        max_z_m=DEFAULT_MAX_Z_M,
        allow_recovery_steps=0,
        terminate_on_base_truncation=True,
    )


def relaxed_termination_limits(action_interface: envs.actions.ActionInterface | str) -> TerminationLimitConfig:
    """Return action-interface-aware relaxed termination limits for training."""
    parsed_interface = envs.actions.parse_action_interface(action_interface)
    if parsed_interface == envs.actions.ActionInterface.DIRECT_RPM:
        return TerminationLimitConfig(
            mode=MODE_RELAXED,
            profile=PROFILE_DIRECT_RPM_RELAXED,
            max_abs_xy_m=DIRECT_RPM_RELAXED_MAX_ABS_XY_M,
            max_roll_pitch_rad=DIRECT_RPM_RELAXED_MAX_ROLL_PITCH_RAD,
            max_speed_mps=DIRECT_RPM_RELAXED_MAX_SPEED_MPS,
            max_angular_velocity_radps=DIRECT_RPM_RELAXED_MAX_ANGULAR_VELOCITY_RADPS,
            min_z_m=DIRECT_RPM_RELAXED_MIN_Z_M,
            max_z_m=DIRECT_RPM_RELAXED_MAX_Z_M,
            allow_recovery_steps=DIRECT_RPM_RELAXED_RECOVERY_STEPS,
            terminate_on_base_truncation=False,
        )
    return TerminationLimitConfig(
        mode=MODE_RELAXED,
        profile=PROFILE_PID_RELAXED,
        max_abs_xy_m=PID_RELAXED_MAX_ABS_XY_M,
        max_roll_pitch_rad=PID_RELAXED_MAX_ROLL_PITCH_RAD,
        max_speed_mps=PID_RELAXED_MAX_SPEED_MPS,
        max_angular_velocity_radps=PID_RELAXED_MAX_ANGULAR_VELOCITY_RADPS,
        min_z_m=PID_RELAXED_MIN_Z_M,
        max_z_m=PID_RELAXED_MAX_Z_M,
        allow_recovery_steps=PID_RELAXED_RECOVERY_STEPS,
        terminate_on_base_truncation=False,
    )


def default_diagnostic_limits() -> DiagnosticLimitConfig:
    """Return strict diagnostic thresholds matching upstream attitude/position checks."""
    return DiagnosticLimitConfig(
        mode=MODE_DEFAULT,
        profile=PROFILE_DEFAULT,
        max_abs_xy_m=DEFAULT_MAX_ABS_XY_M,
        max_roll_pitch_rad=DEFAULT_MAX_ROLL_PITCH_RAD,
        max_speed_mps=DEFAULT_MAX_SPEED_MPS,
        max_angular_velocity_radps=DEFAULT_MAX_ANGULAR_VELOCITY_RADPS,
        min_z_m=DEFAULT_MIN_Z_M,
        max_z_m=DEFAULT_MAX_Z_M,
    )


def parse_termination_limits(
    value: Any,
    action_interface: envs.actions.ActionInterface | str,
) -> TerminationLimitConfig:
    """
    Parse termination-limit settings from a config mapping or preset name.

    Parameters
    ----------
    value
        ``None``, a mode string, an existing config, or a mapping containing ``mode`` and overrides.
    action_interface
        Resolved action interface used when ``mode`` is ``relaxed``.

    Returns
    -------
    TerminationLimitConfig
        Validated hard termination configuration.

    Raises
    ------
    TypeError
        If the value is not a supported config representation.
    ValueError
        If a mode or numeric limit is invalid.

    """
    if isinstance(value, TerminationLimitConfig):
        return value
    if value is None:
        return default_termination_limits()
    if isinstance(value, str):
        return _termination_limits_from_mapping({"mode": value}, action_interface)
    if isinstance(value, Mapping):
        return _termination_limits_from_mapping(value, action_interface)
    message = "termination_limits must be a mapping, string mode, or None"
    raise TypeError(message)


def parse_diagnostic_limits(value: Any) -> DiagnosticLimitConfig:
    """
    Parse diagnostic-limit settings from a config mapping or preset name.

    Parameters
    ----------
    value
        ``None``, a mode string, an existing config, or a mapping containing ``mode`` and overrides.

    Returns
    -------
    DiagnosticLimitConfig
        Validated strict diagnostic threshold configuration.

    Raises
    ------
    TypeError
        If the value is not a supported config representation.
    ValueError
        If a mode or numeric limit is invalid.

    """
    if isinstance(value, DiagnosticLimitConfig):
        return value
    if value is None:
        return default_diagnostic_limits()
    if isinstance(value, str):
        return _diagnostic_limits_from_mapping({"mode": value})
    if isinstance(value, Mapping):
        return _diagnostic_limits_from_mapping(value)
    message = "diagnostic_limits must be a mapping, string mode, or None"
    raise TypeError(message)


def state_limit_violations(state: Any, limits: TerminationLimitConfig | DiagnosticLimitConfig) -> list[str]:
    """
    Return named limit violations visible in a HoverAviary state vector.

    Parameters
    ----------
    state
        Upstream HoverAviary state vector with position, attitude, velocity, and angular velocity fields.
    limits
        Termination or diagnostic threshold configuration.

    Returns
    -------
    list[str]
        Stable violation cause names suitable for info dictionaries and JSON traces.

    """
    state_values = [float(value) for value in state]
    x_position, y_position, z_position = state_values[:3]
    roll, pitch = state_values[7:9]
    velocity = state_values[10:13]
    angular_velocity = state_values[13:16]
    causes: list[str] = []
    if limits.max_abs_xy_m is not None:
        if abs(x_position) > limits.max_abs_xy_m:
            causes.append("x_position_out_of_bounds")
        if abs(y_position) > limits.max_abs_xy_m:
            causes.append("y_position_out_of_bounds")
    if limits.min_z_m is not None and z_position < limits.min_z_m:
        causes.append("z_position_below_limit")
    if limits.max_z_m is not None and z_position > limits.max_z_m:
        causes.append("z_position_above_limit")
    if limits.max_roll_pitch_rad is not None:
        if abs(roll) > limits.max_roll_pitch_rad:
            causes.append("roll_above_limit")
        if abs(pitch) > limits.max_roll_pitch_rad:
            causes.append("pitch_above_limit")
    if limits.max_speed_mps is not None and _norm3(velocity) > limits.max_speed_mps:
        causes.append("speed_above_limit")
    if limits.max_angular_velocity_radps is not None and _norm3(angular_velocity) > limits.max_angular_velocity_radps:
        causes.append("angular_velocity_above_limit")
    return causes


def _termination_limits_from_mapping(
    value: Mapping[str, Any],
    action_interface: envs.actions.ActionInterface | str,
) -> TerminationLimitConfig:
    """Return termination limits from a raw mapping."""
    mode = _mode(value.get("mode", MODE_DEFAULT), allowed=(MODE_DEFAULT, MODE_RELAXED, MODE_CUSTOM), field_name="termination_limits.mode")
    if mode == MODE_DEFAULT:
        base = default_termination_limits()
    elif mode == MODE_RELAXED:
        base = relaxed_termination_limits(action_interface)
    else:
        base = default_termination_limits()
        base = TerminationLimitConfig(
            mode=MODE_CUSTOM,
            profile=PROFILE_CUSTOM,
            max_abs_xy_m=base.max_abs_xy_m,
            max_roll_pitch_rad=base.max_roll_pitch_rad,
            max_speed_mps=base.max_speed_mps,
            max_angular_velocity_radps=base.max_angular_velocity_radps,
            min_z_m=base.min_z_m,
            max_z_m=base.max_z_m,
            allow_recovery_steps=base.allow_recovery_steps,
            terminate_on_base_truncation=base.terminate_on_base_truncation,
        )
    if not _has_overrides(value):
        return base
    return TerminationLimitConfig(
        mode=mode,
        profile=base.profile if mode != MODE_CUSTOM else PROFILE_CUSTOM,
        max_abs_xy_m=_optional_positive_limit(value, "max_abs_xy_m", base.max_abs_xy_m),
        max_roll_pitch_rad=_optional_positive_limit(value, "max_roll_pitch_rad", base.max_roll_pitch_rad),
        max_speed_mps=_optional_positive_limit(value, "max_speed_mps", base.max_speed_mps),
        max_angular_velocity_radps=_optional_positive_limit(
            value,
            "max_angular_velocity_radps",
            base.max_angular_velocity_radps,
        ),
        min_z_m=_optional_finite_limit(value, "min_z_m", base.min_z_m),
        max_z_m=_optional_positive_limit(value, "max_z_m", base.max_z_m),
        allow_recovery_steps=_nonnegative_int(
            value.get("allow_recovery_steps", base.allow_recovery_steps), "termination_limits.allow_recovery_steps"
        ),
        terminate_on_base_truncation=_bool_value(
            value.get("terminate_on_base_truncation", base.terminate_on_base_truncation),
            "termination_limits.terminate_on_base_truncation",
        ),
    )


def _diagnostic_limits_from_mapping(value: Mapping[str, Any]) -> DiagnosticLimitConfig:
    """Return diagnostic limits from a raw mapping."""
    mode = _mode(value.get("mode", MODE_DEFAULT), allowed=(MODE_DEFAULT, MODE_CUSTOM), field_name="diagnostic_limits.mode")
    base = default_diagnostic_limits()
    if not _has_overrides(value):
        return base
    return DiagnosticLimitConfig(
        mode=mode,
        profile=base.profile if mode == MODE_DEFAULT else PROFILE_CUSTOM,
        max_abs_xy_m=_optional_positive_limit(value, "max_abs_xy_m", base.max_abs_xy_m),
        max_roll_pitch_rad=_optional_positive_limit(value, "max_roll_pitch_rad", base.max_roll_pitch_rad),
        max_speed_mps=_optional_positive_limit(value, "max_speed_mps", base.max_speed_mps),
        max_angular_velocity_radps=_optional_positive_limit(
            value,
            "max_angular_velocity_radps",
            base.max_angular_velocity_radps,
        ),
        min_z_m=_optional_finite_limit(value, "min_z_m", base.min_z_m),
        max_z_m=_optional_positive_limit(value, "max_z_m", base.max_z_m),
    )


def _has_overrides(value: Mapping[str, Any]) -> bool:
    """Return whether a raw mapping sets fields besides mode."""
    return any(key in value for key in (*LIMIT_FIELD_NAMES, "allow_recovery_steps", "terminate_on_base_truncation"))


def _mode(value: Any, allowed: tuple[str, ...], field_name: str) -> str:
    """Return a validated mode string."""
    text = str(value).strip()
    if text not in allowed:
        message = f"{field_name} must be one of: {', '.join(allowed)}"
        raise ValueError(message)
    return text


def _optional_positive_limit(value: Mapping[str, Any], key: str, default: float | None) -> float | None:
    """Return a positive finite optional limit value."""
    if key not in value:
        return default
    raw_value = value[key]
    if raw_value is None:
        return None
    resolved = _finite_float(raw_value, f"termination/diagnostic limit {key}")
    if resolved <= 0.0:
        message = f"termination/diagnostic limit {key} must be positive when provided"
        raise ValueError(message)
    return resolved


def _optional_finite_limit(value: Mapping[str, Any], key: str, default: float | None) -> float | None:
    """Return a finite optional limit value."""
    if key not in value:
        return default
    raw_value = value[key]
    if raw_value is None:
        return None
    return _finite_float(raw_value, f"termination/diagnostic limit {key}")


def _finite_float(value: Any, field_name: str) -> float:
    """Return a finite float value for a numeric config field."""
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        message = f"{field_name} must be finite"
        raise ValueError(message) from exc
    if not isfinite(resolved):
        message = f"{field_name} must be finite"
        raise ValueError(message)
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    """Return a nonnegative integer config value."""
    if isinstance(value, bool):
        message = f"{field_name} must be a nonnegative integer"
        raise TypeError(message)
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        message = f"{field_name} must be a nonnegative integer"
        raise ValueError(message) from exc
    if isinstance(value, float) and not value.is_integer():
        message = f"{field_name} must be a nonnegative integer"
        raise ValueError(message)
    if resolved < 0:
        message = f"{field_name} must be a nonnegative integer"
        raise ValueError(message)
    return resolved


def _bool_value(value: Any, field_name: str) -> bool:
    """Return a strict boolean config value."""
    if not isinstance(value, bool):
        message = f"{field_name} must be a boolean"
        raise TypeError(message)
    return value


def _norm3(values: list[float]) -> float:
    """Return the Euclidean norm of a 3D vector represented by floats."""
    return sum(value * value for value in values[:3]) ** 0.5


__all__ = [
    "MODE_CUSTOM",
    "MODE_DEFAULT",
    "MODE_RELAXED",
    "DiagnosticLimitConfig",
    "TerminationLimitConfig",
    "default_diagnostic_limits",
    "default_termination_limits",
    "parse_diagnostic_limits",
    "parse_termination_limits",
    "relaxed_termination_limits",
    "state_limit_violations",
]
