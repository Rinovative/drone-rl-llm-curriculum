"""
===============================================================================
envs_actions.py
===============================================================================
Define explicit action-interface contracts for trajectory-tracking environments.

Responsibilities:
  - Provide canonical action-interface names used by configs and metadata
  - Validate direct-RPM scaling and observation-extension parameters before environments are constructed
  - Keep action-interface semantics independent from PPO and curriculum code

Design principles:
  - Preserve the PID target-position default unless a config opts into direct RPM
  - Reject unknown interface names instead of inferring behavior from legacy flags
  - Keep low-level direct control clearly labelled as experimental

Boundaries:
  - Environment stepping and simulator physics belong in envs_tracking_env.py
  - Training, W&B, and manifest serialization belong in experiments modules
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any


class ActionInterface(str, Enum):
    """
    Canonical trajectory-tracking action interfaces.

    Notes
    -----
    ``PID_POSITION`` preserves the existing high-level target-position PID
    behavior. ``DIRECT_RPM`` exposes normalized per-motor commands that are
    mapped to motor RPMs by the tracking environment before PyBullet physics.

    """

    PID_POSITION = "pid_position"
    DIRECT_RPM = "direct_rpm"


DEFAULT_ACTION_INTERFACE = ActionInterface.PID_POSITION
DEFAULT_RPM_DELTA_SCALE = 0.05
MAX_SAFE_RPM_DELTA_SCALE = 0.5
DEFAULT_INCLUDE_DYNAMICS_OBSERVATION = False
DEFAULT_INCLUDE_PREVIOUS_ACTION = False
DIRECT_RPM_LIMITATIONS = (
    "direct_rpm is experimental low-level motor control; stable learning is harder than PID target-position control",
    "direct_rpm policies usually need dynamics observations such as velocity, attitude, and angular velocity",
)


@dataclass(frozen=True)
class ActionInterfaceConfig:
    """
    Validated action-interface settings shared by environment and training code.

    Parameters
    ----------
    action_interface
        Canonical interface name. Supported values are ``pid_position`` and ``direct_rpm``.
    rpm_delta_scale
        Fractional per-motor RPM delta around hover used by ``direct_rpm``.
    include_dynamics_observation
        Whether tracking observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether tracking observations append the previous PPO-facing action.

    """

    action_interface: ActionInterface | str = DEFAULT_ACTION_INTERFACE
    rpm_delta_scale: float = DEFAULT_RPM_DELTA_SCALE
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION

    def __post_init__(self) -> None:
        """Normalize and validate action-interface settings."""
        object.__setattr__(self, "action_interface", parse_action_interface(self.action_interface))
        object.__setattr__(self, "rpm_delta_scale", validate_rpm_delta_scale(self.rpm_delta_scale))
        if not isinstance(self.include_dynamics_observation, bool):
            message = "include_dynamics_observation must be a boolean"
            raise TypeError(message)
        if not isinstance(self.include_previous_action, bool):
            message = "include_previous_action must be a boolean"
            raise TypeError(message)

    @property
    def parsed_action_interface(self) -> ActionInterface:
        """Return the validated action-interface enum value."""
        return parse_action_interface(self.action_interface)

    @property
    def is_direct_rpm(self) -> bool:
        """Return whether these settings select direct motor-RPM control."""
        return self.parsed_action_interface == ActionInterface.DIRECT_RPM

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable action-interface settings."""
        return {
            "action_interface": self.parsed_action_interface.value,
            "rpm_delta_scale": self.rpm_delta_scale,
            "include_dynamics_observation": self.include_dynamics_observation,
            "include_previous_action": self.include_previous_action,
        }


def parse_action_interface(value: Any, field_name: str = "action_interface") -> ActionInterface:
    """
    Parse a canonical action-interface value.

    Parameters
    ----------
    value
        Raw config or CLI value.
    field_name
        Field name used in validation errors.

    Returns
    -------
    ActionInterface
        Parsed action-interface enum value.

    Raises
    ------
    ValueError
        If ``value`` is not one of the supported canonical names.

    """
    if isinstance(value, ActionInterface):
        return value
    text = str(value).strip()
    for candidate in ActionInterface:
        if text == candidate.value:
            return candidate
    allowed = ", ".join(action_interface_values())
    message = f"{field_name} must be one of: {allowed}"
    raise ValueError(message)


def action_interface_values() -> tuple[str, ...]:
    """Return supported action-interface config values."""
    return tuple(candidate.value for candidate in ActionInterface)


def validate_rpm_delta_scale(value: Any) -> float:
    """
    Validate the fractional direct-RPM scale around hover.

    Parameters
    ----------
    value
        Raw scale value. ``0.05`` means normalized action ``1`` requests
        ``hover_rpm * 1.05`` before physical clipping.

    Returns
    -------
    float
        Validated finite scale.

    Raises
    ------
    ValueError
        If the scale is non-finite, non-positive, or outside the conservative range.

    """
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        message = "rpm_delta_scale must be a finite positive number"
        raise ValueError(message) from exc
    if not isfinite(resolved) or resolved <= 0.0:
        message = "rpm_delta_scale must be a finite positive number"
        raise ValueError(message)
    if resolved > MAX_SAFE_RPM_DELTA_SCALE:
        message = f"rpm_delta_scale must be less than or equal to {MAX_SAFE_RPM_DELTA_SCALE:g}"
        raise ValueError(message)
    return resolved


def direct_control_limitations(action_interface: ActionInterface | str) -> list[str]:
    """Return human-readable limitations for experimental direct-control interfaces."""
    if parse_action_interface(action_interface) == ActionInterface.DIRECT_RPM:
        return list(DIRECT_RPM_LIMITATIONS)
    return []


__all__ = [
    "DEFAULT_ACTION_INTERFACE",
    "DEFAULT_INCLUDE_DYNAMICS_OBSERVATION",
    "DEFAULT_INCLUDE_PREVIOUS_ACTION",
    "DEFAULT_RPM_DELTA_SCALE",
    "DIRECT_RPM_LIMITATIONS",
    "ActionInterface",
    "ActionInterfaceConfig",
    "action_interface_values",
    "direct_control_limitations",
    "parse_action_interface",
    "validate_rpm_delta_scale",
]
