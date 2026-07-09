"""
===============================================================================
envs_initial_state.py
===============================================================================
Define initial-position configuration for trajectory-tracking environments.

Responsibilities:
  - Parse explicit drone initial-state settings from configs and manifests
  - Resolve fixed and reference-derived spawn positions for single-drone tasks
  - Produce reset diagnostics that compare spawn position with the reference start

Design principles:
  - Preserve upstream near-ground behavior unless a config opts into a mode
  - Keep the initial-state contract independent from action and reward semantics
  - Validate all XYZ vectors before they reach gym-pybullet-drones

Boundaries:
  - Simulator construction and reset ordering belong in envs_tracking_env.py
  - Task generation and validation belong in validation and task_distribution modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

XYZ_DIMENSIONS = 3
MODE_DEFAULT = "default"
MODE_FIXED = "fixed"
MODE_REFERENCE_START = "reference_start"
MODE_REFERENCE_START_WITH_OFFSET = "reference_start_with_offset"
SUPPORTED_INITIAL_STATE_MODES = (
    MODE_DEFAULT,
    MODE_FIXED,
    MODE_REFERENCE_START,
    MODE_REFERENCE_START_WITH_OFFSET,
)
DEFAULT_INITIAL_STATE_MODE = MODE_DEFAULT
DEFAULT_INITIAL_STATE_OFFSET_XYZ = (0.0, 0.0, 0.0)
INITIAL_STATE_MATCH_TOLERANCE_M = 1.0e-6
SOURCE_UPSTREAM_DEFAULT = "upstream_default"
SOURCE_CONFIGURED_XYZ = "configured_xyz"
SOURCE_REFERENCE_START = "reference_start"
SOURCE_REFERENCE_START_WITH_OFFSET = "reference_start_with_offset"


@dataclass(frozen=True)
class InitialStateConfig:
    """
    Validated initial drone position configuration.

    Parameters
    ----------
    mode
        Initial-state mode. Supported values are ``default``, ``fixed``,
        ``reference_start`` and ``reference_start_with_offset``.
    xyz
        Fixed XYZ position used only by ``fixed`` mode.
    offset_xyz
        XYZ offset added to the first reference position by
        ``reference_start_with_offset`` mode. A zero offset is kept in metadata
        for all modes so configs can use one consistent shape.

    """

    mode: str = DEFAULT_INITIAL_STATE_MODE
    xyz: tuple[float, float, float] | None = None
    offset_xyz: tuple[float, float, float] = DEFAULT_INITIAL_STATE_OFFSET_XYZ

    def __post_init__(self) -> None:
        """Normalize and validate immutable initial-state settings."""
        mode = parse_initial_state_mode(self.mode)
        xyz = None if self.xyz is None else coerce_xyz(self.xyz, field_name="initial_state.xyz")
        offset_xyz = coerce_xyz(self.offset_xyz, field_name="initial_state.offset_xyz")
        if mode == MODE_FIXED and xyz is None:
            message = "initial_state.xyz is required when initial_state.mode is fixed"
            raise ValueError(message)
        if mode != MODE_FIXED and xyz is not None:
            message = "initial_state.xyz is only supported when initial_state.mode is fixed"
            raise ValueError(message)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "offset_xyz", offset_xyz)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready config mapping."""
        return {
            "mode": self.mode,
            "xyz": None if self.xyz is None else list(self.xyz),
            "offset_xyz": list(self.offset_xyz),
        }


@dataclass(frozen=True)
class InitialStateResolution:
    """
    Resolved single-drone initial position for one trajectory reference.

    Parameters
    ----------
    config
        Validated initial-state configuration that produced this resolution.
    initial_xyz
        Requested or upstream default XYZ spawn position when known.
    initial_reference_xyz
        First XYZ point of the active reference trajectory.
    initial_xyz_source
        Human-readable source label for ``initial_xyz``.
    override_initial_xyzs
        Optional ``(1, 3)`` payload that should be assigned to
        ``HoverAviary.INIT_XYZS`` before reset.

    """

    config: InitialStateConfig
    initial_xyz: tuple[float, float, float] | None
    initial_reference_xyz: tuple[float, float, float]
    initial_xyz_source: str
    override_initial_xyzs: tuple[tuple[float, float, float], ...] | None

    def initial_xyzs_array(self) -> np.ndarray | None:
        """Return the override payload as a ``(1, 3)`` float array."""
        if self.override_initial_xyzs is None:
            return None
        return np.asarray(self.override_initial_xyzs, dtype=float)

    def diagnostics(self, actual_initial_xyz: Any | None = None) -> dict[str, Any]:
        """
        Return reset diagnostics for the resolved and observed spawn position.

        Parameters
        ----------
        actual_initial_xyz
            Optional current simulator XYZ immediately after reset. When provided,
            this is used for error metrics and the ``initial_xyz`` field.

        Returns
        -------
        dict[str, Any]
            JSON-friendly initial-state metadata and error diagnostics.

        """
        actual_xyz = None if actual_initial_xyz is None else coerce_xyz(actual_initial_xyz, field_name="actual_initial_xyz")
        diagnostic_xyz = actual_xyz if actual_xyz is not None else self.initial_xyz
        reference_xyz = self.initial_reference_xyz
        if diagnostic_xyz is None:
            position_error_m = None
            z_error_m = None
            z_error_signed_m = None
            matches_reference = False
        else:
            delta = np.asarray(diagnostic_xyz, dtype=float) - np.asarray(reference_xyz, dtype=float)
            position_error_m = float(np.linalg.norm(delta))
            z_error_signed_m = float(delta[2])
            z_error_m = float(abs(delta[2]))
            matches_reference = bool(position_error_m <= INITIAL_STATE_MATCH_TOLERANCE_M)
        return {
            "initial_state_mode": self.config.mode,
            "initial_state": self.config.to_dict(),
            "initial_xyz": None if diagnostic_xyz is None else np.asarray(diagnostic_xyz, dtype=float),
            "requested_initial_xyz": None if self.initial_xyz is None else np.asarray(self.initial_xyz, dtype=float),
            "actual_initial_xyz": None if actual_xyz is None else np.asarray(actual_xyz, dtype=float),
            "initial_xyz_source": self.initial_xyz_source,
            "initial_xyz_offset": np.asarray(self.config.offset_xyz, dtype=float),
            "initial_reference_xyz": np.asarray(reference_xyz, dtype=float),
            "initial_xyz_matches_reference_start": matches_reference,
            "initial_position_error_m": position_error_m,
            "initial_z_error_m": z_error_m,
            "initial_z_error_signed_m": z_error_signed_m,
            "spawned_at_reference_start": matches_reference,
        }


def parse_initial_state_mode(value: Any) -> str:
    """
    Parse and validate an initial-state mode string.

    Parameters
    ----------
    value
        Raw mode value from a config or caller.

    Returns
    -------
    str
        Canonical mode string.

    Raises
    ------
    ValueError
        If the mode is unknown or empty.

    """
    text = str(value).strip()
    if text in SUPPORTED_INITIAL_STATE_MODES:
        return text
    allowed = ", ".join(SUPPORTED_INITIAL_STATE_MODES)
    message = f"initial_state.mode must be one of: {allowed}"
    raise ValueError(message)


def parse_initial_state_config(value: Any = None) -> InitialStateConfig:
    """
    Parse an initial-state config mapping, mode string, or existing config.

    Parameters
    ----------
    value
        ``None`` for upstream default behavior, a mode string, a mapping with
        ``mode``, ``xyz`` and ``offset_xyz`` keys, or an ``InitialStateConfig``.

    Returns
    -------
    InitialStateConfig
        Validated immutable initial-state configuration.

    Raises
    ------
    TypeError
        If ``value`` is not one of the supported config shapes.
    ValueError
        If fields are invalid.

    """
    if value is None:
        return InitialStateConfig()
    if isinstance(value, InitialStateConfig):
        return value
    if isinstance(value, str):
        return InitialStateConfig(mode=value)
    if isinstance(value, Mapping):
        return InitialStateConfig(
            mode=value.get("mode", DEFAULT_INITIAL_STATE_MODE),
            xyz=value.get("xyz"),
            offset_xyz=value.get("offset_xyz", DEFAULT_INITIAL_STATE_OFFSET_XYZ),
        )
    message = "initial_state must be a mapping, mode string, InitialStateConfig, or None"
    raise TypeError(message)


def coerce_xyz(value: Any, *, field_name: str) -> tuple[float, float, float]:
    """
    Coerce a finite shape-(3,) XYZ vector into a tuple.

    Parameters
    ----------
    value
        Raw vector-like value.
    field_name
        Field name used in validation errors.

    Returns
    -------
    tuple[float, float, float]
        Finite XYZ tuple.

    Raises
    ------
    ValueError
        If the value is not a finite three-element vector.

    """
    array = np.asarray(value, dtype=float)
    if array.shape != (XYZ_DIMENSIONS,):
        message = f"{field_name} must be a finite shape-(3,) XYZ vector"
        raise ValueError(message)
    values = tuple(float(component) for component in array)
    if not all(isfinite(component) for component in values):
        message = f"{field_name} must contain only finite values"
        raise ValueError(message)
    return values  # type: ignore[return-value]


def resolve_initial_state(
    config: InitialStateConfig,
    reference_xyz: Any,
    default_initial_xyz: Any | None = None,
) -> InitialStateResolution:
    """
    Resolve an initial-state config against one active reference start.

    Parameters
    ----------
    config
        Validated initial-state config.
    reference_xyz
        First reference XYZ point for the active episode.
    default_initial_xyz
        Optional upstream default ``HoverAviary.INIT_XYZS[0]`` used for default
        mode diagnostics. It is never written back to the simulator.

    Returns
    -------
    InitialStateResolution
        Requested spawn metadata and optional ``INIT_XYZS`` override.

    """
    reference = coerce_xyz(reference_xyz, field_name="initial reference XYZ")
    if config.mode == MODE_DEFAULT:
        initial_xyz = None if default_initial_xyz is None else coerce_xyz(default_initial_xyz, field_name="default initial XYZ")
        return InitialStateResolution(
            config=config,
            initial_xyz=initial_xyz,
            initial_reference_xyz=reference,
            initial_xyz_source=SOURCE_UPSTREAM_DEFAULT,
            override_initial_xyzs=None,
        )
    if config.mode == MODE_FIXED:
        if config.xyz is None:
            message = "initial_state.xyz is required when initial_state.mode is fixed"
            raise ValueError(message)
        return _override_resolution(config=config, initial_xyz=config.xyz, reference=reference, source=SOURCE_CONFIGURED_XYZ)
    if config.mode == MODE_REFERENCE_START:
        return _override_resolution(config=config, initial_xyz=reference, reference=reference, source=SOURCE_REFERENCE_START)
    if config.mode == MODE_REFERENCE_START_WITH_OFFSET:
        initial_xyz_array = np.asarray(reference, dtype=float) + np.asarray(config.offset_xyz, dtype=float)
        initial_xyz = coerce_xyz(initial_xyz_array, field_name="reference-start initial XYZ with offset")
        return _override_resolution(config=config, initial_xyz=initial_xyz, reference=reference, source=SOURCE_REFERENCE_START_WITH_OFFSET)
    message = f"unsupported initial_state.mode: {config.mode}"
    raise ValueError(message)


def initial_xyzs_for_hover(config: InitialStateConfig, reference_xyz: Any) -> np.ndarray | None:
    """Return the optional ``(1, 3)`` HoverAviary ``initial_xyzs`` payload."""
    return resolve_initial_state(config, reference_xyz).initial_xyzs_array()


def default_reference_start_config() -> InitialStateConfig:
    """Return the active reference-start initial-state policy config."""
    return InitialStateConfig(mode=MODE_REFERENCE_START)


def initial_state_modes() -> tuple[str, ...]:
    """Return supported initial-state mode strings."""
    return SUPPORTED_INITIAL_STATE_MODES


def _override_resolution(
    *,
    config: InitialStateConfig,
    initial_xyz: tuple[float, float, float],
    reference: tuple[float, float, float],
    source: str,
) -> InitialStateResolution:
    """Build a resolution that overrides ``HoverAviary.INIT_XYZS``."""
    return InitialStateResolution(
        config=config,
        initial_xyz=initial_xyz,
        initial_reference_xyz=reference,
        initial_xyz_source=source,
        override_initial_xyzs=(initial_xyz,),
    )


__all__ = [
    "DEFAULT_INITIAL_STATE_MODE",
    "MODE_DEFAULT",
    "MODE_FIXED",
    "MODE_REFERENCE_START",
    "MODE_REFERENCE_START_WITH_OFFSET",
    "SUPPORTED_INITIAL_STATE_MODES",
    "InitialStateConfig",
    "InitialStateResolution",
    "default_reference_start_config",
    "initial_state_modes",
    "initial_xyzs_for_hover",
    "parse_initial_state_config",
    "parse_initial_state_mode",
    "resolve_initial_state",
]
