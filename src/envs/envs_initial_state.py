"""
===============================================================================
envs_initial_state.py
===============================================================================
Define initial-position and randomized spawn configuration for tracking environments.

Responsibilities:
  - Parse explicit drone initial-state settings from configs and manifests
  - Resolve fixed, reference-derived, and randomized spawn positions for single-drone tasks
  - Produce reset diagnostics that compare spawn position with the reference start

Design principles:
  - Preserve upstream near-ground behavior unless a config opts into a mode
  - Keep randomized offsets bounded, deterministic, and independent from action semantics
  - Validate all XYZ vectors before they reach gym-pybullet-drones

Boundaries:
  - Simulator construction and reset ordering belong in envs_tracking_env.py
  - Task generation and validation belong in validation and task_distribution modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt
from typing import Any

import numpy as np

XYZ_DIMENSIONS = 3
RANGE_DIMENSIONS = 2
MODE_DEFAULT = "default"
MODE_FIXED = "fixed"
MODE_REFERENCE_START = "reference_start"
MODE_REFERENCE_START_WITH_OFFSET = "reference_start_with_offset"
MODE_REFERENCE_START_RANDOM_OFFSET = "reference_start_random_offset"
MODE_REFERENCE_START_WITH_RANDOM_OFFSET = "reference_start_with_random_offset"
SUPPORTED_INITIAL_STATE_MODES = (
    MODE_DEFAULT,
    MODE_FIXED,
    MODE_REFERENCE_START,
    MODE_REFERENCE_START_WITH_OFFSET,
    MODE_REFERENCE_START_RANDOM_OFFSET,
    MODE_REFERENCE_START_WITH_RANDOM_OFFSET,
)
INITIAL_STATE_MODE_ALIASES = {
    MODE_REFERENCE_START_WITH_RANDOM_OFFSET: MODE_REFERENCE_START_RANDOM_OFFSET,
}
REFERENCE_DERIVED_INITIAL_STATE_MODES = (
    MODE_REFERENCE_START,
    MODE_REFERENCE_START_WITH_OFFSET,
    MODE_REFERENCE_START_RANDOM_OFFSET,
)
DEFAULT_INITIAL_STATE_MODE = MODE_DEFAULT
DEFAULT_INITIAL_STATE_OFFSET_XYZ = (0.0, 0.0, 0.0)
DEFAULT_INITIAL_STATE_XY_OFFSET_RANGE_M = (0.10, 0.30)
DEFAULT_INITIAL_STATE_Z_OFFSET_RANGE_M = (-0.18, 0.08)
DEFAULT_INITIAL_STATE_Z_OFFSET_BIAS = "below"
DEFAULT_INITIAL_STATE_BELOW_PROBABILITY = 0.70
DEFAULT_INITIAL_STATE_Z_BIAS_SPLIT_M = -0.03
SUPPORTED_Z_OFFSET_BIASES = ("none", "below")
DEFAULT_INITIAL_STATE_MIN_Z_M = 0.2
DEFAULT_INITIAL_STATE_MAX_Z_M = 2.0
DEFAULT_INITIAL_STATE_MAX_ABS_XY_M = 1.5
INITIAL_STATE_MATCH_TOLERANCE_M = 1.0e-6
SOURCE_UPSTREAM_DEFAULT = "upstream_default"
SOURCE_CONFIGURED_XYZ = "configured_xyz"
SOURCE_REFERENCE_START = "reference_start"
SOURCE_REFERENCE_START_WITH_OFFSET = "reference_start_with_offset"
SOURCE_REFERENCE_START_RANDOM_OFFSET = "reference_start_random_offset"


@dataclass(frozen=True)
class InitialStateConfig:
    """
    Validated initial drone position configuration.

    Parameters
    ----------
    mode
        Initial-state mode. Supported values are ``default``, ``fixed``,
        ``reference_start``, ``reference_start_with_offset`` and
        ``reference_start_random_offset``.
    xyz
        Fixed XYZ position used only by ``fixed`` mode.
    offset_xyz
        XYZ offset added to the first reference position by
        ``reference_start_with_offset`` mode. A zero offset is kept in metadata
        for all modes so configs can use one consistent shape.
    xy_offset_range_m
        Inclusive XY offset radius range sampled by ``reference_start_random_offset``.
    z_offset_range_m
        Inclusive signed z-offset range sampled by ``reference_start_random_offset``.
    z_offset_bias
        Optional bias mode for signed z offsets. ``below`` samples from the lower
        subrange more often while still allowing occasional above-reference starts.
    below_probability
        Probability of sampling from the below-reference subrange when ``z_offset_bias`` is ``below``.
    min_z_m
        Lower altitude clamp for randomized reference-start offsets.
    max_z_m
        Upper altitude clamp for randomized reference-start offsets.
    max_abs_xy_m
        Symmetric XY clamp for randomized reference-start offsets.

    """

    mode: str = DEFAULT_INITIAL_STATE_MODE
    xyz: tuple[float, float, float] | None = None
    offset_xyz: tuple[float, float, float] = DEFAULT_INITIAL_STATE_OFFSET_XYZ
    xy_offset_range_m: tuple[float, float] = DEFAULT_INITIAL_STATE_XY_OFFSET_RANGE_M
    z_offset_range_m: tuple[float, float] = DEFAULT_INITIAL_STATE_Z_OFFSET_RANGE_M
    z_offset_bias: str = DEFAULT_INITIAL_STATE_Z_OFFSET_BIAS
    below_probability: float = DEFAULT_INITIAL_STATE_BELOW_PROBABILITY
    min_z_m: float = DEFAULT_INITIAL_STATE_MIN_Z_M
    max_z_m: float = DEFAULT_INITIAL_STATE_MAX_Z_M
    max_abs_xy_m: float = DEFAULT_INITIAL_STATE_MAX_ABS_XY_M

    def __post_init__(self) -> None:
        """Normalize and validate immutable initial-state settings."""
        mode = parse_initial_state_mode(self.mode)
        xyz = None if self.xyz is None else coerce_xyz(self.xyz, field_name="initial_state.xyz")
        offset_xyz = coerce_xyz(self.offset_xyz, field_name="initial_state.offset_xyz")
        xy_offset_range_m = coerce_range(
            self.xy_offset_range_m,
            field_name="initial_state.xy_offset_range_m",
            nonnegative=True,
        )
        z_offset_range_m = coerce_range(self.z_offset_range_m, field_name="initial_state.z_offset_range_m")
        z_offset_bias = coerce_z_offset_bias(self.z_offset_bias)
        below_probability = coerce_probability(self.below_probability, field_name="initial_state.below_probability")
        min_z_m = coerce_finite_float(self.min_z_m, field_name="initial_state.min_z_m")
        max_z_m = coerce_finite_float(self.max_z_m, field_name="initial_state.max_z_m")
        max_abs_xy_m = coerce_positive_float(self.max_abs_xy_m, field_name="initial_state.max_abs_xy_m")
        if max_z_m <= min_z_m:
            message = "initial_state.max_z_m must be greater than initial_state.min_z_m"
            raise ValueError(message)
        if mode == MODE_FIXED and xyz is None:
            message = "initial_state.xyz is required when initial_state.mode is fixed"
            raise ValueError(message)
        if mode != MODE_FIXED and xyz is not None:
            message = "initial_state.xyz is only supported when initial_state.mode is fixed"
            raise ValueError(message)
        if mode == MODE_REFERENCE_START_RANDOM_OFFSET and not _offset_range_can_move(xy_offset_range_m, z_offset_range_m):
            message = "reference_start_random_offset requires a nonzero xy or z offset range"
            raise ValueError(message)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "offset_xyz", offset_xyz)
        object.__setattr__(self, "xy_offset_range_m", xy_offset_range_m)
        object.__setattr__(self, "z_offset_range_m", z_offset_range_m)
        object.__setattr__(self, "z_offset_bias", z_offset_bias)
        object.__setattr__(self, "below_probability", below_probability)
        object.__setattr__(self, "min_z_m", min_z_m)
        object.__setattr__(self, "max_z_m", max_z_m)
        object.__setattr__(self, "max_abs_xy_m", max_abs_xy_m)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready config mapping."""
        return {
            "mode": self.mode,
            "xyz": None if self.xyz is None else list(self.xyz),
            "offset_xyz": list(self.offset_xyz),
            "xy_offset_range_m": list(self.xy_offset_range_m),
            "z_offset_range_m": list(self.z_offset_range_m),
            "z_offset_bias": self.z_offset_bias,
            "below_probability": self.below_probability,
            "min_z_m": self.min_z_m,
            "max_z_m": self.max_z_m,
            "max_abs_xy_m": self.max_abs_xy_m,
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
    initial_xyz_offset
        Offset from the first reference position to the requested initial position.
    initial_offset_policy
        JSON-ready description of how the offset was selected and clamped.
    initial_offset_seed
        Seed that initialized the environment RNG sequence when known.
    initial_offset_sample_index
        Reset-local sample index for deterministic sequence diagnostics.
    override_initial_xyzs
        Optional ``(1, 3)`` payload that should be assigned to
        ``HoverAviary.INIT_XYZS`` before reset.

    """

    config: InitialStateConfig
    initial_xyz: tuple[float, float, float] | None
    initial_reference_xyz: tuple[float, float, float]
    initial_xyz_source: str
    override_initial_xyzs: tuple[tuple[float, float, float], ...] | None
    initial_xyz_offset: tuple[float, float, float] = DEFAULT_INITIAL_STATE_OFFSET_XYZ
    initial_offset_policy: dict[str, Any] | None = None
    initial_offset_seed: int | None = None
    initial_offset_sample_index: int | None = None

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
            offset_xyz = self.initial_xyz_offset
            position_error_m = None
            z_error_m = None
            z_error_signed_m = None
            xy_offset_m = None
            z_offset_m = None
            offset_distance_m = None
            matches_reference = False
            near_reference = False
        else:
            delta = np.asarray(diagnostic_xyz, dtype=float) - np.asarray(reference_xyz, dtype=float)
            offset_xyz = coerce_xyz(delta, field_name="initial XYZ offset")
            position_error_m = float(np.linalg.norm(delta))
            z_error_signed_m = float(delta[2])
            z_error_m = float(abs(delta[2]))
            xy_offset_m = float(np.linalg.norm(delta[:2]))
            z_offset_m = z_error_signed_m
            offset_distance_m = position_error_m
            matches_reference = bool(position_error_m <= INITIAL_STATE_MATCH_TOLERANCE_M)
            near_reference = bool(
                self.config.mode in REFERENCE_DERIVED_INITIAL_STATE_MODES and position_error_m <= _near_reference_threshold_m(self.config, offset_xyz)
            )
        return {
            "initial_state_mode": self.config.mode,
            "initial_state": self.config.to_dict(),
            "initial_xyz": None if diagnostic_xyz is None else list(diagnostic_xyz),
            "requested_initial_xyz": None if self.initial_xyz is None else list(self.initial_xyz),
            "actual_initial_xyz": None if actual_xyz is None else list(actual_xyz),
            "initial_xyz_source": self.initial_xyz_source,
            "initial_xyz_offset": list(offset_xyz),
            "initial_xy_offset_m": xy_offset_m,
            "initial_z_offset_m": z_offset_m,
            "initial_offset_distance_m": offset_distance_m,
            "initial_offset_policy": _initial_offset_policy(self.config, self.initial_offset_policy),
            "initial_offset_seed": self.initial_offset_seed,
            "initial_offset_sample_index": self.initial_offset_sample_index,
            "initial_reference_xyz": list(reference_xyz),
            "initial_xyz_matches_reference_start": matches_reference,
            "initial_position_error_m": position_error_m,
            "initial_z_error_m": z_error_m,
            "initial_z_error_signed_m": z_error_signed_m,
            "spawned_near_reference_start": near_reference,
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
    text = INITIAL_STATE_MODE_ALIASES.get(text, text)
    if text in SUPPORTED_INITIAL_STATE_MODES and text not in INITIAL_STATE_MODE_ALIASES:
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
            xy_offset_range_m=value.get("xy_offset_range_m", DEFAULT_INITIAL_STATE_XY_OFFSET_RANGE_M),
            z_offset_range_m=value.get("z_offset_range_m", DEFAULT_INITIAL_STATE_Z_OFFSET_RANGE_M),
            z_offset_bias=value.get("z_offset_bias", DEFAULT_INITIAL_STATE_Z_OFFSET_BIAS),
            below_probability=value.get("below_probability", DEFAULT_INITIAL_STATE_BELOW_PROBABILITY),
            min_z_m=value.get("min_z_m", DEFAULT_INITIAL_STATE_MIN_Z_M),
            max_z_m=value.get("max_z_m", DEFAULT_INITIAL_STATE_MAX_Z_M),
            max_abs_xy_m=value.get("max_abs_xy_m", DEFAULT_INITIAL_STATE_MAX_ABS_XY_M),
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


def coerce_range(value: Any, *, field_name: str, nonnegative: bool = False) -> tuple[float, float]:
    """
    Coerce a finite inclusive two-value range.

    Parameters
    ----------
    value
        Raw range-like value.
    field_name
        Field name used in validation errors.
    nonnegative
        Whether both range bounds must be greater than or equal to zero.

    Returns
    -------
    tuple[float, float]
        Normalized ``(low, high)`` range.

    Raises
    ------
    ValueError
        If the range is malformed, non-finite, inverted, or negative when disallowed.

    """
    array = np.asarray(value, dtype=float)
    if array.shape != (RANGE_DIMENSIONS,):
        message = f"{field_name} must be a finite two-item range"
        raise ValueError(message)
    low = float(array[0])
    high = float(array[1])
    if not isfinite(low) or not isfinite(high):
        message = f"{field_name} must contain only finite values"
        raise ValueError(message)
    if high < low:
        message = f"{field_name} upper bound must be greater than or equal to lower bound"
        raise ValueError(message)
    if nonnegative and (low < 0.0 or high < 0.0):
        message = f"{field_name} bounds must be nonnegative"
        raise ValueError(message)
    return low, high


def coerce_finite_float(value: Any, *, field_name: str) -> float:
    """Coerce a finite scalar float value."""
    result = float(value)
    if not isfinite(result):
        message = f"{field_name} must be finite"
        raise ValueError(message)
    return result


def coerce_probability(value: Any, *, field_name: str) -> float:
    """Coerce a finite probability value in ``[0, 1]``."""
    result = coerce_finite_float(value, field_name=field_name)
    if result < 0.0 or result > 1.0:
        message = f"{field_name} must be between 0.0 and 1.0"
        raise ValueError(message)
    return result


def coerce_z_offset_bias(value: Any) -> str:
    """Coerce a z-offset bias mode string."""
    bias = str(value).strip().lower()
    if bias not in SUPPORTED_Z_OFFSET_BIASES:
        allowed = ", ".join(SUPPORTED_Z_OFFSET_BIASES)
        message = f"initial_state.z_offset_bias must be one of: {allowed}"
        raise ValueError(message)
    return bias


def coerce_positive_float(value: Any, *, field_name: str) -> float:
    """Coerce a finite positive scalar float value."""
    result = coerce_finite_float(value, field_name=field_name)
    if result <= 0.0:
        message = f"{field_name} must be positive"
        raise ValueError(message)
    return result


def resolve_initial_state(
    config: InitialStateConfig,
    reference_xyz: Any,
    default_initial_xyz: Any | None = None,
    rng: np.random.Generator | None = None,
    offset_seed: int | None = None,
    offset_sample_index: int | None = None,
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
    rng
        Optional environment RNG used by randomized reference-start offsets.
    offset_seed
        Seed that initialized the environment RNG sequence when known.
    offset_sample_index
        Reset-local sample index for deterministic diagnostics.

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
            initial_xyz_offset=_offset_between(initial_xyz, reference),
            initial_offset_policy=_initial_offset_policy(config),
            initial_offset_seed=offset_seed,
            initial_offset_sample_index=offset_sample_index,
        )
    if config.mode == MODE_FIXED:
        if config.xyz is None:
            message = "initial_state.xyz is required when initial_state.mode is fixed"
            raise ValueError(message)
        return _override_resolution(
            config=config,
            initial_xyz=config.xyz,
            reference=reference,
            source=SOURCE_CONFIGURED_XYZ,
            offset_seed=offset_seed,
            offset_sample_index=offset_sample_index,
        )
    if config.mode == MODE_REFERENCE_START:
        return _override_resolution(
            config=config,
            initial_xyz=reference,
            reference=reference,
            source=SOURCE_REFERENCE_START,
            offset_seed=offset_seed,
            offset_sample_index=offset_sample_index,
        )
    if config.mode == MODE_REFERENCE_START_WITH_OFFSET:
        initial_xyz_array = np.asarray(reference, dtype=float) + np.asarray(config.offset_xyz, dtype=float)
        initial_xyz = coerce_xyz(initial_xyz_array, field_name="reference-start initial XYZ with offset")
        return _override_resolution(
            config=config,
            initial_xyz=initial_xyz,
            reference=reference,
            source=SOURCE_REFERENCE_START_WITH_OFFSET,
            offset_seed=offset_seed,
            offset_sample_index=offset_sample_index,
        )
    if config.mode == MODE_REFERENCE_START_RANDOM_OFFSET:
        return _random_offset_resolution(
            config=config,
            reference=reference,
            rng=np.random.default_rng(offset_seed) if rng is None else rng,
            offset_seed=offset_seed,
            offset_sample_index=offset_sample_index,
        )
    message = f"unsupported initial_state.mode: {config.mode}"
    raise ValueError(message)


def initial_xyzs_for_hover(config: InitialStateConfig, reference_xyz: Any) -> np.ndarray | None:
    """Return the optional ``(1, 3)`` HoverAviary ``initial_xyzs`` payload."""
    return resolve_initial_state(config, reference_xyz).initial_xyzs_array()


def default_reference_start_config() -> InitialStateConfig:
    """Return the active randomized reference-start initial-state policy config."""
    return InitialStateConfig(mode=MODE_REFERENCE_START_RANDOM_OFFSET)


def initial_state_modes() -> tuple[str, ...]:
    """Return supported initial-state mode strings."""
    return SUPPORTED_INITIAL_STATE_MODES


def _override_resolution(
    *,
    config: InitialStateConfig,
    initial_xyz: tuple[float, float, float],
    reference: tuple[float, float, float],
    source: str,
    offset_seed: int | None = None,
    offset_sample_index: int | None = None,
) -> InitialStateResolution:
    """Build a resolution that overrides ``HoverAviary.INIT_XYZS``."""
    initial_offset = _offset_between(initial_xyz, reference)
    return InitialStateResolution(
        config=config,
        initial_xyz=initial_xyz,
        initial_reference_xyz=reference,
        initial_xyz_source=source,
        override_initial_xyzs=(initial_xyz,),
        initial_xyz_offset=initial_offset,
        initial_offset_policy=_initial_offset_policy(config),
        initial_offset_seed=offset_seed,
        initial_offset_sample_index=offset_sample_index,
    )


def _random_offset_resolution(
    *,
    config: InitialStateConfig,
    reference: tuple[float, float, float],
    rng: np.random.Generator,
    offset_seed: int | None,
    offset_sample_index: int | None,
) -> InitialStateResolution:
    """Build a reference-start resolution with a sampled bounded offset."""
    sampled_offset = _sample_random_offset(config, rng)
    unclamped_xyz = np.asarray(reference, dtype=float) + np.asarray(sampled_offset, dtype=float)
    initial_xyz = coerce_xyz(_clamp_initial_xyz(unclamped_xyz, config), field_name="randomized reference-start initial XYZ")
    initial_offset = _offset_between(initial_xyz, reference)
    initial_offset_policy = _initial_offset_policy(
        config,
        {
            "xy_sampling": "uniform_radius_range_uniform_direction",
            "z_sampling": _z_sampling_label(config),
            "clamp": "componentwise_xy_and_z_bounds",
            "sampled_offset_xyz": list(sampled_offset),
            "unclamped_initial_xyz": _jsonable_xyz(unclamped_xyz),
            "offset_clamped": bool(
                not np.allclose(unclamped_xyz, np.asarray(initial_xyz, dtype=float), atol=INITIAL_STATE_MATCH_TOLERANCE_M, rtol=0.0)
            ),
        },
    )
    return InitialStateResolution(
        config=config,
        initial_xyz=initial_xyz,
        initial_reference_xyz=reference,
        initial_xyz_source=SOURCE_REFERENCE_START_RANDOM_OFFSET,
        override_initial_xyzs=(initial_xyz,),
        initial_xyz_offset=initial_offset,
        initial_offset_policy=initial_offset_policy,
        initial_offset_seed=offset_seed,
        initial_offset_sample_index=offset_sample_index,
    )


def _sample_random_offset(config: InitialStateConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample a non-axis-aligned-capable bounded random offset."""
    radius = float(rng.uniform(config.xy_offset_range_m[0], config.xy_offset_range_m[1]))
    angle = float(rng.uniform(0.0, 2.0 * pi))
    z_offset = _sample_z_offset(config, rng)
    return radius * cos(angle), radius * sin(angle), z_offset


def _sample_z_offset(config: InitialStateConfig, rng: np.random.Generator) -> float:
    """Sample a signed z offset with the configured bias policy."""
    low, high = config.z_offset_range_m
    if config.z_offset_bias != "below":
        return float(rng.uniform(low, high))
    split = _z_bias_split_m(config)
    if rng.uniform(0.0, 1.0) < config.below_probability:
        return float(rng.uniform(low, split))
    return float(rng.uniform(split, high))


def _z_bias_split_m(config: InitialStateConfig) -> float:
    """Return the z offset split between below-biased and occasional-above subranges."""
    low, high = config.z_offset_range_m
    if high <= 0.0:
        return high
    if low >= 0.0:
        return low
    return float(np.clip(DEFAULT_INITIAL_STATE_Z_BIAS_SPLIT_M, low, high))


def _z_sampling_label(config: InitialStateConfig) -> str:
    """Return a stable metadata label for z-offset sampling."""
    if config.z_offset_bias == "below":
        return "below_biased_uniform_subranges"
    return "uniform_range"


def _clamp_initial_xyz(initial_xyz: np.ndarray, config: InitialStateConfig) -> np.ndarray:
    """Clamp a randomized initial position to conservative simulator-safe bounds."""
    return np.array(
        [
            np.clip(float(initial_xyz[0]), -config.max_abs_xy_m, config.max_abs_xy_m),
            np.clip(float(initial_xyz[1]), -config.max_abs_xy_m, config.max_abs_xy_m),
            np.clip(float(initial_xyz[2]), config.min_z_m, config.max_z_m),
        ],
        dtype=float,
    )


def _offset_between(initial_xyz: tuple[float, float, float] | None, reference: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the finite offset from reference to initial XYZ when known."""
    if initial_xyz is None:
        return DEFAULT_INITIAL_STATE_OFFSET_XYZ
    return coerce_xyz(np.asarray(initial_xyz, dtype=float) - np.asarray(reference, dtype=float), field_name="initial XYZ offset")


def _initial_offset_policy(config: InitialStateConfig, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-ready metadata describing the initial-offset policy."""
    policy: dict[str, Any] = {
        "mode": config.mode,
        "offset_xyz": list(config.offset_xyz),
        "xy_offset_range_m": list(config.xy_offset_range_m),
        "z_offset_range_m": list(config.z_offset_range_m),
        "z_offset_bias": config.z_offset_bias,
        "below_probability": config.below_probability,
        "min_z_m": config.min_z_m,
        "max_z_m": config.max_z_m,
        "max_abs_xy_m": config.max_abs_xy_m,
    }
    if config.mode == MODE_REFERENCE_START_RANDOM_OFFSET:
        policy.setdefault("xy_sampling", "uniform_radius_range_uniform_direction")
        policy.setdefault("z_sampling", _z_sampling_label(config))
        policy.setdefault("clamp", "componentwise_xy_and_z_bounds")
    if extra is not None:
        policy.update(dict(extra))
    return policy


def _near_reference_threshold_m(config: InitialStateConfig, offset_xyz: tuple[float, float, float]) -> float:
    """Return a conservative distance threshold for near-reference spawn diagnostics."""
    if config.mode == MODE_REFERENCE_START:
        return INITIAL_STATE_MATCH_TOLERANCE_M
    if config.mode == MODE_REFERENCE_START_WITH_OFFSET:
        return float(np.linalg.norm(np.asarray(config.offset_xyz, dtype=float))) + INITIAL_STATE_MATCH_TOLERANCE_M
    if config.mode == MODE_REFERENCE_START_RANDOM_OFFSET:
        max_z_offset = max(abs(config.z_offset_range_m[0]), abs(config.z_offset_range_m[1]))
        configured_threshold = sqrt(config.xy_offset_range_m[1] ** 2 + max_z_offset**2)
        observed_threshold = float(np.linalg.norm(np.asarray(offset_xyz, dtype=float)))
        return max(configured_threshold, observed_threshold) + INITIAL_STATE_MATCH_TOLERANCE_M
    return INITIAL_STATE_MATCH_TOLERANCE_M


def _offset_range_can_move(xy_range: tuple[float, float], z_range: tuple[float, float]) -> bool:
    """Return whether a randomized offset range can produce a nonzero displacement."""
    return bool(xy_range[1] > INITIAL_STATE_MATCH_TOLERANCE_M or max(abs(z_range[0]), abs(z_range[1])) > INITIAL_STATE_MATCH_TOLERANCE_M)


def _jsonable_xyz(value: Any) -> list[float]:
    """Return a finite XYZ value as a JSON-ready list."""
    return list(coerce_xyz(value, field_name="initial XYZ metadata"))


__all__ = [
    "DEFAULT_INITIAL_STATE_BELOW_PROBABILITY",
    "DEFAULT_INITIAL_STATE_MAX_ABS_XY_M",
    "DEFAULT_INITIAL_STATE_MAX_Z_M",
    "DEFAULT_INITIAL_STATE_MIN_Z_M",
    "DEFAULT_INITIAL_STATE_MODE",
    "DEFAULT_INITIAL_STATE_XY_OFFSET_RANGE_M",
    "DEFAULT_INITIAL_STATE_Z_OFFSET_BIAS",
    "DEFAULT_INITIAL_STATE_Z_OFFSET_RANGE_M",
    "MODE_DEFAULT",
    "MODE_FIXED",
    "MODE_REFERENCE_START",
    "MODE_REFERENCE_START_RANDOM_OFFSET",
    "MODE_REFERENCE_START_WITH_OFFSET",
    "MODE_REFERENCE_START_WITH_RANDOM_OFFSET",
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
