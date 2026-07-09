"""
===============================================================================
envs_task_distribution.py
===============================================================================
Sample validated training tasks from fixed or randomized task distributions.

Responsibilities:
  - Normalize legacy fixed tasks into degenerate task-distribution settings
  - Validate task-distribution configs and weighted trajectory-family choices
  - Sample conservative validated task variants with deterministic seeded RNG
  - Expose compact metadata for training metrics, manifests, and diagnostics

Design principles:
  - Use existing validation task dictionaries as the only generated task format
  - Keep randomization bounded, deterministic, and independent of simulator state
  - Treat fixed tasks as strength-zero task distributions at integration boundaries

Boundaries:
  - Simulator reset and observation construction belong in envs_tracking_env.py
  - PPO orchestration and artifact writing belong in experiments modules
  - LLM proposal policy belongs in llm modules
===============================================================================

"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src import validation

DISTRIBUTION_CONFIG_KEY = "task_distribution"
MODE_FIXED = "fixed"
MODE_RANDOMIZED = "randomized"
SUPPORTED_MODES = (MODE_FIXED, MODE_RANDOMIZED)
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_DURATION_SEC = 3.0
DEFAULT_Z_M = 1.0
DEFAULT_START_HOLD_SEC = 1.0
DEFAULT_FINAL_HOLD_SEC = 1.0
STANDARD_REFERENCE_HEIGHT_POLICY = "standard_reference_1p0m"
STANDARD_REFERENCE_HEIGHT_RANGE_M = (0.9, 1.1)
STANDARD_START_HOLD_POLICY = "uniform_reference_start_hold_1p0s"
STANDARD_START_HOLD_REWARD_POLICY = "full_tracking_reward_active_during_uniform_reference_start_hold"
MAX_SAMPLE_ATTEMPTS = 64
RANK_SEED_STRIDE = 1
COIN_FLIP_PROBABILITY = 0.5
RANGE_PAIR_LENGTH = 2
MIN_VERTICAL_HEIGHT_DELTA_M = 0.08

FAMILY_HOVER = "hover_stabilization"
FAMILY_TAKEOFF = "takeoff_stabilization"
FAMILY_VERTICAL_UP_DOWN = "vertical_up_down"
FAMILY_ANGLED_VERTICAL = "angled_vertical"
FAMILY_LINE = "line"
FAMILY_START_HOLD_LINE = "start_hold_then_line"
FAMILY_POLYLINE = "polyline"
FAMILY_L_SHAPE = "l_shape"
FAMILY_ZIGZAG = "zigzag"
FAMILY_TRIANGLE = "triangle"
FAMILY_MULTI_HEIGHT_POLYLINE = "multi_height_polyline"
FAMILY_DELAYED_ALTITUDE_POLYLINE = "delayed_altitude_polyline"
FAMILY_RECTANGLE = "rectangle"
FAMILY_SQUARE = "square"
FAMILY_CIRCLE = "circle"
FAMILY_ELLIPSE = "ellipse"
FAMILY_FIGURE_EIGHT = "figure_eight"
FAMILY_BASIC_TRAINING_SHOW = "basic_training_show"

_FAMILY_TO_TASK_SHAPE: dict[str, str] = {
    FAMILY_HOVER: validation.contracts.SHAPE_HOVER_STABILIZATION,
    FAMILY_TAKEOFF: validation.contracts.SHAPE_VERTICAL,
    FAMILY_VERTICAL_UP_DOWN: validation.contracts.SHAPE_VERTICAL,
    FAMILY_ANGLED_VERTICAL: validation.contracts.SHAPE_LINE,
    FAMILY_LINE: validation.contracts.SHAPE_LINE,
    FAMILY_START_HOLD_LINE: validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
    FAMILY_POLYLINE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_L_SHAPE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_ZIGZAG: validation.contracts.SHAPE_POLYLINE,
    FAMILY_TRIANGLE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_MULTI_HEIGHT_POLYLINE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_DELAYED_ALTITUDE_POLYLINE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_RECTANGLE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_SQUARE: validation.contracts.SHAPE_POLYLINE,
    FAMILY_CIRCLE: validation.contracts.SHAPE_CIRCLE,
    FAMILY_ELLIPSE: validation.contracts.SHAPE_ELLIPSE,
    FAMILY_FIGURE_EIGHT: validation.contracts.SHAPE_FIGURE_EIGHT,
    FAMILY_BASIC_TRAINING_SHOW: validation.contracts.SHAPE_BASIC_TRAINING_SHOW,
}

_UNSUPPORTED_REQUESTED_FAMILIES: tuple[str, ...] = ()
_SUPPORTED_FAMILIES = tuple(_FAMILY_TO_TASK_SHAPE)


@dataclass(frozen=True)
class TaskDistributionSettings:
    """
    Validated settings for fixed or randomized task sampling.

    Parameters
    ----------
    name
        Optional distribution identifier used in metrics and manifests.
    enabled
        Whether task-distribution handling is active. Disabled settings behave
        as fixed base-task settings.
    mode
        Sampling mode, either ``fixed`` or ``randomized``.
    seed
        Base RNG seed. Vectorized env ranks add their rank to this value.
    strength
        Randomization strength in ``[0, 1]``. ``0`` returns the base task.
    sample_on_reset
        Whether randomized settings resample at every environment reset.
    base_task
        Valid deterministic task used for fixed behavior and randomization anchors.
    family_weights
        Nonnegative unnormalized sampling weights keyed by supported family id.
    variations
        Optional per-family bounded variation settings.
    validation_limits
        Optional deterministic validation limits for generated tasks.
    config_path
        Optional source config path for metadata only.

    """

    name: str | None
    enabled: bool
    mode: str
    seed: int
    strength: float
    sample_on_reset: bool
    base_task: dict[str, Any]
    family_weights: dict[str, float] = field(default_factory=dict)
    variations: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_limits: validation.tasks.ValidationLimits | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate and normalize immutable task-distribution settings."""
        if self.mode not in SUPPORTED_MODES:
            message = f"task_distribution.mode must be one of: {', '.join(SUPPORTED_MODES)}"
            raise ValueError(message)
        if not 0.0 <= float(self.strength) <= 1.0:
            message = "task_distribution.strength must be in [0.0, 1.0]"
            raise ValueError(message)
        if not isinstance(self.base_task, dict):
            message = "task_distribution.base_task must be a mapping"
            raise TypeError(message)
        validation_result = validation.tasks.validate_task(self.base_task, limits=self.validation_limits)
        if not validation_result.is_valid:
            details = "; ".join(validation_result.messages)
            message = f"task_distribution.base_task is invalid: {details}"
            raise ValueError(message)
        normalized_weights = _normalize_family_weights(self.family_weights, self.base_task)
        normalized_variations = _normalize_variations(self.variations)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "base_task", copy.deepcopy(self.base_task))
        object.__setattr__(self, "family_weights", normalized_weights)
        object.__setattr__(self, "variations", normalized_variations)
        if self.mode == MODE_FIXED and self.sample_on_reset:
            message = "task_distribution.sample_on_reset must be false for fixed mode"
            raise ValueError(message)

    @property
    def normalized_family_weights(self) -> dict[str, float]:
        """Return normalized family weights in deterministic key order."""
        return dict(self.family_weights)

    @property
    def base_task_shape(self) -> str:
        """Return the validated shape of the distribution base task."""
        return str(self.base_task.get(validation.contracts.FIELD_SHAPE, "unknown"))

    def to_metadata(self) -> dict[str, Any]:
        """Return compact JSON-ready task-distribution metadata."""
        return {
            "task_distribution_enabled": bool(self.enabled),
            "task_distribution_mode": self.mode,
            "task_distribution_strength": float(self.strength),
            "task_distribution_sample_on_reset": bool(self.sample_on_reset),
            "task_distribution_seed": int(self.seed),
            "task_distribution_config_path": None if self.config_path is None else str(self.config_path),
            "task_distribution_supported_families": list(supported_task_families()),
            "task_distribution_family_weights": self.normalized_family_weights,
            "task_distribution_name": self.name,
            "task_distribution_base_task_shape": self.base_task_shape,
        }


@dataclass(frozen=True)
class SampledTask:
    """
    One sampled task and compact provenance metadata.

    Parameters
    ----------
    task
        Valid task mapping sampled from the distribution.
    metadata
        JSON-ready metadata describing mode, family, seed, rank, and sample index.

    """

    task: dict[str, Any]
    metadata: dict[str, Any]


class TaskDistributionSampler:
    """
    Deterministic sampler for fixed and randomized task distributions.

    Parameters
    ----------
    settings
        Validated distribution settings.
    env_rank
        Vectorized environment rank. The effective RNG seed is ``seed + rank``.

    """

    def __init__(self, settings: TaskDistributionSettings, env_rank: int = 0) -> None:
        """Initialize deterministic per-rank RNG state."""
        if env_rank < 0:
            message = "env_rank must be nonnegative"
            raise ValueError(message)
        self.settings = settings
        self.env_rank = int(env_rank)
        self.effective_seed = int(settings.seed) + RANK_SEED_STRIDE * self.env_rank
        self._rng = np.random.default_rng(self.effective_seed)
        self._sample_index = 0
        self._stable_sample: SampledTask | None = None
        self.last_sample: SampledTask | None = None
        if not self._should_sample_on_reset():
            self._stable_sample = self._sample_valid_task()
            self.last_sample = self._stable_sample

    def sample_task(self) -> dict[str, Any]:
        """Return the next valid task sampled according to distribution settings."""
        if self._stable_sample is not None:
            self.last_sample = self._stable_sample
            return copy.deepcopy(self._stable_sample.task)
        sample = self._sample_valid_task()
        self.last_sample = sample
        return copy.deepcopy(sample.task)

    def sample_metadata(self) -> dict[str, Any]:
        """Return metadata for the most recent sample."""
        if self.last_sample is None:
            return self.settings.to_metadata()
        return dict(self.last_sample.metadata)

    def _should_sample_on_reset(self) -> bool:
        """Return whether this sampler should produce a new task on every reset."""
        return bool(
            self.settings.enabled and self.settings.mode == MODE_RANDOMIZED and self.settings.sample_on_reset and self.settings.strength > 0.0
        )

    def _sample_valid_task(self) -> SampledTask:
        """Sample until a generated task passes deterministic validation."""
        if not self.settings.enabled or self.settings.mode == MODE_FIXED or self.settings.strength == 0.0:
            return self._fixed_sample()

        errors: list[str] = []
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            family = _sample_family(self._rng, self.settings.family_weights)
            try:
                task = _sample_family_task(family=family, settings=self.settings, rng=self._rng)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
                continue
            result = validation.tasks.validate_task(task, limits=self.settings.validation_limits)
            if result.is_valid:
                metadata = self._metadata(family=family, task=task, validation_messages=result.messages)
                return SampledTask(task=task, metadata=metadata)
            errors.append(f"{family}: {'; '.join(result.messages)}")
        details = "; ".join(errors[-5:])
        message = f"failed to sample a valid task after {MAX_SAMPLE_ATTEMPTS} attempts: {details}"
        raise ValueError(message)

    def _fixed_sample(self) -> SampledTask:
        """Return the base task with fixed-mode sample metadata."""
        family = _family_from_task_shape(self.settings.base_task_shape)
        task = copy.deepcopy(self.settings.base_task)
        result = validation.tasks.validate_task(task, limits=self.settings.validation_limits)
        metadata = self._metadata(family=family, task=task, validation_messages=result.messages)
        return SampledTask(task=task, metadata=metadata)

    def _metadata(self, family: str, task: Mapping[str, Any], validation_messages: Sequence[str]) -> dict[str, Any]:
        """Build metadata for one sampled task."""
        sample_index = self._sample_index
        self._sample_index += 1
        metadata = self.settings.to_metadata()
        metadata.update(
            {
                "task_distribution_env_rank": self.env_rank,
                "task_distribution_effective_seed": self.effective_seed,
                "task_distribution_sample_index": sample_index,
                "task_distribution_sampled_family": family,
                "task_distribution_sampled_task_shape": str(task.get(validation.contracts.FIELD_SHAPE, "unknown")),
                "task_distribution_sampled_task_name": task.get("task_name"),
                "task_distribution_validation_messages": list(validation_messages),
                "task_distribution_sampled_task": copy.deepcopy(dict(task)),
            }
        )
        return metadata


def supported_task_families() -> tuple[str, ...]:
    """Return trajectory families that this sampler can emit through existing task schemas."""
    return _SUPPORTED_FAMILIES


def unsupported_requested_task_families() -> tuple[str, ...]:
    """Return requested broad families intentionally omitted from this schema adapter."""
    return _UNSUPPORTED_REQUESTED_FAMILIES


def load_task_distribution_settings(path_or_mapping: str | Path | Mapping[str, Any]) -> TaskDistributionSettings:
    """
    Load task-distribution settings from YAML or a mapping.

    Parameters
    ----------
    path_or_mapping
        YAML path, raw distribution mapping, or mapping with a top-level
        ``task_distribution`` section.

    Returns
    -------
    TaskDistributionSettings
        Validated task-distribution settings.

    """
    config_path: Path | None = None
    if isinstance(path_or_mapping, (str, Path)):
        config_path = Path(path_or_mapping)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            message = "task distribution config root must be a mapping"
            raise TypeError(message)
        payload = dict(loaded)
    else:
        payload = dict(path_or_mapping)
    distribution_payload = payload.get(DISTRIBUTION_CONFIG_KEY, payload)
    if not isinstance(distribution_payload, Mapping):
        message = "task_distribution must be a mapping"
        raise TypeError(message)
    raw = dict(distribution_payload)
    name = raw.get("name") or payload.get("name")
    limits_payload = raw.get("validation_limits") or payload.get("validation_limits")
    return TaskDistributionSettings(
        name=None if name is None else str(name),
        enabled=_optional_bool(raw.get("enabled", True), "task_distribution.enabled"),
        mode=str(raw.get("mode", MODE_FIXED)),
        seed=int(raw.get("seed", 0)),
        strength=float(raw.get("strength", 0.0)),
        sample_on_reset=_optional_bool(raw.get("sample_on_reset", False), "task_distribution.sample_on_reset"),
        base_task=_required_mapping(raw.get("base_task"), "task_distribution.base_task"),
        family_weights=_optional_float_mapping(raw.get("family_weights"), "task_distribution.family_weights"),
        variations=_optional_nested_mapping(raw.get("variations"), "task_distribution.variations"),
        validation_limits=_validation_limits_from_mapping(limits_payload),
        config_path=config_path,
    )


def normalize_fixed_task_to_distribution(
    task: Mapping[str, Any],
    *,
    seed: int = 0,
    name: str | None = None,
    config_path: str | Path | None = None,
) -> TaskDistributionSettings:
    """
    Represent one fixed task as a degenerate task distribution.

    Parameters
    ----------
    task
        Valid fixed task mapping.
    seed
        Seed recorded in distribution metadata.
    name
        Optional metadata name.
    config_path
        Optional source config path recorded in metadata.

    Returns
    -------
    TaskDistributionSettings
        Fixed-mode strength-zero distribution settings.

    """
    task_copy = copy.deepcopy(dict(task))
    family = _family_from_task_shape(str(task_copy.get(validation.contracts.FIELD_SHAPE, "")))
    return TaskDistributionSettings(
        name=name,
        enabled=True,
        mode=MODE_FIXED,
        seed=int(seed),
        strength=0.0,
        sample_on_reset=False,
        base_task=task_copy,
        family_weights={family: 1.0},
        variations={},
        validation_limits=None,
        config_path=None if config_path is None else Path(config_path),
    )


def task_distribution_with_base_task(settings: TaskDistributionSettings, base_task: Mapping[str, Any]) -> TaskDistributionSettings:
    """Return settings with a different base task while preserving distribution controls."""
    return TaskDistributionSettings(
        name=settings.name,
        enabled=settings.enabled,
        mode=settings.mode,
        seed=settings.seed,
        strength=settings.strength,
        sample_on_reset=settings.sample_on_reset,
        base_task=copy.deepcopy(dict(base_task)),
        family_weights=settings.family_weights,
        variations=settings.variations,
        validation_limits=settings.validation_limits,
        config_path=settings.config_path,
    )


def sample_task(settings: TaskDistributionSettings, env_rank: int = 0) -> dict[str, Any]:
    """Sample one task from settings using a fresh deterministic sampler."""
    return TaskDistributionSampler(settings=settings, env_rank=env_rank).sample_task()


def _normalize_family_weights(raw_weights: Mapping[str, float], base_task: Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize family weights."""
    weights = dict(raw_weights)
    if not weights:
        weights = {_family_from_task_shape(str(base_task.get(validation.contracts.FIELD_SHAPE, ""))): 1.0}
    unknown = sorted(set(weights) - set(_SUPPORTED_FAMILIES))
    if unknown:
        message = f"task_distribution.family_weights contains unsupported families: {', '.join(unknown)}"
        raise ValueError(message)
    negative = [family for family, weight in weights.items() if float(weight) < 0.0]
    if negative:
        message = f"task_distribution.family_weights contains negative weights: {', '.join(sorted(negative))}"
        raise ValueError(message)
    total = float(sum(float(weight) for weight in weights.values()))
    if total <= 0.0:
        message = "task_distribution.family_weights must not be all zero"
        raise ValueError(message)
    return {family: float(weights[family]) / total for family in sorted(weights)}


def _normalize_variations(raw_variations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate variation family keys and return copied dictionaries."""
    unknown = sorted(set(raw_variations) - set(_SUPPORTED_FAMILIES))
    if unknown:
        message = f"task_distribution.variations contains unsupported families: {', '.join(unknown)}"
        raise ValueError(message)
    return {str(family): copy.deepcopy(dict(values)) for family, values in raw_variations.items()}


def _sample_family(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    """Sample one family name from normalized weights."""
    families = list(weights)
    probabilities = np.array([float(weights[family]) for family in families], dtype=float)
    probabilities = probabilities / float(np.sum(probabilities))
    index = int(rng.choice(len(families), p=probabilities))
    return families[index]


def _sample_family_task(family: str, settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample one task for a supported family."""
    family_samplers: dict[str, Callable[[], dict[str, Any]]] = {
        FAMILY_HOVER: lambda: _sample_hover_task(settings, rng),
        FAMILY_TAKEOFF: lambda: _sample_takeoff_task(settings, rng),
        FAMILY_VERTICAL_UP_DOWN: lambda: _sample_vertical_up_down_task(settings, rng),
        FAMILY_ANGLED_VERTICAL: lambda: _sample_angled_vertical_task(settings, rng),
        FAMILY_LINE: lambda: _sample_line_task(settings, rng),
        FAMILY_START_HOLD_LINE: lambda: _sample_start_hold_line_task(settings, rng),
        FAMILY_POLYLINE: lambda: _sample_polyline_task(settings, rng),
        FAMILY_L_SHAPE: lambda: _sample_l_shape_task(settings, rng),
        FAMILY_ZIGZAG: lambda: _sample_zigzag_task(settings, rng),
        FAMILY_TRIANGLE: lambda: _sample_triangle_task(settings, rng),
        FAMILY_MULTI_HEIGHT_POLYLINE: lambda: _sample_multi_height_polyline_task(settings, rng),
        FAMILY_DELAYED_ALTITUDE_POLYLINE: lambda: _sample_delayed_altitude_polyline_task(settings, rng),
        FAMILY_RECTANGLE: lambda: _sample_rectangle_task(settings, rng, square=False),
        FAMILY_SQUARE: lambda: _sample_rectangle_task(settings, rng, square=True),
        FAMILY_CIRCLE: lambda: _sample_circle_task(settings, rng),
        FAMILY_ELLIPSE: lambda: _sample_ellipse_task(settings, rng),
        FAMILY_FIGURE_EIGHT: lambda: _sample_figure_eight_task(settings, rng),
        FAMILY_BASIC_TRAINING_SHOW: lambda: _sample_basic_training_show_task(settings, rng),
    }
    sampler = family_samplers.get(family)
    if sampler is None:
        message = f"unsupported task family: {family}"
        raise ValueError(message)
    return sampler()


def _sample_basic_training_show_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a bounded composed basic training show."""
    variation = settings.variations.get(FAMILY_BASIC_TRAINING_SHOW, {})
    base = settings.base_task
    sample_rate_hz = _sample_rate(base)
    origin = _base_position(base)
    offset = np.array(
        [
            _sample_range(rng, variation.get("x_offset_range_m"), default=(0.0, 0.0), anchor=0.0, strength=settings.strength),
            _sample_range(rng, variation.get("y_offset_range_m"), default=(0.0, 0.0), anchor=0.0, strength=settings.strength),
            _sample_range(rng, variation.get("z_offset_range_m"), default=(0.0, 0.0), anchor=0.0, strength=settings.strength),
        ],
        dtype=float,
    )
    length_scale = _sample_range(rng, variation.get("length_scale_range"), default=(0.95, 1.05), anchor=1.0, strength=settings.strength)
    duration_scale = _sample_range(rng, variation.get("duration_scale_range"), default=(0.95, 1.08), anchor=1.0, strength=settings.strength)
    ellipse_scale = _sample_range(rng, variation.get("ellipse_scale_range"), default=(0.9, 1.1), anchor=1.0, strength=settings.strength)
    heading = math.radians(_float_value(variation.get("heading_jitter_deg", 10.0))) * settings.strength * float(rng.uniform(-1.0, 1.0))
    start_hold_sec = _sample_range(
        rng,
        variation.get("start_hold_range_sec"),
        default=(DEFAULT_START_HOLD_SEC, DEFAULT_START_HOLD_SEC),
        anchor=float(base.get(validation.contracts.FIELD_START_HOLD_SEC, DEFAULT_START_HOLD_SEC)),
        strength=settings.strength,
    )
    final_hold_sec = _sample_range(
        rng,
        variation.get("final_hold_range_sec"),
        default=(0.9, 1.1),
        anchor=float(base.get(validation.contracts.FIELD_FINAL_HOLD_SEC, DEFAULT_FINAL_HOLD_SEC)),
        strength=settings.strength,
    )

    def rotate(delta_xyz: Sequence[float]) -> np.ndarray:
        delta = np.asarray(delta_xyz, dtype=float)
        c = math.cos(heading)
        s = math.sin(heading)
        return np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]], dtype=float)

    def scaled_duration(duration_sec: float) -> float:
        return _round(max(1.0, duration_sec * duration_scale))

    p0 = origin + offset
    p1 = p0 + rotate([0.42 * length_scale, 0.0, 0.0])
    p2 = p1 + rotate([0.28 * length_scale, 0.22 * length_scale, 0.02 * length_scale])
    p3 = p2 + np.array([0.0, 0.0, 0.22 * length_scale], dtype=float)
    radius_x = 0.16 * ellipse_scale * length_scale
    radius_y = 0.11 * ellipse_scale * length_scale
    p4 = p3 + rotate([0.0, 0.30 * length_scale, 0.0])
    p5 = p4 + rotate([0.32 * length_scale, 0.0, -0.04 * length_scale])
    p6 = p5 + rotate([0.18 * length_scale, 0.14 * length_scale, 0.0])
    p7 = p5 + rotate([0.36 * length_scale, -0.12 * length_scale, -0.04 * length_scale])
    p8 = p5 + rotate([0.53 * length_scale, 0.06 * length_scale, -0.04 * length_scale])

    segments = [
        _basic_show_segment("hover_start", "hover_stabilization", p0, p0, scaled_duration(1.2)),
        _basic_show_segment("horizontal_line", "horizontal_line", p0, p1, scaled_duration(2.1)),
        _basic_show_segment("diagonal_line", "diagonal_line", p1, p2, scaled_duration(2.2)),
        _basic_show_segment("vertical_lift", "vertical", p2, p3, scaled_duration(1.8)),
        _basic_show_segment(
            "ellipse_loop",
            "ellipse",
            p3,
            p3,
            scaled_duration(3.6),
            radius_x_m=_round(radius_x),
            radius_y_m=_round(radius_y),
            phase_deg=180.0,
        ),
        _basic_show_segment(
            "polyline_l_shape",
            "l_shape",
            p3,
            p5,
            scaled_duration(2.8),
            segment_points=[_vector(p3), _vector(p4), _vector(p5)],
        ),
        _basic_show_segment(
            "zigzag_finish",
            "zigzag",
            p5,
            p8,
            scaled_duration(3.6),
            segment_points=[_vector(p5), _vector(p6), _vector(p7), _vector(p8)],
        ),
        _basic_show_segment(
            "final_hold",
            "final_hold",
            p8,
            p8,
            _round(final_hold_sec),
            segment_final_hold_sec=_round(final_hold_sec),
        ),
    ]
    metadata = _basic_training_show_metadata(
        segments=segments,
        start_hold_sec=start_hold_sec,
        final_hold_sec=final_hold_sec,
        settings=settings,
        variation=variation,
    )
    return {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_BASIC_TRAINING_SHOW,
        validation.contracts.FIELD_DURATION_SEC: metadata["duration_range_sec"][0],
        validation.contracts.FIELD_SAMPLE_RATE_HZ: sample_rate_hz,
        validation.contracts.FIELD_START_HOLD_ENABLED: True,
        validation.contracts.FIELD_START_HOLD_SEC: _round(start_hold_sec),
        validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
        validation.contracts.FIELD_FINAL_HOLD_ENABLED: True,
        validation.contracts.FIELD_FINAL_HOLD_SEC: _round(final_hold_sec),
        validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS: True,
        validation.contracts.FIELD_SEGMENTS: segments,
        **metadata,
    }


def _basic_show_segment(
    name: str,
    shape: str,
    start: np.ndarray,
    end: np.ndarray,
    duration_sec: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build compact metadata for one basic-training-show segment."""
    return {
        "segment_name": name,
        validation.contracts.FIELD_SEGMENT_SHAPE: shape,
        validation.contracts.FIELD_SEGMENT_START: _vector(start),
        validation.contracts.FIELD_SEGMENT_END: _vector(end),
        validation.contracts.FIELD_SEGMENT_DURATION_SEC: _round(duration_sec),
        **extra,
    }


def _basic_training_show_metadata(
    *,
    segments: Sequence[Mapping[str, Any]],
    start_hold_sec: float,
    final_hold_sec: float,
    settings: TaskDistributionSettings,
    variation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact metadata for a sampled basic training show."""
    durations = [float(segment[validation.contracts.FIELD_SEGMENT_DURATION_SEC]) for segment in segments]
    moving_durations = durations[:-1]
    moving_duration = float(sum(moving_durations))
    total_duration = float(start_hold_sec) + moving_duration + float(final_hold_sec)
    path_lengths = [_basic_show_segment_path_length(segment) for segment in segments]
    nonzero_speeds = [length / duration for length, duration in zip(path_lengths, durations, strict=True) if length > 0.0 and duration > 0.0]
    path_length = float(sum(path_lengths))
    min_speed = min(nonzero_speeds) if nonzero_speeds else 0.0
    max_speed = max(nonzero_speeds) if nonzero_speeds else 0.0
    segment_shapes = ["start_hold", *(str(segment[validation.contracts.FIELD_SEGMENT_SHAPE]) for segment in segments)]
    meaningful_figure_count = len(segments)
    all_durations = [float(start_hold_sec), *durations]
    return {
        "training_task_kind": FAMILY_BASIC_TRAINING_SHOW,
        "task_is_distribution": bool(settings.enabled and settings.mode == MODE_RANDOMIZED),
        "task_is_show": True,
        "show_name": FAMILY_BASIC_TRAINING_SHOW,
        "segment_count": len(segment_shapes),
        "meaningful_figure_count": meaningful_figure_count,
        "segment_shapes": segment_shapes,
        "show_is_continuous": True,
        "continuity_tolerance": 1.0e-6,
        "difficulty_level": "basic",
        "duration_range_sec": [_round(total_duration), _round(total_duration)],
        "move_duration_range_sec": [_round(moving_duration), _round(moving_duration)],
        "segment_duration_range_sec": [_round(min(all_durations)), _round(max(all_durations))],
        "path_length_range_m": [_round(path_length), _round(path_length)],
        "approx_reference_speed_range_mps": [_round(min_speed), _round(max_speed)],
        "segment_speed_bounds": {"min_mps": _round(min_speed), "max_mps": _round(max_speed)},
        "sampled_per_episode": bool(settings.sample_on_reset),
        "constant_within_episode": True,
        "variation_enabled": bool(settings.strength > 0.0),
        "variation_mode": "bounded_per_episode" if settings.strength > 0.0 else "fixed",
        "start_hold_enabled": True,
        "start_hold_sec": _round(start_hold_sec),
        "exclude_start_hold_from_tracking_metrics": True,
        "standard_reference_height_enabled": True,
        "start_height_policy": STANDARD_REFERENCE_HEIGHT_POLICY,
        "base_z_m": _round(float(np.asarray(segments[0][validation.contracts.FIELD_SEGMENT_START], dtype=float)[2])),
        "base_z_range_m": _range_metadata(variation.get("base_z_range_m", variation.get("z_range_m")))
        or [float(STANDARD_REFERENCE_HEIGHT_RANGE_M[0]), float(STANDARD_REFERENCE_HEIGHT_RANGE_M[1])],
        "sampled_start_height_m": _round(float(np.asarray(segments[0][validation.contracts.FIELD_SEGMENT_START], dtype=float)[2])),
        "height_variation_enabled": _height_variation_enabled(variation),
        "start_hold_policy": STANDARD_START_HOLD_POLICY,
        "start_hold_reward_policy": STANDARD_START_HOLD_REWARD_POLICY,
        "tracking_reward_starts_after_start_hold": False,
        "final_hold_enabled": True,
        "final_hold_sec": _round(final_hold_sec),
        "own_task_eval_path": "evaluations/own_task",
        "generalization_eval_path": "evaluations/generalization",
        "scenario_eval_path": "evaluations/scenarios",
        "requested_task_family": FAMILY_BASIC_TRAINING_SHOW,
        "accepted_task_family": FAMILY_BASIC_TRAINING_SHOW,
        "variation_strength": float(settings.strength),
        "proposed_sampling_bounds": copy.deepcopy(dict(variation)),
        "accepted_sampling_bounds": copy.deepcopy(dict(variation)),
        "repair_was_applied": False,
        "repair_reason": None,
    }


def _basic_show_segment_path_length(segment: Mapping[str, Any]) -> float:
    """Approximate path length for one basic-training-show segment."""
    shape = str(segment[validation.contracts.FIELD_SEGMENT_SHAPE])
    duration_shape = shape.lower()
    if duration_shape in {"hover_stabilization", "final_hold"}:
        return 0.0
    if duration_shape == "ellipse":
        radius_x = float(segment["radius_x_m"])
        radius_y = float(segment["radius_y_m"])
        return float(math.pi * (3.0 * (radius_x + radius_y) - math.sqrt((3.0 * radius_x + radius_y) * (radius_x + 3.0 * radius_y))))
    if validation.contracts.FIELD_SEGMENT_POINTS in segment:
        points = np.asarray(segment[validation.contracts.FIELD_SEGMENT_POINTS], dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    start = np.asarray(segment[validation.contracts.FIELD_SEGMENT_START], dtype=float)
    end = np.asarray(segment[validation.contracts.FIELD_SEGMENT_END], dtype=float)
    return float(np.linalg.norm(end - start))


def _sample_hover_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a hover-stabilization task."""
    variation = settings.variations.get(FAMILY_HOVER, {})
    base = settings.base_task
    position = _base_position(base)
    if "x_range_m" in variation or "y_range_m" in variation:
        x = _sample_range(rng, variation.get("x_range_m"), default=(position[0], position[0]), anchor=position[0], strength=settings.strength)
        y = _sample_range(rng, variation.get("y_range_m"), default=(position[1], position[1]), anchor=position[1], strength=settings.strength)
        xy = np.array([x, y], dtype=float)
    else:
        xy = position[:2] + _sample_xy_offset(rng, _float_value(variation.get("xy_radius_m", 0.0)) * settings.strength)
    z = _sample_range(rng, variation.get("z_range_m"), default=(position[2], position[2]), anchor=position[2], strength=settings.strength)
    duration = _sample_range(
        rng,
        variation.get("duration_range_sec"),
        default=(_duration(base), _duration(base)),
        anchor=_duration(base),
        strength=settings.strength,
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_HOVER_STABILIZATION,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_POSITION: _xyz(xy[0], xy[1], z),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_takeoff_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a takeoff-style vertical task using the existing vertical schema."""
    variation = settings.variations.get(FAMILY_TAKEOFF, {})
    base = settings.base_task
    base_xy = np.asarray(base.get(validation.contracts.FIELD_XY, _base_position(base)[:2]), dtype=float)
    xy = base_xy + _sample_xy_offset(rng, _float_value(variation.get("xy_radius_m", 0.0)) * settings.strength)
    start_height = _sample_range(rng, variation.get("start_z_range_m"), default=(0.90, 1.10), anchor=1.0, strength=settings.strength)
    end_anchor = float(base.get(validation.contracts.FIELD_END_HEIGHT, _base_position(base)[2]))
    end_height = _sample_range(rng, variation.get("z_range_m"), default=(1.0, 1.35), anchor=end_anchor, strength=settings.strength)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(3.0, 4.5), anchor=3.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_VERTICAL,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_XY: _xy(xy[0], xy[1]),
        validation.contracts.FIELD_START_HEIGHT: _round(start_height),
        validation.contracts.FIELD_END_HEIGHT: _round(end_height),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_vertical_up_down_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a standard-height vertical task that can climb or descend."""
    variation = settings.variations.get(FAMILY_VERTICAL_UP_DOWN, {})
    base = settings.base_task
    base_xy = np.asarray(base.get(validation.contracts.FIELD_XY, _base_position(base)[:2]), dtype=float)
    xy = base_xy + _sample_xy_offset(rng, _float_value(variation.get("xy_radius_m", 0.0)) * settings.strength)
    start_anchor = float(base.get(validation.contracts.FIELD_START_HEIGHT, _base_position(base)[2]))
    start_height = _sample_range(
        rng,
        variation.get("start_z_range_m", variation.get("base_z_range_m")),
        default=STANDARD_REFERENCE_HEIGHT_RANGE_M,
        anchor=start_anchor,
        strength=settings.strength,
    )
    delta = _sample_range(
        rng,
        variation.get("height_delta_range_m"),
        default=(0.18, 0.42),
        anchor=0.28,
        strength=settings.strength,
    )
    direction = str(variation.get("direction", "up_down"))
    if direction == "up":
        sign = 1.0
    elif direction == "down":
        sign = -1.0
    else:
        sign = 1.0 if rng.random() < COIN_FLIP_PROBABILITY else -1.0
    end_low, end_high = _coerce_range(variation.get("end_z_range_m", variation.get("z_range_m")), default=(0.75, 1.35))
    end_height = float(np.clip(start_height + sign * delta, end_low, end_high))
    if abs(end_height - start_height) < MIN_VERTICAL_HEIGHT_DELTA_M:
        end_height = float(np.clip(start_height - sign * delta, end_low, end_high))
    duration = _sample_range(
        rng,
        variation.get("duration_range_sec"),
        default=(4.0, 6.5),
        anchor=_duration(base, 4.8),
        strength=settings.strength,
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_VERTICAL,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_XY: _xy(xy[0], xy[1]),
        validation.contracts.FIELD_START_HEIGHT: _round(start_height),
        validation.contracts.FIELD_END_HEIGHT: _round(end_height),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_angled_vertical_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a standard-height diagonal climb or descent with lateral motion."""
    variation = settings.variations.get(FAMILY_ANGLED_VERTICAL, {})
    base = settings.base_task
    start_base, _ = _base_line_points(base)
    start_xy_radius = _float_value(variation.get("start_xy_radius_m", variation.get("xy_radius_m", 0.0)))
    start = np.array(start_base, dtype=float, copy=True)
    start[:2] += _sample_xy_offset(rng, start_xy_radius * settings.strength)
    start[2] = _sample_range(
        rng,
        variation.get("start_z_range_m", variation.get("base_z_range_m")),
        default=STANDARD_REFERENCE_HEIGHT_RANGE_M,
        anchor=float(start_base[2]),
        strength=settings.strength,
    )
    delta = _sample_range(
        rng,
        variation.get("height_delta_range_m"),
        default=(0.16, 0.34),
        anchor=0.24,
        strength=settings.strength,
    )
    lateral = _sample_range(
        rng,
        variation.get("lateral_range_m", variation.get("length_range_m")),
        default=(0.22, 0.46),
        anchor=0.32,
        strength=settings.strength,
    )
    direction = str(variation.get("direction", "up_down"))
    if direction == "up":
        vertical_sign = 1.0
    elif direction == "down":
        vertical_sign = -1.0
    else:
        vertical_sign = 1.0 if rng.random() < COIN_FLIP_PROBABILITY else -1.0
    heading_choices = tuple(float(value) for value in variation.get("heading_choices_deg", (35.0, 145.0, -35.0, -145.0)))
    heading_deg = heading_choices[int(rng.integers(0, len(heading_choices)))]
    heading_deg += _float_value(variation.get("heading_jitter_deg", 6.0)) * settings.strength * float(rng.uniform(-1.0, 1.0))
    heading = math.radians(heading_deg)
    end = start + np.array([lateral * math.cos(heading), lateral * math.sin(heading), vertical_sign * delta], dtype=float)
    z_low, z_high = _coerce_range(variation.get("z_range_m"), default=(0.75, 1.35))
    end[2] = min(max(float(end[2]), z_low), z_high)
    duration = _sample_range(
        rng,
        variation.get("duration_range_sec"),
        default=(4.5, 7.0),
        anchor=_duration(base, 5.4),
        strength=settings.strength,
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_LINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_START: _vector(start),
        validation.contracts.FIELD_END: _vector(end),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_line_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a straight-line task."""
    variation = settings.variations.get(FAMILY_LINE, {})
    base = settings.base_task
    start_base, end_base = _base_line_points(base)
    start, end, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    duration_default = (max(3.0, length / 0.2), max(5.0, length / 0.1))
    duration = _sample_range(
        rng, variation.get("duration_range_sec"), default=duration_default, anchor=_duration(base, 4.0), strength=settings.strength
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_LINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_START: _vector(start),
        validation.contracts.FIELD_END: _vector(end),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_start_hold_line_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a held-start then short-line task."""
    variation = settings.variations.get(FAMILY_START_HOLD_LINE, {})
    base = settings.base_task
    start_base, end_base = _base_line_points(base)
    start, end, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    hold_anchor = float(
        base.get(validation.contracts.FIELD_HOLD_DURATION_SEC, base.get(validation.contracts.FIELD_START_HOLD_SEC, DEFAULT_START_HOLD_SEC))
    )
    hold_duration = _sample_range(
        rng,
        variation.get("hold_duration_range_sec", variation.get("start_hold_range_sec")),
        default=(DEFAULT_START_HOLD_SEC, DEFAULT_START_HOLD_SEC),
        anchor=hold_anchor,
        strength=settings.strength,
    )
    move_anchor = float(base.get(validation.contracts.FIELD_MOVE_DURATION_SEC, max(3.0, length / 0.15)))
    move_duration = _sample_range(
        rng,
        variation.get("move_duration_range_sec", variation.get("duration_range_sec")),
        default=(3.0, 6.0),
        anchor=move_anchor,
        strength=settings.strength,
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
        validation.contracts.FIELD_HOLD_DURATION_SEC: _round(hold_duration),
        validation.contracts.FIELD_MOVE_DURATION_SEC: _round(move_duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_START_HOLD_ENABLED: True,
        validation.contracts.FIELD_START_HOLD_SEC: _round(hold_duration),
        validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
        validation.contracts.FIELD_START: _vector(start),
        validation.contracts.FIELD_END: _vector(end),
    }
    return _with_final_hold(task, base, variation, rng, settings.strength)


def _sample_polyline_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a gentle three-point polyline task."""
    variation = settings.variations.get(FAMILY_POLYLINE, {})
    start_base, end_base = _base_line_points(settings.base_task)
    start, end, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    heading = math.atan2(end[1] - start[1], end[0] - start[0])
    turn = rng.uniform(-math.pi / 3.0, math.pi / 3.0) * settings.strength
    mid_length = 0.5 * length
    mid = start + np.array([mid_length * math.cos(heading + turn), mid_length * math.sin(heading + turn), 0.0], dtype=float)
    mid[2] = (start[2] + end[2]) / 2.0
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(5.0, 8.0), anchor=6.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(start), _vector(mid), _vector(end)],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_l_shape_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample an L-shaped polyline task."""
    variation = settings.variations.get(FAMILY_L_SHAPE, {})
    start_base, end_base = _base_line_points(settings.base_task)
    start, _, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    heading = _base_heading(start_base, end_base) + math.radians(
        _float_value(variation.get("heading_jitter_deg", 30.0))
    ) * settings.strength * rng.uniform(-1.0, 1.0)
    leg1 = _sample_range(
        rng, variation.get("leg1_range_m"), default=(0.2, max(0.25, length)), anchor=max(0.25, length * 0.6), strength=settings.strength
    )
    leg2 = _sample_range(
        rng, variation.get("leg2_range_m"), default=(0.2, max(0.25, length)), anchor=max(0.25, length * 0.6), strength=settings.strength
    )
    turn_direction = -1.0 if rng.random() < COIN_FLIP_PROBABILITY else 1.0
    mid = start + np.array([leg1 * math.cos(heading), leg1 * math.sin(heading), 0.0], dtype=float)
    end = mid + np.array(
        [leg2 * math.cos(heading + turn_direction * math.pi / 2.0), leg2 * math.sin(heading + turn_direction * math.pi / 2.0), 0.0], dtype=float
    )
    final_z_offset = _sample_range(
        rng,
        variation.get("final_z_offset_range_m"),
        default=(0.0, 0.0),
        anchor=0.0,
        strength=settings.strength,
    )
    end[2] = start[2] + final_z_offset
    mid[2] = (start[2] + end[2]) / 2.0
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(6.0, 9.0), anchor=7.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(start), _vector(mid), _vector(end)],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_zigzag_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a compact zigzag/slalom polyline task."""
    variation = settings.variations.get(FAMILY_ZIGZAG, {})
    start_base, end_base = _base_line_points(settings.base_task)
    start, _, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    heading = _base_heading(start_base, end_base) + math.radians(
        _float_value(variation.get("heading_jitter_deg", 25.0))
    ) * settings.strength * rng.uniform(-1.0, 1.0)
    amplitude = _sample_range(rng, variation.get("amplitude_range_m"), default=(0.10, 0.22), anchor=0.16, strength=settings.strength)
    z_anchor = float(start[2])
    z_end = _sample_range(rng, variation.get("z_range_m"), default=(z_anchor, z_anchor), anchor=z_anchor, strength=settings.strength)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
    lateral = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=float)
    fractions = (0.0, 0.33, 0.66, 1.0)
    offsets = (0.0, amplitude, -amplitude, 0.5 * amplitude)
    points = []
    for fraction, lateral_offset in zip(fractions, offsets, strict=True):
        point = start + length * fraction * forward + lateral_offset * lateral
        point[2] = z_anchor + (z_end - z_anchor) * fraction
        points.append(point)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(6.0, 9.5), anchor=7.5, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(point) for point in points],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_triangle_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a closed triangular polyline task."""
    variation = settings.variations.get(FAMILY_TRIANGLE, {})
    base_position = _base_position(settings.base_task)
    center = base_position[:2] + _sample_xy_offset(rng, _float_value(variation.get("center_xy_radius_m", 0.08)) * settings.strength)
    side = _sample_range(
        rng, variation.get("side_range_m", variation.get("length_range_m")), default=(0.25, 0.45), anchor=0.35, strength=settings.strength
    )
    heading = math.radians(_float_value(variation.get("heading_jitter_deg", 30.0))) * settings.strength * rng.uniform(-1.0, 1.0)
    z = _sample_range(
        rng, variation.get("z_range_m"), default=(base_position[2], base_position[2]), anchor=base_position[2], strength=settings.strength
    )
    radius = side / math.sqrt(3.0)
    points = []
    for index in range(3):
        angle = heading + math.pi / 2.0 + index * 2.0 * math.pi / 3.0
        points.append(np.array([center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), z], dtype=float))
    points.append(np.array(points[0], dtype=float, copy=True))
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(7.0, 11.0), anchor=9.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(point) for point in points],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_multi_height_polyline_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a polyline with small bounded altitude changes."""
    variation = settings.variations.get(FAMILY_MULTI_HEIGHT_POLYLINE, {})
    start_base, end_base = _base_line_points(settings.base_task)
    start, _, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    heading = _base_heading(start_base, end_base) + math.radians(
        _float_value(variation.get("heading_jitter_deg", 25.0))
    ) * settings.strength * rng.uniform(-1.0, 1.0)
    lateral = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=float)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
    height_delta = _sample_range(
        rng,
        variation.get("height_offset_range_m"),
        default=(-0.12, 0.14),
        anchor=0.08,
        strength=settings.strength,
    )
    z_base = float(start[2])
    points = [
        start,
        start + 0.33 * length * forward + 0.10 * length * lateral + np.array([0.0, 0.0, height_delta], dtype=float),
        start + 0.66 * length * forward - 0.08 * length * lateral + np.array([0.0, 0.0, -0.7 * height_delta], dtype=float),
        start + length * forward + np.array([0.0, 0.0, 0.4 * height_delta], dtype=float),
    ]
    z_low, z_high = _coerce_range(variation.get("z_range_m"), default=(z_base - 0.18, z_base + 0.18))
    for point in points:
        point[2] = min(max(float(point[2]), z_low), z_high)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(6.5, 10.0), anchor=8.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(point) for point in points],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_delayed_altitude_polyline_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a polyline with near-level XY motion before an altitude change."""
    variation = settings.variations.get(FAMILY_DELAYED_ALTITUDE_POLYLINE, {})
    start_base, end_base = _base_line_points(settings.base_task)
    start, _, length = _sample_segment(settings=settings, rng=rng, variation=variation, start_base=start_base, end_base=end_base)
    heading = _base_heading(start_base, end_base) + math.radians(
        _float_value(variation.get("heading_jitter_deg", 20.0))
    ) * settings.strength * rng.uniform(-1.0, 1.0)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
    lateral = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=float)
    first_length = _sample_range(
        rng,
        variation.get("first_segment_length_m"),
        default=(0.25, max(0.30, 0.45 * length)),
        anchor=max(0.28, 0.45 * length),
        strength=settings.strength,
    )
    second_length = _sample_range(
        rng,
        variation.get("second_segment_length_m"),
        default=(0.25, max(0.30, 0.50 * length)),
        anchor=max(0.28, 0.50 * length),
        strength=settings.strength,
    )
    height_delta = _sample_range(
        rng,
        variation.get("height_delta_range_m", variation.get("height_offset_range_m")),
        default=(-0.20, 0.25),
        anchor=0.10,
        strength=settings.strength,
    )
    turn_direction = -1.0 if rng.random() < COIN_FLIP_PROBABILITY else 1.0
    p0 = start
    p1 = p0 + first_length * forward
    p1[2] = p0[2]
    p2 = p1 + second_length * forward + 0.35 * second_length * turn_direction * lateral + np.array([0.0, 0.0, height_delta], dtype=float)
    p3 = p2 + 0.35 * second_length * forward
    p3[2] = p2[2]
    z_low, z_high = _coerce_range(variation.get("z_range_m"), default=(0.75, 1.35))
    for point in (p0, p1, p2, p3):
        point[2] = min(max(float(point[2]), z_low), z_high)
    p1[2] = p0[2]
    duration = _sample_range(
        rng,
        variation.get("duration_range_sec"),
        default=(7.5, 11.5),
        anchor=9.0,
        strength=settings.strength,
    )
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(point) for point in (p0, p1, p2, p3)],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_rectangle_task(settings: TaskDistributionSettings, rng: np.random.Generator, *, square: bool) -> dict[str, Any]:
    """Sample a rectangle or square as a validated polyline task."""
    family = FAMILY_SQUARE if square else FAMILY_RECTANGLE
    variation = settings.variations.get(family, {})
    base_position = _base_position(settings.base_task)
    center = base_position[:2] + _sample_xy_offset(rng, _float_value(variation.get("center_xy_radius_m", 0.05)) * settings.strength)
    width = _sample_range(rng, variation.get("width_range_m"), default=(0.25, 0.45), anchor=0.35, strength=settings.strength)
    height = width if square else _sample_range(rng, variation.get("height_range_m"), default=(0.20, 0.40), anchor=0.30, strength=settings.strength)
    z = _sample_range(
        rng, variation.get("z_range_m"), default=(base_position[2], base_position[2]), anchor=base_position[2], strength=settings.strength
    )
    half_w = width / 2.0
    half_h = height / 2.0
    points = np.array(
        [
            [center[0] - half_w, center[1] - half_h, z],
            [center[0] + half_w, center[1] - half_h, z],
            [center[0] + half_w, center[1] + half_h, z],
            [center[0] - half_w, center[1] + half_h, z],
            [center[0] - half_w, center[1] - half_h, z],
        ],
        dtype=float,
    )
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(8.0, 12.0), anchor=10.0, strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_POLYLINE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(settings.base_task),
        validation.contracts.FIELD_POINTS: [_vector(point) for point in points],
    }
    return _with_start_hold(task, settings.base_task, variation, rng, settings.strength)


def _sample_circle_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a slow circle task."""
    variation = settings.variations.get(FAMILY_CIRCLE, {})
    base = settings.base_task
    base_position = _base_position(base)
    center_default = base.get(validation.contracts.FIELD_CENTER, base_position[:2])
    center_array = np.asarray(center_default, dtype=float)
    center = center_array + _sample_xy_offset(rng, _float_value(variation.get("center_xy_radius_m", 0.05)) * settings.strength)
    radius_anchor = float(base.get(validation.contracts.FIELD_RADIUS, 0.3))
    radius = _sample_range(rng, variation.get("radius_range_m"), default=(0.2, 0.45), anchor=radius_anchor, strength=settings.strength)
    height_anchor = float(base.get(validation.contracts.FIELD_HEIGHT, base_position[2]))
    height = _sample_range(rng, variation.get("z_range_m"), default=(height_anchor, height_anchor), anchor=height_anchor, strength=settings.strength)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(7.0, 10.0), anchor=_duration(base, 8.0), strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_CIRCLE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_RADIUS: _round(radius),
        validation.contracts.FIELD_HEIGHT: _round(height),
        validation.contracts.FIELD_CENTER: _xy(center[0], center[1]),
        validation.contracts.FIELD_CLOCKWISE: bool(rng.random() < COIN_FLIP_PROBABILITY),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_ellipse_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a slow ellipse task."""
    variation = settings.variations.get(FAMILY_ELLIPSE, {})
    base = settings.base_task
    base_position = _base_position(base)
    center_default = base.get(validation.contracts.FIELD_CENTER, base_position[:2])
    center_array = np.asarray(center_default, dtype=float)
    center = center_array + _sample_xy_offset(rng, _float_value(variation.get("center_xy_radius_m", 0.05)) * settings.strength)
    radius_x_anchor = float(base.get(validation.contracts.FIELD_RADIUS_X, base.get(validation.contracts.FIELD_RADIUS, 0.3)))
    radius_y_anchor = float(base.get(validation.contracts.FIELD_RADIUS_Y, 0.2))
    radius_x = _sample_range(
        rng,
        variation.get("radius_x_range_m", variation.get("radius_range_m")),
        default=(0.20, 0.40),
        anchor=radius_x_anchor,
        strength=settings.strength,
    )
    radius_y = _sample_range(
        rng,
        variation.get("radius_y_range_m", variation.get("minor_radius_range_m")),
        default=(0.12, 0.30),
        anchor=radius_y_anchor,
        strength=settings.strength,
    )
    height_anchor = float(base.get(validation.contracts.FIELD_HEIGHT, base_position[2]))
    height = _sample_range(rng, variation.get("z_range_m"), default=(height_anchor, height_anchor), anchor=height_anchor, strength=settings.strength)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(9.0, 13.0), anchor=_duration(base, 10.0), strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_ELLIPSE,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_RADIUS_X: _round(radius_x),
        validation.contracts.FIELD_RADIUS_Y: _round(radius_y),
        validation.contracts.FIELD_HEIGHT: _round(height),
        validation.contracts.FIELD_CENTER: _xy(center[0], center[1]),
        validation.contracts.FIELD_CLOCKWISE: bool(rng.random() < COIN_FLIP_PROBABILITY),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_figure_eight_task(settings: TaskDistributionSettings, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a slow figure-eight task."""
    variation = settings.variations.get(FAMILY_FIGURE_EIGHT, {})
    base = settings.base_task
    base_position = _base_position(base)
    center_default = base.get(validation.contracts.FIELD_CENTER, base_position[:2])
    center_array = np.asarray(center_default, dtype=float)
    center = center_array + _sample_xy_offset(rng, _float_value(variation.get("center_xy_radius_m", 0.05)) * settings.strength)
    radius_x_anchor = float(base.get(validation.contracts.FIELD_RADIUS_X, base.get(validation.contracts.FIELD_RADIUS, 0.28)))
    radius_y_anchor = float(base.get(validation.contracts.FIELD_RADIUS_Y, 0.18))
    radius_x = _sample_range(
        rng,
        variation.get("radius_x_range_m", variation.get("radius_range_m")),
        default=(0.18, 0.35),
        anchor=radius_x_anchor,
        strength=settings.strength,
    )
    radius_y = _sample_range(
        rng,
        variation.get("radius_y_range_m", variation.get("minor_radius_range_m")),
        default=(0.10, 0.25),
        anchor=radius_y_anchor,
        strength=settings.strength,
    )
    height_anchor = float(base.get(validation.contracts.FIELD_HEIGHT, base_position[2]))
    height = _sample_range(rng, variation.get("z_range_m"), default=(height_anchor, height_anchor), anchor=height_anchor, strength=settings.strength)
    duration = _sample_range(rng, variation.get("duration_range_sec"), default=(10.0, 15.0), anchor=_duration(base, 12.0), strength=settings.strength)
    task = {
        validation.contracts.FIELD_TASK_TYPE: validation.contracts.TASK_TYPE_TRAJECTORY,
        validation.contracts.FIELD_SHAPE: validation.contracts.SHAPE_FIGURE_EIGHT,
        validation.contracts.FIELD_DURATION_SEC: _round(duration),
        validation.contracts.FIELD_SAMPLE_RATE_HZ: _sample_rate(base),
        validation.contracts.FIELD_RADIUS_X: _round(radius_x),
        validation.contracts.FIELD_RADIUS_Y: _round(radius_y),
        validation.contracts.FIELD_HEIGHT: _round(height),
        validation.contracts.FIELD_CENTER: _xy(center[0], center[1]),
        validation.contracts.FIELD_CLOCKWISE: bool(rng.random() < COIN_FLIP_PROBABILITY),
    }
    return _with_start_hold(task, base, variation, rng, settings.strength)


def _sample_segment(
    *,
    settings: TaskDistributionSettings,
    rng: np.random.Generator,
    variation: Mapping[str, Any],
    start_base: np.ndarray,
    end_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample a conservative horizontal segment anchored to base line points."""
    base_vector = end_base - start_base
    base_length = float(np.linalg.norm(base_vector[:2]))
    base_length = base_length if base_length > 0.0 else 0.4
    length = _sample_range(rng, variation.get("length_range_m"), default=(0.2, max(0.3, base_length)), anchor=base_length, strength=settings.strength)
    heading = _base_heading(start_base, end_base)
    heading += math.radians(_float_value(variation.get("heading_jitter_deg", 0.0))) * settings.strength * rng.uniform(-1.0, 1.0)
    start_xy_radius = _float_value(variation.get("start_xy_radius_m", variation.get("xy_radius_m", 0.0)))
    start = np.array(start_base, dtype=float, copy=True)
    start[:2] += _sample_xy_offset(rng, start_xy_radius * settings.strength)
    z_anchor = float((start_base[2] + end_base[2]) / 2.0)
    z = _sample_range(
        rng,
        variation.get("base_z_range_m", variation.get("z_range_m")),
        default=(z_anchor, z_anchor),
        anchor=z_anchor,
        strength=settings.strength,
    )
    start[2] = z
    end = np.array([start[0] + length * math.cos(heading), start[1] + length * math.sin(heading), z], dtype=float)
    return start, end, float(length)


def _with_start_hold(
    task: dict[str, Any],
    base_task: Mapping[str, Any],
    variation: Mapping[str, Any],
    rng: np.random.Generator,
    strength: float,
) -> dict[str, Any]:
    """Add explicit start-hold fields from variation, base task, or project defaults."""
    if validation.contracts.FIELD_START_HOLD_SEC in variation or "start_hold_range_sec" in variation:
        hold_sec = _sample_range(
            rng,
            variation.get("start_hold_range_sec", variation.get(validation.contracts.FIELD_START_HOLD_SEC)),
            default=(DEFAULT_START_HOLD_SEC, DEFAULT_START_HOLD_SEC),
            anchor=float(base_task.get(validation.contracts.FIELD_START_HOLD_SEC, DEFAULT_START_HOLD_SEC)),
            strength=strength,
        )
        task[validation.contracts.FIELD_START_HOLD_ENABLED] = True
        task[validation.contracts.FIELD_START_HOLD_SEC] = _round(hold_sec)
        task[validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS] = True
        return _with_final_hold(task, base_task, variation, rng, strength)
    copied = False
    for key in (
        validation.contracts.FIELD_START_HOLD_ENABLED,
        validation.contracts.FIELD_START_HOLD_SEC,
        validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
    ):
        if key in base_task:
            task[key] = copy.deepcopy(base_task[key])
            copied = True
    if copied:
        task.setdefault(
            validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS, bool(task.get(validation.contracts.FIELD_START_HOLD_ENABLED, False))
        )
        return _with_final_hold(task, base_task, variation, rng, strength)
    task[validation.contracts.FIELD_START_HOLD_ENABLED] = True
    task[validation.contracts.FIELD_START_HOLD_SEC] = DEFAULT_START_HOLD_SEC
    task[validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS] = True
    return _with_final_hold(task, base_task, variation, rng, strength)


def _with_final_hold(
    task: dict[str, Any],
    base_task: Mapping[str, Any],
    variation: Mapping[str, Any],
    rng: np.random.Generator,
    strength: float,
) -> dict[str, Any]:
    """Add explicit final-hold fields from variation, base task, or project defaults."""
    if validation.contracts.FIELD_FINAL_HOLD_SEC in variation or "final_hold_range_sec" in variation:
        hold_sec = _sample_range(
            rng,
            variation.get("final_hold_range_sec", variation.get(validation.contracts.FIELD_FINAL_HOLD_SEC)),
            default=(0.75, 1.0),
            anchor=float(base_task.get(validation.contracts.FIELD_FINAL_HOLD_SEC, DEFAULT_FINAL_HOLD_SEC)),
            strength=strength,
        )
        task[validation.contracts.FIELD_FINAL_HOLD_ENABLED] = True
        task[validation.contracts.FIELD_FINAL_HOLD_SEC] = _round(hold_sec)
        task[validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS] = True
        return _with_policy_metadata(task, base_task, variation)
    copied = False
    for key in (
        validation.contracts.FIELD_FINAL_HOLD_ENABLED,
        validation.contracts.FIELD_FINAL_HOLD_SEC,
        validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
    ):
        if key in base_task:
            task[key] = copy.deepcopy(base_task[key])
            copied = True
    if copied:
        task.setdefault(
            validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS, bool(task.get(validation.contracts.FIELD_FINAL_HOLD_ENABLED, False))
        )
        return _with_policy_metadata(task, base_task, variation)
    task[validation.contracts.FIELD_FINAL_HOLD_ENABLED] = True
    task[validation.contracts.FIELD_FINAL_HOLD_SEC] = DEFAULT_FINAL_HOLD_SEC
    task[validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS] = True
    return _with_policy_metadata(task, base_task, variation)


def _task_initial_reference_height(task: Mapping[str, Any]) -> float:
    """Return the first reference height encoded by a task mapping."""
    if validation.contracts.FIELD_POSITION in task:
        return float(np.asarray(task[validation.contracts.FIELD_POSITION], dtype=float)[2])
    if validation.contracts.FIELD_START in task:
        return float(np.asarray(task[validation.contracts.FIELD_START], dtype=float)[2])
    if validation.contracts.FIELD_POINTS in task:
        return float(np.asarray(task[validation.contracts.FIELD_POINTS], dtype=float)[0, 2])
    if validation.contracts.FIELD_START_HEIGHT in task:
        return float(task[validation.contracts.FIELD_START_HEIGHT])
    if validation.contracts.FIELD_HEIGHT in task:
        return float(task[validation.contracts.FIELD_HEIGHT])
    return float(_base_position(task)[2])


def _with_policy_metadata(task: dict[str, Any], base_task: Mapping[str, Any], variation: Mapping[str, Any]) -> dict[str, Any]:
    """Attach standard-height and start-hold reward policy metadata to sampled tasks."""
    start_height = _task_initial_reference_height(task)
    task.setdefault("standard_reference_height_enabled", True)
    task.setdefault("start_height_policy", _standard_policy_value(base_task.get("start_height_policy"), STANDARD_REFERENCE_HEIGHT_POLICY))
    range_source = variation.get("base_z_range_m", variation.get("start_z_range_m", variation.get("z_range_m")))
    range_metadata = _range_metadata(range_source)
    if range_metadata is None:
        range_metadata = [float(STANDARD_REFERENCE_HEIGHT_RANGE_M[0]), float(STANDARD_REFERENCE_HEIGHT_RANGE_M[1])]
    task.setdefault("base_z_range_m", range_metadata)
    task.setdefault("base_z_m", _round(start_height))
    task.setdefault("sampled_start_height_m", _round(start_height))
    task.setdefault("height_variation_enabled", _height_variation_enabled(variation))
    task.setdefault("start_hold_policy", _standard_policy_value(base_task.get("start_hold_policy"), STANDARD_START_HOLD_POLICY))
    task.setdefault(
        "start_hold_reward_policy",
        _standard_policy_value(base_task.get("start_hold_reward_policy"), STANDARD_START_HOLD_REWARD_POLICY),
    )
    task.setdefault("tracking_reward_starts_after_start_hold", bool(base_task.get("tracking_reward_starts_after_start_hold", False)))
    return task


def _standard_policy_value(value: Any, default: str) -> str:
    """Return a current policy string, replacing stale lower-start policy names."""
    if not isinstance(value, str):
        return default
    stale_markers = ("lower_start", "lower_reference", "adjusted_lower_reference", "short_lower_start_hold", "1p2s", "1p8s", "2p5s")
    if any(marker in value for marker in stale_markers):
        return default
    return value


def _range_metadata(raw_range: Any) -> list[float] | None:
    """Return a JSON-ready numeric range when one is configured."""
    if raw_range is None or isinstance(raw_range, str):
        return None
    if isinstance(raw_range, (int, float)):
        value = _round(float(raw_range))
        return [value, value]
    if not isinstance(raw_range, Sequence) or len(raw_range) != RANGE_PAIR_LENGTH:
        return None
    return [_round(float(raw_range[0])), _round(float(raw_range[1]))]


def _height_variation_enabled(variation: Mapping[str, Any]) -> bool:
    """Return whether variation settings can change reference height."""
    return any(
        key in variation
        for key in (
            "base_z_range_m",
            "base_z_offset_range_m",
            "z_range_m",
            "start_z_range_m",
            "end_z_range_m",
            "height_offset_range_m",
            "height_delta_range_m",
            "final_z_offset_range_m",
            "z_offset_range_m",
        )
    )


def _base_line_points(task: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return base start/end points inferred from any supported task shape."""
    if validation.contracts.FIELD_START in task and validation.contracts.FIELD_END in task:
        return np.asarray(task[validation.contracts.FIELD_START], dtype=float), np.asarray(task[validation.contracts.FIELD_END], dtype=float)
    if validation.contracts.FIELD_POINTS in task:
        points = np.asarray(task[validation.contracts.FIELD_POINTS], dtype=float)
        return np.array(points[0], dtype=float), np.array(points[-1], dtype=float)
    position = _base_position(task)
    return position, np.array([position[0] + 0.4, position[1], position[2]], dtype=float)


def _base_position(task: Mapping[str, Any]) -> np.ndarray:
    """Return a representative XYZ position for a task mapping."""
    if validation.contracts.FIELD_POSITION in task:
        return np.asarray(task[validation.contracts.FIELD_POSITION], dtype=float)
    if task.get(validation.contracts.FIELD_SHAPE) == validation.contracts.SHAPE_BASIC_TRAINING_SHOW:
        segments = task.get(validation.contracts.FIELD_SEGMENTS, [])
        if isinstance(segments, Sequence) and not isinstance(segments, str) and segments:
            first_segment = segments[0]
            if isinstance(first_segment, Mapping) and validation.contracts.FIELD_SEGMENT_START in first_segment:
                return np.asarray(first_segment[validation.contracts.FIELD_SEGMENT_START], dtype=float)
    if validation.contracts.FIELD_START in task:
        return np.asarray(task[validation.contracts.FIELD_START], dtype=float)
    if validation.contracts.FIELD_POINTS in task:
        points = np.asarray(task[validation.contracts.FIELD_POINTS], dtype=float)
        return np.array(points[0], dtype=float)
    if validation.contracts.FIELD_XY in task and validation.contracts.FIELD_END_HEIGHT in task:
        xy = np.asarray(task[validation.contracts.FIELD_XY], dtype=float)
        return np.array([xy[0], xy[1], float(task[validation.contracts.FIELD_END_HEIGHT])], dtype=float)
    if validation.contracts.FIELD_HEIGHT in task:
        center = np.asarray(task.get(validation.contracts.FIELD_CENTER, [0.0, 0.0]), dtype=float)
        return np.array([center[0], center[1], float(task[validation.contracts.FIELD_HEIGHT])], dtype=float)
    return np.array([0.0, 0.0, DEFAULT_Z_M], dtype=float)


def _family_from_task_shape(shape: str) -> str:
    """Map an existing task shape to the closest task-distribution family."""
    if shape in {validation.contracts.SHAPE_HOVER, validation.contracts.SHAPE_HOVER_STABILIZATION, validation.contracts.SHAPE_NEARBY_TARGET_HOVER}:
        return FAMILY_HOVER
    if shape == validation.contracts.SHAPE_VERTICAL:
        return FAMILY_TAKEOFF
    if shape in {validation.contracts.SHAPE_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE}:
        return FAMILY_LINE
    if shape == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE:
        return FAMILY_START_HOLD_LINE
    if shape == validation.contracts.SHAPE_POLYLINE:
        return FAMILY_POLYLINE
    if shape == validation.contracts.SHAPE_CIRCLE:
        return FAMILY_CIRCLE
    if shape == validation.contracts.SHAPE_ELLIPSE:
        return FAMILY_ELLIPSE
    if shape == validation.contracts.SHAPE_FIGURE_EIGHT:
        return FAMILY_FIGURE_EIGHT
    if shape == validation.contracts.SHAPE_BASIC_TRAINING_SHOW:
        return FAMILY_BASIC_TRAINING_SHOW
    message = f"unsupported base task shape for task distribution: {shape}"
    raise ValueError(message)


def _validation_limits_from_mapping(raw: Any) -> validation.tasks.ValidationLimits | None:
    """Build optional validation limits from a mapping."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        message = "validation_limits must be a mapping"
        raise TypeError(message)
    return validation.tasks.ValidationLimits(**dict(raw))


def _required_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a required mapping as a copied dict."""
    if not isinstance(value, Mapping):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _optional_nested_mapping(value: Any, field_name: str) -> dict[str, dict[str, Any]]:
    """Return an optional nested mapping copied into plain dictionaries."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    nested: dict[str, dict[str, Any]] = {}
    for key, nested_value in value.items():
        if not isinstance(nested_value, Mapping):
            message = f"{field_name}.{key} must be a mapping"
            raise TypeError(message)
        nested[str(key)] = copy.deepcopy(dict(nested_value))
    return nested


def _optional_float_mapping(value: Any, field_name: str) -> dict[str, float]:
    """Return an optional mapping of finite float values."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    output: dict[str, float] = {}
    for key, raw_value in value.items():
        resolved = float(raw_value)
        if not np.isfinite(resolved):
            message = f"{field_name}.{key} must be finite"
            raise ValueError(message)
        output[str(key)] = resolved
    return output


def _optional_bool(value: Any, field_name: str) -> bool:
    """Return a boolean value without treating strings as truthy."""
    if not isinstance(value, bool):
        message = f"{field_name} must be a boolean"
        raise TypeError(message)
    return value


def _sample_range(
    rng: np.random.Generator,
    raw_range: Any,
    *,
    default: tuple[float, float],
    anchor: float,
    strength: float,
) -> float:
    """Sample a finite numeric range and blend toward an anchor by strength."""
    low, high = _coerce_range(raw_range, default)
    raw = float(rng.uniform(low, high))
    value = float(anchor) + float(strength) * (raw - float(anchor))
    return float(np.clip(value, min(low, high), max(low, high)))


def _coerce_range(raw_range: Any, default: tuple[float, float]) -> tuple[float, float]:
    """Return a finite low/high pair."""
    if raw_range is None:
        return default
    if isinstance(raw_range, (int, float)):
        value = float(raw_range)
        return value, value
    if isinstance(raw_range, str) or not isinstance(raw_range, Sequence) or len(raw_range) != RANGE_PAIR_LENGTH:
        message = "range values must be a two-item numeric sequence"
        raise ValueError(message)
    low = float(raw_range[0])
    high = float(raw_range[1])
    if not np.isfinite(low) or not np.isfinite(high):
        message = "range bounds must be finite"
        raise ValueError(message)
    if high < low:
        message = "range upper bound must be greater than or equal to lower bound"
        raise ValueError(message)
    return low, high


def _sample_xy_offset(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Sample a uniform XY offset inside a disk."""
    if radius <= 0.0:
        return np.zeros(2, dtype=float)
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    magnitude = float(radius) * math.sqrt(float(rng.uniform(0.0, 1.0)))
    return np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)], dtype=float)


def _base_heading(start: np.ndarray, end: np.ndarray) -> float:
    """Return horizontal heading from start to end or zero for degenerate points."""
    delta = end[:2] - start[:2]
    if float(np.linalg.norm(delta)) <= 0.0:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def _duration(task: Mapping[str, Any], default: float = DEFAULT_DURATION_SEC) -> float:
    """Return task duration or a default."""
    if validation.contracts.FIELD_DURATION_SEC in task:
        return float(task[validation.contracts.FIELD_DURATION_SEC])
    if validation.contracts.FIELD_MOVE_DURATION_SEC in task and validation.contracts.FIELD_HOLD_DURATION_SEC in task:
        return float(task[validation.contracts.FIELD_MOVE_DURATION_SEC]) + float(task[validation.contracts.FIELD_HOLD_DURATION_SEC])
    return float(default)


def _sample_rate(task: Mapping[str, Any]) -> float:
    """Return task sample rate or a conservative default."""
    return float(task.get(validation.contracts.FIELD_SAMPLE_RATE_HZ, DEFAULT_SAMPLE_RATE_HZ))


def _float_value(value: Any) -> float:
    """Return a finite float, using zero for omitted values."""
    if value is None:
        return 0.0
    resolved = float(value)
    if not np.isfinite(resolved):
        message = "numeric variation value must be finite"
        raise ValueError(message)
    return resolved


def _round(value: float) -> float:
    """Round sampled scalars to stable config precision."""
    return float(round(float(value), 6))


def _vector(values: Any) -> list[float]:
    """Return a rounded XYZ vector list."""
    return [_round(float(value)) for value in values]


def _xyz(x: float, y: float, z: float) -> list[float]:
    """Return a rounded XYZ list."""
    return [_round(x), _round(y), _round(z)]


def _xy(x: float, y: float) -> list[float]:
    """Return a rounded XY list."""
    return [_round(x), _round(y)]


__all__ = [
    "DISTRIBUTION_CONFIG_KEY",
    "FAMILY_BASIC_TRAINING_SHOW",
    "MODE_FIXED",
    "MODE_RANDOMIZED",
    "SampledTask",
    "TaskDistributionSampler",
    "TaskDistributionSettings",
    "load_task_distribution_settings",
    "normalize_fixed_task_to_distribution",
    "sample_task",
    "supported_task_families",
    "task_distribution_with_base_task",
    "unsupported_requested_task_families",
]
