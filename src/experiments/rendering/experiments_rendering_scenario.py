"""
===============================================================================
experiments_rendering_scenario.py
===============================================================================
Render continuous multi-phase evaluation scenarios as one simulator rollout.

Responsibilities:
  - Load scenario configs that explicitly order concrete local phase geometry
  - Compose local phase references into one continuous global trajectory
  - Run PPO or scripted-reference controllers once against the combined reference
  - Write scenario GIF, trace, plots, and manifest artifacts under evaluation runs

Design principles:
  - Treat task configs as primitive/default catalogs, not implicit scenario playback
  - Preserve simulator continuity by resetting only once per scenario render
  - Reuse trained-policy render helpers for camera, overlay, trace, and plot behavior

Boundaries:
  - PPO training behavior belongs in experiments_ppo_tracking.py
  - Single-task render compatibility belongs in experiments_policy_render.py
  - Storage layout, Docker scripts, W&B, and notebooks are not changed here
===============================================================================

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src import envs, evaluation, trajectories, utils, validation
from src.experiments import experiments_config as config_loader

from . import experiments_rendering_policy as policy_render

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_SCENARIO_CONFIG_PATH = Path("configs/evaluation/scenarios/show_easy.yaml")
DEFAULT_TASK_CONFIG_PATH = Path("configs/evaluation/scenario_task_catalog.yaml")
DEFAULT_MAX_STEPS: int | None = None
DEFAULT_MAX_STEP_SAFETY_MARGIN = 5
DEFAULT_BASE_EPISODE_TIME_MARGIN_SEC = 1.0
DEFAULT_SEED = 0
DEFAULT_GIF_FILENAME = "scenario_rollout.gif"
DEFAULT_MANIFEST_FILENAME = "scenario_render_manifest.json"
DEFAULT_TRACE_FILENAME = "scenario_rollout_trace.jsonl"
DEFAULT_PHASE_SAMPLE_RATE_HZ = 10.0
DEFAULT_PHASE_Z_M = 1.0
DEFAULT_START_HOLD_SEC = 1.0
DEFAULT_FINAL_HOLD_SEC = 0.0
SCRIPTED_REFERENCE_SCENARIO_PREFIX = "scripted_reference_"
PPO_SCENARIO_PREFIX = "ppo_"
START_HOLD_NAME = "start_hold"
FINAL_HOLD_NAME = "final_hold"
NORMAL_TERMINATION_REASONS = {
    "tracking_reference_complete",
    "terminated_reference_complete",
    "terminated_reference_complete_and_requested_max_steps",
}
CONTINUITY_TOLERANCE_M = 1.0e-9
START_MODE_INITIAL = "initial"
START_MODE_PREVIOUS_END = "previous_end"
SUPPORTED_START_MODES = (START_MODE_INITIAL, START_MODE_PREVIOUS_END)
PHASE_TYPE_HOVER = "hover"
PHASE_TYPE_LINE = "line"
PHASE_TYPE_VERTICAL = "vertical"
PHASE_TYPE_CIRCLE = "circle"
PHASE_TYPE_ELLIPSE = "ellipse"
PHASE_TYPE_FIGURE_EIGHT = "figure_eight"
PHASE_TYPE_POLYLINE = "polyline"
SUPPORTED_PHASE_TYPES = (
    PHASE_TYPE_HOVER,
    PHASE_TYPE_LINE,
    PHASE_TYPE_VERTICAL,
    PHASE_TYPE_CIRCLE,
    PHASE_TYPE_ELLIPSE,
    PHASE_TYPE_FIGURE_EIGHT,
    PHASE_TYPE_POLYLINE,
)
SCENARIO_TASK_SHAPE = "scenario"
SCENARIO_EVALUATION_TYPE = "scenario"
XYZ_DIMENSIONS = 3
XY_DIMENSIONS = 2
POINT_ARRAY_NDIM = 2
MIN_WAYPOINT_ROWS = 2


@dataclass(frozen=True, init=False)
class ScenarioPhase:
    """
    One ordered phase from a scenario configuration.

    Parameters
    ----------
    name
        Stable phase name used in traces and manifests.
    phase_type
        Concrete primitive type: ``hover``, ``line``, ``vertical``, ``circle``, ``ellipse``, ``figure_eight``, or ``polyline``.
    task_shape
        Alias for ``phase_type`` used by existing task-shape-oriented callers.
    duration_sec
        Optional phase duration. Falls back to the task-catalog primitive duration.
    start_mode
        Phase placement mode. ``previous_end`` maps the local phase start to the previous endpoint.
    sample_rate_hz
        Optional phase sample rate. Falls back to the matching task-catalog primitive sample rate.
    hold_after_sec
        Optional stationary hold appended at this phase endpoint before the next segment.
    z
        Optional base local height in meters. Falls back to task-catalog primitive height/start Z.
    geometry
        Primitive-specific local geometry fields copied from the scenario config.

    """

    name: str
    phase_type: str
    duration_sec: float | None
    start_mode: str
    sample_rate_hz: float | None
    hold_after_sec: float
    z: float | None
    geometry: Mapping[str, Any]

    def __init__(
        self,
        name: str,
        phase_type: str | None = None,
        task_shape: str | None = None,
        duration_sec: float | None = None,
        start_mode: str = START_MODE_PREVIOUS_END,
        sample_rate_hz: float | None = None,
        hold_after_sec: float = 0.0,
        z: float | None = None,
        geometry: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize and validate a scenario phase."""
        resolved_type = phase_type if phase_type is not None else task_shape
        if resolved_type is None:
            message = "phase type or task_shape must be provided"
            raise ValueError(message)
        name_text = str(name)
        phase_type_text = str(resolved_type).strip().lower()
        if not name_text.strip():
            message = "phase name must be non-empty"
            raise ValueError(message)
        if phase_type_text not in SUPPORTED_PHASE_TYPES:
            message = f"phase type must be one of: {', '.join(SUPPORTED_PHASE_TYPES)}"
            raise ValueError(message)
        if duration_sec is not None and duration_sec <= 0.0:
            message = "phase duration_sec must be positive when provided"
            raise ValueError(message)
        if sample_rate_hz is not None and sample_rate_hz <= 0.0:
            message = "phase sample_rate_hz must be positive when provided"
            raise ValueError(message)
        if not np.isfinite(float(hold_after_sec)) or float(hold_after_sec) < 0.0:
            message = "phase hold_after_sec must be finite and nonnegative"
            raise ValueError(message)
        if z is not None and not np.isfinite(float(z)):
            message = "phase z must be finite when provided"
            raise ValueError(message)
        if start_mode not in SUPPORTED_START_MODES:
            message = f"phase start_mode must be one of: {', '.join(SUPPORTED_START_MODES)}"
            raise ValueError(message)

        object.__setattr__(self, "name", name_text)
        object.__setattr__(self, "phase_type", phase_type_text)
        object.__setattr__(self, "duration_sec", None if duration_sec is None else float(duration_sec))
        object.__setattr__(self, "start_mode", start_mode)
        object.__setattr__(self, "sample_rate_hz", None if sample_rate_hz is None else float(sample_rate_hz))
        object.__setattr__(self, "hold_after_sec", float(hold_after_sec))
        object.__setattr__(self, "z", None if z is None else float(z))
        object.__setattr__(self, "geometry", MappingProxyType(dict(geometry or {})))

    @property
    def task_shape(self) -> str:
        """Return the phase type as a task-shape-compatible alias."""
        return self.phase_type


@dataclass(frozen=True)
class ScenarioRenderSettings:
    """
    Settings for continuous scenario rollout rendering.

    Parameters
    ----------
    scenario_config_path
        YAML scenario file defining ordered phases and renderer defaults.
    scenario_name
        Stable scenario name. Defaults to the value in the scenario config.
    task_config_path
        Task catalog YAML path containing primitive defaults and validation limits.
    phases
        Ordered scenario phases. Defaults to the phases in the scenario config.
    controller
        ``ppo`` or ``scripted_reference`` controller for the rollout.
    model_run_name
        Optional training run name used for metadata and the default PPO model lookup.
    model_path
        Optional explicit PPO zip path. When provided, it overrides the model-run-name lookup.
    run_name
        Optional evaluation run name. Derived from controller and scenario when omitted.
    output_dir
        Optional direct output directory override for tests or custom workflows.
    max_steps
        Optional upper safety cap for the continuous rollout. When omitted, the cap is derived from the composed reference length.
    seed
        Deterministic simulator seed.
    camera_mode
        External camera mode from the policy renderer.
    camera_distance
        External camera distance in meters.
    camera_yaw
        External camera yaw in degrees.
    camera_pitch
        External camera pitch in degrees.
    gif_filename
        GIF filename written under the scenario renders directory.
    manifest_filename
        Manifest filename written under the scenario manifests directory.
    frame_interval
        Number of environment steps between captured frames.
    image_width
        Captured frame width in pixels.
    image_height
        Captured frame height in pixels.
    start_hold_sec
        Optional initial stationary reference duration prepended before the first phase.
    final_hold_sec
        Optional final stationary reference duration appended after the last phase.
    normalize_actions
        Whether to wrap the scenario action space the same way training did.
    action_interface
        PPO-facing action interface used by the evaluated model.
    rpm_delta_scale
        RPM delta scale used by direct-RPM policies.
    include_dynamics_observation
        Whether the scenario env includes dynamics observations expected by training.
    include_previous_action
        Whether the scenario env includes previous-action observations expected by training.
    initial_state
        Initial drone spawn-position policy used by the scenario environment.
    source_manifest_path
        Optional direct run or curriculum root manifest path used for diagnostics.
    training_config_path
        Optional training config path used for diagnostics.
    final_stage_manifest_path
        Optional curriculum final-stage manifest path used for diagnostics.
    evaluated_model_source
        Label describing whether the evaluated checkpoint came from best or last.

    """

    scenario_config_path: Path = DEFAULT_SCENARIO_CONFIG_PATH
    scenario_name: str | None = None
    task_config_path: Path = DEFAULT_TASK_CONFIG_PATH
    phases: tuple[ScenarioPhase, ...] = ()
    controller: str = policy_render.SCRIPTED_REFERENCE_CONTROLLER
    model_run_name: str | None = None
    model_path: Path | None = None
    run_name: str | None = None
    output_dir: Path | None = None
    max_steps: int | None = DEFAULT_MAX_STEPS
    seed: int | None = DEFAULT_SEED
    camera_mode: str = policy_render.DEFAULT_CAMERA_MODE
    camera_distance: float = policy_render.DEFAULT_CAMERA_DISTANCE_M
    camera_yaw: float = policy_render.DEFAULT_CAMERA_YAW_DEG
    camera_pitch: float = policy_render.DEFAULT_CAMERA_PITCH_DEG
    gif_filename: str = DEFAULT_GIF_FILENAME
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME
    frame_interval: int = policy_render.DEFAULT_FRAME_INTERVAL
    image_width: int = policy_render.DEFAULT_IMAGE_WIDTH
    image_height: int = policy_render.DEFAULT_IMAGE_HEIGHT
    start_hold_sec: float = DEFAULT_START_HOLD_SEC
    final_hold_sec: float = DEFAULT_FINAL_HOLD_SEC
    normalize_actions: bool = True
    action_interface: str = "pid_position"
    rpm_delta_scale: float = 0.05
    pid_target_z_min_m: float = envs.actions.DEFAULT_PID_TARGET_Z_MIN_M
    pid_target_z_max_m: float = envs.actions.DEFAULT_PID_TARGET_Z_MAX_M
    include_dynamics_observation: bool = False
    include_previous_action: bool = False
    initial_state: envs.initial_state.InitialStateConfig | Mapping[str, Any] | str | None = None
    source_manifest_path: Path | None = None
    training_config_path: Path | None = None
    final_stage_manifest_path: Path | None = None
    evaluated_model_source: str | None = None

    def __post_init__(self) -> None:
        """Validate scenario-render settings."""
        if self.scenario_name is not None and not self.scenario_name.strip():
            message = "scenario_name must be non-empty when provided"
            raise ValueError(message)
        if self.controller not in policy_render.SUPPORTED_CONTROLLERS:
            message = f"controller must be one of: {', '.join(policy_render.SUPPORTED_CONTROLLERS)}"
            raise ValueError(message)
        action_config = envs.actions.ActionInterfaceConfig(
            action_interface=self.action_interface,
            rpm_delta_scale=self.rpm_delta_scale,
            pid_target_z_min_m=self.pid_target_z_min_m,
            pid_target_z_max_m=self.pid_target_z_max_m,
            include_dynamics_observation=self.include_dynamics_observation,
            include_previous_action=self.include_previous_action,
        )
        object.__setattr__(self, "action_interface", action_config.parsed_action_interface.value)
        object.__setattr__(self, "rpm_delta_scale", action_config.rpm_delta_scale)
        object.__setattr__(self, "pid_target_z_min_m", action_config.pid_target_z_min_m)
        object.__setattr__(self, "pid_target_z_max_m", action_config.pid_target_z_max_m)
        object.__setattr__(self, "include_dynamics_observation", action_config.include_dynamics_observation)
        object.__setattr__(self, "include_previous_action", action_config.include_previous_action)
        object.__setattr__(self, "initial_state", envs.initial_state.parse_initial_state_config(self.initial_state))
        if self.controller == policy_render.PPO_CONTROLLER and not self.model_run_name and self.model_path is None:
            message = "PPO scenario rendering requires model_run_name or model_path"
            raise ValueError(message)
        if self.model_run_name is not None:
            utils.artifacts.get_run_dir(self.model_run_name)
        if self.model_path is not None:
            self.model_path.expanduser().resolve(strict=False)
        if self.run_name is not None:
            utils.artifacts.get_run_evaluation_dir(self.run_name, "scenario")
        if self.max_steps is not None and self.max_steps <= 0:
            message = "max_steps must be positive when provided"
            raise ValueError(message)
        if self.seed is not None and self.seed < 0:
            message = "seed must be nonnegative"
            raise ValueError(message)
        if self.camera_mode not in policy_render.SUPPORTED_CAMERA_MODES:
            message = f"camera_mode must be one of: {', '.join(policy_render.SUPPORTED_CAMERA_MODES)}"
            raise ValueError(message)
        if not np.isfinite(self.camera_distance) or self.camera_distance <= 0.0:
            message = "camera_distance must be finite and positive"
            raise ValueError(message)
        if not np.isfinite(self.camera_yaw):
            message = "camera_yaw must be finite"
            raise ValueError(message)
        if not np.isfinite(self.camera_pitch):
            message = "camera_pitch must be finite"
            raise ValueError(message)
        if self.start_hold_sec < 0.0 or not np.isfinite(self.start_hold_sec):
            message = "start_hold_sec must be finite and nonnegative"
            raise ValueError(message)
        if self.final_hold_sec < 0.0 or not np.isfinite(self.final_hold_sec):
            message = "final_hold_sec must be finite and nonnegative"
            raise ValueError(message)
        if self.frame_interval <= 0:
            message = "frame_interval must be positive"
            raise ValueError(message)
        if self.image_width <= 0 or self.image_height <= 0:
            message = "image dimensions must be positive"
            raise ValueError(message)
        if not self.gif_filename.endswith(".gif"):
            message = "gif_filename must end with .gif"
            raise ValueError(message)
        if not self.manifest_filename.endswith(".json"):
            message = "manifest_filename must end with .json"
            raise ValueError(message)


def _settings_initial_state(settings: ScenarioRenderSettings) -> envs.initial_state.InitialStateConfig:
    """Return the normalized initial-state config from scenario settings."""
    return cast("envs.initial_state.InitialStateConfig", settings.initial_state)


@dataclass(frozen=True)
class ScenarioComposition:
    """
    Combined scenario reference and phase metadata.

    Parameters
    ----------
    reference
        Environment-ready reference for one continuous scenario trajectory.
    task
        Scenario metadata task passed to visual overlay helpers.
    phases
        Ordered phase definitions used to build the reference.
    phase_step_ranges
        Per-phase ``[start, end)`` step ranges in the combined reference.
    phase_time_ranges
        Per-phase ``[start, end]`` time ranges in seconds.
    phase_start_positions
        Per-phase shifted XYZ start positions.
    phase_end_positions
        Per-phase shifted XYZ end positions.
    phase_offsets
        Per-phase XYZ offsets applied to local primitive positions.
    phase_geometry
        Manifest-ready local geometry and global endpoints for each phase.
    start_hold_sec
        Initial stationary hold duration prepended before the first phase.
    start_hold_steps
        Number of reference samples in the initial start hold.
    final_hold_sec
        Final stationary hold duration appended after the last phase.
    reference_motion_steps
        Number of non-hold phase motion samples.
    phase_hold_steps
        Number of explicit per-phase hold samples.
    final_hold_steps
        Number of reference samples in the final hold.
    total_reference_steps
        Total composed reference sample count including final hold.
    phase_hold_step_ranges
        Per-phase hold ``[start, end)`` sample ranges with phase identifiers.
    phase_hold_time_ranges
        Per-phase hold ``[start, end]`` time ranges with phase identifiers.
    start_hold_step_range
        Optional ``[start, end)`` sample range for the initial start hold.
    start_hold_time_range
        Optional ``[start, end]`` time range for the initial start hold.
    final_hold_step_range
        Optional ``[start, end)`` sample range for the final hold.
    final_hold_time_range
        Optional ``[start, end]`` time range for the final hold.

    """

    reference: envs.task_adapter.EnvironmentTaskReference
    task: dict[str, Any]
    phases: tuple[ScenarioPhase, ...]
    phase_step_ranges: tuple[dict[str, int], ...]
    phase_time_ranges: tuple[dict[str, float], ...]
    phase_start_positions: tuple[list[float], ...]
    phase_end_positions: tuple[list[float], ...]
    phase_offsets: tuple[list[float], ...]
    phase_geometry: tuple[dict[str, Any], ...]
    start_hold_sec: float
    start_hold_steps: int
    start_hold_step_range: dict[str, int] | None
    start_hold_time_range: dict[str, float] | None
    final_hold_sec: float
    scenario_duration_sec: float
    reference_motion_steps: int
    reference_motion_end_step: int
    phase_hold_steps: int
    phase_hold_end_step: int
    phase_hold_step_ranges: tuple[dict[str, Any], ...]
    phase_hold_time_ranges: tuple[dict[str, Any], ...]
    final_hold_steps: int
    total_reference_steps: int
    final_hold_step_range: dict[str, int] | None
    final_hold_time_range: dict[str, float] | None


@dataclass(frozen=True)
class _PhaseBuildResult:
    """Local phase trajectory plus metadata used during scenario composition."""

    trajectory: trajectories.primitives.Trajectory
    task: dict[str, Any]
    geometry: dict[str, Any]
    waypoint_positions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ScenarioRenderResult:
    """
    Summary returned by a continuous scenario render run.

    Parameters
    ----------
    gif_path
        Path to the written scenario GIF.
    manifest_path
        Path to the written scenario manifest.
    metrics
        JSON-serializable scenario rollout metrics.
    warnings
        Nonfatal warning strings generated during the run.

    """

    gif_path: str
    manifest_path: str
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()


def load_scenario_render_settings(path: str | Path) -> ScenarioRenderSettings:
    """
    Load scenario-render settings from a YAML scenario config.

    Parameters
    ----------
    path
        Scenario YAML path.

    Returns
    -------
    ScenarioRenderSettings
        Validated settings populated from the scenario file.

    """
    config_path = Path(path)
    config = config_loader.load_experiment_config(config_path)
    return _settings_from_mapping(config=config, scenario_config_path=config_path)


def run_scenario_render(settings: ScenarioRenderSettings | None = None) -> ScenarioRenderResult:
    """
    Render a continuous multi-phase scenario as one simulator rollout.

    Parameters
    ----------
    settings
        Optional scenario-render settings. Defaults load the scripted reference scenario.

    Returns
    -------
    ScenarioRenderResult
        Paths and metrics for the generated scenario artifacts.

    Raises
    ------
    FileNotFoundError
        If a PPO scenario model cannot be found under the training-run model directory.
    RuntimeError
        If simulator rendering, rollout collection, or artifact writing fails.

    """
    active_settings = settings or load_scenario_render_settings(DEFAULT_SCENARIO_CONFIG_PATH)
    scenario_name = _scenario_name(active_settings)
    evaluation_run_name = _evaluation_run_name(active_settings)
    model_path = _resolve_model_path(active_settings)
    training_metadata, metadata_warnings = policy_render._load_training_metadata(active_settings.model_run_name)  # noqa: SLF001
    training_task_shape = policy_render._training_task_shape(training_metadata)  # noqa: SLF001

    if active_settings.controller == policy_render.PPO_CONTROLLER and model_path is not None and not model_path.exists():
        message = (
            "trained PPO model was not found at "
            f"{model_path}. Scenario PPO rendering only loads models from "
            "storage/runs/<model_run_name>/training/models/."
        )
        raise FileNotFoundError(message)

    composition = compose_scenario_reference(active_settings)
    effective_max_steps = _effective_max_steps(active_settings=active_settings, composition=composition)
    base_time_limit_sec = _base_time_limit_sec(composition)
    policy_settings = _policy_settings(
        active_settings=active_settings,
        model_path=model_path,
        evaluation_run_name=evaluation_run_name,
        effective_max_steps=effective_max_steps,
    )
    output_dir = _resolve_output_dir(active_settings.output_dir, evaluation_run_name)
    renders_dir, manifests_dir = policy_render._artifact_dirs(output_dir)  # noqa: SLF001
    traces_dir, plots_dir = policy_render._review_artifact_dirs(output_dir)  # noqa: SLF001
    renders_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    gif_path = renders_dir / active_settings.gif_filename
    manifest_path = manifests_dir / active_settings.manifest_filename
    trace_path = traces_dir / DEFAULT_TRACE_FILENAME

    model: Any | None = None
    if active_settings.controller == policy_render.PPO_CONTROLLER:
        try:
            from stable_baselines3 import PPO  # noqa: PLC0415
        except ImportError as exc:
            message = f"stable_baselines3 is required for PPO scenario rendering: {exc}"
            raise RuntimeError(message) from exc
        if model_path is None:
            message = "PPO scenario rendering requires a resolved model path"
            raise RuntimeError(message)
        model = PPO.load(str(model_path), device="cpu")

    tracking_env = _make_scenario_tracking_env(
        settings=active_settings,
        composition=composition,
        effective_max_steps=effective_max_steps,
        base_time_limit_sec=base_time_limit_sec,
    )
    try:
        rollout_payload = policy_render._run_policy_rollout(  # noqa: SLF001
            model=model,
            tracking_env=tracking_env,
            settings=policy_settings,
            seed=active_settings.seed or DEFAULT_SEED,
            task=composition.task,
            task_shape=SCENARIO_TASK_SHAPE,
        )
    finally:
        tracking_env.close()

    trace_records = _add_phase_fields(rollout_payload["trace_records"], composition)
    policy_render._write_gif(rollout_payload["frames"], gif_path, active_settings.frame_interval)  # noqa: SLF001
    trace_result = evaluation.rollout.write_policy_rollout_trace(trace_records, trace_path)
    plot_result = evaluation.plots.write_policy_rollout_trace_plots(trace_path, plots_dir)

    actual_steps = len(rollout_payload["rewards"])
    final_info = dict(rollout_payload.get("final_info", {}))
    reference_sample_count = int(composition.reference.positions.shape[0])
    termination_reason = str(
        final_info.get(
            "termination_reason",
            policy_render._termination_reason(  # noqa: SLF001
                terminated=rollout_payload["terminated"],
                truncated=rollout_payload["truncated"],
                actual_steps=actual_steps,
                requested_max_steps=effective_max_steps,
                reference_sample_count=reference_sample_count,
            ),
        )
    )
    warnings = [*metadata_warnings]
    warnings.extend(
        policy_render._rollout_warnings(  # noqa: SLF001
            actual_steps=actual_steps,
            requested_max_steps=effective_max_steps,
            terminated=bool(rollout_payload["terminated"]),
            truncated=bool(rollout_payload["truncated"]),
            termination_reason=termination_reason,
        )
    )
    actual_positions = [np.asarray(position, dtype=float) for position in rollout_payload["current_positions"]]
    reference_positions = [np.asarray(position, dtype=float) for position in rollout_payload["reference_positions"]]
    completion = _scenario_completion_summary(
        actual_steps=actual_steps,
        requested_max_steps=active_settings.max_steps,
        effective_max_steps=effective_max_steps,
        reference_sample_count=reference_sample_count,
        reference_motion_steps=composition.reference_motion_steps,
        reference_motion_end_step=composition.reference_motion_end_step,
        start_hold_steps=composition.start_hold_steps,
        phase_hold_steps=composition.phase_hold_steps,
        phase_hold_end_step=composition.phase_hold_end_step,
        final_hold_steps=composition.final_hold_steps,
        terminated=bool(rollout_payload["terminated"]),
        truncated=bool(rollout_payload["truncated"]),
        termination_reason=termination_reason,
    )
    metrics = _build_scenario_metrics(
        settings=active_settings,
        evaluation_run_name=evaluation_run_name,
        scenario_name=scenario_name,
        model_path=model_path,
        training_task_shape=training_task_shape,
        rewards=rollout_payload["rewards"],
        position_errors=rollout_payload["position_errors"],
        terminated=bool(rollout_payload["terminated"]),
        truncated=bool(rollout_payload["truncated"]),
        termination_reason=termination_reason,
        completion=completion,
        policy_predict_used=bool(rollout_payload["policy_predict_used"]),
        warnings=tuple(warnings),
        observation_check=rollout_payload.get("observation_check"),
    )
    manifest = _build_scenario_manifest(
        settings=active_settings,
        evaluation_run_name=evaluation_run_name,
        scenario_name=scenario_name,
        composition=composition,
        model_path=model_path,
        training_task_shape=training_task_shape,
        gif_path=gif_path,
        trace_path=Path(trace_result.output_path),
        plot_paths=plot_result.plot_paths,
        actual_steps=actual_steps,
        requested_max_steps=active_settings.max_steps,
        effective_max_steps=effective_max_steps,
        base_time_limit_sec=base_time_limit_sec,
        completion=completion,
        termination_reason=termination_reason,
        true_simulator_rendering=True,
        policy_predict_used=bool(rollout_payload["policy_predict_used"]),
        metrics=metrics,
        warnings=tuple(warnings),
        final_info=final_info,
        final_action=rollout_payload.get("final_action"),
        actual_positions=actual_positions,
        rollout_reference_positions=reference_positions,
        output_dir=output_dir,
        observation_check=rollout_payload.get("observation_check"),
    )
    policy_render._write_manifest(manifest_path, manifest)  # noqa: SLF001
    return ScenarioRenderResult(gif_path=str(gif_path), manifest_path=str(manifest_path), metrics=metrics, warnings=tuple(warnings))


def compose_scenario_reference(settings: ScenarioRenderSettings) -> ScenarioComposition:
    """
    Compose ordered scenario phases into one continuous environment reference.

    Parameters
    ----------
    settings
        Scenario settings containing a task catalog path and ordered concrete phases.

    Returns
    -------
    ScenarioComposition
        Combined reference trajectory plus manifest-ready phase metadata.

    Raises
    ------
    ValueError
        If phase geometry is invalid or adjacent phases are discontinuous.

    """
    phases = settings.phases or _phases_from_config(settings.scenario_config_path)
    if not phases:
        message = "scenario must define at least one phase"
        raise ValueError(message)
    task_catalog, limits = _load_task_catalog(settings.task_config_path)

    combined_times: list[np.ndarray] = []
    combined_positions: list[np.ndarray] = []
    phase_step_ranges: list[dict[str, int]] = []
    phase_time_ranges: list[dict[str, float]] = []
    phase_hold_step_ranges: list[dict[str, Any]] = []
    phase_hold_time_ranges: list[dict[str, Any]] = []
    phase_start_positions: list[list[float]] = []
    phase_end_positions: list[list[float]] = []
    phase_offsets: list[list[float]] = []
    phase_geometry: list[dict[str, Any]] = []
    waypoint_positions: list[np.ndarray] = []

    previous_end: np.ndarray | None = None
    current_time_end = 0.0
    reference_motion_steps = 0
    for phase_index, phase in enumerate(phases):
        catalog_task = _catalog_task_for_type(task_catalog=task_catalog, phase_type=phase.phase_type)
        phase_build = _build_phase_reference(phase=phase, catalog_task=catalog_task, limits=limits)
        local_positions = np.asarray(phase_build.trajectory.positions, dtype=float)
        local_times = np.asarray(phase_build.trajectory.times, dtype=float) - float(phase_build.trajectory.times[0])
        start_mode = START_MODE_INITIAL if phase_index == 0 and phase.start_mode == START_MODE_PREVIOUS_END else phase.start_mode
        offset = _phase_offset(positions=local_positions, previous_end=previous_end, start_mode=start_mode)
        shifted_positions = local_positions + offset
        adjusted_times = _phase_times_after_previous(
            local_times=local_times,
            current_time_end=current_time_end,
            is_first_phase=phase_index == 0,
        )

        step_start = sum(positions.shape[0] for positions in combined_positions)
        step_end = step_start + int(shifted_positions.shape[0])
        phase_start = shifted_positions[0]
        phase_end = shifted_positions[-1]
        phase_step_ranges.append({"start": int(step_start), "end": int(step_end)})
        phase_time_ranges.append({"start": float(adjusted_times[0]), "end": float(adjusted_times[-1])})
        phase_start_positions.append(_array_to_floats(phase_start))
        phase_end_positions.append(_array_to_floats(phase_end))
        phase_offsets.append(_array_to_floats(offset))
        phase_geometry.append(
            {
                **phase_build.geometry,
                "global_start_position": _array_to_floats(phase_start),
                "global_end_position": _array_to_floats(phase_end),
                "applied_offset": _array_to_floats(offset),
            }
        )
        waypoint_positions.extend(position + offset for position in phase_build.waypoint_positions)
        combined_times.append(adjusted_times)
        combined_positions.append(shifted_positions)
        reference_motion_steps += int(shifted_positions.shape[0])
        previous_end = np.array(phase_end, dtype=float, copy=True)
        current_time_end = float(adjusted_times[-1])

        hold_step_range, hold_time_range = _append_phase_hold(
            combined_times=combined_times,
            combined_positions=combined_positions,
            current_time_end=current_time_end,
            phase=phase,
            phase_index=phase_index,
            hold_position=previous_end,
            sample_rate_hz=_nominal_sample_rate(adjusted_times),
        )
        if hold_step_range is not None and hold_time_range is not None:
            phase_hold_step_ranges.append(hold_step_range)
            phase_hold_time_ranges.append(hold_time_range)
            current_time_end = float(hold_time_range["end"])

    final_hold_step_range, final_hold_time_range = _append_final_hold(
        combined_times=combined_times,
        combined_positions=combined_positions,
        current_time_end=current_time_end,
        final_position=previous_end,
        final_hold_sec=settings.final_hold_sec,
    )
    times = np.concatenate(combined_times)
    positions = np.vstack(combined_positions)
    start_hold = _prepend_start_hold(times=times, positions=positions, start_hold_sec=settings.start_hold_sec)
    times = start_hold["times"]
    positions = start_hold["positions"]
    start_hold_steps = int(start_hold["steps"])
    start_hold_step_range = start_hold["step_range"]
    start_hold_time_range = start_hold["time_range"]
    if start_hold_steps > 0:
        phase_step_ranges = _shift_step_ranges(phase_step_ranges, start_hold_steps)
        phase_hold_step_ranges = _shift_step_ranges(phase_hold_step_ranges, start_hold_steps)
        final_hold_step_range = _shift_optional_step_range(final_hold_step_range, start_hold_steps)
        phase_time_ranges = _shift_time_ranges(phase_time_ranges, float(start_hold["time_shift_sec"]))
        phase_hold_time_ranges = _shift_time_ranges(phase_hold_time_ranges, float(start_hold["time_shift_sec"]))
        final_hold_time_range = _shift_optional_time_range(final_hold_time_range, float(start_hold["time_shift_sec"]))
    final_hold_steps = 0 if final_hold_step_range is None else final_hold_step_range["end"] - final_hold_step_range["start"]
    phase_hold_steps = sum(int(step_range["end"] - step_range["start"]) for step_range in phase_hold_step_ranges)
    reference_motion_end_step = int(phase_step_ranges[-1]["end"])
    phase_hold_end_step = max((int(step_range["end"]) for step_range in phase_hold_step_ranges), default=start_hold_steps)
    scenario_duration_sec = float(times[-1] - times[0])
    _validate_combined_reference(times=times, positions=positions, phase_step_ranges=phase_step_ranges, limits=limits)

    scenario_task = {
        "task_type": "trajectory",
        "shape": SCENARIO_TASK_SHAPE,
        "scenario_name": _scenario_name(settings),
        "points": _unique_positions(waypoint_positions).tolist(),
        "duration_sec": scenario_duration_sec,
        "sample_rate_hz": _nominal_sample_rate(times),
        "start_hold_enabled": bool(settings.start_hold_sec > 0.0),
        "start_hold_sec": float(settings.start_hold_sec),
        "exclude_start_hold_from_tracking_metrics": bool(settings.start_hold_sec > 0.0),
        "final_hold_sec": float(settings.final_hold_sec),
    }
    final_hold_start_step = positions.shape[0] if final_hold_step_range is None else int(final_hold_step_range["start"])
    final_hold_end_time = float(times[-1] if final_hold_step_range is None else times[max(final_hold_start_step - 1, 0)])
    tracking_phase_start_step = start_hold_steps
    tracking_phase_start_time = float(times[tracking_phase_start_step]) if tracking_phase_start_step < times.shape[0] else float(times[0])
    reference = envs.task_adapter.EnvironmentTaskReference(
        task=MappingProxyType(dict(scenario_task)),
        shape=SCENARIO_TASK_SHAPE,
        times=np.array(times, dtype=float, copy=True),
        positions=np.array(positions, dtype=float, copy=True),
        validation_messages=("scenario reference composed from validated local phase geometry",),
        start_hold_enabled=bool(settings.start_hold_sec > 0.0),
        start_hold_sec=float(settings.start_hold_sec),
        exclude_start_hold_from_tracking_metrics=bool(settings.start_hold_sec > 0.0),
        tracking_phase_start_step=int(tracking_phase_start_step),
        tracking_phase_start_time_sec=tracking_phase_start_time,
        final_hold_enabled=bool(settings.final_hold_sec > 0.0),
        final_hold_sec=float(settings.final_hold_sec),
        exclude_final_hold_from_tracking_metrics=bool(settings.final_hold_sec > 0.0),
        tracking_phase_end_step=int(final_hold_start_step),
        tracking_phase_end_time_sec=final_hold_end_time,
    )
    return ScenarioComposition(
        reference=reference,
        task=scenario_task,
        phases=phases,
        phase_step_ranges=tuple(phase_step_ranges),
        phase_time_ranges=tuple(phase_time_ranges),
        phase_start_positions=tuple(phase_start_positions),
        phase_end_positions=tuple(phase_end_positions),
        phase_offsets=tuple(phase_offsets),
        phase_geometry=tuple(phase_geometry),
        start_hold_sec=float(settings.start_hold_sec),
        start_hold_steps=int(start_hold_steps),
        start_hold_step_range=start_hold_step_range,
        start_hold_time_range=start_hold_time_range,
        final_hold_sec=float(settings.final_hold_sec),
        scenario_duration_sec=scenario_duration_sec,
        reference_motion_steps=int(reference_motion_steps),
        reference_motion_end_step=reference_motion_end_step,
        phase_hold_steps=int(phase_hold_steps),
        phase_hold_end_step=int(phase_hold_end_step),
        phase_hold_step_ranges=tuple(phase_hold_step_ranges),
        phase_hold_time_ranges=tuple(phase_hold_time_ranges),
        final_hold_steps=int(final_hold_steps),
        total_reference_steps=int(positions.shape[0]),
        final_hold_step_range=final_hold_step_range,
        final_hold_time_range=final_hold_time_range,
    )


def _settings_from_mapping(config: dict[str, Any], scenario_config_path: Path) -> ScenarioRenderSettings:
    """Build scenario-render settings from a loaded YAML mapping."""
    camera = config.get("camera", {})
    if camera is None:
        camera = {}
    if not isinstance(camera, dict):
        message = "scenario camera must be a mapping when provided"
        raise TypeError(message)
    scenario_name = _required_text(config, "scenario_name")
    controller = str(config.get("controller", policy_render.SCRIPTED_REFERENCE_CONTROLLER))
    requested_max_steps = None if config.get("max_steps") is None else int(config["max_steps"])
    return ScenarioRenderSettings(
        scenario_config_path=scenario_config_path,
        scenario_name=scenario_name,
        task_config_path=Path(config.get("task_config_path", DEFAULT_TASK_CONFIG_PATH)),
        phases=_parse_phases(config.get("phases")),
        controller=controller,
        model_run_name=None if config.get("model_run_name") is None else str(config["model_run_name"]),
        run_name=None if config.get("run_name") is None else str(config["run_name"]),
        max_steps=requested_max_steps,
        seed=None if config.get("seed") is None else int(config.get("seed", DEFAULT_SEED)),
        camera_mode=str(camera.get("mode", policy_render.DEFAULT_CAMERA_MODE)),
        camera_distance=float(camera.get("distance", policy_render.DEFAULT_CAMERA_DISTANCE_M)),
        camera_yaw=float(camera.get("yaw", policy_render.DEFAULT_CAMERA_YAW_DEG)),
        camera_pitch=float(camera.get("pitch", policy_render.DEFAULT_CAMERA_PITCH_DEG)),
        start_hold_sec=float(config.get("start_hold_sec", DEFAULT_START_HOLD_SEC)),
        final_hold_sec=float(config.get("final_hold_sec", DEFAULT_FINAL_HOLD_SEC)),
        normalize_actions=bool(config.get("normalize_actions", True)),
        action_interface=str(config.get("action_interface", "pid_position")),
        rpm_delta_scale=float(config.get("rpm_delta_scale") or 0.05),
        pid_target_z_min_m=float(config.get("pid_target_z_min_m") or envs.actions.DEFAULT_PID_TARGET_Z_MIN_M),
        pid_target_z_max_m=float(config.get("pid_target_z_max_m") or envs.actions.DEFAULT_PID_TARGET_Z_MAX_M),
        include_dynamics_observation=bool(config.get("include_dynamics_observation", False)),
        include_previous_action=bool(config.get("include_previous_action", False)),
        initial_state=config.get("initial_state"),
    )


def _parse_phases(raw_phases: Any) -> tuple[ScenarioPhase, ...]:
    """Parse and validate phase mappings from config."""
    if not isinstance(raw_phases, list) or not raw_phases:
        message = "scenario phases must be a non-empty list"
        raise ValueError(message)
    phases: list[ScenarioPhase] = []
    common_keys = {"name", "type", "task_shape", "duration_sec", "start_mode", "sample_rate_hz", "hold_after_sec", "z"}
    for index, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict):
            message = f"scenario phase {index} must be a mapping"
            raise TypeError(message)
        default_start_mode = START_MODE_INITIAL if index == 0 else START_MODE_PREVIOUS_END
        geometry = {str(key): value for key, value in raw_phase.items() if key not in common_keys}
        phases.append(
            ScenarioPhase(
                name=_required_text(raw_phase, "name"),
                phase_type=None if raw_phase.get("type") is None else str(raw_phase["type"]),
                task_shape=None if raw_phase.get("task_shape") is None else str(raw_phase["task_shape"]),
                duration_sec=None if raw_phase.get("duration_sec") is None else float(raw_phase["duration_sec"]),
                start_mode=str(raw_phase.get("start_mode", default_start_mode)),
                sample_rate_hz=None if raw_phase.get("sample_rate_hz") is None else float(raw_phase["sample_rate_hz"]),
                hold_after_sec=float(raw_phase.get("hold_after_sec", 0.0)),
                z=None if raw_phase.get("z") is None else float(raw_phase["z"]),
                geometry=geometry,
            )
        )
    return tuple(phases)


def _phases_from_config(path: Path) -> tuple[ScenarioPhase, ...]:
    """Load phases from a scenario config path."""
    return load_scenario_render_settings(path).phases


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string field from a mapping."""
    value = mapping.get(key)
    if value is None or not str(value).strip():
        message = f"{key} must be a non-empty string"
        raise ValueError(message)
    return str(value)


def _load_task_catalog(path: Path) -> tuple[list[dict[str, Any]], validation.tasks.ValidationLimits | None]:
    """Load task mappings and validation limits from the configured task catalog."""
    config = config_loader.load_experiment_config(path)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        message = "task config must contain a top-level tasks list"
        raise TypeError(message)
    catalog: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            message = f"task catalog entry {index} must be a mapping"
            raise TypeError(message)
        catalog.append(dict(task))
    return catalog, _validation_limits_from_config(config.get("validation_limits"))


def _validation_limits_from_config(raw_limits: Any) -> validation.tasks.ValidationLimits | None:
    """Return validation limits from a task config mapping, when provided."""
    if raw_limits is None:
        return None
    if not isinstance(raw_limits, dict):
        message = "validation_limits must be a mapping when provided"
        raise TypeError(message)
    return validation.tasks.ValidationLimits(**raw_limits)


def _catalog_task_for_type(task_catalog: list[dict[str, Any]], phase_type: str) -> dict[str, Any] | None:
    """Return the first catalog task matching a phase type, if available."""
    requested_type = phase_type.strip().lower()
    for task in task_catalog:
        if str(task.get("shape", "")).lower() == requested_type:
            return dict(task)
    return None


def _build_phase_reference(
    phase: ScenarioPhase,
    catalog_task: dict[str, Any] | None,
    limits: validation.tasks.ValidationLimits | None,
) -> _PhaseBuildResult:
    """Build a validated local reference trajectory for one concrete phase."""
    if phase.phase_type == PHASE_TYPE_LINE:
        result = _build_line_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_HOVER:
        result = _build_hover_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_VERTICAL:
        result = _build_vertical_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_POLYLINE:
        result = _build_polyline_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_CIRCLE:
        result = _build_circle_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_ELLIPSE:
        result = _build_ellipse_phase(phase=phase, catalog_task=catalog_task)
    elif phase.phase_type == PHASE_TYPE_FIGURE_EIGHT:
        result = _build_figure_eight_phase(phase=phase, catalog_task=catalog_task)
    else:
        message = f"unsupported phase type: {phase.phase_type}"
        raise ValueError(message)

    validation_result = validation.tasks.validate_trajectory(result.trajectory, limits=limits)
    if not validation_result.is_valid:
        message = f"invalid scenario phase '{phase.name}': " + "; ".join(validation_result.messages)
        raise ValueError(message)
    return result


def _build_line_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local line phase from delta_position or end_offset."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    delta = _line_delta(phase.geometry)
    start = np.array([0.0, 0.0, z], dtype=float)
    end = start + delta
    trajectory = trajectories.primitives.make_line_trajectory(
        start=start.tolist(),
        end=end.tolist(),
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_LINE,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "start": _array_to_floats(start),
        "end": _array_to_floats(end),
    }
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=start,
        local_end=end,
        extra={"delta_position": _array_to_floats(delta)},
    )
    return _PhaseBuildResult(trajectory=trajectory, task=task, geometry=geometry, waypoint_positions=(start, end))


def _build_hover_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local hover phase."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    hold_current_position = bool(phase.geometry.get("hold_current_position", False))
    offset = np.zeros(XYZ_DIMENSIONS, dtype=float) if hold_current_position else _optional_xyz(phase.geometry, "position_offset")
    position = np.array([offset[0], offset[1], z + offset[2]], dtype=float)
    trajectory = trajectories.primitives.make_hover_trajectory(
        position=position.tolist(),
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_HOVER,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "position": _array_to_floats(position),
    }
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=position,
        local_end=position,
        extra={"hold_current_position": hold_current_position, "position_offset": _array_to_floats(offset)},
    )
    return _PhaseBuildResult(trajectory=trajectory, task=task, geometry=geometry, waypoint_positions=(position,))


def _build_vertical_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local vertical phase with configurable start and end heights."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    start_height = float(phase.geometry.get("start_height", phase.geometry.get("start_z", z)))
    end_height = float(phase.geometry.get("end_height", phase.geometry.get("end_z", start_height + float(phase.geometry.get("delta_z", 0.35)))))
    if not np.isfinite(start_height) or not np.isfinite(end_height):
        message = "vertical start and end heights must be finite"
        raise ValueError(message)
    xy_offset = _optional_xy(phase.geometry, "xy_offset")
    start = np.array([xy_offset[0], xy_offset[1], start_height], dtype=float)
    end = np.array([xy_offset[0], xy_offset[1], end_height], dtype=float)
    trajectory = trajectories.primitives.make_vertical_trajectory(
        xy=xy_offset.tolist(),
        start_height=start_height,
        end_height=end_height,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_VERTICAL,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "xy": _array_to_floats(xy_offset),
        "start_height": float(start_height),
        "end_height": float(end_height),
    }
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=start,
        local_end=end,
        extra={"xy_offset": _array_to_floats(xy_offset), "delta_z": float(end_height - start_height)},
    )
    return _PhaseBuildResult(trajectory=trajectory, task=task, geometry=geometry, waypoint_positions=(start, end))


def _build_polyline_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local polyline phase from local waypoint offsets."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    if "waypoints" not in phase.geometry:
        message = f"polyline phase '{phase.name}' requires waypoints"
        raise ValueError(message)
    waypoint_offsets = _xyz_rows(phase.geometry["waypoints"], name="waypoints")
    points = waypoint_offsets + np.array([0.0, 0.0, z], dtype=float)
    trajectory = trajectories.primitives.make_polyline_trajectory(
        points=points,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_POLYLINE,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "points": points.tolist(),
    }
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=points[0],
        local_end=points[-1],
        extra={"waypoints": waypoint_offsets.tolist()},
    )
    return _PhaseBuildResult(
        trajectory=trajectory,
        task=task,
        geometry=geometry,
        waypoint_positions=tuple(np.array(point, dtype=float, copy=True) for point in points),
    )


def _build_circle_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local XY circle phase with configurable radius and center."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    radius = _positive_float(phase.geometry.get("radius_m", _catalog_float(catalog_task, "radius", None)), "radius_m")
    plane = str(phase.geometry.get("plane", "xy")).lower()
    if plane != "xy":
        message = "circle phases currently support only plane: xy"
        raise ValueError(message)
    direction = str(phase.geometry.get("direction", "ccw")).lower()
    if direction not in {"cw", "ccw"}:
        message = "circle direction must be cw or ccw"
        raise ValueError(message)
    revolutions = _positive_float(phase.geometry.get("revolutions", 1.0), "revolutions")
    start_angle_deg = float(phase.geometry.get("start_angle_deg", 0.0))
    if not np.isfinite(start_angle_deg):
        message = "start_angle_deg must be finite"
        raise ValueError(message)
    default_center_offset = [-radius * math.cos(math.radians(start_angle_deg)), -radius * math.sin(math.radians(start_angle_deg)), 0.0]
    center_offset = _optional_xyz(phase.geometry, "center_offset", default=default_center_offset)
    center = np.array([center_offset[0], center_offset[1], z + center_offset[2]], dtype=float)
    trajectory = _make_circle_phase_trajectory(
        radius=radius,
        center=center,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        direction=direction,
        revolutions=revolutions,
        start_angle_deg=start_angle_deg,
    )
    local_start = trajectory.positions[0]
    local_end = trajectory.positions[-1]
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_CIRCLE,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "radius": radius,
        "height": float(center[2]),
        "center": _array_to_floats(center[:2]),
        "clockwise": direction == "cw",
    }
    anchor_indices = np.linspace(0, trajectory.positions.shape[0] - 1, num=5, dtype=int)
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=local_start,
        local_end=local_end,
        extra={
            "radius_m": float(radius),
            "center_offset": _array_to_floats(center_offset),
            "plane": plane,
            "direction": direction,
            "revolutions": float(revolutions),
            "start_angle_deg": float(start_angle_deg),
        },
    )
    return _PhaseBuildResult(
        trajectory=trajectory,
        task=task,
        geometry=geometry,
        waypoint_positions=tuple(np.array(trajectory.positions[index], dtype=float, copy=True) for index in anchor_indices),
    )


def _build_ellipse_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local XY ellipse phase with configurable radii and center."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    radius_x = _positive_float(phase.geometry.get("radius_x_m", _catalog_float(catalog_task, "radius_x", None)), "radius_x_m")
    radius_y = _positive_float(phase.geometry.get("radius_y_m", _catalog_float(catalog_task, "radius_y", None)), "radius_y_m")
    direction = str(phase.geometry.get("direction", "ccw")).lower()
    if direction not in {"cw", "ccw"}:
        message = "ellipse direction must be cw or ccw"
        raise ValueError(message)
    revolutions = _positive_float(phase.geometry.get("revolutions", 1.0), "revolutions")
    start_angle_deg = float(phase.geometry.get("start_angle_deg", 0.0))
    if not np.isfinite(start_angle_deg):
        message = "start_angle_deg must be finite"
        raise ValueError(message)
    default_center_offset = [-radius_x * math.cos(math.radians(start_angle_deg)), -radius_y * math.sin(math.radians(start_angle_deg)), 0.0]
    center_offset = _optional_xyz(phase.geometry, "center_offset", default=default_center_offset)
    center = np.array([center_offset[0], center_offset[1], z + center_offset[2]], dtype=float)
    trajectory = _make_ellipse_phase_trajectory(
        radius_x=radius_x,
        radius_y=radius_y,
        center=center,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        direction=direction,
        revolutions=revolutions,
        start_angle_deg=start_angle_deg,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_ELLIPSE,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "radius_x": float(radius_x),
        "radius_y": float(radius_y),
        "height": float(center[2]),
        "center": _array_to_floats(center[:2]),
        "clockwise": direction == "cw",
    }
    anchor_indices = np.linspace(0, trajectory.positions.shape[0] - 1, num=5, dtype=int)
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=trajectory.positions[0],
        local_end=trajectory.positions[-1],
        extra={
            "radius_x_m": float(radius_x),
            "radius_y_m": float(radius_y),
            "center_offset": _array_to_floats(center_offset),
            "direction": direction,
            "revolutions": float(revolutions),
            "start_angle_deg": float(start_angle_deg),
        },
    )
    return _PhaseBuildResult(
        trajectory=trajectory,
        task=task,
        geometry=geometry,
        waypoint_positions=tuple(np.array(trajectory.positions[index], dtype=float, copy=True) for index in anchor_indices),
    )


def _build_figure_eight_phase(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> _PhaseBuildResult:
    """Build a local XY figure-eight phase with configurable radii and center."""
    duration_sec = _phase_duration(phase=phase, catalog_task=catalog_task)
    sample_rate_hz = _phase_sample_rate(phase=phase, catalog_task=catalog_task)
    z = _phase_z(phase=phase, catalog_task=catalog_task)
    radius_x = _positive_float(phase.geometry.get("radius_x_m", _catalog_float(catalog_task, "radius_x", None)), "radius_x_m")
    radius_y = _positive_float(phase.geometry.get("radius_y_m", _catalog_float(catalog_task, "radius_y", None)), "radius_y_m")
    direction = str(phase.geometry.get("direction", "ccw")).lower()
    if direction not in {"cw", "ccw"}:
        message = "figure_eight direction must be cw or ccw"
        raise ValueError(message)
    revolutions = _positive_float(phase.geometry.get("revolutions", 1.0), "revolutions")
    center_offset = _optional_xyz(phase.geometry, "center_offset")
    center = np.array([center_offset[0], center_offset[1], z + center_offset[2]], dtype=float)
    trajectory = _make_figure_eight_phase_trajectory(
        radius_x=radius_x,
        radius_y=radius_y,
        center=center,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        direction=direction,
        revolutions=revolutions,
    )
    task = {
        "task_type": "trajectory",
        "shape": PHASE_TYPE_FIGURE_EIGHT,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "radius_x": float(radius_x),
        "radius_y": float(radius_y),
        "height": float(center[2]),
        "center": _array_to_floats(center[:2]),
        "clockwise": direction == "cw",
    }
    anchor_indices = np.linspace(0, trajectory.positions.shape[0] - 1, num=9, dtype=int)
    geometry = _phase_geometry_payload(
        phase=phase,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        z=z,
        local_start=trajectory.positions[0],
        local_end=trajectory.positions[-1],
        extra={
            "radius_x_m": float(radius_x),
            "radius_y_m": float(radius_y),
            "center_offset": _array_to_floats(center_offset),
            "direction": direction,
            "revolutions": float(revolutions),
        },
    )
    return _PhaseBuildResult(
        trajectory=trajectory,
        task=task,
        geometry=geometry,
        waypoint_positions=tuple(np.array(trajectory.positions[index], dtype=float, copy=True) for index in anchor_indices),
    )


def _make_circle_phase_trajectory(
    radius: float,
    center: np.ndarray,
    duration_sec: float,
    sample_rate_hz: float,
    direction: str,
    revolutions: float,
    start_angle_deg: float,
) -> trajectories.primitives.Trajectory:
    """Generate a local XY circle trajectory supporting fractional revolutions."""
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    sign = -1.0 if direction == "cw" else 1.0
    start_angle_rad = math.radians(start_angle_deg)
    angles = start_angle_rad + sign * 2.0 * math.pi * revolutions * times / duration_sec
    positions = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(times.shape, center[2], dtype=float),
        )
    )
    return trajectories.primitives.Trajectory(times=times, positions=positions)


def _make_ellipse_phase_trajectory(
    radius_x: float,
    radius_y: float,
    center: np.ndarray,
    duration_sec: float,
    sample_rate_hz: float,
    direction: str,
    revolutions: float,
    start_angle_deg: float,
) -> trajectories.primitives.Trajectory:
    """Generate a local XY ellipse trajectory supporting fractional revolutions."""
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    sign = -1.0 if direction == "cw" else 1.0
    start_angle_rad = math.radians(start_angle_deg)
    angles = start_angle_rad + sign * 2.0 * math.pi * revolutions * times / duration_sec
    positions = np.column_stack(
        (
            center[0] + radius_x * np.cos(angles),
            center[1] + radius_y * np.sin(angles),
            np.full(times.shape, center[2], dtype=float),
        )
    )
    return trajectories.primitives.Trajectory(times=times, positions=positions)


def _make_figure_eight_phase_trajectory(
    radius_x: float,
    radius_y: float,
    center: np.ndarray,
    duration_sec: float,
    sample_rate_hz: float,
    direction: str,
    revolutions: float,
) -> trajectories.primitives.Trajectory:
    """Generate a local XY Gerono figure-eight trajectory."""
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    sign = -1.0 if direction == "cw" else 1.0
    theta = sign * 2.0 * math.pi * revolutions * times / duration_sec
    positions = np.column_stack(
        (
            center[0] + radius_x * np.sin(theta),
            center[1] + radius_y * np.sin(theta) * np.cos(theta),
            np.full(times.shape, center[2], dtype=float),
        )
    )
    return trajectories.primitives.Trajectory(times=times, positions=positions)


def _make_times(duration_sec: float, sample_rate_hz: float) -> np.ndarray:
    """Create inclusive sample times for scenario-only primitive generation."""
    sample_count = max(2, round(duration_sec * sample_rate_hz) + 1)
    return np.linspace(0.0, duration_sec, num=sample_count, dtype=float)


def _phase_geometry_payload(
    phase: ScenarioPhase,
    duration_sec: float,
    sample_rate_hz: float,
    z: float,
    local_start: np.ndarray,
    local_end: np.ndarray,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build a manifest-ready geometry payload for one phase."""
    return {
        "name": phase.name,
        "type": phase.phase_type,
        "duration_sec": float(duration_sec),
        "sample_rate_hz": float(sample_rate_hz),
        "start_mode": phase.start_mode,
        "hold_after_sec": float(phase.hold_after_sec),
        "z": float(z),
        "local_start_position": _array_to_floats(local_start),
        "local_end_position": _array_to_floats(local_end),
        **extra,
    }


def _phase_duration(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> float:
    """Return phase duration from the scenario or catalog default."""
    if phase.duration_sec is not None:
        return float(phase.duration_sec)
    value = _catalog_float(catalog_task, "duration_sec", None)
    if value is None:
        message = f"phase '{phase.name}' requires duration_sec when no catalog default is available"
        raise ValueError(message)
    return value


def _phase_sample_rate(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> float:
    """Return phase sample rate from the scenario, catalog, or project default."""
    if phase.sample_rate_hz is not None:
        return float(phase.sample_rate_hz)
    value = _catalog_float(catalog_task, "sample_rate_hz", None)
    return DEFAULT_PHASE_SAMPLE_RATE_HZ if value is None else value


def _phase_z(phase: ScenarioPhase, catalog_task: dict[str, Any] | None) -> float:
    """Return phase base height from the scenario, catalog, or project default."""
    if phase.z is not None:
        return float(phase.z)
    if catalog_task is None:
        return DEFAULT_PHASE_Z_M
    if "position" in catalog_task:
        return float(_xyz(catalog_task["position"], name="position")[2])
    if "start" in catalog_task:
        return float(_xyz(catalog_task["start"], name="start")[2])
    if "points" in catalog_task:
        return float(_xyz_rows(catalog_task["points"], name="points")[0, 2])
    if "height" in catalog_task:
        return float(catalog_task["height"])
    if "start_height" in catalog_task:
        return float(catalog_task["start_height"])
    return DEFAULT_PHASE_Z_M


def _catalog_float(catalog_task: dict[str, Any] | None, key: str, default: float | None) -> float | None:
    """Return a finite float from a catalog task when present."""
    if catalog_task is None or key not in catalog_task:
        return default
    value = float(catalog_task[key])
    if not np.isfinite(value):
        message = f"catalog {key} must be finite"
        raise ValueError(message)
    return value


def _line_delta(geometry: Mapping[str, Any]) -> np.ndarray:
    """Read line delta_position or end_offset from phase geometry."""
    if "delta_position" in geometry:
        return _xyz(geometry["delta_position"], name="delta_position")
    if "end_offset" in geometry:
        return _xyz(geometry["end_offset"], name="end_offset")
    message = "line phase requires delta_position or end_offset"
    raise ValueError(message)


def _optional_xy(geometry: Mapping[str, Any], key: str, default: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Return an optional finite XY vector from geometry."""
    value = geometry.get(key, default)
    array = np.asarray(value, dtype=float)
    if array.shape != (XY_DIMENSIONS,):
        message = f"{key} must have shape (2,)"
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = f"{key} must contain only finite values"
        raise ValueError(message)
    return array


def _optional_xyz(geometry: Mapping[str, Any], key: str, default: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Return an optional finite XYZ vector from geometry."""
    if key not in geometry:
        return _xyz(default, name=key)
    return _xyz(geometry[key], name=key)


def _xyz(value: Any, name: str) -> np.ndarray:
    """Return a finite XYZ vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (XYZ_DIMENSIONS,):
        message = f"{name} must have shape (3,)"
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return array


def _xyz_rows(value: Any, name: str) -> np.ndarray:
    """Return a finite two-dimensional array of XYZ rows."""
    array = np.asarray(value, dtype=float)
    if array.ndim != POINT_ARRAY_NDIM or array.shape[1:] != (XYZ_DIMENSIONS,):
        message = f"{name} must have shape (num_points, 3)"
        raise ValueError(message)
    if array.shape[0] < MIN_WAYPOINT_ROWS:
        message = f"{name} must contain at least two rows"
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return array


def _positive_float(value: Any, name: str) -> float:
    """Return a finite positive float."""
    if value is None:
        message = f"{name} is required"
        raise ValueError(message)
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        message = f"{name} must be finite and positive"
        raise ValueError(message)
    return number


def _append_phase_hold(
    combined_times: list[np.ndarray],
    combined_positions: list[np.ndarray],
    current_time_end: float,
    phase: ScenarioPhase,
    phase_index: int,
    hold_position: np.ndarray | None,
    sample_rate_hz: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Append an explicit stationary hold after one phase, when configured."""
    hold_range, time_range = _append_stationary_hold(
        combined_times=combined_times,
        combined_positions=combined_positions,
        current_time_end=current_time_end,
        hold_position=hold_position,
        hold_sec=phase.hold_after_sec,
        sample_rate_hz=sample_rate_hz,
        missing_position_message="cannot append phase hold without a phase endpoint",
    )
    if hold_range is None or time_range is None:
        return None, None
    return (
        {
            "phase_index": int(phase_index),
            "phase_name": phase.name,
            "phase_type": phase.phase_type,
            **hold_range,
        },
        {
            "phase_index": int(phase_index),
            "phase_name": phase.name,
            "phase_type": phase.phase_type,
            **time_range,
        },
    )


def _prepend_start_hold(times: np.ndarray, positions: np.ndarray, start_hold_sec: float) -> dict[str, Any]:
    """Prepend a stationary start hold before the first scenario phase."""
    if start_hold_sec <= 0.0:
        return {
            "times": times,
            "positions": positions,
            "steps": 0,
            "time_shift_sec": 0.0,
            "step_range": None,
            "time_range": None,
        }
    sample_rate_hz = _nominal_sample_rate(times)
    hold_steps = max(1, round(start_hold_sec * sample_rate_hz))
    sample_interval = 1.0 / sample_rate_hz
    effective_hold_sec = float(hold_steps * sample_interval)
    hold_times = float(times[0]) + sample_interval * np.arange(hold_steps, dtype=float)
    hold_positions = np.repeat(np.asarray(positions[0], dtype=float).reshape(1, XYZ_DIMENSIONS), repeats=hold_steps, axis=0)
    shifted_times = times + effective_hold_sec
    return {
        "times": np.concatenate((hold_times, shifted_times)),
        "positions": np.vstack((hold_positions, positions)),
        "steps": int(hold_steps),
        "time_shift_sec": effective_hold_sec,
        "step_range": {"start": 0, "end": int(hold_steps)},
        "time_range": {"start": float(hold_times[0]), "end": float(hold_times[-1])},
    }


def _shift_step_ranges(ranges: Sequence[Mapping[str, Any]], offset: int) -> list[dict[str, Any]]:
    """Return copied step ranges shifted forward by ``offset`` samples."""
    shifted: list[dict[str, Any]] = []
    for step_range in ranges:
        copied = dict(step_range)
        copied["start"] = int(copied["start"]) + int(offset)
        copied["end"] = int(copied["end"]) + int(offset)
        shifted.append(copied)
    return shifted


def _shift_optional_step_range(step_range: Mapping[str, int] | None, offset: int) -> dict[str, int] | None:
    """Return an optional step range shifted forward by ``offset`` samples."""
    if step_range is None:
        return None
    return {"start": int(step_range["start"]) + int(offset), "end": int(step_range["end"]) + int(offset)}


def _shift_time_ranges(ranges: Sequence[Mapping[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    """Return copied time ranges shifted forward by ``offset_sec`` seconds."""
    shifted: list[dict[str, Any]] = []
    for time_range in ranges:
        copied = dict(time_range)
        copied["start"] = float(copied["start"]) + float(offset_sec)
        copied["end"] = float(copied["end"]) + float(offset_sec)
        shifted.append(copied)
    return shifted


def _shift_optional_time_range(time_range: Mapping[str, float] | None, offset_sec: float) -> dict[str, float] | None:
    """Return an optional time range shifted forward by ``offset_sec`` seconds."""
    if time_range is None:
        return None
    return {"start": float(time_range["start"]) + float(offset_sec), "end": float(time_range["end"]) + float(offset_sec)}


def _append_final_hold(
    combined_times: list[np.ndarray],
    combined_positions: list[np.ndarray],
    current_time_end: float,
    final_position: np.ndarray | None,
    final_hold_sec: float,
) -> tuple[dict[str, int] | None, dict[str, float] | None]:
    """Append a stationary final hold segment to the composed reference."""
    sample_rate_hz = DEFAULT_PHASE_SAMPLE_RATE_HZ if not combined_times else _nominal_sample_rate(np.concatenate(combined_times))
    return _append_stationary_hold(
        combined_times=combined_times,
        combined_positions=combined_positions,
        current_time_end=current_time_end,
        hold_position=final_position,
        hold_sec=final_hold_sec,
        sample_rate_hz=sample_rate_hz,
        missing_position_message="cannot append final hold without a final phase position",
    )


def _append_stationary_hold(
    combined_times: list[np.ndarray],
    combined_positions: list[np.ndarray],
    current_time_end: float,
    hold_position: np.ndarray | None,
    hold_sec: float,
    sample_rate_hz: float,
    missing_position_message: str,
) -> tuple[dict[str, int] | None, dict[str, float] | None]:
    """Append stationary hold samples after the current combined reference end."""
    if hold_sec <= 0.0:
        return None, None
    if hold_position is None:
        raise ValueError(missing_position_message)
    if sample_rate_hz <= 0.0:
        sample_rate_hz = DEFAULT_PHASE_SAMPLE_RATE_HZ
    hold_steps = max(1, round(hold_sec * sample_rate_hz))
    sample_interval = 1.0 / sample_rate_hz
    hold_times = current_time_end + sample_interval * np.arange(1, hold_steps + 1, dtype=float)
    hold_positions = np.repeat(np.asarray(hold_position, dtype=float).reshape(1, XYZ_DIMENSIONS), repeats=hold_steps, axis=0)
    start_index = sum(positions.shape[0] for positions in combined_positions)
    end_index = start_index + hold_steps
    combined_times.append(hold_times)
    combined_positions.append(hold_positions)
    return (
        {"start": int(start_index), "end": int(end_index)},
        {"start": float(hold_times[0]), "end": float(hold_times[-1])},
    )


def _validate_combined_reference(
    times: np.ndarray,
    positions: np.ndarray,
    phase_step_ranges: list[dict[str, int]],
    limits: validation.tasks.ValidationLimits | None,
) -> None:
    """Validate the combined scenario reference and its phase boundaries."""
    if np.any(np.diff(times) <= 0.0):
        message = "composed scenario reference times must be strictly increasing"
        raise ValueError(message)
    _validate_phase_continuity(positions=positions, phase_step_ranges=phase_step_ranges)
    result = validation.tasks.validate_trajectory(
        trajectories.primitives.Trajectory(times=np.array(times, dtype=float, copy=True), positions=np.array(positions, dtype=float, copy=True)),
        limits=limits,
    )
    if not result.is_valid:
        message = "invalid composed scenario reference: " + "; ".join(result.messages)
        raise ValueError(message)


def _validate_phase_continuity(positions: np.ndarray, phase_step_ranges: list[dict[str, int]]) -> None:
    """Raise when adjacent phases have a boundary position jump."""
    for phase_index, (previous_range, next_range) in enumerate(pairwise(phase_step_ranges), start=1):
        previous_end = positions[previous_range["end"] - 1]
        next_start = positions[next_range["start"]]
        gap = float(np.linalg.norm(next_start - previous_end))
        if gap > CONTINUITY_TOLERANCE_M:
            message = f"scenario phase boundary {phase_index - 1}->{phase_index} is discontinuous by {gap:.6g} m"
            raise ValueError(message)


def _phase_offset(positions: np.ndarray, previous_end: np.ndarray | None, start_mode: str) -> np.ndarray:
    """Return XYZ offset needed to place a phase according to start mode."""
    if start_mode == START_MODE_INITIAL or previous_end is None:
        return np.zeros(XYZ_DIMENSIONS, dtype=float)
    if start_mode == START_MODE_PREVIOUS_END:
        return np.asarray(previous_end, dtype=float) - np.asarray(positions[0], dtype=float)
    message = f"unsupported start_mode: {start_mode}"
    raise ValueError(message)


def _phase_times_after_previous(local_times: np.ndarray, current_time_end: float, is_first_phase: bool) -> np.ndarray:
    """Shift local phase times so the combined scenario remains strictly increasing."""
    if is_first_phase:
        return np.array(local_times, dtype=float, copy=True)
    sample_interval = _first_positive_step(local_times)
    return np.array(current_time_end + sample_interval + local_times, dtype=float, copy=True)


def _first_positive_step(times: np.ndarray) -> float:
    """Return the first positive sample interval from a time vector."""
    steps = np.diff(np.asarray(times, dtype=float))
    positive_steps = steps[steps > 0.0]
    if positive_steps.size == 0:
        message = "phase times must contain a positive sample interval"
        raise ValueError(message)
    return float(positive_steps[0])


def _add_phase_fields(trace_records: list[dict[str, Any]], composition: ScenarioComposition) -> list[dict[str, Any]]:
    """Add scenario phase and final-hold columns to rollout trace records."""
    enriched: list[dict[str, Any]] = []
    for fallback_step, record in enumerate(trace_records):
        global_step = int(record.get("step_index", fallback_step))
        phase_metadata = _phase_metadata_for_step(global_step, composition)
        enriched_record = dict(record)
        enriched_record.update(
            {
                "global_step": global_step,
                **phase_metadata,
                "reference_position": record.get("reference_position_xyz_m"),
                "current_position": record.get("actual_position_xyz_m"),
                "position_error": record.get("position_error_m"),
            }
        )
        enriched.append(enriched_record)
    return enriched


def _phase_metadata_for_step(step_index: int, composition: ScenarioComposition) -> dict[str, Any]:
    """Return trace phase metadata for a combined-reference step index."""
    if _is_start_hold_step(step_index, composition.start_hold_step_range):
        return {
            "phase_index": -1,
            "phase_name": START_HOLD_NAME,
            "phase_type": START_HOLD_NAME,
            "phase_task_shape": START_HOLD_NAME,
            "is_start_hold": True,
            "is_phase_hold": False,
            "is_final_hold": False,
        }
    if _is_final_hold_step(step_index, composition.final_hold_step_range):
        return {
            "phase_index": len(composition.phases),
            "phase_name": FINAL_HOLD_NAME,
            "phase_type": FINAL_HOLD_NAME,
            "phase_task_shape": FINAL_HOLD_NAME,
            "is_start_hold": False,
            "is_phase_hold": False,
            "is_final_hold": True,
        }
    phase_hold = _phase_hold_metadata_for_step(step_index, composition.phase_hold_step_ranges)
    if phase_hold is not None:
        phase = composition.phases[int(phase_hold["phase_index"])]
        return {
            "phase_index": int(phase_hold["phase_index"]),
            "phase_name": phase.name,
            "phase_type": phase.phase_type,
            "phase_task_shape": phase.task_shape,
            "is_start_hold": False,
            "is_phase_hold": True,
            "is_final_hold": False,
        }
    phase_index = _phase_index_for_step(step_index, composition.phase_step_ranges)
    phase = composition.phases[phase_index]
    return {
        "phase_index": phase_index,
        "phase_name": phase.name,
        "phase_type": phase.phase_type,
        "phase_task_shape": phase.task_shape,
        "is_start_hold": False,
        "is_phase_hold": False,
        "is_final_hold": False,
    }


def _is_start_hold_step(step_index: int, start_hold_step_range: dict[str, int] | None) -> bool:
    """Return whether a step index belongs to the prepended start hold."""
    if start_hold_step_range is None:
        return False
    return start_hold_step_range["start"] <= step_index < start_hold_step_range["end"]


def _is_final_hold_step(step_index: int, final_hold_step_range: dict[str, int] | None) -> bool:
    """Return whether a step index belongs to the appended final hold."""
    if final_hold_step_range is None:
        return False
    return final_hold_step_range["start"] <= step_index < final_hold_step_range["end"]


def _phase_hold_metadata_for_step(step_index: int, ranges: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    """Return phase-hold metadata for a combined-reference step index, if any."""
    for step_range in ranges:
        if int(step_range["start"]) <= step_index < int(step_range["end"]):
            return step_range
    return None


def _phase_index_for_step(step_index: int, ranges: tuple[dict[str, int], ...]) -> int:
    """Return the phase index containing a combined-reference step index."""
    for index, step_range in enumerate(ranges):
        if step_range["start"] <= step_index < step_range["end"]:
            return index
    return max(len(ranges) - 1, 0)


def _effective_max_steps(active_settings: ScenarioRenderSettings, composition: ScenarioComposition) -> int:
    """Return the rollout safety cap after validating it covers the composed reference."""
    required_steps = int(composition.total_reference_steps)
    if active_settings.max_steps is None:
        return required_steps + DEFAULT_MAX_STEP_SAFETY_MARGIN
    requested_steps = int(active_settings.max_steps)
    if requested_steps < required_steps:
        message = (
            f"max_steps ({requested_steps}) is smaller than the composed scenario reference length "
            f"({required_steps} samples). Increase max_steps or omit it to derive a safe cap."
        )
        raise ValueError(message)
    return requested_steps


def _base_time_limit_sec(composition: ScenarioComposition) -> float:
    """Return a base simulator episode duration long enough for the scenario reference."""
    return float(composition.scenario_duration_sec + DEFAULT_BASE_EPISODE_TIME_MARGIN_SEC)


def _make_scenario_tracking_env(
    *,
    settings: ScenarioRenderSettings,
    composition: ScenarioComposition,
    effective_max_steps: int,
    base_time_limit_sec: float,
) -> Any:
    """Build the scenario env with the evaluated model's training action/observation flags."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(
        composition.reference,
        gui=False,
        record=False,
        max_steps=effective_max_steps,
        episode_len_sec=base_time_limit_sec,
        action_interface=settings.action_interface,
        rpm_delta_scale=settings.rpm_delta_scale,
        pid_target_z_min_m=settings.pid_target_z_min_m,
        pid_target_z_max_m=settings.pid_target_z_max_m,
        include_dynamics_observation=settings.include_dynamics_observation,
        include_previous_action=settings.include_previous_action,
        initial_state=settings.initial_state,
    )
    if settings.normalize_actions or envs.actions.parse_action_interface(settings.action_interface) == envs.actions.ActionInterface.DIRECT_RPM:
        return envs.tracking_env.make_normalized_action_env(real_env)
    return real_env


def _resolve_model_path(settings: ScenarioRenderSettings) -> Path | None:
    """Resolve the PPO model path from storage/runs/<model_run_name>/training/models."""
    if settings.controller != policy_render.PPO_CONTROLLER:
        return None
    if settings.model_path is not None:
        return settings.model_path.expanduser().resolve(strict=False)
    if settings.model_run_name is None:
        message = "PPO scenario rendering requires model_run_name or model_path"
        raise ValueError(message)
    model_dir = utils.artifacts.get_run_training_models_dir(settings.model_run_name).expanduser().resolve(strict=False)
    preferred = model_dir / f"{settings.model_run_name}.zip"
    if preferred.exists():
        return preferred.resolve(strict=False)
    candidates = sorted(model_dir.glob("*.zip"))
    if candidates:
        return candidates[0].resolve(strict=False)
    return preferred.resolve(strict=False)


def _policy_settings(
    active_settings: ScenarioRenderSettings,
    model_path: Path | None,
    evaluation_run_name: str,
    effective_max_steps: int,
) -> policy_render.PolicyRenderSettings:
    """Build a compatible policy-render settings object for shared rollout helpers."""
    return policy_render.PolicyRenderSettings(
        model_path=model_path or policy_render.default_model_path(),
        config_path=active_settings.task_config_path,
        model_run_name=active_settings.model_run_name,
        controller=active_settings.controller,
        run_name=evaluation_run_name,
        max_steps=effective_max_steps,
        seed=active_settings.seed,
        camera_mode=active_settings.camera_mode,
        camera_distance=active_settings.camera_distance,
        camera_yaw=active_settings.camera_yaw,
        camera_pitch=active_settings.camera_pitch,
        gif_filename=active_settings.gif_filename,
        manifest_filename=active_settings.manifest_filename,
        frame_interval=active_settings.frame_interval,
        image_width=active_settings.image_width,
        image_height=active_settings.image_height,
        normalize_actions=active_settings.normalize_actions,
        action_interface=active_settings.action_interface,
        rpm_delta_scale=active_settings.rpm_delta_scale,
        pid_target_z_min_m=active_settings.pid_target_z_min_m,
        pid_target_z_max_m=active_settings.pid_target_z_max_m,
        include_dynamics_observation=active_settings.include_dynamics_observation,
        include_previous_action=active_settings.include_previous_action,
        initial_state=active_settings.initial_state,
        source_manifest_path=active_settings.source_manifest_path,
        training_config_path=active_settings.training_config_path,
        final_stage_manifest_path=active_settings.final_stage_manifest_path,
        evaluated_model_source=active_settings.evaluated_model_source,
        scenario_name=_scenario_name(active_settings),
        scenario_config_path=active_settings.scenario_config_path,
    )


def _resolve_output_dir(output_dir: Path | None, evaluation_run_name: str) -> Path:
    """Resolve an output directory for a scenario evaluation run."""
    if output_dir is not None:
        return output_dir.expanduser().resolve(strict=False)
    return utils.artifacts.get_run_evaluation_dir(evaluation_run_name, "scenario")


def _scenario_name(settings: ScenarioRenderSettings) -> str:
    """Return the scenario name from settings or config."""
    if settings.scenario_name is not None:
        return settings.scenario_name
    return load_scenario_render_settings(settings.scenario_config_path).scenario_name or settings.scenario_config_path.stem


def _evaluation_run_name(settings: ScenarioRenderSettings) -> str:
    """Return explicit or derived evaluation run name for a scenario."""
    if settings.run_name is not None:
        return settings.run_name
    scenario_slug = _scenario_slug(settings)
    if settings.controller == policy_render.PPO_CONTROLLER:
        if settings.model_run_name is None:
            message = "PPO scenario rendering requires model_run_name"
            raise ValueError(message)
        return f"eval_{settings.model_run_name}_on_{scenario_slug}"
    if settings.controller == policy_render.SCRIPTED_REFERENCE_CONTROLLER:
        return f"eval_scripted_reference_on_{scenario_slug}"
    message = f"unsupported controller: {settings.controller}"
    raise ValueError(message)


def _scenario_slug(settings: ScenarioRenderSettings) -> str:
    """Return a scenario slug without duplicated controller prefixes."""
    slug = _scenario_name(settings)
    prefixes = []
    if settings.controller == policy_render.SCRIPTED_REFERENCE_CONTROLLER:
        prefixes.append(SCRIPTED_REFERENCE_SCENARIO_PREFIX)
    if settings.controller == policy_render.PPO_CONTROLLER:
        prefixes.extend((PPO_SCENARIO_PREFIX, SCRIPTED_REFERENCE_SCENARIO_PREFIX))
        if settings.model_run_name is not None:
            prefixes.append(f"{settings.model_run_name}_")
    for prefix in prefixes:
        if slug.startswith(prefix):
            return slug[len(prefix) :]
    return slug


def _scenario_completion_summary(
    actual_steps: int,
    requested_max_steps: int | None,
    effective_max_steps: int,
    reference_sample_count: int,
    reference_motion_steps: int,
    reference_motion_end_step: int,
    start_hold_steps: int,
    phase_hold_steps: int,
    phase_hold_end_step: int,
    final_hold_steps: int,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
) -> dict[str, Any]:
    """Build explicit scenario completion and fraction metrics."""
    rollout_step_fraction = 0.0 if effective_max_steps <= 0 else float(actual_steps / effective_max_steps)
    reference_completion_fraction = 0.0 if reference_sample_count <= 0 else float(min(actual_steps / reference_sample_count, 1.0))
    normal_reference_done = termination_reason in NORMAL_TERMINATION_REASONS
    completed_reference_motion = actual_steps >= reference_motion_end_step or normal_reference_done
    completed_phase_holds = phase_hold_steps == 0 or actual_steps >= phase_hold_end_step or normal_reference_done
    completed_final_hold = final_hold_steps == 0 or actual_steps >= reference_sample_count or normal_reference_done
    completed_reference = actual_steps >= reference_sample_count or normal_reference_done
    ended_normally = bool(completed_reference and not truncated and (terminated or normal_reference_done))
    survived_fraction = 1.0 if ended_normally else rollout_step_fraction
    return {
        "requested_max_steps": None if requested_max_steps is None else int(requested_max_steps),
        "effective_max_steps": int(effective_max_steps),
        "rollout_step_fraction": rollout_step_fraction,
        "reference_completion_fraction": reference_completion_fraction,
        "reference_motion_steps": int(reference_motion_steps),
        "start_hold_steps": int(start_hold_steps),
        "phase_hold_steps": int(phase_hold_steps),
        "final_hold_steps": int(final_hold_steps),
        "completed_reference": bool(completed_reference),
        "completed_reference_motion": bool(completed_reference_motion),
        "completed_phase_holds": bool(completed_phase_holds),
        "completed_final_hold": bool(completed_final_hold),
        "ended_normally": bool(ended_normally),
        "survived_fraction": float(survived_fraction),
    }


def _build_scenario_metrics(
    settings: ScenarioRenderSettings,
    evaluation_run_name: str,
    scenario_name: str,
    model_path: Path | None,
    training_task_shape: str | None,
    rewards: list[float],
    position_errors: list[float],
    terminated: bool,
    truncated: bool,
    termination_reason: str,
    completion: dict[str, Any],
    policy_predict_used: bool,
    warnings: tuple[str, ...],
    observation_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable metrics for a scenario rollout."""
    if not rewards or not position_errors:
        message = "scenario rollout produced no step metrics"
        raise RuntimeError(message)
    return {
        "run_type": "evaluation",
        "evaluation_type": SCENARIO_EVALUATION_TYPE,
        "evaluation_run_name": evaluation_run_name,
        "scenario_name": scenario_name,
        "controller_type": settings.controller,
        "model_run_name": settings.model_run_name,
        "model_path": None if model_path is None else str(model_path),
        "configured_model_path": None if settings.model_path is None else str(settings.model_path),
        "policy_predict_used": bool(policy_predict_used),
        "training_task_shape": training_task_shape,
        "steps": len(rewards),
        "total_steps": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(position_errors)),
        "final_position_error_m": float(position_errors[-1]),
        "min_position_error_m": float(np.min(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reason,
        "warnings": list(warnings),
        "action_interface": settings.action_interface,
        "rpm_delta_scale": settings.rpm_delta_scale if settings.action_interface == "direct_rpm" else None,
        "pid_target_z_min_m": settings.pid_target_z_min_m if settings.action_interface == "pid_position" else None,
        "pid_target_z_max_m": settings.pid_target_z_max_m if settings.action_interface == "pid_position" else None,
        "normalize_actions": bool(settings.normalize_actions),
        "include_dynamics_observation": bool(settings.include_dynamics_observation),
        "include_previous_action": bool(settings.include_previous_action),
        "initial_state_mode": _settings_initial_state(settings).mode,
        "initial_state": _settings_initial_state(settings).to_dict(),
        "source_manifest_path": None if settings.source_manifest_path is None else str(settings.source_manifest_path),
        "training_config_path": None if settings.training_config_path is None else str(settings.training_config_path),
        "final_stage_manifest_path": None if settings.final_stage_manifest_path is None else str(settings.final_stage_manifest_path),
        "evaluated_model_source": settings.evaluated_model_source,
        "observation_check": {} if observation_check is None else dict(observation_check),
        "model_observation_space": None if observation_check is None else observation_check.get("model_observation_space"),
        "model_observation_space_shape": None if observation_check is None else observation_check.get("model_observation_space_shape"),
        "env_observation_space": None if observation_check is None else observation_check.get("env_observation_space"),
        "env_observation_space_shape": None if observation_check is None else observation_check.get("env_observation_space_shape"),
        "actual_reset_observation_shape": None if observation_check is None else observation_check.get("actual_reset_observation_shape"),
        "actual_step_observation_shape": None if observation_check is None else observation_check.get("actual_step_observation_shape"),
        **completion,
    }


def _build_scenario_manifest(
    settings: ScenarioRenderSettings,
    evaluation_run_name: str,
    scenario_name: str,
    composition: ScenarioComposition,
    model_path: Path | None,
    training_task_shape: str | None,
    gif_path: Path,
    trace_path: Path,
    plot_paths: dict[str, str],
    actual_steps: int,
    requested_max_steps: int | None,
    effective_max_steps: int,
    base_time_limit_sec: float,
    completion: dict[str, Any],
    termination_reason: str,
    true_simulator_rendering: bool,
    policy_predict_used: bool,
    metrics: dict[str, Any],
    warnings: tuple[str, ...],
    final_info: dict[str, Any] | None = None,
    final_action: Any | None = None,
    actual_positions: list[np.ndarray] | None = None,
    rollout_reference_positions: list[np.ndarray] | None = None,
    output_dir: Path | None = None,
    observation_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the scenario render manifest payload."""
    info = {} if final_info is None else final_info
    positions = actual_positions or ([np.asarray(info.get("current_position", []), dtype=float)] if info.get("current_position") is not None else [])
    reference_positions = [np.asarray(row, dtype=float) for row in composition.reference.positions]
    position_bounds = policy_render._position_bounds(positions)  # noqa: SLF001
    reference_position_bounds = policy_render._position_bounds(reference_positions)  # noqa: SLF001
    rollout_reference_position_bounds = policy_render._position_bounds(rollout_reference_positions or [])  # noqa: SLF001
    return {
        "run_type": "evaluation",
        "evaluation_type": SCENARIO_EVALUATION_TYPE,
        "mode": policy_render._mode_for_controller(settings.controller),  # noqa: SLF001
        "evaluation_run_name": evaluation_run_name,
        "scenario_name": scenario_name,
        "render_mode": "simulator_external_camera_gif",
        "controller_type": settings.controller,
        "model_run_name": settings.model_run_name,
        "model_path": None if model_path is None else str(model_path),
        "configured_model_path": None if settings.model_path is None else str(settings.model_path),
        "policy_predict_used": bool(policy_predict_used),
        "true_simulator_rendering": bool(true_simulator_rendering),
        "training_task_shape": training_task_shape,
        "evaluated_model_source": settings.evaluated_model_source,
        "source_manifest_path": None if settings.source_manifest_path is None else str(settings.source_manifest_path),
        "training_config_path": None if settings.training_config_path is None else str(settings.training_config_path),
        "final_stage_manifest_path": None if settings.final_stage_manifest_path is None else str(settings.final_stage_manifest_path),
        "action_interface": settings.action_interface,
        "rpm_delta_scale": settings.rpm_delta_scale if settings.action_interface == "direct_rpm" else None,
        "pid_target_z_min_m": settings.pid_target_z_min_m if settings.action_interface == "pid_position" else None,
        "pid_target_z_max_m": settings.pid_target_z_max_m if settings.action_interface == "pid_position" else None,
        "normalize_actions": bool(settings.normalize_actions),
        "include_dynamics_observation": bool(settings.include_dynamics_observation),
        "include_previous_action": bool(settings.include_previous_action),
        "observation_check": {} if observation_check is None else dict(observation_check),
        "model_observation_space": None if observation_check is None else observation_check.get("model_observation_space"),
        "model_observation_space_shape": None if observation_check is None else observation_check.get("model_observation_space_shape"),
        "env_observation_space": None if observation_check is None else observation_check.get("env_observation_space"),
        "env_observation_space_shape": None if observation_check is None else observation_check.get("env_observation_space_shape"),
        "actual_reset_observation_shape": None if observation_check is None else observation_check.get("actual_reset_observation_shape"),
        "actual_step_observation_shape": None if observation_check is None else observation_check.get("actual_step_observation_shape"),
        "task_config_path": str(settings.task_config_path),
        "scenario_config_path": str(settings.scenario_config_path),
        "run_name": settings.run_name,
        "output_dir": None if output_dir is None else str(output_dir),
        "phases": [_phase_to_manifest(phase) for phase in composition.phases],
        "phase_names": [phase.name for phase in composition.phases],
        "phase_types": [phase.phase_type for phase in composition.phases],
        "phase_task_shapes": [phase.task_shape for phase in composition.phases],
        "phase_step_ranges": [dict(step_range) for step_range in composition.phase_step_ranges],
        "phase_time_ranges": [dict(time_range) for time_range in composition.phase_time_ranges],
        "phase_start_positions": [list(position) for position in composition.phase_start_positions],
        "phase_end_positions": [list(position) for position in composition.phase_end_positions],
        "phase_offsets": [list(offset) for offset in composition.phase_offsets],
        "phase_geometry": [dict(geometry) for geometry in composition.phase_geometry],
        "phase_hold_step_ranges": [dict(step_range) for step_range in composition.phase_hold_step_ranges],
        "phase_hold_time_ranges": [dict(time_range) for time_range in composition.phase_hold_time_ranges],
        "scenario_duration_sec": float(composition.scenario_duration_sec),
        "total_steps": int(actual_steps),
        "requested_max_steps": None if requested_max_steps is None else int(requested_max_steps),
        "effective_max_steps": int(effective_max_steps),
        "base_time_limit_sec": float(base_time_limit_sec),
        "reference_sample_count": int(composition.reference.positions.shape[0]),
        "reference_motion_steps": int(composition.reference_motion_steps),
        "reference_motion_end_step": int(composition.reference_motion_end_step),
        "start_hold_enabled": bool(composition.start_hold_sec > 0.0),
        "start_hold_sec": float(composition.start_hold_sec),
        "start_hold_steps": int(composition.start_hold_steps),
        "start_hold_step_range": None if composition.start_hold_step_range is None else dict(composition.start_hold_step_range),
        "start_hold_time_range": None if composition.start_hold_time_range is None else dict(composition.start_hold_time_range),
        "phase_hold_steps": int(composition.phase_hold_steps),
        "phase_hold_end_step": int(composition.phase_hold_end_step),
        "final_hold_steps": int(composition.final_hold_steps),
        "total_reference_steps": int(composition.total_reference_steps),
        "final_hold_sec": float(composition.final_hold_sec),
        "final_hold_step_range": None if composition.final_hold_step_range is None else dict(composition.final_hold_step_range),
        "final_hold_time_range": None if composition.final_hold_time_range is None else dict(composition.final_hold_time_range),
        "completed_reference_motion": bool(completion["completed_reference_motion"]),
        "completed_phase_holds": bool(completion["completed_phase_holds"]),
        "completed_final_hold": bool(completion["completed_final_hold"]),
        "completed_reference": bool(completion["completed_reference"]),
        "ended_normally": bool(completion["ended_normally"]),
        "rollout_step_fraction": float(completion["rollout_step_fraction"]),
        "reference_completion_fraction": float(completion["reference_completion_fraction"]),
        "survived_fraction": float(completion["survived_fraction"]),
        "termination_reason": termination_reason,
        "reset_count": 1,
        "camera_mode": settings.camera_mode,
        "camera_distance": float(settings.camera_distance),
        "camera_yaw": float(settings.camera_yaw),
        "camera_pitch": float(settings.camera_pitch),
        "camera_settings": {
            "mode": settings.camera_mode,
            "distance_m": float(settings.camera_distance),
            "yaw_deg": float(settings.camera_yaw),
            "pitch_deg": float(settings.camera_pitch),
            "image_width": int(settings.image_width),
            "image_height": int(settings.image_height),
            "frame_interval": int(settings.frame_interval),
        },
        "position_bounds": position_bounds,
        "reference_position_bounds": reference_position_bounds,
        "rollout_reference_position_bounds": rollout_reference_position_bounds,
        "final_position": policy_render._array_to_jsonable(info.get("current_position", [])),  # noqa: SLF001
        "final_reference_position": policy_render._array_to_jsonable(info.get("reference_position", [])),  # noqa: SLF001
        "final_position_error_m": float(info.get("position_error_m", 0.0)),
        "final_action": policy_render._array_to_jsonable(final_action if final_action is not None else []),  # noqa: SLF001
        "gif_path": str(gif_path),
        "trace_path": str(trace_path),
        "plot_paths": dict(plot_paths),
        "output_files": [str(gif_path), str(trace_path), *sorted(plot_paths.values())],
        "reference_path_overlay_enabled": True,
        "waypoint_markers_enabled": True,
        "active_target_marker_enabled": True,
        "actual_path_trail_enabled": True,
        "overlay_visual_roles": policy_render.OVERLAY_VISUAL_ROLES,
        "overlay_geometry_mode": "pybullet_visual_only_no_collision",
        "metrics": metrics,
        "warnings": list(warnings),
    }


def _phase_to_manifest(phase: ScenarioPhase) -> dict[str, Any]:
    """Convert a scenario phase to a JSON-serializable manifest mapping."""
    return {
        "name": phase.name,
        "type": phase.phase_type,
        "task_shape": phase.task_shape,
        "duration_sec": phase.duration_sec,
        "sample_rate_hz": phase.sample_rate_hz,
        "start_mode": phase.start_mode,
        "hold_after_sec": float(phase.hold_after_sec),
        "z": phase.z,
        "geometry": _json_ready(dict(phase.geometry)),
    }


def _array_to_floats(value: np.ndarray) -> list[float]:
    """Return an array as a list of floats."""
    return [float(item) for item in np.asarray(value, dtype=float)]


def _unique_positions(positions: list[np.ndarray]) -> np.ndarray:
    """Return finite unique XYZ positions while preserving order."""
    unique_rows: list[np.ndarray] = []
    for position in positions:
        row = np.asarray(position, dtype=float)
        if row.shape != (XYZ_DIMENSIONS,) or not np.all(np.isfinite(row)):
            message = "scenario waypoint positions must be finite XYZ rows"
            raise ValueError(message)
        if not any(np.allclose(row, existing) for existing in unique_rows):
            unique_rows.append(np.array(row, dtype=float, copy=True))
    return np.vstack(unique_rows) if unique_rows else np.empty((0, XYZ_DIMENSIONS), dtype=float)


def _nominal_sample_rate(times: np.ndarray) -> float:
    """Estimate a representative sample rate for metadata."""
    steps = np.diff(np.asarray(times, dtype=float))
    positive_steps = steps[steps > 0.0]
    if positive_steps.size == 0:
        return 0.0
    return float(1.0 / np.median(positive_steps))


def _json_ready(value: Any) -> Any:
    """Return a JSON-compatible copy of a nested value."""
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "CONTINUITY_TOLERANCE_M",
    "DEFAULT_GIF_FILENAME",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_PHASE_SAMPLE_RATE_HZ",
    "DEFAULT_PHASE_Z_M",
    "DEFAULT_SCENARIO_CONFIG_PATH",
    "DEFAULT_SEED",
    "DEFAULT_TASK_CONFIG_PATH",
    "DEFAULT_TRACE_FILENAME",
    "PHASE_TYPE_CIRCLE",
    "PHASE_TYPE_HOVER",
    "PHASE_TYPE_LINE",
    "PHASE_TYPE_POLYLINE",
    "SCENARIO_EVALUATION_TYPE",
    "START_MODE_INITIAL",
    "START_MODE_PREVIOUS_END",
    "SUPPORTED_PHASE_TYPES",
    "SUPPORTED_START_MODES",
    "ScenarioComposition",
    "ScenarioPhase",
    "ScenarioRenderResult",
    "ScenarioRenderSettings",
    "compose_scenario_reference",
    "load_scenario_render_settings",
    "run_scenario_render",
]
