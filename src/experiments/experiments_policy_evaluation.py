"""
===============================================================================
experiments_policy_evaluation.py
===============================================================================
Run one shared PPO policy evaluation pipeline for report-ready artifacts.

Responsibilities:
  - Evaluate one PPO model deterministically on one configured trajectory task
  - Write diagnostics, traces, plots, renders, metrics, and manifests consistently
  - Expose one reusable result contract for normal and curriculum evaluation flows

Design principles:
  - Keep artifact paths deterministic and review-friendly
  - Reuse existing diagnostics, plotting, and policy-render rollout helpers
  - Fail clearly when model, task, or required simulator interfaces are invalid

Boundaries:
  - Curriculum mode planning and benchmark selection belong in curriculum evaluation
  - PPO training and persistence belong in ppo_tracking
===============================================================================

"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import envs, evaluation, utils, validation
from src.experiments import experiments_config as config

DEFAULT_RENDER_FPS = 20
SIMULATOR_CAPTURE_FPS = 30


@dataclass(frozen=True)
class PolicyEvaluationArtifactOptions:
    """
    Artifact controls for one policy evaluation run.

    Parameters
    ----------
    render_enabled
        Whether to write a simulator GIF rollout.
    plots_enabled
        Whether to write trajectory plots derived from the evaluation trace.
    trace_enabled
        Whether to copy the evaluation trace into ``traces/evaluation_trace.jsonl``.
    diagnostics_enabled
        Whether to write diagnostics JSON artifacts under ``diagnostics/``.
    render_fps
        Requested GIF playback frame rate.
    render_max_steps
        Optional render rollout length override. Uses ``eval_steps`` when omitted.

    """

    render_enabled: bool = True
    plots_enabled: bool = True
    trace_enabled: bool = True
    diagnostics_enabled: bool = True
    render_fps: int = DEFAULT_RENDER_FPS
    render_max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate artifact options."""
        if self.render_fps <= 0:
            message = "render_fps must be positive"
            raise ValueError(message)
        if self.render_max_steps is not None and self.render_max_steps <= 0:
            message = "render_max_steps must be positive when provided"
            raise ValueError(message)


@dataclass(frozen=True)
class _RenderArtifactResult:
    """In-memory render artifact payload used to keep report plots aligned with GIFs."""

    gif_path: Path
    warnings: list[str]
    trace_records: list[dict[str, Any]]


@dataclass(frozen=True)
class PolicyEvaluationSpec:
    """
    One concrete model/task evaluation specification.

    Parameters
    ----------
    label
        Human-readable result label used in metrics and manifest file names.
    model_role
        Stable role label such as ``stage`` or ``baseline``.
    model_path
        Stable-Baselines3 PPO model path.
    task_config_path
        Task config path with a top-level ``tasks`` list.
    task_shape
        Expected trajectory shape used for validation and metadata.
    output_dir
        Directory where report-ready artifacts are written.
    eval_steps
        Deterministic evaluation rollout steps.
    seed
        Deterministic reset seed for diagnostics and rendering.
    task_index
        Task index selected from ``task_config_path``.
    total_timesteps
        Optional training budget metadata.
    normalize_actions
        Whether the model expects normalized action wrappers.

    """

    label: str
    model_role: str
    model_path: Path
    task_config_path: Path
    task_shape: str
    output_dir: Path
    eval_steps: int
    seed: int
    task_index: int = 0
    total_timesteps: int = 0
    normalize_actions: bool = True

    def __post_init__(self) -> None:
        """Validate required spec fields."""
        if not self.label.strip():
            message = "label must be non-empty"
            raise ValueError(message)
        if not self.model_role.strip():
            message = "model_role must be non-empty"
            raise ValueError(message)
        if not self.task_shape.strip():
            message = "task_shape must be non-empty"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "eval_steps must be positive"
            raise ValueError(message)
        if self.seed < 0:
            message = "seed must be nonnegative"
            raise ValueError(message)
        if self.task_index < 0:
            message = "task_index must be nonnegative"
            raise ValueError(message)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Result contract returned by ``run_policy_evaluation``."""

    label: str
    model_role: str
    model_path: str
    task_config_path: str
    task_shape: str
    output_dir: str
    diagnostics_dir: str
    traces_dir: str
    plots_dir: str
    renders_dir: str
    metrics_path: str
    manifest_path: str
    trace_path: str | None
    gif_path: str | None
    plot_paths: dict[str, str]
    plot_trace_scope: str
    failure_report_path: str | None
    episode_summaries_path: str | None
    curriculum_feedback_path: str | None
    render_enabled: bool
    plots_enabled: bool
    trace_enabled: bool
    metrics: dict[str, Any]


def run_policy_evaluation(
    spec: PolicyEvaluationSpec,
    artifacts: PolicyEvaluationArtifactOptions | None = None,
) -> PolicyEvaluationResult:
    """
    Evaluate one PPO model on one configured task and write report-ready artifacts.

    Parameters
    ----------
    spec
        Concrete model/task specification.
    artifacts
        Optional artifact controls. Defaults enable diagnostics, traces, plots, and renders.

    Returns
    -------
    PolicyEvaluationResult
        Stable artifact paths and compact metrics for one evaluated model.

    """
    options = artifacts or PolicyEvaluationArtifactOptions()
    _validate_model_path(spec.model_path)
    task = _load_and_validate_task(spec)

    diagnostics_dir = spec.output_dir / utils.artifacts.DIAGNOSTICS_DIRNAME
    traces_dir = spec.output_dir / utils.artifacts.TRACES_DIRNAME
    plots_dir = spec.output_dir / utils.artifacts.PLOTS_DIRNAME
    renders_dir = spec.output_dir / utils.artifacts.RENDERS_DIRNAME
    metrics_dir = spec.output_dir / utils.artifacts.METRICS_DIRNAME
    manifests_dir = spec.output_dir / utils.artifacts.MANIFESTS_DIRNAME

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    diagnostics, artifact_fields = _collect_diagnostics(spec=spec, task=task, diagnostics_dir=diagnostics_dir)

    trace_path: Path | None = None
    if options.trace_enabled:
        trace_source = Path(str(artifact_fields["evaluation_trace_path"]))
        trace_path = traces_dir / evaluation.diagnostics.EVALUATION_TRACE_FILENAME
        shutil.copyfile(trace_source, trace_path)

    gif_path: Path | None = None
    render_warnings: list[str] = []
    render_result: _RenderArtifactResult | None = None
    if options.render_enabled:
        render_result = _write_render_artifact(
            spec=spec,
            task=task,
            renders_dir=renders_dir,
            render_steps=int(options.render_max_steps or spec.eval_steps),
            render_fps=options.render_fps,
        )
        gif_path = render_result.gif_path
        render_warnings = list(render_result.warnings)

    plot_paths: dict[str, str] = {}
    plot_trace_scope = "disabled"
    plot_trace_records: list[dict[str, Any]] = []
    if options.plots_enabled:
        if render_result is not None:
            plot_trace_scope = "render_rollout"
            plot_trace_records = render_result.trace_records
        else:
            plot_trace_scope = "full_evaluation"
            plot_trace_records = diagnostics.trace_records
        plot_result = evaluation.plots.write_policy_rollout_trace_plots(plot_trace_records, plots_dir)
        plot_paths = dict(plot_result.plot_paths)

    metrics_payload = {
        "label": spec.label,
        "model_role": spec.model_role,
        "model_path": str(spec.model_path),
        "task_config_path_used_for_evaluation": str(spec.task_config_path),
        "task_shape_used_for_evaluation": spec.task_shape,
        "evaluation_dir": str(spec.output_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "traces_dir": str(traces_dir),
        "plots_dir": str(plots_dir),
        "renders_dir": str(renders_dir),
        "trace_path": None if trace_path is None else str(trace_path),
        "gif_path": None if gif_path is None else str(gif_path),
        "plot_paths": dict(plot_paths),
        "plot_trace_scope": plot_trace_scope,
        "plot_trace_step_count": len(plot_trace_records),
        "plot_trace_terminated": bool(plot_trace_records[-1].get("terminated", False)) if plot_trace_records else None,
        "plot_trace_truncated": bool(plot_trace_records[-1].get("truncated", False)) if plot_trace_records else None,
        "failure_report_path": str(artifact_fields.get("failure_report_path")) if artifact_fields.get("failure_report_path") else None,
        "episode_summaries_path": str(artifact_fields.get("episode_summaries_path")) if artifact_fields.get("episode_summaries_path") else None,
        "curriculum_feedback_path": str(artifact_fields.get("curriculum_feedback_path")) if artifact_fields.get("curriculum_feedback_path") else None,
        "render_enabled": options.render_enabled,
        "plots_enabled": options.plots_enabled,
        "trace_enabled": options.trace_enabled,
        "diagnostics_enabled": options.diagnostics_enabled,
        "eval_steps": spec.eval_steps,
        "seed": spec.seed,
        "render_warnings": list(render_warnings),
        **diagnostics.metrics,
        **artifact_fields,
    }

    label_stem = _safe_name(spec.label)
    metrics_path = metrics_dir / f"{label_stem}_metrics.json"
    manifest_path = manifests_dir / f"{label_stem}_manifest.json"
    metrics_payload["metrics_path"] = str(metrics_path)
    metrics_payload["manifest_path"] = str(manifest_path)

    manifest = _manifest_from_metrics(metrics_payload)
    _write_json(metrics_path, metrics_payload)
    _write_json(manifest_path, manifest)

    return PolicyEvaluationResult(
        label=spec.label,
        model_role=spec.model_role,
        model_path=str(spec.model_path),
        task_config_path=str(spec.task_config_path),
        task_shape=spec.task_shape,
        output_dir=str(spec.output_dir),
        diagnostics_dir=str(diagnostics_dir),
        traces_dir=str(traces_dir),
        plots_dir=str(plots_dir),
        renders_dir=str(renders_dir),
        metrics_path=str(metrics_path),
        manifest_path=str(manifest_path),
        trace_path=None if trace_path is None else str(trace_path),
        gif_path=None if gif_path is None else str(gif_path),
        plot_paths=plot_paths,
        plot_trace_scope=plot_trace_scope,
        failure_report_path=metrics_payload["failure_report_path"],
        episode_summaries_path=metrics_payload["episode_summaries_path"],
        curriculum_feedback_path=metrics_payload["curriculum_feedback_path"],
        render_enabled=options.render_enabled,
        plots_enabled=options.plots_enabled,
        trace_enabled=options.trace_enabled,
        metrics=metrics_payload,
    )


def _collect_diagnostics(
    spec: PolicyEvaluationSpec,
    task: dict[str, Any],
    diagnostics_dir: Path,
) -> tuple[evaluation.diagnostics.PolicyEvaluationDiagnostics, dict[str, Any]]:
    """Run deterministic diagnostics and write diagnostics artifacts."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env) if spec.normalize_actions else real_env
    try:
        model = PPO.load(str(spec.model_path), env=tracking_env, device="cpu")
        diagnostics = evaluation.diagnostics.collect_policy_evaluation_diagnostics(
            model=model,
            tracking_env=tracking_env,
            eval_steps=spec.eval_steps,
            seed=spec.seed,
            training_run_name=f"{spec.model_role}_{spec.label}",
            task_shape=spec.task_shape,
            total_timesteps=spec.total_timesteps,
        )
    finally:
        tracking_env.close()
    artifact_fields = evaluation.diagnostics.write_policy_evaluation_diagnostics(diagnostics, diagnostics_dir)
    return diagnostics, artifact_fields


def _write_render_artifact(
    spec: PolicyEvaluationSpec,
    task: dict[str, Any],
    renders_dir: Path,
    render_steps: int,
    render_fps: int,
) -> _RenderArtifactResult:
    """Write one simulator rollout GIF and return the path, warnings, and plotted trace."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    from src.experiments import experiments_policy_render as policy_render  # noqa: PLC0415

    settings = policy_render.PolicyRenderSettings(
        model_path=spec.model_path,
        max_steps=render_steps,
        seed=spec.seed,
        gif_filename="scenario_rollout.gif",
        frame_interval=_frame_interval_for_fps(render_fps),
    )
    real_render_env = envs.tracking_env.make_trajectory_tracking_env(
        task,
        gui=False,
        record=False,
        max_steps=render_steps,
    )
    render_env = envs.tracking_env.make_normalized_action_env(real_render_env) if spec.normalize_actions else real_render_env
    try:
        model = PPO.load(str(spec.model_path), env=render_env, device="cpu")
        rollout_payload = policy_render._run_policy_rollout(  # noqa: SLF001
            model=model,
            tracking_env=render_env,
            settings=settings,
            seed=spec.seed,
            task=task,
            task_shape=spec.task_shape,
        )
    finally:
        render_env.close()

    gif_path = renders_dir / "scenario_rollout.gif"
    policy_render._write_gif(rollout_payload["frames"], gif_path, settings.frame_interval)  # noqa: SLF001
    return _RenderArtifactResult(gif_path=gif_path, warnings=[], trace_records=list(rollout_payload["trace_records"]))


def _load_and_validate_task(spec: PolicyEvaluationSpec) -> dict[str, Any]:
    """Load one task from config and validate its deterministic contract."""
    config_payload = config.load_experiment_config(spec.task_config_path)
    tasks = config_payload.get("tasks")
    if not isinstance(tasks, list) or spec.task_index >= len(tasks):
        message = f"cannot load task index {spec.task_index} from {spec.task_config_path}"
        raise ValueError(message)
    raw_task = tasks[spec.task_index]
    if not isinstance(raw_task, dict):
        message = f"task index {spec.task_index} in {spec.task_config_path} must be a mapping"
        raise TypeError(message)
    task = dict(raw_task)
    if str(task.get("shape", "")) != spec.task_shape:
        message = (
            f"task shape mismatch: spec task_shape={spec.task_shape!r}, config shape={task.get('shape')!r}, task_config_path={spec.task_config_path}"
        )
        raise ValueError(message)
    validation_result = validation.tasks.validate_task(task)
    if not validation_result.is_valid:
        details = "; ".join(validation_result.messages)
        message = f"invalid evaluation task in {spec.task_config_path}: {details}"
        raise ValueError(message)
    return task


def _validate_model_path(model_path: Path) -> None:
    """Raise when the PPO model path is missing."""
    if not model_path.exists():
        message = f"model_path does not exist: {model_path}"
        raise FileNotFoundError(message)


def _frame_interval_for_fps(render_fps: int) -> int:
    """Return the simulator frame interval nearest to the requested GIF FPS."""
    return max(1, round(SIMULATOR_CAPTURE_FPS / float(render_fps)))


def _manifest_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build a compact manifest payload for one evaluated model."""
    keys = (
        "label",
        "model_role",
        "model_path",
        "task_config_path_used_for_evaluation",
        "task_shape_used_for_evaluation",
        "evaluation_dir",
        "diagnostics_dir",
        "traces_dir",
        "plots_dir",
        "renders_dir",
        "metrics_path",
        "manifest_path",
        "trace_path",
        "gif_path",
        "plot_paths",
        "plot_trace_scope",
        "plot_trace_step_count",
        "plot_trace_terminated",
        "plot_trace_truncated",
        "failure_report_path",
        "episode_summaries_path",
        "curriculum_feedback_path",
        "render_enabled",
        "plots_enabled",
        "trace_enabled",
        "diagnostics_enabled",
        "eval_steps",
        "seed",
        "start_hold_enabled",
        "start_hold_sec",
        "exclude_start_hold_from_tracking_metrics",
        "tracking_phase_start_step",
        "tracking_phase_start_time_sec",
        "mean_position_error_m",
        "mean_position_error_tracking_m",
        "final_position_error_m",
        "max_position_error_m",
        "actual_xy_span_m",
        "reference_xy_span_m",
        "xy_tracking_ratio",
        "action_saturation_fraction",
        "real_action_saturation_fraction",
        "failure_overall_status",
        "failure_primary_mode",
        "failure_modes",
    )
    return {key: metrics.get(key) for key in keys}


def _safe_name(value: str) -> str:
    """Return a filesystem-safe label component."""
    text = value.strip().replace(" ", "_")
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable-formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_RENDER_FPS",
    "PolicyEvaluationArtifactOptions",
    "PolicyEvaluationResult",
    "PolicyEvaluationSpec",
    "run_policy_evaluation",
]
