"""
===============================================================================
experiments_evaluation_policy.py
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
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from src import envs, evaluation, utils, validation
from src.experiments import experiments_config as config
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites
from src.experiments.rendering import experiments_rendering_policy as policy_render
from src.experiments.rendering import experiments_rendering_scenario as scenario_render

DEFAULT_RENDER_FPS = 20
SIMULATOR_CAPTURE_FPS = 30
OWN_TASK_EVALUATION_NAME = "own_task"
STANDARD_EVALUATION_PROFILE = "standard"
STANDARD_GENERALIZATION_SUITE_PATH = Path("configs/evaluation/generalization_eval_suite.yaml")
STANDARD_SCENARIO_EVALUATION_NAME = "scenarios"
STANDARD_SCENARIO_CONFIG_PATHS = {
    "easy": Path("configs/evaluation/scenarios/show_easy.yaml"),
    "medium": Path("configs/evaluation/scenarios/show_medium.yaml"),
    "hard": Path("configs/evaluation/scenarios/show_hard.yaml"),
}


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
    action_interface: str = "pid_position"
    rpm_delta_scale: float = 0.05
    include_dynamics_observation: bool = False
    include_previous_action: bool = False
    source_manifest_path: Path | None = None
    evaluation_name: str | None = None
    evaluation_suite_name: str | None = None
    suite_task_name: str | None = None
    suite_task_names: tuple[str, ...] = ()
    suite_config_snapshot_path: Path | None = None
    suite_config_snapshot_path_relative: str | None = None
    suite_config_sha256: str | None = None
    source_run_name: str | None = None
    source_run_kind: str | None = None
    source_curriculum_kind: str | None = None
    source_stage: dict[str, Any] | None = None
    model_scope: str | None = None
    evaluated_model_source: str | None = None
    own_task_source: str | None = None
    own_task_config_path: Path | None = None
    own_task_distribution_config_path: Path | None = None
    own_task_shape: str | None = None
    own_task_is_show: bool | None = None
    own_task_fallback_used: bool = False
    own_task_fallback_reason: str | None = None

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
class OwnTaskResolution:
    """Resolved task/config pair for one own-task evaluation."""

    task_config_path: Path
    task_index: int
    task_shape: str
    source: str
    config_path: Path
    distribution_config_path: Path | None
    task_is_show: bool
    fallback_used: bool
    fallback_reason: str | None


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


@dataclass(frozen=True)
class PolicySuiteEvaluationResult:
    """Aggregate result returned after evaluating one direct PPO run on a suite."""

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PolicyStandardEvaluationResult:
    """Aggregate result returned after running the standard direct PPO profile."""

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PolicyScenarioEvaluationResult:
    """Aggregate result returned after evaluating the standard show scenarios."""

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _ScenarioEvaluationEnvSettings:
    """PPO-facing env settings resolved for standard scenario evaluation."""

    normalize_actions: bool
    action_interface: str
    rpm_delta_scale: float
    include_dynamics_observation: bool
    include_previous_action: bool
    source_manifest_path: Path | None
    training_config_path: Path | None
    final_stage_manifest_path: Path | None


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
        "evaluation_name": spec.evaluation_name,
        "evaluation_suite_name": spec.evaluation_suite_name,
        "suite_task_name": spec.suite_task_name,
        "suite_task_names": list(spec.suite_task_names),
        "suite_task_count": len(spec.suite_task_names),
        "suite_config_snapshot_path": None if spec.suite_config_snapshot_path is None else str(spec.suite_config_snapshot_path),
        "suite_config_snapshot_path_relative": spec.suite_config_snapshot_path_relative,
        "suite_config_sha256": spec.suite_config_sha256,
        "evaluation_suite": spec.evaluation_suite_name or spec.evaluation_name,
        "evaluated_task_name": spec.suite_task_name,
        "source_run_name": spec.source_run_name,
        "source_run_kind": spec.source_run_kind,
        "source_curriculum_kind": spec.source_curriculum_kind,
        "source_stage": spec.source_stage,
        "model_scope": spec.model_scope,
        "model_path": str(spec.model_path),
        "evaluated_model_path": str(spec.model_path),
        "evaluated_model_source": spec.evaluated_model_source or "specified",
        "task_config_path_used_for_evaluation": str(spec.task_config_path),
        "task_shape_used_for_evaluation": spec.task_shape,
        "own_task_source": spec.own_task_source,
        "own_task_config_path": None if spec.own_task_config_path is None else str(spec.own_task_config_path),
        "own_task_distribution_config_path": None if spec.own_task_distribution_config_path is None else str(spec.own_task_distribution_config_path),
        "own_task_shape": spec.own_task_shape,
        "own_task_is_show": spec.own_task_is_show,
        "own_task_fallback_used": bool(spec.own_task_fallback_used),
        "own_task_fallback_reason": spec.own_task_fallback_reason,
        "source_manifest_path": None if spec.source_manifest_path is None else str(spec.source_manifest_path),
        "action_interface": spec.action_interface,
        "rpm_delta_scale": spec.rpm_delta_scale if spec.action_interface == "direct_rpm" else None,
        "normalize_actions": spec.normalize_actions,
        "include_dynamics_observation": spec.include_dynamics_observation,
        "include_previous_action": spec.include_previous_action,
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


def run_direct_policy_own_task_evaluation(
    run_manifest_path: str | Path,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
) -> PolicySuiteEvaluationResult:
    """Evaluate a direct PPO model on its recorded training task snapshot."""
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)
    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("run_kind") != "direct_ppo":
        message = "own-task policy evaluation requires a direct PPO run manifest"
        raise ValueError(message)

    run_name = _required_text(run_manifest.get("run_name"), "run_name")
    run_root = manifest_path.expanduser().resolve(strict=False).parent
    storage_root = utils.artifacts.storage_root_from_run_dir(run_root)
    training = _mapping(run_manifest.get("training"), "training")
    model_path, evaluated_model_source = _select_manifest_model_path(run_root=run_root, payload=training, field_prefix="training")
    own_task = _resolve_direct_own_task(run_manifest=run_manifest, run_root=run_root, run_name=run_name, storage_root=storage_root)
    total_timesteps = int(run_manifest.get("total_timesteps", 0))
    normalize_actions = bool(run_manifest.get("normalize_actions", True))
    eval_steps = int(run_manifest.get("eval_steps", 120))
    seed = int(run_manifest.get("seed", 0))
    output_root = utils.artifacts.get_run_evaluation_dir(run_name, OWN_TASK_EVALUATION_NAME, storage_root=storage_root)

    result = run_policy_evaluation(
        PolicyEvaluationSpec(
            label=f"{run_name}_{OWN_TASK_EVALUATION_NAME}",
            model_role="direct_ppo",
            model_path=model_path,
            task_config_path=own_task.task_config_path,
            task_index=own_task.task_index,
            task_shape=own_task.task_shape,
            output_dir=output_root,
            eval_steps=eval_steps,
            seed=seed,
            total_timesteps=total_timesteps,
            normalize_actions=normalize_actions,
            **_evaluation_env_kwargs_from_manifest(run_manifest, manifest_path),
            evaluation_name=OWN_TASK_EVALUATION_NAME,
            evaluation_suite_name=None,
            suite_task_name=OWN_TASK_EVALUATION_NAME,
            suite_task_names=(OWN_TASK_EVALUATION_NAME,),
            source_run_name=run_name,
            source_run_kind="direct_ppo",
            model_scope="direct",
            evaluated_model_source=evaluated_model_source,
            own_task_source=own_task.source,
            own_task_config_path=own_task.config_path,
            own_task_distribution_config_path=own_task.distribution_config_path,
            own_task_shape=own_task.task_shape,
            own_task_is_show=own_task.task_is_show,
            own_task_fallback_used=own_task.fallback_used,
            own_task_fallback_reason=own_task.fallback_reason,
        ),
        PolicyEvaluationArtifactOptions(),
    )
    evaluated_models = [_evaluated_model_entry(result=result, run_root=run_root, suite_task_name=OWN_TASK_EVALUATION_NAME)]
    update_run_evaluation_index(
        manifest_path,
        _evaluation_index_entry(
            run_root=run_root,
            run_name=run_name,
            run_kind="direct_ppo",
            evaluation_name=OWN_TASK_EVALUATION_NAME,
            evaluation_suite_name=None,
            suite_config_snapshot_path=None,
            suite_config_snapshot_path_relative=None,
            suite_config_sha256=None,
            aggregate_metrics_path=Path(result.metrics_path),
            aggregate_manifest_path=Path(result.manifest_path),
            model_label=run_name,
            model_role="direct_ppo",
            model_path=model_path,
            task_names=[OWN_TASK_EVALUATION_NAME],
            evaluated_models=evaluated_models,
            mode="direct_policy_own_task_evaluation",
        ),
    )
    return PolicySuiteEvaluationResult(metrics_path=result.metrics_path, manifest_path=result.manifest_path, metrics=result.metrics)


def run_direct_policy_standard_evaluation(
    run_manifest_path: str | Path,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
) -> PolicyStandardEvaluationResult:
    """Run the standard direct PPO evaluation profile."""
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)
    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("run_kind") != "direct_ppo":
        message = "standard policy evaluation requires a direct PPO run manifest"
        raise ValueError(message)
    run_root = manifest_path.expanduser().resolve(strict=False).parent
    run_name = _required_text(run_manifest.get("run_name"), "run_name")

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if _direct_training_task_config_path(run_manifest=run_manifest, run_root=run_root) is None:
        skipped.append({"evaluation_name": OWN_TASK_EVALUATION_NAME, "reason": "training task snapshot unavailable"})
    else:
        own_task = run_direct_policy_own_task_evaluation(run_manifest_path=manifest_path, wandb_mode=wandb_mode)
        results.append(_profile_result_entry(OWN_TASK_EVALUATION_NAME, own_task))

    for suite_path in _standard_suite_paths():
        suite = evaluation_suites.load_evaluation_suite(suite_path)
        result = run_direct_policy_suite_evaluation(
            run_manifest_path=manifest_path,
            suite_path=suite_path,
            wandb_mode=wandb_mode,
        )
        results.append(_profile_result_entry(suite.evaluation_name, result))

    scenarios = run_direct_policy_scenario_evaluation(run_manifest_path=manifest_path, wandb_mode=wandb_mode)
    results.append(_profile_result_entry(STANDARD_SCENARIO_EVALUATION_NAME, scenarios))

    updated_manifest = _read_json(manifest_path)
    index_path = _evaluation_index_path_from_manifest(run_manifest=updated_manifest, run_root=run_root)
    index_payload = _read_json(index_path) if index_path.exists() else {"run_name": run_name, "evaluations": []}
    profile_payload = {
        "run_type": "evaluation",
        "run_kind": "direct_ppo",
        "mode": "direct_policy_standard_evaluation",
        "profile_name": STANDARD_EVALUATION_PROFILE,
        "run_name": run_name,
        "evaluation_names": [entry["evaluation_name"] for entry in results],
        "evaluations": results,
        "skipped_evaluations": skipped,
        "evaluation_index_path": str(index_path),
        "evaluation_index_path_relative": utils.artifacts.path_relative_to(index_path, run_root),
        "evaluation_index": index_payload,
    }
    return PolicyStandardEvaluationResult(metrics_path=str(index_path), manifest_path=str(index_path), metrics=profile_payload)


def run_direct_policy_scenario_evaluation(
    run_manifest_path: str | Path,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
) -> PolicyScenarioEvaluationResult:
    """Evaluate a direct PPO run on the standard easy/medium/hard show scenarios."""
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)
    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("run_kind") != "direct_ppo":
        message = "policy scenario evaluation requires a direct PPO run manifest"
        raise ValueError(message)

    run_name = _required_text(run_manifest.get("run_name"), "run_name")
    run_root = manifest_path.expanduser().resolve(strict=False).parent
    training = _mapping(run_manifest.get("training"), "training")
    model_path, evaluated_model_source = _select_manifest_model_path(run_root=run_root, payload=training, field_prefix="training")
    return run_standard_scenario_evaluations(
        run_root=run_root,
        run_name=run_name,
        model_path=model_path,
        model_run_name=run_name,
        source_run_kind="direct_ppo",
        source_curriculum_kind=None,
        model_scope="direct",
        evaluated_model_source=evaluated_model_source,
        run_manifest_path=manifest_path,
    )


def run_standard_scenario_evaluations(
    *,
    run_root: Path,
    run_name: str,
    model_path: Path,
    model_run_name: str | None,
    source_run_kind: str,
    source_curriculum_kind: str | None,
    model_scope: str,
    evaluated_model_source: str | None,
    run_manifest_path: Path | None = None,
    scenario_config_paths: Mapping[str, Path] | None = None,
    normalize_actions: bool | None = None,
    action_interface: str | None = None,
    rpm_delta_scale: float | None = None,
    include_dynamics_observation: bool | None = None,
    include_previous_action: bool | None = None,
    training_config_path: Path | None = None,
    final_stage_manifest_path: Path | None = None,
    output_root: Path | None = None,
    source_stage: Mapping[str, Any] | None = None,
) -> PolicyScenarioEvaluationResult:
    """Evaluate one trained model on the standard easy, medium, and hard show scenarios."""
    scenario_paths = dict(scenario_config_paths or STANDARD_SCENARIO_CONFIG_PATHS)
    env_settings = _scenario_evaluation_env_settings(
        run_manifest_path=run_manifest_path,
        run_root=run_root,
        source_run_kind=source_run_kind,
        normalize_actions=normalize_actions,
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        training_config_path=training_config_path,
        final_stage_manifest_path=final_stage_manifest_path,
    )
    scenario_output_root = output_root or run_root / utils.artifacts.EVALUATIONS_DIRNAME / STANDARD_SCENARIO_EVALUATION_NAME
    entries: list[dict[str, Any]] = []
    for scenario_label, scenario_path in scenario_paths.items():
        loaded = scenario_render.load_scenario_render_settings(scenario_path)
        composition = scenario_render.compose_scenario_reference(loaded)
        output_dir = scenario_output_root / _safe_name(scenario_label)
        settings = scenario_render.ScenarioRenderSettings(
            scenario_config_path=loaded.scenario_config_path,
            scenario_name=loaded.scenario_name,
            task_config_path=loaded.task_config_path,
            phases=loaded.phases,
            controller=policy_render.PPO_CONTROLLER,
            model_run_name=model_run_name,
            model_path=model_path,
            run_name=f"{run_name}_{scenario_label}_scenario",
            output_dir=output_dir,
            max_steps=loaded.max_steps,
            seed=loaded.seed,
            camera_mode=loaded.camera_mode,
            camera_distance=loaded.camera_distance,
            camera_yaw=loaded.camera_yaw,
            camera_pitch=loaded.camera_pitch,
            gif_filename=loaded.gif_filename,
            manifest_filename=loaded.manifest_filename,
            frame_interval=loaded.frame_interval,
            image_width=loaded.image_width,
            image_height=loaded.image_height,
            start_hold_sec=loaded.start_hold_sec,
            final_hold_sec=loaded.final_hold_sec,
            normalize_actions=env_settings.normalize_actions,
            action_interface=env_settings.action_interface,
            rpm_delta_scale=env_settings.rpm_delta_scale,
            include_dynamics_observation=env_settings.include_dynamics_observation,
            include_previous_action=env_settings.include_previous_action,
            source_manifest_path=env_settings.source_manifest_path,
            training_config_path=env_settings.training_config_path,
            final_stage_manifest_path=env_settings.final_stage_manifest_path,
            evaluated_model_source=evaluated_model_source,
        )
        scenario_result = scenario_render.run_scenario_render(settings)
        scenario_artifacts = _write_standard_scenario_metrics_and_diagnostics(
            run_root=run_root,
            run_name=run_name,
            output_dir=output_dir,
            scenario_label=scenario_label,
            scenario_path=scenario_path,
            settings=settings,
            composition=composition,
            scenario_result=scenario_result,
            model_path=model_path,
            model_run_name=model_run_name,
            source_run_kind=source_run_kind,
            source_curriculum_kind=source_curriculum_kind,
            model_scope=model_scope,
            evaluated_model_source=evaluated_model_source,
            source_stage=source_stage,
        )
        entry = {
            "scenario_label": scenario_label,
            "scenario_name": scenario_artifacts["metrics"].get("scenario_name"),
            "evaluation_name": STANDARD_SCENARIO_EVALUATION_NAME,
            "scenario_config_path": str(scenario_path),
            "output_dir": str(output_dir),
            "output_dir_relative": utils.artifacts.path_relative_to(output_dir, run_root),
            "metrics_path": str(scenario_artifacts["metrics_path"]),
            "metrics_path_relative": utils.artifacts.path_relative_to(scenario_artifacts["metrics_path"], run_root),
            "diagnostics_path": str(scenario_artifacts["diagnostics_path"]),
            "diagnostics_path_relative": utils.artifacts.path_relative_to(scenario_artifacts["diagnostics_path"], run_root),
            "manifest_path": scenario_result.manifest_path,
            "manifest_path_relative": utils.artifacts.path_relative_to(scenario_result.manifest_path, run_root),
            "gif_path": scenario_result.gif_path,
            "gif_path_relative": utils.artifacts.path_relative_to(scenario_result.gif_path, run_root),
            "model_path": str(model_path),
            "model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
            "model_run_name": model_run_name,
            "source_run_name": run_name,
            "source_run_kind": source_run_kind,
            "source_curriculum_kind": source_curriculum_kind,
            "source_stage": None if source_stage is None else dict(source_stage),
            "model_scope": model_scope,
            "evaluated_model_source": evaluated_model_source,
            "failure_overall_status": scenario_artifacts["diagnostics"].get("failure_overall_status"),
            "failure_primary_mode": scenario_artifacts["diagnostics"].get("failure_primary_mode"),
            "failure_modes": scenario_artifacts["diagnostics"].get("failure_modes"),
            "warnings": list(scenario_result.warnings),
            "metrics": dict(scenario_artifacts["metrics"]),
            "diagnostics": dict(scenario_artifacts["diagnostics"]),
        }
        entries.append(entry)

    metrics_dir = scenario_output_root / utils.artifacts.METRICS_DIRNAME
    manifests_dir = scenario_output_root / utils.artifacts.MANIFESTS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = f"{run_name}_{STANDARD_SCENARIO_EVALUATION_NAME}"
    metrics_path = metrics_dir / f"{filename_stem}_metrics.json"
    manifest_path = manifests_dir / f"{filename_stem}_manifest.json"
    aggregate_metrics = {
        "run_type": "evaluation",
        "run_kind": source_run_kind,
        "mode": "standard_scenario_evaluation",
        "run_name": run_name,
        "evaluation_name": STANDARD_SCENARIO_EVALUATION_NAME,
        "scenario_labels": list(scenario_paths),
        "scenario_names": [str(entry.get("scenario_name")) for entry in entries],
        "scenario_count": len(entries),
        "model_path": str(model_path),
        "model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
        "model_run_name": model_run_name,
        "source_run_name": run_name,
        "source_run_kind": source_run_kind,
        "source_curriculum_kind": source_curriculum_kind,
        "source_stage": None if source_stage is None else dict(source_stage),
        "model_scope": model_scope,
        "evaluated_model_source": evaluated_model_source,
        "canonical_evaluation_owner": "curriculum_stage" if source_stage is not None else "run_root",
        "final_stage_index": None if source_stage is None else int(source_stage["stage_index"]),
        "final_stage_name": None if source_stage is None else str(source_stage["stage_name"]),
        "final_stage_evaluation_path": None if source_stage is None else str(scenario_output_root),
        "final_stage_evaluation_path_relative": None if source_stage is None else utils.artifacts.path_relative_to(scenario_output_root, run_root),
        "root_evaluation_outputs_duplicated": False,
        "evaluated_models": entries,
        "summary_metrics_path": str(metrics_path),
        "summary_metrics_path_relative": utils.artifacts.path_relative_to(metrics_path, run_root),
        "summary_manifest_path": str(manifest_path),
        "summary_manifest_path_relative": utils.artifacts.path_relative_to(manifest_path, run_root),
        "entry_count": len(entries),
    }
    aggregate_manifest = {
        key: aggregate_metrics[key]
        for key in (
            "run_type",
            "run_kind",
            "mode",
            "run_name",
            "evaluation_name",
            "scenario_labels",
            "scenario_names",
            "scenario_count",
            "model_path",
            "model_path_relative",
            "model_run_name",
            "source_run_name",
            "source_run_kind",
            "source_curriculum_kind",
            "source_stage",
            "model_scope",
            "evaluated_model_source",
            "canonical_evaluation_owner",
            "final_stage_index",
            "final_stage_name",
            "final_stage_evaluation_path",
            "final_stage_evaluation_path_relative",
            "root_evaluation_outputs_duplicated",
            "summary_metrics_path",
            "summary_metrics_path_relative",
            "summary_manifest_path",
            "summary_manifest_path_relative",
            "entry_count",
        )
    }
    _write_json(metrics_path, aggregate_metrics)
    _write_json(manifest_path, aggregate_manifest)

    if run_manifest_path is not None:
        update_run_evaluation_index(
            run_manifest_path,
            _evaluation_index_entry(
                run_root=run_root,
                run_name=run_name,
                run_kind=source_run_kind,
                evaluation_name=STANDARD_SCENARIO_EVALUATION_NAME,
                evaluation_suite_name=None,
                suite_config_snapshot_path=None,
                suite_config_snapshot_path_relative=None,
                suite_config_sha256=None,
                aggregate_metrics_path=metrics_path,
                aggregate_manifest_path=manifest_path,
                model_label=run_name,
                model_role=model_scope,
                model_path=model_path,
                task_names=list(scenario_paths),
                evaluated_models=entries,
                mode="standard_scenario_evaluation",
            ),
        )

    return PolicyScenarioEvaluationResult(metrics_path=str(metrics_path), manifest_path=str(manifest_path), metrics=aggregate_metrics)


def _write_standard_scenario_metrics_and_diagnostics(
    *,
    run_root: Path,
    run_name: str,
    output_dir: Path,
    scenario_label: str,
    scenario_path: Path,
    settings: scenario_render.ScenarioRenderSettings,
    composition: scenario_render.ScenarioComposition,
    scenario_result: scenario_render.ScenarioRenderResult,
    model_path: Path,
    model_run_name: str | None,
    source_run_kind: str,
    source_curriculum_kind: str | None,
    model_scope: str,
    evaluated_model_source: str | None,
    source_stage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write per-scenario metrics and diagnostics under one scenario output folder."""
    metrics_dir = output_dir / utils.artifacts.METRICS_DIRNAME
    diagnostics_dir = output_dir / utils.artifacts.DIAGNOSTICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    filename_stem = f"{_safe_name(run_name)}_{_safe_name(scenario_label)}_scenario"
    metrics_path = metrics_dir / f"{filename_stem}_metrics.json"
    diagnostics_path = diagnostics_dir / f"{filename_stem}_diagnostics.json"
    reference_metadata = _scenario_reference_metadata(composition)
    metrics_payload = {
        **dict(scenario_result.metrics),
        **reference_metadata,
        "run_type": "evaluation",
        "run_kind": source_run_kind,
        "mode": "standard_scenario_evaluation",
        "evaluation_name": STANDARD_SCENARIO_EVALUATION_NAME,
        "evaluation_suite_name": STANDARD_SCENARIO_EVALUATION_NAME,
        "scenario_label": scenario_label,
        "scenario_name": scenario_result.metrics.get("scenario_name") or settings.scenario_name,
        "scenario_config_path": str(scenario_path),
        "task_config_path": str(settings.task_config_path),
        "output_dir": str(output_dir),
        "output_dir_relative": utils.artifacts.path_relative_to(output_dir, run_root),
        "model_path": str(model_path),
        "model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
        "model_run_name": model_run_name,
        "source_run_name": run_name,
        "source_run_kind": source_run_kind,
        "source_curriculum_kind": source_curriculum_kind,
        "source_stage": None if source_stage is None else dict(source_stage),
        "model_scope": model_scope,
        "evaluated_model_source": evaluated_model_source,
        "source_manifest_path": None if settings.source_manifest_path is None else str(settings.source_manifest_path),
        "training_config_path": None if settings.training_config_path is None else str(settings.training_config_path),
        "final_stage_manifest_path": None if settings.final_stage_manifest_path is None else str(settings.final_stage_manifest_path),
        "action_interface": settings.action_interface,
        "rpm_delta_scale": settings.rpm_delta_scale if settings.action_interface == "direct_rpm" else None,
        "normalize_actions": bool(settings.normalize_actions),
        "include_dynamics_observation": bool(settings.include_dynamics_observation),
        "include_previous_action": bool(settings.include_previous_action),
        "evaluated_model_path": str(model_path),
        "evaluated_model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
        "manifest_path": scenario_result.manifest_path,
        "manifest_path_relative": utils.artifacts.path_relative_to(scenario_result.manifest_path, run_root),
        "gif_path": scenario_result.gif_path,
        "gif_path_relative": utils.artifacts.path_relative_to(scenario_result.gif_path, run_root),
        "metrics_path": str(metrics_path),
        "metrics_path_relative": utils.artifacts.path_relative_to(metrics_path, run_root),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_path_relative": utils.artifacts.path_relative_to(diagnostics_path, run_root),
    }
    diagnostics_payload = _standard_scenario_failure_diagnostics(metrics_payload)
    metrics_payload.update(
        {
            "failure_overall_status": diagnostics_payload["failure_overall_status"],
            "failure_primary_mode": diagnostics_payload["failure_primary_mode"],
            "failure_modes": list(diagnostics_payload["failure_modes"]),
            "diagnostics_summary": diagnostics_payload,
        }
    )
    _write_json(diagnostics_path, diagnostics_payload)
    _write_json(metrics_path, metrics_payload)
    _augment_standard_scenario_manifest(
        manifest_path=Path(scenario_result.manifest_path),
        run_root=run_root,
        metrics_path=metrics_path,
        diagnostics_path=diagnostics_path,
        output_dir=output_dir,
        metrics=metrics_payload,
    )
    return {
        "metrics_path": metrics_path,
        "diagnostics_path": diagnostics_path,
        "metrics": metrics_payload,
        "diagnostics": diagnostics_payload,
    }


def _augment_standard_scenario_manifest(
    *,
    manifest_path: Path,
    run_root: Path,
    metrics_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    metrics: Mapping[str, Any],
) -> None:
    """Add standard-evaluation paths and completion metadata to the scenario manifest."""
    manifest = _read_json(manifest_path)
    trace_path = _required_existing_path(manifest.get("trace_path"), "trace_path")
    gif_path = _required_existing_path(manifest.get("gif_path"), "gif_path")
    plot_paths = manifest.get("plot_paths")
    if not isinstance(plot_paths, Mapping) or not plot_paths:
        message = f"scenario manifest must include non-empty plot_paths: {manifest_path}"
        raise ValueError(message)
    for plot_name, plot_path in plot_paths.items():
        _required_existing_path(plot_path, f"plot_paths.{plot_name}")

    renders_dir = output_dir / utils.artifacts.RENDERS_DIRNAME
    traces_dir = output_dir / utils.artifacts.TRACES_DIRNAME
    plots_dir = output_dir / utils.artifacts.PLOTS_DIRNAME
    manifests_dir = output_dir / utils.artifacts.MANIFESTS_DIRNAME
    manifest.update(
        {
            "evaluation_name": STANDARD_SCENARIO_EVALUATION_NAME,
            "metrics_path": str(metrics_path),
            "metrics_path_relative": utils.artifacts.path_relative_to(metrics_path, run_root),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_path_relative": utils.artifacts.path_relative_to(diagnostics_path, run_root),
            "trace_path_relative": utils.artifacts.path_relative_to(trace_path, run_root),
            "gif_path_relative": utils.artifacts.path_relative_to(gif_path, run_root),
            "plots_dir": str(plots_dir),
            "plots_dir_relative": utils.artifacts.path_relative_to(plots_dir, run_root),
            "traces_dir": str(traces_dir),
            "traces_dir_relative": utils.artifacts.path_relative_to(traces_dir, run_root),
            "renders_dir": str(renders_dir),
            "renders_dir_relative": utils.artifacts.path_relative_to(renders_dir, run_root),
            "manifests_dir": str(manifests_dir),
            "manifests_dir_relative": utils.artifacts.path_relative_to(manifests_dir, run_root),
            "scenario_complete": True,
            "scenario_completion_requirements": {
                "metrics": True,
                "diagnostics": True,
                "manifest": True,
                "trace": True,
                "plot": True,
                "render": True,
            },
            "render_source": "evaluated_policy_rollout_trace",
            "policy_rollout_render_required": True,
            "standard_scenario_metrics": dict(metrics),
        }
    )
    _write_json(manifest_path, manifest)


def _required_existing_path(value: Any, field_name: str) -> Path:
    """Return an existing artifact path or raise a scenario-completeness error."""
    if value is None or not str(value).strip():
        message = f"scenario artifact manifest missing {field_name}"
        raise ValueError(message)
    path = Path(str(value))
    if not path.exists():
        message = f"scenario artifact {field_name} does not exist: {path}"
        raise FileNotFoundError(message)
    return path


def _scenario_reference_metadata(composition: scenario_render.ScenarioComposition) -> dict[str, Any]:
    """Return deterministic reference geometry metadata for scenario metrics."""
    positions = composition.reference.positions
    path_length_m = _reference_path_length_m(positions)
    moving_duration_sec = max(float(composition.scenario_duration_sec) - float(composition.start_hold_sec) - float(composition.final_hold_sec), 0.0)
    return {
        "scenario_duration_sec": float(composition.scenario_duration_sec),
        "scenario_reference_path_length_m": path_length_m,
        "scenario_reference_mean_speed_mps": None if moving_duration_sec <= 0.0 else float(path_length_m / moving_duration_sec),
        "scenario_phase_count": len(composition.phases),
        "scenario_phase_names": [phase.name for phase in composition.phases],
        "scenario_phase_types": [phase.phase_type for phase in composition.phases],
        "scenario_segments": _scenario_segments(composition),
        "start_hold_enabled": bool(composition.start_hold_sec > 0.0),
        "start_hold_sec": float(composition.start_hold_sec),
        "start_hold_steps": int(composition.start_hold_steps),
        "start_hold_step_range": None if composition.start_hold_step_range is None else dict(composition.start_hold_step_range),
        "final_hold_enabled": bool(composition.final_hold_sec > 0.0),
        "final_hold_sec": float(composition.final_hold_sec),
        "final_hold_steps": int(composition.final_hold_steps),
        "final_hold_step_range": None if composition.final_hold_step_range is None else dict(composition.final_hold_step_range),
        "reference_motion_steps": int(composition.reference_motion_steps),
        "total_reference_steps": int(composition.total_reference_steps),
    }


def _scenario_segments(composition: scenario_render.ScenarioComposition) -> list[dict[str, Any]]:
    """Return manifest-ready segment metadata for a composed scenario."""
    segments: list[dict[str, Any]] = []
    for index, phase in enumerate(composition.phases):
        segments.append(
            {
                "index": index,
                "name": phase.name,
                "type": phase.phase_type,
                "task_shape": phase.task_shape,
                "duration_sec": phase.duration_sec,
                "step_range": dict(composition.phase_step_ranges[index]),
                "time_range": dict(composition.phase_time_ranges[index]),
                "start_position": list(composition.phase_start_positions[index]),
                "end_position": list(composition.phase_end_positions[index]),
                "geometry": dict(composition.phase_geometry[index]),
            }
        )
    return segments


def _reference_path_length_m(positions: Any) -> float:
    """Return cumulative XYZ path length for a sampled reference position array."""
    rows = [tuple(float(value) for value in row) for row in positions]
    return float(sum(math.dist(previous, current) for previous, current in pairwise(rows)))


def _standard_scenario_failure_diagnostics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build compact failure diagnostics from scenario rollout metrics."""
    failure_modes: list[str] = []
    if metrics.get("truncated") is True:
        failure_modes.append("truncated")
    if metrics.get("completed_reference_motion") is False:
        failure_modes.append("incomplete_reference_motion")
    if metrics.get("completed_phase_holds") is False:
        failure_modes.append("incomplete_phase_holds")
    if metrics.get("completed_final_hold") is False:
        failure_modes.append("incomplete_final_hold")
    if metrics.get("completed_reference") is False:
        failure_modes.append("incomplete_reference")
    ended_normally = metrics.get("ended_normally")
    if ended_normally is False and not failure_modes:
        failure_modes.append("abnormal_termination")

    deduplicated_modes = list(dict.fromkeys(failure_modes))
    status = "passed" if not deduplicated_modes else "failed"
    primary_mode = "none" if not deduplicated_modes else deduplicated_modes[0]
    return {
        "run_type": "evaluation",
        "evaluation_name": STANDARD_SCENARIO_EVALUATION_NAME,
        "scenario_label": metrics.get("scenario_label"),
        "scenario_name": metrics.get("scenario_name"),
        "failure_overall_status": status,
        "failure_primary_mode": primary_mode,
        "failure_modes": deduplicated_modes,
        "termination_reason": metrics.get("termination_reason"),
        "terminated": metrics.get("terminated"),
        "truncated": metrics.get("truncated"),
        "completed_reference": metrics.get("completed_reference"),
        "completed_reference_motion": metrics.get("completed_reference_motion"),
        "completed_phase_holds": metrics.get("completed_phase_holds"),
        "completed_final_hold": metrics.get("completed_final_hold"),
        "mean_position_error_m": metrics.get("mean_position_error_m"),
        "final_position_error_m": metrics.get("final_position_error_m"),
        "max_position_error_m": metrics.get("max_position_error_m"),
        "warnings": list(metrics.get("warnings") or []),
        "evaluated_model_path": metrics.get("evaluated_model_path"),
        "evaluated_model_source": metrics.get("evaluated_model_source"),
        "model_scope": metrics.get("model_scope"),
    }


def _scenario_evaluation_env_settings(
    *,
    run_manifest_path: Path | None,
    run_root: Path,
    source_run_kind: str,
    normalize_actions: bool | None,
    action_interface: str | None,
    rpm_delta_scale: float | None,
    include_dynamics_observation: bool | None,
    include_previous_action: bool | None,
    training_config_path: Path | None,
    final_stage_manifest_path: Path | None,
) -> _ScenarioEvaluationEnvSettings:
    """Resolve scenario env flags from run/final-stage manifests, with explicit overrides."""
    inferred = _infer_scenario_evaluation_env_settings(
        run_manifest_path=run_manifest_path,
        run_root=run_root,
        source_run_kind=source_run_kind,
    )
    return _ScenarioEvaluationEnvSettings(
        normalize_actions=inferred.normalize_actions if normalize_actions is None else bool(normalize_actions),
        action_interface=inferred.action_interface if action_interface is None else str(action_interface),
        rpm_delta_scale=inferred.rpm_delta_scale if rpm_delta_scale is None else float(rpm_delta_scale),
        include_dynamics_observation=inferred.include_dynamics_observation
        if include_dynamics_observation is None
        else bool(include_dynamics_observation),
        include_previous_action=inferred.include_previous_action if include_previous_action is None else bool(include_previous_action),
        source_manifest_path=inferred.source_manifest_path,
        training_config_path=inferred.training_config_path if training_config_path is None else training_config_path,
        final_stage_manifest_path=inferred.final_stage_manifest_path if final_stage_manifest_path is None else final_stage_manifest_path,
    )


def _infer_scenario_evaluation_env_settings(
    *,
    run_manifest_path: Path | None,
    run_root: Path,
    source_run_kind: str,
) -> _ScenarioEvaluationEnvSettings:
    """Infer scenario env settings from direct run or curriculum summary manifests."""
    default = _default_scenario_evaluation_env_settings(run_manifest_path)
    if run_manifest_path is None or not Path(run_manifest_path).exists():
        return default

    manifest_path = Path(run_manifest_path)
    manifest = _read_json(manifest_path)
    run_kind = str(manifest.get("run_kind") or source_run_kind)
    if run_kind == "direct_ppo":
        return _direct_scenario_evaluation_env_settings(manifest=manifest, manifest_path=manifest_path, run_root=run_root)
    if run_kind == "curriculum":
        return _curriculum_scenario_evaluation_env_settings(manifest=manifest, manifest_path=manifest_path, run_root=run_root)
    return default


def _default_scenario_evaluation_env_settings(source_manifest_path: Path | None) -> _ScenarioEvaluationEnvSettings:
    """Return conservative default scenario env settings for compatibility-only callers."""
    return _ScenarioEvaluationEnvSettings(
        normalize_actions=True,
        action_interface="pid_position",
        rpm_delta_scale=0.05,
        include_dynamics_observation=False,
        include_previous_action=False,
        source_manifest_path=source_manifest_path,
        training_config_path=None,
        final_stage_manifest_path=None,
    )


def _direct_scenario_evaluation_env_settings(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    run_root: Path,
) -> _ScenarioEvaluationEnvSettings:
    """Resolve scenario env settings from a direct PPO run manifest."""
    env_kwargs = _evaluation_env_kwargs_from_manifest(manifest, manifest_path)
    config_payload = _mapping_or_empty(manifest.get("config"))
    return _ScenarioEvaluationEnvSettings(
        normalize_actions=bool(manifest.get("normalize_actions", config_payload.get("normalize_actions", True))),
        action_interface=str(env_kwargs["action_interface"]),
        rpm_delta_scale=float(env_kwargs["rpm_delta_scale"]),
        include_dynamics_observation=bool(env_kwargs["include_dynamics_observation"]),
        include_previous_action=bool(env_kwargs["include_previous_action"]),
        source_manifest_path=manifest_path,
        training_config_path=_training_config_path_from_manifest_payload(manifest, run_root=run_root),
        final_stage_manifest_path=None,
    )


def _curriculum_scenario_evaluation_env_settings(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    run_root: Path,
) -> _ScenarioEvaluationEnvSettings:
    """Resolve scenario env settings from the final stage of a curriculum manifest."""
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        return _default_scenario_evaluation_env_settings(manifest_path)
    final_stage = _mapping_or_empty(stages[-1])
    stage_manifest_path = None
    if isinstance(final_stage.get("manifest_path"), str) and str(final_stage.get("manifest_path")).strip():
        stage_manifest_path = _resolve_manifest_path(str(final_stage["manifest_path"]), run_root=run_root)
    stage_manifest = _mapping_or_empty(_read_json(stage_manifest_path) if stage_manifest_path is not None and stage_manifest_path.exists() else {})
    stage_config = _mapping_or_empty(stage_manifest.get("config"))
    return _ScenarioEvaluationEnvSettings(
        normalize_actions=bool(
            final_stage.get("normalize_actions", stage_manifest.get("normalize_actions", stage_config.get("normalize_actions", True)))
        ),
        action_interface=str(
            final_stage.get("action_interface") or stage_manifest.get("action_interface") or stage_config.get("action_interface") or "pid_position"
        ),
        rpm_delta_scale=float(
            final_stage.get("rpm_delta_scale") or stage_manifest.get("rpm_delta_scale") or stage_config.get("rpm_delta_scale") or 0.05
        ),
        include_dynamics_observation=bool(
            final_stage.get(
                "include_dynamics_observation",
                stage_manifest.get("include_dynamics_observation", stage_config.get("include_dynamics_observation", False)),
            )
        ),
        include_previous_action=bool(
            final_stage.get(
                "include_previous_action",
                stage_manifest.get("include_previous_action", stage_config.get("include_previous_action", False)),
            )
        ),
        source_manifest_path=manifest_path,
        training_config_path=_training_config_path_from_manifest_payload(stage_manifest, run_root=run_root),
        final_stage_manifest_path=stage_manifest_path,
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return a mapping payload or an empty mapping for optional manifest sections."""
    return value if isinstance(value, Mapping) else {}


def _training_config_path_from_manifest_payload(manifest: Mapping[str, Any], *, run_root: Path) -> Path | None:
    """Resolve the training config path recorded in a run or stage manifest."""
    config_payload = _mapping_or_empty(manifest.get("config"))
    for candidate in (
        manifest.get("training_config_path"),
        manifest.get("source_config_path"),
        config_payload.get("training_config_snapshot_path_relative"),
        config_payload.get("training_config_snapshot_path"),
        config_payload.get("training_config_path"),
        config_payload.get("source_config_path"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return _resolve_manifest_path(candidate, run_root=run_root)
    return None


def run_direct_policy_suite_evaluation(
    run_manifest_path: str | Path,
    suite_path: str | Path = evaluation_suites.DEFAULT_EVALUATION_SUITE_PATH,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
) -> PolicySuiteEvaluationResult:
    """Evaluate a direct PPO run on a suite and store outputs under that run."""
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)
    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("run_kind") != "direct_ppo":
        message = "policy suite evaluation requires a direct PPO run manifest"
        raise ValueError(message)

    run_name = _required_text(run_manifest.get("run_name"), "run_name")
    run_root = manifest_path.expanduser().resolve(strict=False).parent
    storage_root = utils.artifacts.storage_root_from_run_dir(run_root)
    suite = evaluation_suites.load_evaluation_suite(suite_path)
    snapshot = evaluation_suites.write_evaluation_suite_snapshot(
        run_name=run_name,
        suite=suite,
        suite_path=suite_path,
        storage_root=storage_root,
    )
    output_root = utils.artifacts.get_run_evaluation_dir(run_name, suite.evaluation_name, storage_root=storage_root)
    output_root.mkdir(parents=True, exist_ok=True)

    training = _mapping(run_manifest.get("training"), "training")
    model_path, evaluated_model_source = _select_manifest_model_path(run_root=run_root, payload=training, field_prefix="training")
    total_timesteps = int(run_manifest.get("total_timesteps", 0))
    normalize_actions = bool(run_manifest.get("normalize_actions", True))
    artifact_options = PolicyEvaluationArtifactOptions(
        render_enabled=suite.render.enabled,
        plots_enabled=suite.plots.enabled,
        trace_enabled=suite.traces.enabled,
        diagnostics_enabled=True,
        render_fps=suite.render.fps,
        render_max_steps=suite.render.max_steps,
    )

    evaluated_models: list[dict[str, Any]] = []
    for suite_task in suite.tasks:
        result = run_policy_evaluation(
            PolicyEvaluationSpec(
                label=f"{run_name}_{suite_task.task_name}",
                model_role="direct_ppo",
                model_path=model_path,
                task_config_path=snapshot.task_config_paths[suite_task.task_name],
                task_index=0,
                task_shape=suite_task.task_shape,
                output_dir=output_root / _safe_name(suite_task.task_name),
                eval_steps=suite.eval_steps,
                seed=suite.seed,
                total_timesteps=total_timesteps,
                normalize_actions=normalize_actions,
                **_evaluation_env_kwargs_from_manifest(run_manifest, manifest_path),
                evaluation_name=suite.evaluation_name,
                evaluation_suite_name=suite.evaluation_name,
                suite_task_name=suite_task.task_name,
                suite_task_names=tuple(suite.task_names),
                suite_config_snapshot_path=snapshot.suite_config_path,
                suite_config_snapshot_path_relative=snapshot.suite_config_path_relative,
                suite_config_sha256=snapshot.suite_config_sha256,
                source_run_name=run_name,
                source_run_kind="direct_ppo",
                model_scope="direct",
                evaluated_model_source=evaluated_model_source,
            ),
            artifact_options,
        )
        evaluated_models.append(_evaluated_model_entry(result=result, run_root=run_root, suite_task_name=suite_task.task_name))

    filename_stem = f"{run_name}_{suite.evaluation_name}"
    metrics_dir = output_root / utils.artifacts.METRICS_DIRNAME
    manifests_dir = output_root / utils.artifacts.MANIFESTS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{filename_stem}_metrics.json"
    aggregate_manifest_path = manifests_dir / f"{filename_stem}_manifest.json"

    aggregate_metrics = {
        "run_type": "evaluation",
        "run_kind": "direct_ppo",
        "mode": "direct_policy_suite_evaluation",
        "run_name": run_name,
        "evaluation_name": suite.evaluation_name,
        "evaluation_suite_name": suite.evaluation_name,
        "evaluation_suite_path": str(Path(suite_path)),
        "suite_config_snapshot_path": str(snapshot.suite_config_path),
        "suite_config_snapshot_path_relative": snapshot.suite_config_path_relative,
        "suite_config_sha256": snapshot.suite_config_sha256,
        "suite_task_names": suite.task_names,
        "suite_task_count": len(suite.task_names),
        "model_label": run_name,
        "model_role": "direct_ppo",
        "model_path": str(model_path),
        "model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
        "evaluated_model_path": str(model_path),
        "evaluated_model_path_relative": utils.artifacts.path_relative_to(model_path, run_root),
        "evaluated_model_source": evaluated_model_source,
        "source_run_name": run_name,
        "source_run_kind": "direct_ppo",
        "source_curriculum_kind": None,
        "source_stage": None,
        "model_scope": "direct",
        "evaluated_models": evaluated_models,
        "summary_metrics_path": str(metrics_path),
        "summary_metrics_path_relative": utils.artifacts.path_relative_to(metrics_path, run_root),
        "summary_manifest_path": str(aggregate_manifest_path),
        "summary_manifest_path_relative": utils.artifacts.path_relative_to(aggregate_manifest_path, run_root),
        "entry_count": len(evaluated_models),
    }
    aggregate_manifest = {
        key: aggregate_metrics[key]
        for key in (
            "run_type",
            "run_kind",
            "mode",
            "run_name",
            "evaluation_name",
            "evaluation_suite_name",
            "evaluation_suite_path",
            "suite_config_snapshot_path",
            "suite_config_snapshot_path_relative",
            "suite_config_sha256",
            "suite_task_names",
            "suite_task_count",
            "model_label",
            "model_role",
            "model_path",
            "model_path_relative",
            "evaluated_model_path",
            "evaluated_model_path_relative",
            "evaluated_model_source",
            "summary_metrics_path",
            "summary_metrics_path_relative",
            "summary_manifest_path",
            "summary_manifest_path_relative",
            "entry_count",
        )
    }
    _write_json(metrics_path, aggregate_metrics)
    _write_json(aggregate_manifest_path, aggregate_manifest)

    update_run_evaluation_index(
        manifest_path,
        _evaluation_index_entry(
            run_root=run_root,
            run_name=run_name,
            run_kind="direct_ppo",
            evaluation_name=suite.evaluation_name,
            evaluation_suite_name=suite.evaluation_name,
            suite_config_snapshot_path=snapshot.suite_config_path,
            suite_config_snapshot_path_relative=snapshot.suite_config_path_relative,
            suite_config_sha256=snapshot.suite_config_sha256,
            aggregate_metrics_path=metrics_path,
            aggregate_manifest_path=aggregate_manifest_path,
            model_label=run_name,
            model_role="direct_ppo",
            model_path=model_path,
            task_names=suite.task_names,
            evaluated_models=evaluated_models,
            mode="direct_policy_suite_evaluation",
        ),
    )
    return PolicySuiteEvaluationResult(metrics_path=str(metrics_path), manifest_path=str(aggregate_manifest_path), metrics=aggregate_metrics)


def update_run_evaluation_index(run_manifest_path: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Upsert one linked evaluation entry into a run-scoped evaluation index."""
    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    run_root = manifest_path.expanduser().resolve(strict=False).parent
    run_name = _required_text(run_manifest.get("run_name"), "run_name")
    index_path = _evaluation_index_path_from_manifest(run_manifest=run_manifest, run_root=run_root)
    index_payload = _read_json(index_path) if index_path.exists() else {"run_name": run_name, "evaluations": []}
    existing_entries = index_payload.get("evaluations", [])
    if not isinstance(existing_entries, list):
        existing_entries = []
    entry_payload = dict(entry)
    entry_key = str(entry_payload.get("index_key") or entry_payload.get("evaluation_name"))
    entries = [candidate for candidate in existing_entries if not isinstance(candidate, Mapping) or candidate.get("index_key") != entry_key]
    entries.append(entry_payload)
    index_payload = {
        "run_name": run_name,
        "run_kind": run_manifest.get("run_kind"),
        "index_path": str(index_path),
        "index_path_relative": utils.artifacts.path_relative_to(index_path, run_root),
        "entry_count": len(entries),
        "evaluations": entries,
    }
    _write_json(index_path, index_payload)
    run_manifest["evaluation_index"] = {
        "path": str(index_path),
        "path_relative": utils.artifacts.path_relative_to(index_path, run_root),
        "entry_count": len(entries),
        "evaluations": entries,
    }
    _write_json(manifest_path, run_manifest)
    return index_payload


def _evaluated_model_entry(result: PolicyEvaluationResult, run_root: Path, suite_task_name: str | None) -> dict[str, Any]:
    """Build a manifest/index entry for one evaluated model-task pair."""
    metrics = result.metrics
    return {
        "label": result.label,
        "model_label": result.label,
        "model_role": result.model_role,
        "model_path": result.model_path,
        "model_path_relative": utils.artifacts.path_relative_to(result.model_path, run_root),
        "evaluated_model_path": metrics.get("evaluated_model_path"),
        "evaluated_model_path_relative": utils.artifacts.path_relative_to(metrics.get("evaluated_model_path"), run_root),
        "evaluated_model_source": metrics.get("evaluated_model_source"),
        "suite_task_name": suite_task_name,
        "task_config_path": result.task_config_path,
        "task_config_path_relative": utils.artifacts.path_relative_to(result.task_config_path, run_root),
        "task_shape": result.task_shape,
        "evaluation_dir": result.output_dir,
        "evaluation_dir_relative": utils.artifacts.path_relative_to(result.output_dir, run_root),
        "metrics_path": result.metrics_path,
        "metrics_path_relative": utils.artifacts.path_relative_to(result.metrics_path, run_root),
        "manifest_path": result.manifest_path,
        "manifest_path_relative": utils.artifacts.path_relative_to(result.manifest_path, run_root),
        "trace_path": result.trace_path,
        "trace_path_relative": utils.artifacts.path_relative_to(result.trace_path, run_root),
        "gif_path": result.gif_path,
        "gif_path_relative": utils.artifacts.path_relative_to(result.gif_path, run_root),
        "plot_paths": dict(result.plot_paths),
        "plot_paths_relative": {key: utils.artifacts.path_relative_to(value, run_root) for key, value in result.plot_paths.items()},
        "eval_steps": metrics.get("eval_steps"),
        "seed": metrics.get("seed"),
        "source_run_name": metrics.get("source_run_name"),
        "source_run_kind": metrics.get("source_run_kind"),
        "source_curriculum_kind": metrics.get("source_curriculum_kind"),
        "source_stage": metrics.get("source_stage"),
        "model_scope": metrics.get("model_scope"),
        "source_manifest_path": metrics.get("source_manifest_path"),
        "action_interface": metrics.get("action_interface"),
        "rpm_delta_scale": metrics.get("rpm_delta_scale"),
        "normalize_actions": metrics.get("normalize_actions"),
        "include_dynamics_observation": metrics.get("include_dynamics_observation"),
        "include_previous_action": metrics.get("include_previous_action"),
        "termination_limits_mode": metrics.get("termination_limits_mode"),
        "base_truncation_policy": metrics.get("base_truncation_policy"),
        "strict_limit_violation_count": metrics.get("strict_limit_violation_count"),
    }


def _evaluation_index_entry(
    *,
    run_root: Path,
    run_name: str,
    run_kind: str,
    evaluation_name: str,
    evaluation_suite_name: str | None,
    suite_config_snapshot_path: Path | None,
    suite_config_snapshot_path_relative: str | None,
    suite_config_sha256: str | None,
    aggregate_metrics_path: Path,
    aggregate_manifest_path: Path,
    model_label: str,
    model_role: str,
    model_path: Path | str | None,
    task_names: list[str],
    evaluated_models: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Build a link-only evaluation index entry."""
    model_path_text = None if model_path is None else str(model_path)
    source_stage = next(
        (entry.get("source_stage") for entry in evaluated_models if isinstance(entry.get("source_stage"), Mapping)),
        None,
    )
    stage_evaluation_path = None if source_stage is None else Path(aggregate_metrics_path).parent.parent
    return {
        "index_key": f"{mode}:{evaluation_name}",
        "run_name": run_name,
        "run_kind": run_kind,
        "mode": mode,
        "evaluation_name": evaluation_name,
        "evaluation_suite_name": evaluation_suite_name,
        "suite_name": evaluation_suite_name,
        "suite_config_snapshot_path": None if suite_config_snapshot_path is None else str(suite_config_snapshot_path),
        "suite_config_snapshot_path_relative": suite_config_snapshot_path_relative,
        "suite_config_snapshot_relative": suite_config_snapshot_path_relative,
        "suite_config_sha256": suite_config_sha256,
        "aggregate_metrics_path": str(aggregate_metrics_path),
        "aggregate_metrics_path_relative": utils.artifacts.path_relative_to(aggregate_metrics_path, run_root),
        "aggregate_metrics_relative": utils.artifacts.path_relative_to(aggregate_metrics_path, run_root),
        "evaluation_manifest_path": str(aggregate_manifest_path),
        "evaluation_manifest_path_relative": utils.artifacts.path_relative_to(aggregate_manifest_path, run_root),
        "evaluation_manifest_relative": utils.artifacts.path_relative_to(aggregate_manifest_path, run_root),
        "model_label": model_label,
        "model_role": model_role,
        "model_path": model_path_text,
        "model_path_relative": utils.artifacts.path_relative_to(model_path_text, run_root),
        "source_stage": None if source_stage is None else dict(source_stage),
        "final_stage_index": None if source_stage is None else int(source_stage["stage_index"]),
        "final_stage_name": None if source_stage is None else str(source_stage["stage_name"]),
        "final_stage_evaluation_path": None if stage_evaluation_path is None else str(stage_evaluation_path),
        "final_stage_evaluation_path_relative": None
        if stage_evaluation_path is None
        else utils.artifacts.path_relative_to(stage_evaluation_path, run_root),
        "root_evaluation_outputs_duplicated": False,
        "task_names": list(task_names),
        "evaluated_models": evaluated_models,
    }


def _evaluation_index_path_from_manifest(run_manifest: Mapping[str, Any], run_root: Path) -> Path:
    """Resolve the canonical root evaluation index path for a run manifest."""
    del run_manifest
    return run_root / utils.artifacts.EVALUATION_INDEX_FILENAME


def _resolve_direct_own_task(
    *,
    run_manifest: Mapping[str, Any],
    run_root: Path,
    run_name: str,
    storage_root: Path | None,
) -> OwnTaskResolution:
    """Resolve the exact direct-PPO own task or distribution representative."""
    distribution_config_path = _direct_training_task_distribution_config_path(run_manifest=run_manifest, run_root=run_root)
    if distribution_config_path is not None:
        task, source, fallback_used, fallback_reason = _own_task_from_distribution_config(distribution_config_path)
        task_shape = str(task.get(validation.contracts.FIELD_SHAPE) or "")
        config_path = _write_direct_own_task_config(
            run_name=run_name,
            storage_root=storage_root,
            source=source,
            task=task,
        )
        return OwnTaskResolution(
            task_config_path=config_path,
            task_index=0,
            task_shape=task_shape,
            source=source,
            config_path=config_path,
            distribution_config_path=distribution_config_path,
            task_is_show=_task_is_show(task),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    task_config_path = _direct_training_task_config_path(run_manifest=run_manifest, run_root=run_root)
    if task_config_path is None:
        message = "direct PPO run manifest must include a training task config snapshot for own_task evaluation"
        raise ValueError(message)
    task_shape = _direct_training_task_shape(run_manifest)
    return OwnTaskResolution(
        task_config_path=task_config_path,
        task_index=_direct_training_task_index(run_manifest),
        task_shape=task_shape,
        source="training_task_snapshot",
        config_path=task_config_path,
        distribution_config_path=None,
        task_is_show=task_shape == validation.contracts.SHAPE_BASIC_TRAINING_SHOW,
        fallback_used=False,
        fallback_reason=None,
    )


def _direct_training_task_distribution_config_path(run_manifest: Mapping[str, Any], run_root: Path) -> Path | None:
    """Return the configured training distribution path for direct PPO, when present."""
    config_payload = run_manifest.get("config")
    candidates: list[Any] = [run_manifest.get("task_distribution_config_path")]
    if isinstance(config_payload, Mapping):
        candidates.append(config_payload.get("task_distribution_config_path"))
        task_distribution = config_payload.get("task_distribution")
        if isinstance(task_distribution, Mapping):
            candidates.append(task_distribution.get("task_distribution_config_path"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return _resolve_manifest_path(candidate, run_root=run_root)
    return None


def _own_task_from_distribution_config(distribution_config_path: Path) -> tuple[dict[str, Any], str, bool, str | None]:
    """Load a deterministic own-task representative from a task-distribution config."""
    payload = yaml.safe_load(distribution_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        message = f"task distribution config root must be a mapping: {distribution_config_path}"
        raise TypeError(message)
    raw_distribution = payload.get(envs.task_distribution.DISTRIBUTION_CONFIG_KEY, {})
    if raw_distribution is not None and not isinstance(raw_distribution, Mapping):
        message = f"task_distribution must be a mapping: {distribution_config_path}"
        raise TypeError(message)
    distribution_payload = dict(raw_distribution or {})
    representative = payload.get("own_task_representative", distribution_payload.get("own_task_representative"))
    settings = envs.task_distribution.load_task_distribution_settings(distribution_config_path)
    if isinstance(representative, Mapping):
        task = dict(representative)
        source = "task_distribution_own_task_representative"
        fallback_used = False
        fallback_reason = None
    else:
        task = dict(settings.base_task)
        source = "task_distribution_base_task"
        fallback_used = settings.base_task_shape != validation.contracts.SHAPE_BASIC_TRAINING_SHOW
        fallback_reason = None
        if fallback_used:
            fallback_reason = f"no own_task_representative found in {distribution_config_path}; used distribution base_task"
    result = validation.tasks.validate_task(task, limits=settings.validation_limits)
    if not result.is_valid:
        details = "; ".join(result.messages)
        message = f"invalid own_task representative in {distribution_config_path}: {details}"
        raise ValueError(message)
    return task, source, fallback_used, fallback_reason


def _write_direct_own_task_config(
    *,
    run_name: str,
    storage_root: Path | None,
    source: str,
    task: Mapping[str, Any],
) -> Path:
    """Write one run-local own-task config used by distribution-trained direct PPO evaluation."""
    config_dir = utils.artifacts.get_run_config_dir(run_name, storage_root=storage_root) / "own_task"
    config_dir.mkdir(parents=True, exist_ok=True)
    shape = str(task.get(validation.contracts.FIELD_SHAPE) or "task")
    config_path = config_dir / f"{_safe_name(source)}_{_safe_name(shape)}.yaml"
    payload = {
        "name": f"{run_name}_own_task",
        "own_task_source": source,
        "tasks": [dict(task)],
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _resolve_manifest_path(value: str, *, run_root: Path) -> Path:
    """Resolve a manifest path relative to the run root when appropriate."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve(strict=False)
    run_relative = (run_root / path).resolve(strict=False)
    if run_relative.exists():
        return run_relative
    return path.resolve(strict=False)


def _task_is_show(task: Mapping[str, Any]) -> bool:
    """Return whether an own-task representative is a show-style task."""
    return bool(
        task.get("task_is_show")
        or task.get("show_name")
        or str(task.get(validation.contracts.FIELD_SHAPE) or "") == validation.contracts.SHAPE_BASIC_TRAINING_SHOW
    )


def _direct_training_task_config_path(run_manifest: Mapping[str, Any], run_root: Path) -> Path | None:
    """Return the task config snapshot used for direct PPO training when available."""
    config_payload = run_manifest.get("config")
    if not isinstance(config_payload, Mapping):
        return None
    relative_path = config_payload.get("task_config_snapshot_path_relative")
    absolute_path = config_payload.get("task_config_snapshot_path")
    if isinstance(relative_path, str) and relative_path:
        return (run_root / relative_path).resolve(strict=False)
    if isinstance(absolute_path, str) and absolute_path:
        path = Path(absolute_path)
        return path.resolve(strict=False) if path.is_absolute() else (run_root / path).resolve(strict=False)
    source_path = config_payload.get("task_config_path")
    if isinstance(source_path, str) and source_path:
        path = Path(source_path)
        return path.resolve(strict=False)
    return None


def _direct_training_task_index(run_manifest: Mapping[str, Any]) -> int:
    """Return the task index selected during direct PPO training."""
    config_payload = run_manifest.get("config")
    if isinstance(config_payload, Mapping):
        return int(config_payload.get("task_index", 0))
    return int(run_manifest.get("task_index", 0))


def _direct_training_task_shape(run_manifest: Mapping[str, Any]) -> str:
    """Return the task shape selected during direct PPO training."""
    config_payload = run_manifest.get("config")
    if isinstance(config_payload, Mapping):
        shape = config_payload.get("task_shape") or config_payload.get("training_task_shape")
        if shape is not None and str(shape).strip():
            return str(shape)
    return _required_text(run_manifest.get("task_shape") or run_manifest.get("training_task_shape"), "config.task_shape")


def _standard_suite_paths() -> list[Path]:
    """Return suite configs included in the standard direct PPO profile."""
    return [STANDARD_GENERALIZATION_SUITE_PATH]


def _profile_result_entry(evaluation_name: str, result: PolicySuiteEvaluationResult | PolicyScenarioEvaluationResult) -> dict[str, Any]:
    """Return a compact profile result link for one evaluation step."""
    return {
        "evaluation_name": evaluation_name,
        "metrics_path": result.metrics_path,
        "manifest_path": result.manifest_path,
    }


def _resolve_manifest_path_value(
    *,
    run_root: Path,
    absolute_value: Any,
    relative_value: Any,
    field_name: str,
) -> Path:
    """Resolve a manifest path using a relative value first, then an absolute/local value."""
    if isinstance(relative_value, str) and relative_value:
        return (run_root / relative_value).resolve(strict=False)
    if not isinstance(absolute_value, str) or not absolute_value:
        message = f"run manifest must contain {field_name}"
        raise ValueError(message)
    path = Path(absolute_value)
    return path.resolve(strict=False) if path.is_absolute() else (run_root / path).resolve(strict=False)


def _select_manifest_model_path(*, run_root: Path, payload: Mapping[str, Any], field_prefix: str) -> tuple[Path, str]:
    """Select the best available model path from a manifest payload."""
    for source, absolute_key, relative_key in (
        ("best", "best_model_path", "best_model_path_relative"),
        ("last", "last_model_path", "last_model_path_relative"),
        ("last", "model_path", "model_path_relative"),
    ):
        absolute_value = payload.get(absolute_key)
        relative_value = payload.get(relative_key)
        if (isinstance(relative_value, str) and relative_value) or (isinstance(absolute_value, str) and absolute_value):
            return (
                _resolve_manifest_path_value(
                    run_root=run_root,
                    absolute_value=absolute_value,
                    relative_value=relative_value,
                    field_name=f"{field_prefix}.{absolute_key}",
                ),
                source,
            )
    message = f"run manifest must contain {field_prefix}.best_model_path, {field_prefix}.last_model_path, or {field_prefix}.model_path"
    raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"expected JSON object at {path}"
        raise TypeError(message)
    return payload


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a mapping field from a manifest."""
    if not isinstance(value, Mapping):
        message = f"run manifest must contain a mapping at {field_name}"
        raise TypeError(message)
    return value


def _required_text(value: Any, field_name: str) -> str:
    """Return a required non-empty text field."""
    if value is None or not str(value).strip():
        message = f"run manifest must contain {field_name}"
        raise ValueError(message)
    return str(value)


def _evaluation_env_kwargs_from_manifest(run_manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Return environment identity flags recorded in a training manifest."""
    raw_config_payload = run_manifest.get("config")
    config_payload = raw_config_payload if isinstance(raw_config_payload, Mapping) else {}
    return {
        "action_interface": str(run_manifest.get("action_interface") or config_payload.get("action_interface") or "pid_position"),
        "rpm_delta_scale": float(run_manifest.get("rpm_delta_scale") or config_payload.get("rpm_delta_scale") or 0.05),
        "include_dynamics_observation": bool(
            run_manifest.get("include_dynamics_observation", config_payload.get("include_dynamics_observation", False))
        ),
        "include_previous_action": bool(run_manifest.get("include_previous_action", config_payload.get("include_previous_action", False))),
        "source_manifest_path": manifest_path,
    }


def _evaluation_env_kwargs_from_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Return environment identity flags recorded for a curriculum stage."""
    return {
        "action_interface": str(stage.get("action_interface") or "pid_position"),
        "rpm_delta_scale": float(stage.get("rpm_delta_scale") or 0.05),
        "include_dynamics_observation": bool(stage.get("include_dynamics_observation", False)),
        "include_previous_action": bool(stage.get("include_previous_action", False)),
        "source_manifest_path": Path(str(stage["manifest_path"])) if stage.get("manifest_path") else None,
    }


def _make_policy_evaluation_env(spec: PolicyEvaluationSpec, task: Mapping[str, Any], record: bool, max_steps: int | None) -> Any:
    """Build the PPO-facing evaluation environment from training manifest flags."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(
        task,
        gui=False,
        record=record,
        max_steps=max_steps,
        action_interface=spec.action_interface,
        rpm_delta_scale=spec.rpm_delta_scale,
        include_dynamics_observation=spec.include_dynamics_observation,
        include_previous_action=spec.include_previous_action,
    )
    if spec.normalize_actions or envs.actions.parse_action_interface(spec.action_interface) == envs.actions.ActionInterface.DIRECT_RPM:
        return envs.tracking_env.make_normalized_action_env(real_env)
    return real_env


def _load_ppo_with_evaluation_env(ppo_class: Any, *, spec: PolicyEvaluationSpec, tracking_env: Any) -> Any:
    """Load a PPO model and re-raise observation mismatches with manifest context."""
    try:
        return ppo_class.load(str(spec.model_path), env=tracking_env, device="cpu")
    except ValueError as exc:
        if "Observation spaces do not match" not in str(exc):
            raise
        model_observation_space = _model_observation_space_for_error(ppo_class, spec.model_path)
        message = (
            "evaluation environment observation space does not match the saved PPO model: "
            f"model_observation_space={model_observation_space}; "
            f"env_observation_space={getattr(tracking_env, 'observation_space', None)}; "
            f"manifest_path={spec.source_manifest_path}; model_path={spec.model_path}; "
            f"action_interface={spec.action_interface}; "
            f"include_dynamics_observation={spec.include_dynamics_observation}; "
            f"include_previous_action={spec.include_previous_action}"
        )
        raise ValueError(message) from exc


def _model_observation_space_for_error(ppo_class: Any, model_path: Path) -> Any:
    """Best-effort saved model observation-space lookup for error messages."""
    try:
        model = ppo_class.load(str(model_path), device="cpu")
    except Exception:  # noqa: BLE001 - best-effort diagnostic context.
        return "unavailable"
    return getattr(model, "observation_space", "unavailable")


def _collect_diagnostics(
    spec: PolicyEvaluationSpec,
    task: dict[str, Any],
    diagnostics_dir: Path,
) -> tuple[evaluation.diagnostics.PolicyEvaluationDiagnostics, dict[str, Any]]:
    """Run deterministic diagnostics and write diagnostics artifacts."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    tracking_env = _make_policy_evaluation_env(spec=spec, task=task, record=False, max_steps=None)
    try:
        model = _load_ppo_with_evaluation_env(PPO, spec=spec, tracking_env=tracking_env)
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

    from src.experiments.rendering import experiments_rendering_policy as policy_render  # noqa: PLC0415

    settings = policy_render.PolicyRenderSettings(
        model_path=spec.model_path,
        max_steps=render_steps,
        seed=spec.seed,
        gif_filename="scenario_rollout.gif",
        frame_interval=_frame_interval_for_fps(render_fps),
        normalize_actions=spec.normalize_actions,
        action_interface=spec.action_interface,
        rpm_delta_scale=spec.rpm_delta_scale,
        include_dynamics_observation=spec.include_dynamics_observation,
        include_previous_action=spec.include_previous_action,
        source_manifest_path=spec.source_manifest_path,
        evaluated_model_source=spec.evaluated_model_source,
    )
    render_env = _make_policy_evaluation_env(spec=spec, task=task, record=False, max_steps=render_steps)
    try:
        model = _load_ppo_with_evaluation_env(PPO, spec=spec, tracking_env=render_env)
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
        "evaluation_name",
        "evaluation_suite_name",
        "suite_task_name",
        "suite_task_names",
        "suite_task_count",
        "suite_config_snapshot_path",
        "suite_config_snapshot_path_relative",
        "suite_config_sha256",
        "evaluation_suite",
        "evaluated_task_name",
        "source_run_name",
        "source_run_kind",
        "source_curriculum_kind",
        "source_stage",
        "model_scope",
        "model_path",
        "evaluated_model_path",
        "evaluated_model_source",
        "task_config_path_used_for_evaluation",
        "task_shape_used_for_evaluation",
        "own_task_source",
        "own_task_config_path",
        "own_task_distribution_config_path",
        "own_task_shape",
        "own_task_is_show",
        "own_task_fallback_used",
        "own_task_fallback_reason",
        "source_manifest_path",
        "action_interface",
        "rpm_delta_scale",
        "normalize_actions",
        "include_dynamics_observation",
        "include_previous_action",
        "termination_limits_mode",
        "termination_limits",
        "diagnostic_limits",
        "base_truncation_policy",
        "terminate_on_base_truncation",
        "strict_limit_violation_count",
        "strict_limit_violation_causes",
        "base_truncated_count",
        "base_truncation_effective_count",
        "base_truncation_ignored_count",
        "base_truncation_causes",
        "project_truncated_count",
        "project_truncation_causes",
        "recovery_allowed_after_limit_violation_count",
        "max_abs_roll_pitch_rad",
        "max_speed_mps",
        "max_angular_velocity_radps",
        "actual_z_min_m",
        "actual_z_max_m",
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
        "final_hold_enabled",
        "final_hold_sec",
        "exclude_final_hold_from_tracking_metrics",
        "tracking_phase_end_step",
        "tracking_phase_end_time_sec",
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
        "failure_overall_status",
        "curriculum_feedback_version",
        "curriculum_feedback_summary",
        "curriculum_current_task_family",
        "curriculum_current_difficulty_level",
        "curriculum_primary_skill_gaps",
        "curriculum_diagnostic_signals",
        "curriculum_strategy",
        "curriculum_recommended_next_task_families",
        "curriculum_avoid_next_task_families",
        "curriculum_constraints_for_next",
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
    "OWN_TASK_EVALUATION_NAME",
    "STANDARD_EVALUATION_PROFILE",
    "STANDARD_GENERALIZATION_SUITE_PATH",
    "STANDARD_SCENARIO_CONFIG_PATHS",
    "STANDARD_SCENARIO_EVALUATION_NAME",
    "PolicyEvaluationArtifactOptions",
    "PolicyEvaluationResult",
    "PolicyEvaluationSpec",
    "PolicyScenarioEvaluationResult",
    "PolicyStandardEvaluationResult",
    "PolicySuiteEvaluationResult",
    "run_direct_policy_own_task_evaluation",
    "run_direct_policy_scenario_evaluation",
    "run_direct_policy_standard_evaluation",
    "run_direct_policy_suite_evaluation",
    "run_policy_evaluation",
    "run_standard_scenario_evaluations",
    "update_run_evaluation_index",
]
