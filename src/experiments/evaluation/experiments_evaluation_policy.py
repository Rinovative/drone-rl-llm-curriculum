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
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import envs, evaluation, utils, validation
from src.experiments import experiments_config as config
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites

DEFAULT_RENDER_FPS = 20
SIMULATOR_CAPTURE_FPS = 30
OWN_TASK_EVALUATION_NAME = "own_task"
STANDARD_EVALUATION_PROFILE = "standard"
STANDARD_LINE_EVALUATION_SUITE_PATH = Path("configs/evaluation/line_eval_suite.yaml")
STANDARD_FINAL_BENCHMARK_SUITE_PATH = Path("configs/evaluation/final_benchmark_eval_suite.yaml")
STANDARD_GENERALIZATION_SUITE_PATH = Path("configs/evaluation/generalization_eval_suite.yaml")


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
    model_path = _resolve_manifest_path_value(
        run_root=run_root,
        absolute_value=training.get("model_path"),
        relative_value=training.get("model_path_relative"),
        field_name="training.model_path",
    )
    task_config_path = _direct_training_task_config_path(run_manifest=run_manifest, run_root=run_root)
    if task_config_path is None:
        message = "direct PPO run manifest must include a training task config snapshot for own_task evaluation"
        raise ValueError(message)
    task_index = _direct_training_task_index(run_manifest)
    task_shape = _direct_training_task_shape(run_manifest)
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
            task_config_path=task_config_path,
            task_index=task_index,
            task_shape=task_shape,
            output_dir=output_root,
            eval_steps=eval_steps,
            seed=seed,
            total_timesteps=total_timesteps,
            normalize_actions=normalize_actions,
            evaluation_name=OWN_TASK_EVALUATION_NAME,
            evaluation_suite_name=None,
            suite_task_name=OWN_TASK_EVALUATION_NAME,
            suite_task_names=(OWN_TASK_EVALUATION_NAME,),
            source_run_name=run_name,
            source_run_kind="direct_ppo",
            model_scope="direct",
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
    model_path = _resolve_manifest_path_value(
        run_root=run_root,
        absolute_value=training.get("model_path"),
        relative_value=training.get("model_path_relative"),
        field_name="training.model_path",
    )
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
        "task_names": list(task_names),
        "evaluated_models": evaluated_models,
    }


def _evaluation_index_path_from_manifest(run_manifest: Mapping[str, Any], run_root: Path) -> Path:
    """Resolve the canonical root evaluation index path for a run manifest."""
    del run_manifest
    return run_root / utils.artifacts.EVALUATION_INDEX_FILENAME


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
    paths = [STANDARD_LINE_EVALUATION_SUITE_PATH, STANDARD_FINAL_BENCHMARK_SUITE_PATH]
    if STANDARD_GENERALIZATION_SUITE_PATH.is_file():
        paths.append(STANDARD_GENERALIZATION_SUITE_PATH)
    return paths


def _profile_result_entry(evaluation_name: str, result: PolicySuiteEvaluationResult) -> dict[str, Any]:
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

    from src.experiments.rendering import experiments_rendering_policy as policy_render  # noqa: PLC0415

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
    "OWN_TASK_EVALUATION_NAME",
    "STANDARD_EVALUATION_PROFILE",
    "STANDARD_FINAL_BENCHMARK_SUITE_PATH",
    "STANDARD_GENERALIZATION_SUITE_PATH",
    "STANDARD_LINE_EVALUATION_SUITE_PATH",
    "PolicyEvaluationArtifactOptions",
    "PolicyEvaluationResult",
    "PolicyEvaluationSpec",
    "PolicyStandardEvaluationResult",
    "PolicySuiteEvaluationResult",
    "run_direct_policy_own_task_evaluation",
    "run_direct_policy_standard_evaluation",
    "run_direct_policy_suite_evaluation",
    "run_policy_evaluation",
    "update_run_evaluation_index",
]
