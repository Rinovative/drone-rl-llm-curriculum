"""
===============================================================================
experiments_curriculum_evaluation.py
===============================================================================
Evaluate manual-curriculum PPO checkpoints through shared evaluation suites.

Responsibilities:
  - Load curriculum summaries and canonical evaluation suite definitions
  - Build concrete own-stage and suite-driven evaluation specs
  - Delegate per-model execution to the shared policy evaluation helper
  - Aggregate compact curriculum-level metrics and manifests

Design principles:
  - Keep curriculum evaluation focused on planning and aggregation
  - Keep benchmark tasks suite-driven and deterministic
  - Fail clearly when required summary fields, suite tasks, or model paths are invalid

Boundaries:
  - Rollout, diagnostics, plotting, and rendering logic belong in shared helpers
  - Training behavior and environment physics are out of scope
===============================================================================

"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import utils
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites

DEFAULT_EVALUATION_SUITE_PATH = evaluation_suites.DEFAULT_EVALUATION_SUITE_PATH
SUPPORTED_EVALUATION_MODES = ("own-stage", "suite")
DEFAULT_EVALUATION_MODE = "suite"
SUPPORTED_MODEL_SCOPES = ("all-stages", "final-stage")
DEFAULT_MODEL_SCOPE = "all-stages"
DEFAULT_RENDER_FPS = policy_evaluation.DEFAULT_RENDER_FPS


@dataclass(frozen=True)
class CurriculumEvaluationResult:
    """
    Aggregate result returned after one curriculum evaluation mode run.

    Parameters
    ----------
    metrics_path
        Path to the aggregate summary metrics JSON.
    manifest_path
        Path to the aggregate summary manifest JSON.
    metrics
        JSON-serializable aggregate summary payload.

    """

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


def run_curriculum_evaluation(
    summary_path: str | Path,
    mode: str = DEFAULT_EVALUATION_MODE,
    suite_path: str | Path | None = DEFAULT_EVALUATION_SUITE_PATH,
    model_scope: str = DEFAULT_MODEL_SCOPE,
    include_baseline_model: str | Path | None = None,
    baseline_label: str = "baseline",
    eval_steps: int | None = None,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
    render: bool | None = None,
    render_fps: int | None = None,
    render_max_steps: int | None = None,
    plots: bool | None = None,
    traces: bool | None = None,
) -> CurriculumEvaluationResult:
    """
    Run curriculum evaluation using own-stage tasks or a canonical evaluation suite.

    Parameters
    ----------
    summary_path
        Curriculum summary JSON path.
    mode
        Evaluation mode: ``suite`` or ``own-stage``. Defaults to suite evaluation.
    suite_path
        Canonical evaluation suite YAML path required for ``suite`` mode.
    model_scope
        Stage model selection scope: ``all-stages`` or ``final-stage``.
    include_baseline_model
        Optional baseline model path evaluated alongside selected curriculum models in suite mode.
    baseline_label
        Human-readable baseline label.
    eval_steps
        Optional evaluation-step override. Suite mode defaults to the suite ``eval_steps`` value.
    wandb_mode
        Accepted for CLI symmetry.
    render
        Optional render-enabled override. Uses suite settings when omitted in suite mode.
    render_fps
        Optional render FPS override. Uses suite settings when omitted in suite mode.
    render_max_steps
        Optional render-step override. Uses suite settings when omitted in suite mode.
    plots
        Optional plot-enabled override. Uses suite settings when omitted in suite mode.
    traces
        Optional trace-enabled override. Uses suite settings when omitted in suite mode.

    Returns
    -------
    CurriculumEvaluationResult
        Aggregate summary metrics and manifest paths.

    """
    _validate_mode_and_wandb(mode=mode, wandb_mode=wandb_mode)
    _validate_model_scope(model_scope)
    summary = _read_json(Path(summary_path))
    stages = _stages(summary)
    suite = _load_suite_for_mode(mode=mode, suite_path=suite_path)

    curriculum_run_name = _curriculum_run_name(summary)
    evaluation_name = _evaluation_name(mode=mode, suite=suite)
    output_root = _mode_output_root(summary=summary, evaluation_name=evaluation_name)
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_options = _artifact_options_from_suite(
        suite=suite,
        render=render,
        render_fps=render_fps,
        render_max_steps=render_max_steps,
        plots=plots,
        traces=traces,
    )

    spec_payloads = _evaluation_spec_payloads(
        summary=summary,
        stages=stages,
        mode=mode,
        output_root=output_root,
        curriculum_run_name=curriculum_run_name,
        evaluation_name=evaluation_name,
        suite=suite,
        model_scope=model_scope,
        include_baseline_model=include_baseline_model,
        baseline_label=baseline_label,
        eval_steps_override=eval_steps,
    )

    evaluated_models: list[dict[str, Any]] = []
    for payload in spec_payloads:
        result = policy_evaluation.run_policy_evaluation(payload["spec"], artifact_options)
        evaluated_models.append(
            _evaluated_model_entry(
                result=result,
                stage_index=payload.get("stage_index"),
                stage_name=payload.get("stage_name"),
                source_run_name=payload.get("source_run_name"),
                is_final_stage=bool(payload.get("is_final_stage", False)),
                evaluation_suite_name=None if suite is None else suite.evaluation_name,
                suite_task_name=payload.get("suite_task_name"),
            )
        )

    filename_stem = _summary_filename_stem(summary=summary, mode=mode, suite=suite)
    metrics_dir = output_root / utils.artifacts.METRICS_DIRNAME
    manifests_dir = output_root / utils.artifacts.MANIFESTS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{filename_stem}_metrics.json"
    manifest_path = manifests_dir / f"{filename_stem}_manifest.json"
    suite_path_for_metrics = None if suite is None or suite_path is None else str(Path(suite_path))
    suite_task_names = [] if suite is None else suite.task_names

    aggregate_metrics = {
        "run_type": "evaluation",
        "run_kind": "curriculum",
        "curriculum_kind": summary.get("curriculum_kind"),
        "mode": "curriculum_evaluation",
        "curriculum_name": str(summary["curriculum_name"]),
        "seed": int(summary.get("seed", 0)),
        "evaluation_mode": mode,
        "evaluation_name": evaluation_name,
        "evaluation_suite_name": None if suite is None else suite.evaluation_name,
        "evaluation_suite_path": suite_path_for_metrics,
        "suite_task_names": suite_task_names,
        "suite_task_count": len(suite_task_names),
        "curriculum_run_name": curriculum_run_name,
        "model_scope": model_scope,
        "evaluated_models": evaluated_models,
        "summary_metrics_path": str(metrics_path),
        "summary_manifest_path": str(manifest_path),
        "entry_count": len(evaluated_models),
    }
    aggregate_manifest = {
        "run_type": "evaluation",
        "run_kind": "curriculum",
        "curriculum_kind": summary.get("curriculum_kind"),
        "mode": "curriculum_evaluation",
        "curriculum_name": str(summary["curriculum_name"]),
        "seed": int(summary.get("seed", 0)),
        "evaluation_mode": mode,
        "evaluation_name": evaluation_name,
        "evaluation_suite_name": None if suite is None else suite.evaluation_name,
        "evaluation_suite_path": suite_path_for_metrics,
        "suite_task_names": suite_task_names,
        "suite_task_count": len(suite_task_names),
        "curriculum_run_name": curriculum_run_name,
        "model_scope": model_scope,
        "summary_metrics_path": str(metrics_path),
        "summary_manifest_path": str(manifest_path),
        "entry_count": len(evaluated_models),
    }

    _write_json(metrics_path, aggregate_metrics)
    _write_json(manifest_path, aggregate_manifest)
    return CurriculumEvaluationResult(
        metrics_path=str(metrics_path),
        manifest_path=str(manifest_path),
        metrics=aggregate_metrics,
    )


def _validate_mode_and_wandb(mode: str, wandb_mode: str) -> None:
    """Validate supported evaluation mode and W&B mode values."""
    if mode not in SUPPORTED_EVALUATION_MODES:
        message = f"mode must be one of: {', '.join(SUPPORTED_EVALUATION_MODES)}"
        raise ValueError(message)
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)


def _validate_model_scope(model_scope: str) -> None:
    """Validate stage model selection scope."""
    if model_scope not in SUPPORTED_MODEL_SCOPES:
        message = f"model_scope must be one of: {', '.join(SUPPORTED_MODEL_SCOPES)}"
        raise ValueError(message)


def _load_suite_for_mode(
    mode: str,
    suite_path: str | Path | None,
) -> evaluation_suites.EvaluationSuite | None:
    """Load the suite required by suite mode."""
    if mode == "own-stage":
        return None
    if suite_path is None:
        message = "--suite is required for suite mode"
        raise ValueError(message)
    return evaluation_suites.load_evaluation_suite(suite_path)


def _artifact_options_from_suite(
    suite: evaluation_suites.EvaluationSuite | None,
    render: bool | None,
    render_fps: int | None,
    render_max_steps: int | None,
    plots: bool | None,
    traces: bool | None,
) -> policy_evaluation.PolicyEvaluationArtifactOptions:
    """Resolve artifact options from a suite plus explicit overrides."""
    render_enabled = (True if suite is None else suite.render.enabled) if render is None else render
    plots_enabled = (True if suite is None else suite.plots.enabled) if plots is None else plots
    traces_enabled = (True if suite is None else suite.traces.enabled) if traces is None else traces
    resolved_render_fps = (DEFAULT_RENDER_FPS if suite is None else suite.render.fps) if render_fps is None else render_fps
    resolved_render_max_steps = (None if suite is None else suite.render.max_steps) if render_max_steps is None else render_max_steps
    return policy_evaluation.PolicyEvaluationArtifactOptions(
        render_enabled=render_enabled,
        plots_enabled=plots_enabled,
        trace_enabled=traces_enabled,
        diagnostics_enabled=True,
        render_fps=resolved_render_fps,
        render_max_steps=resolved_render_max_steps,
    )


def _evaluation_spec_payloads(
    summary: Mapping[str, Any],
    stages: list[Mapping[str, Any]],
    mode: str,
    output_root: Path,
    curriculum_run_name: str,
    evaluation_name: str,
    suite: evaluation_suites.EvaluationSuite | None,
    model_scope: str,
    include_baseline_model: str | Path | None,
    baseline_label: str,
    eval_steps_override: int | None,
) -> list[dict[str, Any]]:
    """Build model-evaluation spec payloads for the requested mode."""
    selected_stages = _selected_stages(stages=stages, model_scope=model_scope)
    if mode == "own-stage":
        return _own_stage_payloads(
            stages=selected_stages,
            all_stages=stages,
            curriculum_run_name=curriculum_run_name,
            evaluation_name=evaluation_name,
            eval_steps_override=eval_steps_override,
            default_seed=int(summary.get("seed", 0)),
        )

    if suite is None:
        message = "suite must be provided for suite mode"
        raise ValueError(message)
    return _suite_payloads(
        stages=stages,
        selected_stages=selected_stages,
        suite=suite,
        output_root=output_root,
        curriculum_run_name=curriculum_run_name,
        evaluation_name=evaluation_name,
        model_scope=model_scope,
        include_baseline_model=include_baseline_model,
        baseline_label=baseline_label,
        eval_steps_override=eval_steps_override,
    )


def _suite_payloads(
    stages: list[Mapping[str, Any]],
    selected_stages: list[Mapping[str, Any]],
    suite: evaluation_suites.EvaluationSuite,
    output_root: Path,
    curriculum_run_name: str,
    evaluation_name: str,
    model_scope: str,
    include_baseline_model: str | Path | None,
    baseline_label: str,
    eval_steps_override: int | None,
) -> list[dict[str, Any]]:
    """Build suite-task evaluation payloads for selected curriculum models."""
    task_config_paths = _write_suite_task_configs(curriculum_run_name=curriculum_run_name, suite=suite)
    payloads: list[dict[str, Any]] = []
    final_stage_index = int(stages[-1]["stage_index"])
    final_stage_run_level = model_scope == "final-stage"

    for stage in selected_stages:
        stage_index = int(stage["stage_index"])
        stage_name = str(stage["stage_name"])
        stage_dir_name = f"stage{stage_index:02d}_{stage_name}"
        stage_output_root = (
            output_root
            if final_stage_run_level
            else utils.artifacts.get_curriculum_stage_evaluation_dir(
                curriculum_run_name,
                stage_index,
                stage_name,
                evaluation_name,
            )
        )
        label_prefix = "final_stage" if final_stage_run_level else stage_dir_name
        payloads.extend(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=f"{label_prefix}_{suite_task.task_name}",
                    model_role="stage",
                    model_path=Path(str(stage["model_path"])),
                    task_config_path=task_config_paths[suite_task.task_name],
                    task_index=0,
                    task_shape=suite_task.task_shape,
                    output_dir=stage_output_root / _safe_name(suite_task.task_name),
                    eval_steps=int(eval_steps_override or suite.eval_steps),
                    seed=suite.seed,
                    total_timesteps=int(stage.get("total_timesteps", 0)),
                    normalize_actions=bool(stage.get("normalize_actions", True)),
                ),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "source_run_name": stage.get("run_name"),
                "is_final_stage": stage_index == final_stage_index,
                "suite_task_name": suite_task.task_name,
            }
            for suite_task in suite.tasks
        )

    if include_baseline_model is not None:
        baseline_dir_name = f"baseline_{_safe_name(baseline_label)}"
        payloads.extend(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=f"{baseline_dir_name}_{suite_task.task_name}",
                    model_role="baseline",
                    model_path=Path(str(include_baseline_model)),
                    task_config_path=task_config_paths[suite_task.task_name],
                    task_index=0,
                    task_shape=suite_task.task_shape,
                    output_dir=output_root / "baselines" / baseline_dir_name / _safe_name(suite_task.task_name),
                    eval_steps=int(eval_steps_override or suite.eval_steps),
                    seed=suite.seed,
                    total_timesteps=0,
                    normalize_actions=True,
                ),
                "stage_index": None,
                "stage_name": None,
                "source_run_name": baseline_dir_name,
                "is_final_stage": False,
                "suite_task_name": suite_task.task_name,
            }
            for suite_task in suite.tasks
        )

    return payloads


def _selected_stages(stages: list[Mapping[str, Any]], model_scope: str) -> list[Mapping[str, Any]]:
    """Return curriculum stages selected by model scope."""
    if model_scope == "all-stages":
        return stages
    if model_scope == "final-stage":
        return [stages[-1]]
    message = f"unsupported model_scope: {model_scope}"
    raise ValueError(message)


def _own_stage_payloads(
    stages: list[Mapping[str, Any]],
    all_stages: list[Mapping[str, Any]],
    curriculum_run_name: str,
    evaluation_name: str,
    eval_steps_override: int | None,
    default_seed: int,
) -> list[dict[str, Any]]:
    """Build stage-indexed evaluation payloads for own-stage mode."""
    payloads: list[dict[str, Any]] = []
    final_stage_index = int(all_stages[-1]["stage_index"])
    for stage in stages:
        stage_manifest = _read_json(Path(str(stage["manifest_path"])))
        stage_index = int(stage["stage_index"])
        stage_name = str(stage["stage_name"])
        stage_dir_name = f"stage{stage_index:02d}_{stage_name}"
        stage_output_dir = utils.artifacts.get_curriculum_stage_evaluation_dir(
            curriculum_run_name,
            stage_index,
            stage_name,
            evaluation_name,
        )
        payloads.append(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=stage_dir_name,
                    model_role="stage",
                    model_path=Path(str(stage["model_path"])),
                    task_config_path=Path(str(stage_manifest["task_config_path"])),
                    task_index=int(stage_manifest.get("task_index", 0)),
                    task_shape=str(stage["task_shape"]),
                    output_dir=stage_output_dir,
                    eval_steps=int(eval_steps_override or stage.get("eval_steps") or stage_manifest.get("eval_steps") or 120),
                    seed=int(stage.get("seed", default_seed)),
                    total_timesteps=int(stage.get("total_timesteps", stage_manifest.get("total_timesteps", 0))),
                    normalize_actions=bool(stage.get("normalize_actions", stage_manifest.get("normalize_actions", True))),
                ),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "source_run_name": stage.get("run_name"),
                "is_final_stage": stage_index == final_stage_index,
                "suite_task_name": None,
            }
        )
    return payloads


def _mode_output_root(summary: Mapping[str, Any], evaluation_name: str) -> Path:
    """Return mode-scoped curriculum evaluation output root."""
    return utils.artifacts.get_run_evaluation_dir(_curriculum_run_name(summary), evaluation_name)


def _curriculum_run_name(summary: Mapping[str, Any]) -> str:
    """Return the canonical curriculum run name from a summary payload."""
    run_name = summary.get("run_name")
    if run_name is None or not str(run_name).strip():
        message = "curriculum summary must include canonical run_name"
        raise ValueError(message)
    return str(run_name)


def _evaluation_name(mode: str, suite: evaluation_suites.EvaluationSuite | None) -> str:
    """Return the canonical evaluation name for a curriculum evaluation mode."""
    if mode == "own-stage":
        return "own_stage"
    if suite is None:
        message = "suite must be provided for suite evaluation naming"
        raise ValueError(message)
    return suite.evaluation_name


def _summary_filename_stem(
    summary: Mapping[str, Any],
    mode: str,
    suite: evaluation_suites.EvaluationSuite | None,
) -> str:
    """Return summary filename stem for aggregate metrics/manifest outputs."""
    run_name = _curriculum_run_name(summary)
    if mode == "own-stage":
        return f"{run_name}_own_stage"
    if suite is None:
        message = "suite must be provided for suite summary naming"
        raise ValueError(message)
    return f"{run_name}_{suite.evaluation_name}"


def _write_suite_task_configs(
    curriculum_run_name: str,
    suite: evaluation_suites.EvaluationSuite,
) -> dict[str, Path]:
    """Copy a canonical suite and write one-task configs consumed by policy evaluation."""
    config_dir = utils.artifacts.get_run_config_evaluation_suites_dir(curriculum_run_name)
    config_dir.mkdir(parents=True, exist_ok=True)
    suite_stem = _safe_name(suite.evaluation_name)
    suite_copy_path = config_dir / f"{suite_stem}_eval_suite.yaml"
    suite_copy_path.write_text(_to_yaml(suite.to_dict()), encoding="utf-8")

    task_config_dir = config_dir / suite_stem
    task_config_dir.mkdir(parents=True, exist_ok=True)
    task_config_paths: dict[str, Path] = {}
    for suite_task in suite.tasks:
        task_config_path = task_config_dir / f"{_safe_name(suite_task.task_name)}_task.yaml"
        payload = {
            "name": suite_task.task_name,
            "evaluation_name": suite.evaluation_name,
            "suite_task_name": suite_task.task_name,
            "tasks": [copy.deepcopy(suite_task.task)],
        }
        task_config_path.write_text(_to_yaml(payload), encoding="utf-8")
        task_config_paths[suite_task.task_name] = task_config_path
    return task_config_paths


def _evaluated_model_entry(
    result: policy_evaluation.PolicyEvaluationResult,
    stage_index: int | None,
    stage_name: str | None,
    source_run_name: str | None,
    is_final_stage: bool,
    evaluation_suite_name: str | None,
    suite_task_name: str | None,
) -> dict[str, Any]:
    """Build one evaluated-model summary entry from a shared helper result."""
    metrics = result.metrics
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
    entry = {key: metrics.get(key) for key in keys}
    entry["stage_index"] = stage_index
    entry["stage_name"] = stage_name
    entry["is_final_stage"] = bool(is_final_stage)
    entry["evaluation_suite_name"] = evaluation_suite_name
    entry["suite_task_name"] = suite_task_name
    entry["source_run_name"] = source_run_name
    return entry


def _stages(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return validated stage entries from a curriculum summary payload."""
    raw_stages = summary.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        message = "curriculum summary must contain non-empty 'stages'"
        raise ValueError(message)
    stages = [stage for stage in raw_stages if isinstance(stage, Mapping)]
    if len(stages) != len(raw_stages):
        message = "curriculum summary contains non-mapping stage entries"
        raise TypeError(message)
    return stages


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name component."""
    text = value.strip().replace(" ", "_")
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in text)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"expected JSON object at {path}"
        raise TypeError(message)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable-formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize a compact evaluation config to YAML."""
    import yaml  # noqa: PLC0415

    return yaml.safe_dump(dict(payload), sort_keys=False)


__all__ = [
    "DEFAULT_EVALUATION_MODE",
    "DEFAULT_EVALUATION_SUITE_PATH",
    "DEFAULT_MODEL_SCOPE",
    "DEFAULT_RENDER_FPS",
    "SUPPORTED_EVALUATION_MODES",
    "SUPPORTED_MODEL_SCOPES",
    "CurriculumEvaluationResult",
    "run_curriculum_evaluation",
]
