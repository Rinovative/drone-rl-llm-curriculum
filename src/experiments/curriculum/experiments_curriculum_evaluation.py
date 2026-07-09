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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import utils
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites

DEFAULT_EVALUATION_SUITE_PATH = evaluation_suites.DEFAULT_EVALUATION_SUITE_PATH
OWN_TASK_EVALUATION_NAME = policy_evaluation.OWN_TASK_EVALUATION_NAME
STANDARD_EVALUATION_PROFILE = policy_evaluation.STANDARD_EVALUATION_PROFILE
STANDARD_GENERALIZATION_SUITE_PATH = policy_evaluation.STANDARD_GENERALIZATION_SUITE_PATH
STANDARD_SCENARIO_EVALUATION_NAME = policy_evaluation.STANDARD_SCENARIO_EVALUATION_NAME
STANDARD_SCENARIO_CONFIG_PATHS = policy_evaluation.STANDARD_SCENARIO_CONFIG_PATHS
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
        Path to the root evaluation summary JSON.
    manifest_path
        Path to the root evaluation summary JSON.
    metrics
        JSON-serializable aggregate summary payload for this evaluation.

    """

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CurriculumStandardEvaluationResult:
    """Aggregate result returned after running the standard curriculum profile."""

    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]


def run_curriculum_standard_evaluation(
    summary_path: str | Path,
    eval_steps: int | None = None,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
    model_scope: str = DEFAULT_MODEL_SCOPE,
    render: bool | None = None,
    render_fps: int | None = None,
    render_max_steps: int | None = None,
    plots: bool | None = None,
    traces: bool | None = None,
) -> CurriculumStandardEvaluationResult:
    """Run the standard stage-owned curriculum evaluation profile."""
    if wandb_mode not in utils.wandb.WANDB_MODES:
        message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
        raise ValueError(message)
    _validate_model_scope(model_scope)

    results: list[dict[str, str]] = []
    own_task = run_curriculum_evaluation(
        summary_path=summary_path,
        mode="own-stage",
        suite_path=None,
        model_scope=model_scope,
        eval_steps=eval_steps,
        wandb_mode=wandb_mode,
        render=render,
        render_fps=render_fps,
        render_max_steps=render_max_steps,
        plots=plots,
        traces=traces,
    )
    results.append(_standard_profile_result_entry(OWN_TASK_EVALUATION_NAME, model_scope, own_task))

    for suite_path, suite_model_scope in _standard_curriculum_suite_plan(model_scope):
        suite = evaluation_suites.load_evaluation_suite(suite_path)
        result = run_curriculum_evaluation(
            summary_path=summary_path,
            mode="suite",
            suite_path=suite_path,
            model_scope=suite_model_scope,
            eval_steps=eval_steps,
            wandb_mode=wandb_mode,
            render=render,
            render_fps=render_fps,
            render_max_steps=render_max_steps,
            plots=plots,
            traces=traces,
        )
        results.append(_standard_profile_result_entry(suite.evaluation_name, suite_model_scope, result))

    run_manifest_path = Path(summary_path)
    run_root = run_manifest_path.expanduser().resolve(strict=False).parent
    summary = _read_json(run_manifest_path)
    stages = _stages(summary)
    final_stage = stages[-1]
    final_model_path, final_model_source = _stage_model_path_and_source(final_stage)
    curriculum_run_name = _curriculum_run_name(summary)
    scenario_result = policy_evaluation.run_standard_scenario_evaluations(
        run_root=run_root,
        run_name=curriculum_run_name,
        model_path=final_model_path,
        model_run_name=str(final_stage.get("run_name") or curriculum_run_name),
        source_run_kind="curriculum",
        source_curriculum_kind=str(summary.get("curriculum_kind") or ""),
        model_scope="final-stage",
        evaluated_model_source=final_model_source,
        run_manifest_path=run_manifest_path,
    )
    results.append(_standard_profile_result_entry(STANDARD_SCENARIO_EVALUATION_NAME, "final-stage", scenario_result))

    run_name = curriculum_run_name
    storage_root = utils.artifacts.storage_root_from_run_dir(run_root)
    evaluation_summary_path = utils.artifacts.get_run_evaluation_summary_path(run_name, storage_root=storage_root)
    scenario_summary_entry = {
        **scenario_result.metrics,
        "curriculum_run_name": run_name,
        "model_scope": "final-stage",
        "index_key": f"curriculum_evaluation:{STANDARD_SCENARIO_EVALUATION_NAME}:final-stage",
    }
    _update_curriculum_evaluation_summary(
        run_name=run_name,
        run_root=run_root,
        summary_path=evaluation_summary_path,
        aggregate_entry=scenario_summary_entry,
    )
    evaluation_summary = _read_json(evaluation_summary_path) if evaluation_summary_path.exists() else {}
    profile_payload = {
        "run_type": "evaluation",
        "run_kind": "curriculum",
        "mode": "curriculum_standard_evaluation",
        "profile_name": STANDARD_EVALUATION_PROFILE,
        "curriculum_run_name": run_name,
        "evaluation_names": [entry["evaluation_name"] for entry in results],
        "evaluations": results,
        "evaluation_summary_path": str(evaluation_summary_path),
        "evaluation_summary_path_relative": utils.artifacts.path_relative_to(evaluation_summary_path, run_root),
        "evaluation_summary": evaluation_summary,
    }
    return CurriculumStandardEvaluationResult(
        metrics_path=str(evaluation_summary_path),
        manifest_path=str(evaluation_summary_path),
        metrics=profile_payload,
    )


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
        Unsupported legacy convenience baseline path. Direct PPO should be evaluated through its own run.
    baseline_label
        Deprecated baseline label retained for CLI argument validation.
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
    if include_baseline_model is not None:
        message = "curriculum evaluation no longer writes convenience baseline artifacts inside curriculum runs; evaluate direct PPO separately"
        raise ValueError(message)
    run_manifest_path = Path(summary_path)
    run_root = run_manifest_path.expanduser().resolve(strict=False).parent
    storage_root = utils.artifacts.storage_root_from_run_dir(run_root)
    summary = _read_json(run_manifest_path)
    stages = _stages(summary)
    suite = _load_suite_for_mode(mode=mode, suite_path=suite_path)

    curriculum_run_name = _curriculum_run_name(summary)
    evaluation_name = _evaluation_name(mode=mode, suite=suite)
    suite_snapshot = (
        None
        if suite is None
        else evaluation_suites.write_evaluation_suite_snapshot(
            run_name=curriculum_run_name,
            suite=suite,
            suite_path=suite_path,
            storage_root=storage_root,
        )
    )
    output_root = run_root
    summary_output_path = utils.artifacts.get_run_evaluation_summary_path(curriculum_run_name, storage_root=storage_root)

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
        curriculum_kind=str(summary.get("curriculum_kind") or ""),
        stages=stages,
        mode=mode,
        output_root=output_root,
        curriculum_run_name=curriculum_run_name,
        evaluation_name=evaluation_name,
        suite=suite,
        suite_snapshot=suite_snapshot,
        model_scope=model_scope,
        include_baseline_model=include_baseline_model,
        baseline_label=baseline_label,
        eval_steps_override=eval_steps,
        storage_root=storage_root,
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
                run_root=run_root,
            )
        )

    metrics_path = summary_output_path
    manifest_path = summary_output_path
    suite_path_for_metrics = None if suite is None or suite_path is None else str(Path(suite_path))
    suite_task_names = [] if suite is None else suite.task_names
    suite_snapshot_path = None if suite_snapshot is None else str(suite_snapshot.suite_config_path)
    suite_snapshot_path_relative = None if suite_snapshot is None else suite_snapshot.suite_config_path_relative
    suite_snapshot_sha256 = None if suite_snapshot is None else suite_snapshot.suite_config_sha256
    contains_convenience_baseline = include_baseline_model is not None

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
        "suite_config_snapshot_path": suite_snapshot_path,
        "suite_config_snapshot_path_relative": suite_snapshot_path_relative,
        "suite_config_sha256": suite_snapshot_sha256,
        "suite_task_names": suite_task_names,
        "suite_task_count": len(suite_task_names),
        "curriculum_run_name": curriculum_run_name,
        "source_run_name": curriculum_run_name,
        "source_run_kind": "curriculum",
        "source_curriculum_kind": summary.get("curriculum_kind"),
        "source_stage": None,
        "model_scope": model_scope,
        "contains_convenience_baseline": contains_convenience_baseline,
        "baseline_ownership": _baseline_ownership_note(contains_convenience_baseline),
        "summary_role": "derived_aggregate_link_summary",
        "canonical_evaluation_owner": "curriculum_stage",
        "detailed_stage_artifacts_duplicated_at_run_root": False,
        "evaluated_models": evaluated_models,
        "canonical_stage_evaluation_manifest_paths": _stage_manifest_paths(evaluated_models),
        "canonical_stage_evaluation_manifest_paths_relative": _stage_manifest_paths_relative(evaluated_models),
        "summary_metrics_path": str(metrics_path),
        "summary_metrics_path_relative": utils.artifacts.path_relative_to(metrics_path, run_root),
        "summary_manifest_path": str(manifest_path),
        "summary_manifest_path_relative": utils.artifacts.path_relative_to(manifest_path, run_root),
        "entry_count": len(evaluated_models),
        "index_key": f"curriculum_evaluation:{evaluation_name}:{model_scope}",
    }
    _update_curriculum_evaluation_summary(
        run_name=curriculum_run_name,
        run_root=run_root,
        summary_path=summary_output_path,
        aggregate_entry=aggregate_metrics,
    )
    _update_curriculum_evaluation_index(
        run_manifest_path=run_manifest_path,
        summary=summary,
        run_root=run_root,
        evaluation_name=evaluation_name,
        mode=mode,
        model_scope=model_scope,
        suite_name=None if suite is None else suite.evaluation_name,
        suite_snapshot=suite_snapshot,
        aggregate_metrics_path=metrics_path,
        aggregate_manifest_path=manifest_path,
        evaluated_models=evaluated_models,
        task_names=suite_task_names if suite is not None else [str(stage.get("stage_name")) for stage in stages],
    )
    return CurriculumEvaluationResult(
        metrics_path=str(metrics_path),
        manifest_path=str(manifest_path),
        metrics=aggregate_metrics,
    )


def _standard_curriculum_suite_plan(model_scope: str) -> list[tuple[Path, str]]:
    """Return suite configs and model scopes for the standard curriculum profile."""
    return [(STANDARD_GENERALIZATION_SUITE_PATH, model_scope)]


def _standard_profile_result_entry(
    evaluation_name: str,
    model_scope: str,
    result: CurriculumEvaluationResult | policy_evaluation.PolicyScenarioEvaluationResult,
) -> dict[str, str]:
    """Return a compact standard-profile result link."""
    return {
        "evaluation_name": evaluation_name,
        "model_scope": model_scope,
        "metrics_path": result.metrics_path,
        "manifest_path": result.manifest_path,
    }


def _update_curriculum_evaluation_summary(
    *,
    run_name: str,
    run_root: Path,
    summary_path: Path,
    aggregate_entry: Mapping[str, Any],
) -> None:
    """Upsert one link-only curriculum evaluation entry into the root summary."""
    existing = _read_json(summary_path) if summary_path.exists() else {"evaluations": []}
    raw_entries = existing.get("evaluations") if isinstance(existing, Mapping) else None
    entries = raw_entries if isinstance(raw_entries, list) else []
    entry = dict(aggregate_entry)
    entry_key = str(entry.get("index_key") or f"curriculum_evaluation:{entry.get('evaluation_name')}:{entry.get('model_scope')}")
    entry["index_key"] = entry_key
    updated_entries = [candidate for candidate in entries if not isinstance(candidate, Mapping) or candidate.get("index_key") != entry_key]
    updated_entries.append(entry)
    payload = {
        "run_type": "evaluation",
        "run_kind": "curriculum",
        "mode": "curriculum_evaluation_summary",
        "summary_role": "derived_aggregate_link_summary",
        "run_name": run_name,
        "curriculum_run_name": run_name,
        "evaluation_summary_path": str(summary_path),
        "evaluation_summary_path_relative": utils.artifacts.path_relative_to(summary_path, run_root),
        "entry_count": len(updated_entries),
        "evaluations": updated_entries,
    }
    _write_json(summary_path, payload)


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
    curriculum_kind: str | None,
    stages: list[Mapping[str, Any]],
    mode: str,
    output_root: Path,
    curriculum_run_name: str,
    evaluation_name: str,
    suite: evaluation_suites.EvaluationSuite | None,
    suite_snapshot: evaluation_suites.EvaluationSuiteSnapshot | None,
    model_scope: str,
    include_baseline_model: str | Path | None,
    baseline_label: str,
    eval_steps_override: int | None,
    storage_root: Path | None,
) -> list[dict[str, Any]]:
    """Build model-evaluation spec payloads for the requested mode."""
    selected_stages = _selected_stages(stages=stages, model_scope=model_scope)
    if mode == "own-stage":
        return _own_stage_payloads(
            stages=selected_stages,
            all_stages=stages,
            curriculum_run_name=curriculum_run_name,
            curriculum_kind=curriculum_kind,
            evaluation_name=evaluation_name,
            model_scope=model_scope,
            eval_steps_override=eval_steps_override,
            default_seed=int(summary.get("seed", 0)),
            storage_root=storage_root,
        )

    if suite is None:
        message = "suite must be provided for suite mode"
        raise ValueError(message)
    if suite_snapshot is None:
        message = "suite snapshot must be available for suite mode"
        raise ValueError(message)
    return _suite_payloads(
        stages=stages,
        selected_stages=selected_stages,
        suite=suite,
        suite_snapshot=suite_snapshot,
        output_root=output_root,
        curriculum_run_name=curriculum_run_name,
        evaluation_name=evaluation_name,
        curriculum_kind=curriculum_kind,
        model_scope=model_scope,
        include_baseline_model=include_baseline_model,
        baseline_label=baseline_label,
        eval_steps_override=eval_steps_override,
        storage_root=storage_root,
    )


def _suite_payloads(
    stages: list[Mapping[str, Any]],
    selected_stages: list[Mapping[str, Any]],
    suite: evaluation_suites.EvaluationSuite,
    suite_snapshot: evaluation_suites.EvaluationSuiteSnapshot,
    output_root: Path,
    curriculum_run_name: str,
    evaluation_name: str,
    curriculum_kind: str | None,
    model_scope: str,
    include_baseline_model: str | Path | None,
    baseline_label: str,
    eval_steps_override: int | None,
    storage_root: Path | None,
) -> list[dict[str, Any]]:
    """Build suite-task evaluation payloads for selected curriculum models."""
    task_config_paths = suite_snapshot.task_config_paths
    payloads: list[dict[str, Any]] = []
    final_stage_index = int(stages[-1]["stage_index"])

    for stage in selected_stages:
        stage_manifest = _read_json(Path(str(stage["manifest_path"]))) if stage.get("manifest_path") else {}
        model_path, evaluated_model_source = _stage_model_path_and_source(stage)
        stage_index = int(stage["stage_index"])
        stage_name = str(stage["stage_name"])
        stage_dir_name = f"stage{stage_index:02d}_{stage_name}"
        stage_output_root = utils.artifacts.get_curriculum_stage_evaluation_dir(
            curriculum_run_name,
            stage_index,
            stage_name,
            evaluation_name,
            storage_root=storage_root,
        )
        label_prefix = stage_dir_name
        payloads.extend(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=f"{label_prefix}_{suite_task.task_name}",
                    model_role="stage",
                    model_path=model_path,
                    task_config_path=task_config_paths[suite_task.task_name],
                    task_index=0,
                    task_shape=suite_task.task_shape,
                    output_dir=stage_output_root / _safe_name(suite_task.task_name),
                    eval_steps=int(eval_steps_override or suite.eval_steps),
                    seed=suite.seed,
                    total_timesteps=int(stage.get("total_timesteps", stage_manifest.get("total_timesteps", 0))),
                    normalize_actions=bool(stage.get("normalize_actions", stage_manifest.get("normalize_actions", True))),
                    **_stage_evaluation_env_kwargs(stage, stage_manifest),
                    evaluation_name=evaluation_name,
                    evaluation_suite_name=suite.evaluation_name,
                    suite_task_name=suite_task.task_name,
                    suite_task_names=tuple(suite.task_names),
                    suite_config_snapshot_path=suite_snapshot.suite_config_path,
                    suite_config_snapshot_path_relative=suite_snapshot.suite_config_path_relative,
                    suite_config_sha256=suite_snapshot.suite_config_sha256,
                    source_run_name=str(stage.get("run_name") or ""),
                    source_run_kind="curriculum_stage",
                    source_curriculum_kind=curriculum_kind,
                    source_stage={"stage_index": stage_index, "stage_name": stage_name},
                    model_scope=model_scope,
                    evaluated_model_source=evaluated_model_source,
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
                    evaluation_name=evaluation_name,
                    evaluation_suite_name=suite.evaluation_name,
                    suite_task_name=suite_task.task_name,
                    suite_task_names=tuple(suite.task_names),
                    suite_config_snapshot_path=suite_snapshot.suite_config_path,
                    suite_config_snapshot_path_relative=suite_snapshot.suite_config_path_relative,
                    suite_config_sha256=suite_snapshot.suite_config_sha256,
                    source_run_name=baseline_dir_name,
                    source_run_kind="baseline",
                    source_curriculum_kind=curriculum_kind,
                    source_stage=None,
                    model_scope=model_scope,
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
    curriculum_kind: str | None,
    evaluation_name: str,
    model_scope: str,
    eval_steps_override: int | None,
    default_seed: int,
    storage_root: Path | None,
) -> list[dict[str, Any]]:
    """Build stage-indexed evaluation payloads for own-stage mode."""
    payloads: list[dict[str, Any]] = []
    final_stage_index = int(all_stages[-1]["stage_index"])
    for stage in stages:
        stage_manifest = _read_json(Path(str(stage["manifest_path"])))
        model_path, evaluated_model_source = _stage_model_path_and_source(stage)
        stage_index = int(stage["stage_index"])
        stage_name = str(stage["stage_name"])
        stage_dir_name = f"stage{stage_index:02d}_{stage_name}"
        stage_output_dir = utils.artifacts.get_curriculum_stage_evaluation_dir(
            curriculum_run_name,
            stage_index,
            stage_name,
            evaluation_name,
            storage_root=storage_root,
        )
        payloads.append(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=stage_dir_name,
                    model_role="stage",
                    model_path=model_path,
                    task_config_path=Path(str(stage_manifest["task_config_path"])),
                    task_index=int(stage_manifest.get("task_index", 0)),
                    task_shape=str(stage["task_shape"]),
                    output_dir=stage_output_dir,
                    eval_steps=int(eval_steps_override or stage.get("eval_steps") or stage_manifest.get("eval_steps") or 120),
                    seed=int(stage.get("seed", default_seed)),
                    total_timesteps=int(stage.get("total_timesteps", stage_manifest.get("total_timesteps", 0))),
                    normalize_actions=bool(stage.get("normalize_actions", stage_manifest.get("normalize_actions", True))),
                    **_stage_evaluation_env_kwargs(stage, stage_manifest),
                    evaluation_name=evaluation_name,
                    evaluation_suite_name=None,
                    suite_task_name=stage_name,
                    suite_task_names=tuple(str(candidate.get("stage_name")) for candidate in stages),
                    source_run_name=str(stage.get("run_name") or ""),
                    source_run_kind="curriculum_stage",
                    source_curriculum_kind=curriculum_kind,
                    source_stage={"stage_index": stage_index, "stage_name": stage_name},
                    model_scope=model_scope,
                    evaluated_model_source=evaluated_model_source,
                ),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "source_run_name": stage.get("run_name"),
                "is_final_stage": stage_index == final_stage_index,
                "suite_task_name": None,
            }
        )
    return payloads


def _stage_model_path_and_source(stage: Mapping[str, Any]) -> tuple[Path, str]:
    """Return the best available stage model path and its source label."""
    for source, key in (("best", "best_model_path"), ("last", "last_model_path"), ("last", "model_path")):
        value = stage.get(key)
        if isinstance(value, str) and value:
            return Path(value), source
    message = "curriculum stage must include best_model_path, last_model_path, or model_path"
    raise ValueError(message)


def _stage_model_path_relative(stage: Mapping[str, Any]) -> str | None:
    """Return the relative path matching the selected stage model when available."""
    for key in ("best_model_path_relative", "last_model_path_relative", "model_path_relative"):
        value = stage.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _stage_evaluation_env_kwargs(stage: Mapping[str, Any], stage_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return PPO-facing env flags from a curriculum stage and optional manifest."""
    manifest = stage_manifest or {}
    return {
        "action_interface": str(stage.get("action_interface") or manifest.get("action_interface") or "pid_position"),
        "rpm_delta_scale": float(stage.get("rpm_delta_scale") or manifest.get("rpm_delta_scale") or 0.05),
        "include_dynamics_observation": bool(stage.get("include_dynamics_observation", manifest.get("include_dynamics_observation", False))),
        "include_previous_action": bool(stage.get("include_previous_action", manifest.get("include_previous_action", False))),
        "source_manifest_path": Path(str(stage["manifest_path"])) if stage.get("manifest_path") else None,
    }


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
        return OWN_TASK_EVALUATION_NAME
    if suite is None:
        message = "suite must be provided for suite evaluation naming"
        raise ValueError(message)
    return suite.evaluation_name


def _summary_filename_stem(
    summary: Mapping[str, Any],
    mode: str,
    suite: evaluation_suites.EvaluationSuite | None,
    model_scope: str,
) -> str:
    """Return summary filename stem for aggregate metrics/manifest outputs."""
    run_name = _curriculum_run_name(summary)
    scope_name = model_scope.replace("-", "_")
    if mode == "own-stage":
        return f"{run_name}_{OWN_TASK_EVALUATION_NAME}_{scope_name}"
    if suite is None:
        message = "suite must be provided for suite summary naming"
        raise ValueError(message)
    return f"{run_name}_{suite.evaluation_name}_{scope_name}"


def _evaluated_model_entry(
    result: policy_evaluation.PolicyEvaluationResult,
    stage_index: int | None,
    stage_name: str | None,
    source_run_name: str | None,
    is_final_stage: bool,
    evaluation_suite_name: str | None,
    suite_task_name: str | None,
    run_root: Path,
) -> dict[str, Any]:
    """Build one link-oriented evaluated-model summary entry."""
    metrics = result.metrics
    entry = {
        "label": result.label,
        "model_label": result.label,
        "model_role": result.model_role,
        "model_path": result.model_path,
        "model_path_relative": utils.artifacts.path_relative_to(result.model_path, run_root),
        "evaluated_model_path": metrics.get("evaluated_model_path"),
        "evaluated_model_path_relative": utils.artifacts.path_relative_to(metrics.get("evaluated_model_path"), run_root),
        "evaluated_model_source": metrics.get("evaluated_model_source"),
        "task_config_path_used_for_evaluation": result.task_config_path,
        "task_config_path_relative": utils.artifacts.path_relative_to(result.task_config_path, run_root),
        "task_shape_used_for_evaluation": result.task_shape,
        "evaluation_dir": result.output_dir,
        "evaluation_dir_relative": utils.artifacts.path_relative_to(result.output_dir, run_root),
        "diagnostics_dir": result.diagnostics_dir,
        "diagnostics_dir_relative": utils.artifacts.path_relative_to(result.diagnostics_dir, run_root),
        "traces_dir": result.traces_dir,
        "traces_dir_relative": utils.artifacts.path_relative_to(result.traces_dir, run_root),
        "plots_dir": result.plots_dir,
        "plots_dir_relative": utils.artifacts.path_relative_to(result.plots_dir, run_root),
        "renders_dir": result.renders_dir,
        "renders_dir_relative": utils.artifacts.path_relative_to(result.renders_dir, run_root),
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
        "render_enabled": result.render_enabled,
        "plots_enabled": result.plots_enabled,
        "trace_enabled": result.trace_enabled,
        "diagnostics_enabled": metrics.get("diagnostics_enabled"),
        "eval_steps": metrics.get("eval_steps"),
        "seed": metrics.get("seed"),
        "stage_index": stage_index,
        "stage_name": stage_name,
        "is_final_stage": bool(is_final_stage),
        "evaluation_suite_name": evaluation_suite_name,
        "suite_task_name": suite_task_name,
        "source_run_name": source_run_name,
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
        "canonical_owner": "convenience_baseline" if result.model_role == "baseline" else "curriculum_stage",
    }
    if result.model_role == "baseline":
        entry["baseline_ownership"] = _baseline_ownership_note(True)
    return entry


def _baseline_ownership_note(contains_baseline: bool) -> str | None:
    """Return the ownership note for convenience baseline outputs."""
    if not contains_baseline:
        return None
    return "convenience_only_not_primary_comparison_truth; evaluate the baseline run separately through experiments_cli_evaluate_policy"


def _update_curriculum_evaluation_index(
    *,
    run_manifest_path: Path,
    summary: Mapping[str, Any],
    run_root: Path,
    evaluation_name: str,
    mode: str,
    model_scope: str,
    suite_name: str | None,
    suite_snapshot: evaluation_suites.EvaluationSuiteSnapshot | None,
    aggregate_metrics_path: Path,
    aggregate_manifest_path: Path,
    evaluated_models: list[dict[str, Any]],
    task_names: list[str],
) -> None:
    """Update the owning curriculum run manifest with evaluation links."""
    final_stage = _stages(summary)[-1]
    final_model_path, final_model_source = _stage_model_path_and_source(final_stage)
    final_model_path_relative = _stage_model_path_relative(final_stage) or utils.artifacts.path_relative_to(final_model_path, run_root)
    index_entry = {
        "index_key": f"curriculum_evaluation:{evaluation_name}:{model_scope}",
        "run_name": _curriculum_run_name(summary),
        "run_kind": "curriculum",
        "mode": "curriculum_evaluation",
        "evaluation_mode": mode,
        "model_scope": model_scope,
        "evaluation_name": evaluation_name,
        "evaluation_suite_name": suite_name,
        "source_run_name": _curriculum_run_name(summary),
        "source_run_kind": "curriculum",
        "source_curriculum_kind": summary.get("curriculum_kind"),
        "suite_name": suite_name,
        "suite_config_snapshot_path": None if suite_snapshot is None else str(suite_snapshot.suite_config_path),
        "suite_config_snapshot_path_relative": None if suite_snapshot is None else suite_snapshot.suite_config_path_relative,
        "suite_config_snapshot_relative": None if suite_snapshot is None else suite_snapshot.suite_config_path_relative,
        "suite_config_sha256": None if suite_snapshot is None else suite_snapshot.suite_config_sha256,
        "aggregate_metrics_path": str(aggregate_metrics_path),
        "aggregate_metrics_path_relative": utils.artifacts.path_relative_to(aggregate_metrics_path, run_root),
        "aggregate_metrics_relative": utils.artifacts.path_relative_to(aggregate_metrics_path, run_root),
        "evaluation_manifest_path": str(aggregate_manifest_path),
        "evaluation_manifest_path_relative": utils.artifacts.path_relative_to(aggregate_manifest_path, run_root),
        "evaluation_manifest_relative": utils.artifacts.path_relative_to(aggregate_manifest_path, run_root),
        "model_label": f"curriculum_{model_scope}",
        "model_role": "stage",
        "model_path": str(final_model_path) if model_scope == "final-stage" else None,
        "model_path_relative": final_model_path_relative if model_scope == "final-stage" else None,
        "evaluated_model_source": final_model_source if model_scope == "final-stage" else None,
        "task_names": list(task_names),
        "summary_role": "derived_aggregate_link_summary",
        "canonical_evaluation_owner": "curriculum_stage",
        "detailed_stage_artifacts_duplicated_at_run_root": False,
        "canonical_stage_evaluation_manifest_paths": _stage_manifest_paths(evaluated_models),
        "canonical_stage_evaluation_manifest_paths_relative": _stage_manifest_paths_relative(evaluated_models),
        "final_stage_index": int(final_stage["stage_index"]),
        "final_stage_name": str(final_stage["stage_name"]),
        "evaluated_models": evaluated_models,
    }
    policy_evaluation.update_run_evaluation_index(run_manifest_path, index_entry)
    _update_curriculum_manifest_evaluation_links(
        run_manifest_path=run_manifest_path,
        summary=summary,
        run_root=run_root,
        index_entry=index_entry,
    )


def _stage_manifest_paths(evaluated_models: list[dict[str, Any]]) -> list[str]:
    """Return canonical stage-owned per-task evaluation manifest paths."""
    return [
        str(entry["manifest_path"]) for entry in evaluated_models if entry.get("model_role") == "stage" and entry.get("manifest_path") is not None
    ]


def _stage_manifest_paths_relative(evaluated_models: list[dict[str, Any]]) -> list[str]:
    """Return canonical stage-owned per-task evaluation manifest paths relative to the run."""
    return [
        str(entry["manifest_path_relative"])
        for entry in evaluated_models
        if entry.get("model_role") == "stage" and entry.get("manifest_path_relative") is not None
    ]


def _update_curriculum_manifest_evaluation_links(
    *,
    run_manifest_path: Path,
    summary: Mapping[str, Any],
    run_root: Path,
    index_entry: Mapping[str, Any],
) -> None:
    """Add notebook-friendly curriculum evaluation links to the run manifest."""
    manifest = _read_json(run_manifest_path)
    final_stage = _stages(summary)[-1]
    final_model_path, final_model_source = _stage_model_path_and_source(final_stage)
    final_model_path_relative = _stage_model_path_relative(final_stage) or utils.artifacts.path_relative_to(final_model_path, run_root)
    existing_final_stage = manifest.get("final_stage")
    final_stage_link = dict(existing_final_stage) if isinstance(existing_final_stage, Mapping) else {}
    final_stage_link.update(
        {
            "stage_index": int(final_stage["stage_index"]),
            "stage_name": str(final_stage["stage_name"]),
            "model_path": str(final_model_path),
            "model_path_relative": final_model_path_relative,
            "selected_model_source": final_model_source,
            "last_model_path": final_stage.get("last_model_path") or final_stage.get("model_path"),
            "last_model_path_relative": final_stage.get("last_model_path_relative") or final_stage.get("model_path_relative"),
            "best_model_path": final_stage.get("best_model_path"),
            "best_model_path_relative": final_stage.get("best_model_path_relative"),
        }
    )
    manifest["final_stage"] = final_stage_link
    link = {
        "index_key": index_entry.get("index_key"),
        "evaluation_name": index_entry.get("evaluation_name"),
        "evaluation_suite_name": index_entry.get("evaluation_suite_name"),
        "model_scope": index_entry.get("model_scope"),
        "aggregate_metrics_path": index_entry.get("aggregate_metrics_path"),
        "aggregate_metrics_path_relative": index_entry.get("aggregate_metrics_path_relative"),
        "evaluation_manifest_path": index_entry.get("evaluation_manifest_path"),
        "evaluation_manifest_path_relative": index_entry.get("evaluation_manifest_path_relative"),
        "canonical_stage_evaluation_manifest_paths": index_entry.get("canonical_stage_evaluation_manifest_paths", []),
        "canonical_stage_evaluation_manifest_paths_relative": index_entry.get("canonical_stage_evaluation_manifest_paths_relative", []),
        "suite_config_snapshot_path": index_entry.get("suite_config_snapshot_path"),
        "suite_config_snapshot_path_relative": index_entry.get("suite_config_snapshot_path_relative"),
        "suite_config_sha256": index_entry.get("suite_config_sha256"),
        "task_names": index_entry.get("task_names", []),
    }
    if index_entry.get("model_scope") == "final-stage":
        manifest["final_stage_evaluation"] = {
            **link,
            "final_stage_index": int(final_stage["stage_index"]),
            "final_stage_name": str(final_stage["stage_name"]),
            "final_model_path": str(final_model_path),
            "final_model_path_relative": final_model_path_relative,
            "final_model_source": final_model_source,
        }
    elif index_entry.get("model_scope") == "all-stages":
        existing = manifest.get("all_stage_evaluations")
        entries = existing if isinstance(existing, list) else []
        key = str(link.get("index_key"))
        manifest["all_stage_evaluations"] = [entry for entry in entries if not isinstance(entry, Mapping) or entry.get("index_key") != key] + [link]
    _write_json(run_manifest_path, manifest)


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


__all__ = [
    "DEFAULT_EVALUATION_MODE",
    "DEFAULT_EVALUATION_SUITE_PATH",
    "DEFAULT_MODEL_SCOPE",
    "DEFAULT_RENDER_FPS",
    "OWN_TASK_EVALUATION_NAME",
    "STANDARD_EVALUATION_PROFILE",
    "STANDARD_GENERALIZATION_SUITE_PATH",
    "STANDARD_SCENARIO_CONFIG_PATHS",
    "STANDARD_SCENARIO_EVALUATION_NAME",
    "SUPPORTED_EVALUATION_MODES",
    "SUPPORTED_MODEL_SCOPES",
    "CurriculumEvaluationResult",
    "CurriculumStandardEvaluationResult",
    "run_curriculum_evaluation",
    "run_curriculum_standard_evaluation",
]
