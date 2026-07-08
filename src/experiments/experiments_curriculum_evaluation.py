"""
===============================================================================
experiments_curriculum_evaluation.py
===============================================================================
Evaluate manual-curriculum PPO checkpoints through one shared policy pipeline.

Responsibilities:
  - Load curriculum summaries and config-driven benchmark definitions
  - Build concrete own-stage, benchmark, and generalization evaluation specs
  - Delegate per-model execution to the shared policy evaluation helper
  - Aggregate compact curriculum-level metrics and manifests

Design principles:
  - Keep curriculum evaluation focused on planning and aggregation
  - Keep benchmark selection config-driven and deterministic
  - Fail clearly when required summary fields, benchmarks, or model paths are invalid

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

from src import utils, validation
from src.experiments import experiments_config as config
from src.experiments import experiments_policy_evaluation as policy_evaluation

DEFAULT_BENCHMARK_CONFIG_PATH = Path("configs/evaluation/curriculum_benchmarks.yaml")
SUPPORTED_EVALUATION_MODES = ("own-stage", "benchmark", "generalization")
SUPPORTED_MODEL_SCOPES = ("all-stages", "final-stage")
DEFAULT_MODEL_SCOPE = "all-stages"
DEFAULT_RENDER_FPS = policy_evaluation.DEFAULT_RENDER_FPS


@dataclass(frozen=True)
class CurriculumBenchmark:
    """
    One named benchmark task loaded from configuration.

    Parameters
    ----------
    benchmark_name
        Stable benchmark identifier.
    task_shape
        Expected task shape for the benchmark evaluation.
    eval_steps
        Deterministic rollout steps for benchmark evaluation.
    task
        Validated trajectory task mapping.

    """

    benchmark_name: str
    task_shape: str
    eval_steps: int
    task: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate benchmark metadata."""
        if not self.benchmark_name.strip():
            message = "benchmark_name must be non-empty"
            raise ValueError(message)
        if not self.task_shape.strip():
            message = "task_shape must be non-empty"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "eval_steps must be positive"
            raise ValueError(message)
        if str(self.task.get("shape", "")) != self.task_shape:
            message = f"benchmark {self.benchmark_name!r} task shape must match task_shape {self.task_shape!r}"
            raise ValueError(message)


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


def load_curriculum_benchmarks(path: str | Path = DEFAULT_BENCHMARK_CONFIG_PATH) -> dict[str, CurriculumBenchmark]:
    """
    Load and validate benchmark tasks from YAML.

    Parameters
    ----------
    path
        Benchmark YAML path.

    Returns
    -------
    dict[str, CurriculumBenchmark]
        Benchmarks keyed by benchmark name.

    """
    payload = config.load_experiment_config(Path(path))
    raw_benchmarks = payload.get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        message = "benchmark config must contain a top-level 'benchmarks' list"
        raise TypeError(message)
    benchmarks = {benchmark.benchmark_name: benchmark for benchmark in (_benchmark_from_mapping(raw) for raw in raw_benchmarks)}
    for benchmark in benchmarks.values():
        _validate_task(benchmark.task, label=f"benchmark {benchmark.benchmark_name!r}")
    return benchmarks


def run_curriculum_evaluation(
    summary_path: str | Path,
    mode: str,
    benchmark: str | None = None,
    benchmark_config_path: str | Path = DEFAULT_BENCHMARK_CONFIG_PATH,
    model_scope: str = DEFAULT_MODEL_SCOPE,
    include_baseline_model: str | Path | None = None,
    baseline_label: str = "baseline",
    eval_steps: int | None = None,
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED,
    render: bool = True,
    render_fps: int = DEFAULT_RENDER_FPS,
    render_max_steps: int | None = None,
    plots: bool = True,
    traces: bool = True,
) -> CurriculumEvaluationResult:
    """
    Run curriculum evaluation for one mode using the shared model-evaluation pipeline.

    Parameters
    ----------
    summary_path
        Curriculum summary JSON path.
    mode
        Evaluation mode: ``own-stage``, ``benchmark``, or ``generalization``.
    benchmark
        Benchmark name required for benchmark/generalization modes.
    benchmark_config_path
        Benchmark YAML path.
    model_scope
        Stage model selection scope: ``all-stages`` or ``final-stage``.
    include_baseline_model
        Optional baseline model path evaluated alongside selected curriculum models.
    baseline_label
        Human-readable baseline label.
    eval_steps
        Optional evaluation-step override.
    wandb_mode
        Accepted for CLI symmetry.
    render
        Whether GIF rendering is enabled for each evaluated model.
    render_fps
        Requested GIF playback frame rate.
    render_max_steps
        Optional render-step override.
    plots
        Whether plot generation is enabled for each evaluated model.
    traces
        Whether trace-copy artifacts are enabled for each evaluated model.

    Returns
    -------
    CurriculumEvaluationResult
        Aggregate summary metrics and manifest paths.

    """
    _validate_mode_and_wandb(mode=mode, wandb_mode=wandb_mode)
    _validate_model_scope(model_scope)
    summary = _read_json(Path(summary_path))
    stages = _stages(summary)
    benchmarks = load_curriculum_benchmarks(benchmark_config_path)

    benchmark_name, benchmark_task_shape = _benchmark_metadata(mode=mode, benchmark_name=benchmark, benchmarks=benchmarks)
    output_root = _mode_output_root(summary=summary, mode=mode, benchmark_name=benchmark_name)
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_options = policy_evaluation.PolicyEvaluationArtifactOptions(
        render_enabled=render,
        plots_enabled=plots,
        trace_enabled=traces,
        diagnostics_enabled=True,
        render_fps=render_fps,
        render_max_steps=render_max_steps,
    )

    spec_payloads = _evaluation_spec_payloads(
        summary=summary,
        stages=stages,
        mode=mode,
        output_root=output_root,
        benchmark_name=benchmark_name,
        benchmarks=benchmarks,
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
                benchmark_name=benchmark_name,
            )
        )

    filename_stem = _summary_filename_stem(summary=summary, mode=mode, benchmark_name=benchmark_name)
    metrics_dir = output_root / utils.artifacts.METRICS_DIRNAME
    manifests_dir = output_root / utils.artifacts.MANIFESTS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{filename_stem}_metrics.json"
    manifest_path = manifests_dir / f"{filename_stem}_manifest.json"

    aggregate_metrics = {
        "run_type": "evaluation",
        "mode": "curriculum_evaluation",
        "curriculum_name": str(summary["curriculum_name"]),
        "seed": int(summary.get("seed", 0)),
        "evaluation_mode": mode,
        "benchmark_name": benchmark_name,
        "benchmark_task_shape": benchmark_task_shape,
        "model_scope": model_scope,
        "evaluated_models": evaluated_models,
        "summary_metrics_path": str(metrics_path),
        "summary_manifest_path": str(manifest_path),
        "entry_count": len(evaluated_models),
    }
    aggregate_manifest = {
        "run_type": "evaluation",
        "mode": "curriculum_evaluation",
        "curriculum_name": str(summary["curriculum_name"]),
        "seed": int(summary.get("seed", 0)),
        "evaluation_mode": mode,
        "benchmark_name": benchmark_name,
        "benchmark_task_shape": benchmark_task_shape,
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


def _benchmark_metadata(
    mode: str,
    benchmark_name: str | None,
    benchmarks: Mapping[str, CurriculumBenchmark],
) -> tuple[str | None, str | None]:
    """Resolve benchmark name and shape metadata for the requested mode."""
    if mode == "own-stage":
        return None, None
    if benchmark_name is None:
        message = f"--benchmark is required for {mode} mode"
        raise ValueError(message)
    benchmark = _require_benchmark(benchmarks, benchmark_name)
    return benchmark.benchmark_name, benchmark.task_shape


def _evaluation_spec_payloads(
    summary: Mapping[str, Any],
    stages: list[Mapping[str, Any]],
    mode: str,
    output_root: Path,
    benchmark_name: str | None,
    benchmarks: Mapping[str, CurriculumBenchmark],
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
            output_root=output_root,
            eval_steps_override=eval_steps_override,
            default_seed=int(summary.get("seed", 0)),
        )

    if benchmark_name is None:
        message = "benchmark_name must be provided for benchmark/generalization modes"
        raise ValueError(message)
    benchmark = _require_benchmark(benchmarks, benchmark_name)
    benchmark_task_config = _write_benchmark_task_config(output_root=output_root, benchmark=benchmark)

    payloads: list[dict[str, Any]] = []
    final_stage_index = int(stages[-1]["stage_index"])
    for stage in selected_stages:
        stage_index = int(stage["stage_index"])
        stage_name = str(stage["stage_name"])
        stage_dir_name = f"stage{stage_index:02d}_{stage_name}"
        payloads.append(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=stage_dir_name,
                    model_role="stage",
                    model_path=Path(str(stage["model_path"])),
                    task_config_path=benchmark_task_config,
                    task_index=0,
                    task_shape=benchmark.task_shape,
                    output_dir=output_root / "models" / stage_dir_name,
                    eval_steps=int(eval_steps_override or benchmark.eval_steps),
                    seed=int(summary.get("seed", stage.get("seed", 0))),
                    total_timesteps=int(stage.get("total_timesteps", 0)),
                    normalize_actions=bool(stage.get("normalize_actions", True)),
                ),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "source_run_name": stage.get("run_name"),
                "is_final_stage": stage_index == final_stage_index,
            }
        )

    if include_baseline_model is not None:
        baseline_dir_name = f"baseline_{_safe_name(baseline_label)}"
        payloads.append(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=baseline_dir_name,
                    model_role="baseline",
                    model_path=Path(str(include_baseline_model)),
                    task_config_path=benchmark_task_config,
                    task_index=0,
                    task_shape=benchmark.task_shape,
                    output_dir=output_root / "baselines" / baseline_dir_name,
                    eval_steps=int(eval_steps_override or benchmark.eval_steps),
                    seed=int(summary.get("seed", 0)),
                    total_timesteps=0,
                    normalize_actions=True,
                ),
                "stage_index": None,
                "stage_name": None,
                "source_run_name": baseline_dir_name,
                "is_final_stage": False,
            }
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
    output_root: Path,
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
        payloads.append(
            {
                "spec": policy_evaluation.PolicyEvaluationSpec(
                    label=stage_dir_name,
                    model_role="stage",
                    model_path=Path(str(stage["model_path"])),
                    task_config_path=Path(str(stage_manifest["task_config_path"])),
                    task_index=int(stage_manifest.get("task_index", 0)),
                    task_shape=str(stage["task_shape"]),
                    output_dir=output_root / "models" / stage_dir_name,
                    eval_steps=int(eval_steps_override or stage.get("eval_steps") or stage_manifest.get("eval_steps") or 120),
                    seed=int(stage.get("seed", default_seed)),
                    total_timesteps=int(stage.get("total_timesteps", stage_manifest.get("total_timesteps", 0))),
                    normalize_actions=bool(stage.get("normalize_actions", stage_manifest.get("normalize_actions", True))),
                ),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "source_run_name": stage.get("run_name"),
                "is_final_stage": stage_index == final_stage_index,
            }
        )
    return payloads


def _mode_output_root(summary: Mapping[str, Any], mode: str, benchmark_name: str | None) -> Path:
    """Return mode-scoped curriculum evaluation output root."""
    root = (
        utils.artifacts.get_storage_root()
        / utils.artifacts.EVALUATION_RUNS_DIRNAME
        / "curricula"
        / str(summary["curriculum_name"])
        / f"seed{int(summary.get('seed', 0))}"
        / mode.replace("-", "_")
    )
    if benchmark_name is not None:
        return root / benchmark_name
    return root


def _summary_filename_stem(summary: Mapping[str, Any], mode: str, benchmark_name: str | None) -> str:
    """Return summary filename stem for aggregate metrics/manifest outputs."""
    mode_slug = mode.replace("-", "_")
    if benchmark_name is None:
        return f"{summary['curriculum_name']}_seed{int(summary.get('seed', 0))}_{mode_slug}"
    return f"{summary['curriculum_name']}_seed{int(summary.get('seed', 0))}_{mode_slug}_{benchmark_name}"


def _write_benchmark_task_config(output_root: Path, benchmark: CurriculumBenchmark) -> Path:
    """Write a one-task benchmark config used by the shared evaluator."""
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{benchmark.benchmark_name}_task.yaml"
    payload = {
        "name": f"benchmark_{benchmark.benchmark_name}",
        "tasks": [benchmark.task],
    }
    config_path.write_text(_to_yaml(payload), encoding="utf-8")
    return config_path


def _evaluated_model_entry(
    result: policy_evaluation.PolicyEvaluationResult,
    stage_index: int | None,
    stage_name: str | None,
    source_run_name: str | None,
    is_final_stage: bool,
    benchmark_name: str | None,
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
    entry["benchmark_name"] = benchmark_name
    entry["source_run_name"] = source_run_name
    return entry


def _benchmark_from_mapping(raw: Any) -> CurriculumBenchmark:
    """Build one benchmark from a raw YAML mapping."""
    if not isinstance(raw, Mapping):
        message = "benchmark entry must be a mapping"
        raise TypeError(message)
    task = raw.get("task")
    if not isinstance(task, Mapping):
        message = "benchmark entry must contain an explicit task mapping"
        raise TypeError(message)
    return CurriculumBenchmark(
        benchmark_name=str(raw.get("benchmark_name") or ""),
        task_shape=str(raw.get("task_shape") or ""),
        eval_steps=int(raw.get("eval_steps", 0)),
        task=dict(task),
    )


def _require_benchmark(benchmarks: Mapping[str, CurriculumBenchmark], benchmark_name: str) -> CurriculumBenchmark:
    """Return a benchmark or raise a clear error with available names."""
    try:
        return benchmarks[benchmark_name]
    except KeyError as exc:
        available = ", ".join(sorted(benchmarks))
        message = f"benchmark {benchmark_name!r} not found; available: {available}"
        raise ValueError(message) from exc


def _validate_task(task: Mapping[str, Any], label: str) -> None:
    """Raise when a benchmark task fails deterministic validation."""
    result = validation.tasks.validate_task(dict(task))
    if not result.is_valid:
        details = "; ".join(result.messages)
        message = f"invalid {label} task: {details}"
        raise ValueError(message)


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
    """Serialize a compact one-task config to YAML."""
    import yaml  # noqa: PLC0415

    return yaml.safe_dump(dict(payload), sort_keys=False)


__all__ = [
    "DEFAULT_BENCHMARK_CONFIG_PATH",
    "DEFAULT_MODEL_SCOPE",
    "DEFAULT_RENDER_FPS",
    "SUPPORTED_EVALUATION_MODES",
    "SUPPORTED_MODEL_SCOPES",
    "CurriculumBenchmark",
    "CurriculumEvaluationResult",
    "load_curriculum_benchmarks",
    "run_curriculum_evaluation",
]
