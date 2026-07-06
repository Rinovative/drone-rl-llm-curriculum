"""
===============================================================================
cli_mvp.py
===============================================================================
Command-line helper for running or printing the complete MVP smoke sequence.

Responsibilities:
  - Print reviewer-friendly commands for reproducing the MVP path
  - Run tiny deterministic training, rollout evaluation, and visualization steps
  - Summarize generated artifacts under the approved storage results directory

Design principles:
  - Keep orchestration thin and reuse importable MVP helper modules
  - Keep defaults deterministic, headless, and suitable for smoke execution

Boundaries:
  - Core reward, training, rollout, and plotting logic belongs in dedicated modules
  - Documentation, notebooks, long training, and external services stay outside this CLI
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import evaluation, experiments, utils

DEFAULT_OUTPUT_DIR = utils.artifacts.get_training_run_dir("mvp_smoke")
DEFAULT_MAX_STEPS = 16


def build_parser() -> argparse.ArgumentParser:
    """Build the MVP CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run or print the tiny deterministic MVP smoke sequence.")
    parser.add_argument("--config", type=Path, default=experiments.training_smoke.DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    return parser


def build_repro_commands(
    config_path: Path = experiments.training_smoke.DEFAULT_TRAINING_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[str]:
    """
    Build reviewer-facing MVP reproduction commands.

    Parameters
    ----------
    config_path
        Smoke training config path.
    output_dir
        Output directory for generated MVP artifacts.
    max_steps
        Tiny max-step setting used by both commands.

    Returns
    -------
    list[str]
        Commands that can be run from the repository root.

    """
    training_command = (
        f"python -m src.experiments.cli_training_smoke --config {config_path.as_posix()} --output-dir {output_dir.as_posix()} --max-steps {max_steps}"
    )
    mvp_command = f"python -m src.experiments.cli_mvp --config {config_path.as_posix()} --output-dir {output_dir.as_posix()} --max-steps {max_steps}"
    return [training_command, mvp_command]


def run_mvp_sequence(
    config_path: Path = experiments.training_smoke.DEFAULT_TRAINING_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    task_index: int | None = None,
    skip_plot: bool = False,
) -> dict[str, Any]:
    """
    Run the tiny deterministic MVP smoke sequence.

    Parameters
    ----------
    config_path
        Smoke training config path.
    output_dir
        Output directory for generated artifacts.
    max_steps
        Maximum deterministic training-smoke steps.
    task_index
        Optional task index override.
    skip_plot
        Whether to skip trajectory comparison output.

    Returns
    -------
    dict[str, Any]
        JSON-serializable summary of generated outputs and warnings.

    """
    _ensure_prerequisites()
    metrics_dir, plots_dir = _artifact_dirs(output_dir)
    training_result = experiments.training_smoke.run_training_smoke_from_config(
        config_path=config_path,
        output_dir=metrics_dir,
        max_steps=max_steps,
        task_index=task_index,
    )
    settings = experiments.training_smoke.load_training_smoke_settings(config_path)
    selected_task_index = settings.task_index if task_index is None else task_index
    task = _load_task(settings.task_config_path, selected_task_index)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rollout_result = evaluation.rollout.write_task_rollout_evaluation(
        task,
        metrics_dir / evaluation.rollout.DEFAULT_OUTPUT_FILENAME,
    )

    plot_output: dict[str, Any] | None = None
    if not skip_plot:
        rollout = evaluation.rollout.evaluate_task_rollout(task)
        plot_result = evaluation.plots.write_trajectory_comparison(
            rollout.reference,
            rollout.actual,
            plots_dir / "trajectory_comparison.png",
        )
        plot_output = {
            "output_path": plot_result.output_path,
            "output_kind": plot_result.output_kind,
            "sample_count": plot_result.sample_count,
        }

    return {
        "training_output_path": training_result.output_path,
        "rollout_metrics_path": rollout_result.output_path,
        "visualization": plot_output,
        "metrics": {
            "training": training_result.metrics,
            "rollout": rollout_result.metrics,
        },
        "warnings": list(training_result.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the MVP CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    if args.print_commands:
        for command in build_repro_commands(config_path=args.config, output_dir=args.output_dir, max_steps=args.max_steps):
            print(command)
        return 0

    summary = run_mvp_sequence(
        config_path=args.config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        task_index=args.task_index,
        skip_plot=args.skip_plot,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _artifact_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Return metrics and plots directories for a training run or explicit override."""
    return output_dir / "metrics", output_dir / "plots"


def _ensure_prerequisites() -> None:
    """Raise a clear error if required MVP helper modules are unavailable."""
    missing: list[str] = []
    if not hasattr(experiments.training_smoke, "run_training_smoke_from_config"):
        missing.append("experiments.training_smoke.run_training_smoke_from_config")
    if not hasattr(evaluation, "rollout"):
        missing.append("evaluation.rollout")
    if not hasattr(evaluation, "plots"):
        missing.append("evaluation.plots")
    if missing:
        message = "missing MVP prerequisites: " + ", ".join(missing)
        raise RuntimeError(message)


def _load_task(task_config_path: Path, task_index: int) -> dict[str, Any]:
    """Load a copied task mapping for CLI orchestration."""
    config = experiments.config.load_experiment_config(task_config_path)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        message = "task config must contain a top-level tasks list"
        raise ValueError(message)  # noqa: TRY004 - CLI contract reports config errors as ValueError.
    if task_index < 0 or task_index >= len(tasks):
        message = "task_index is outside the configured task list"
        raise ValueError(message)
    task = tasks[task_index]
    if not isinstance(task, dict):
        message = "selected task must be a mapping"
        raise ValueError(message)  # noqa: TRY004 - CLI contract reports config errors as ValueError.
    return dict(task)


if __name__ == "__main__":
    raise SystemExit(main())
