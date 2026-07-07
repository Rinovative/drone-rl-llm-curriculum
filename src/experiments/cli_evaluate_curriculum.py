"""
===============================================================================
cli_evaluate_curriculum.py
===============================================================================
Command-line entry point for curriculum benchmark evaluation.

Responsibilities:
  - Parse curriculum summary, mode, benchmark, and baseline arguments
  - Run reusable curriculum evaluation helpers
  - Print aggregate metrics and manifest paths for operators

Design principles:
  - Keep CLI behavior thin and report-oriented
  - Leave rollout and artifact details in experiment modules

Boundaries:
  - This CLI does not train policies or render videos
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import experiments, utils


def build_parser() -> argparse.ArgumentParser:
    """Build the curriculum evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate trained curriculum PPO models on own-stage or named benchmark tasks.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--mode", choices=experiments.curriculum_evaluation.SUPPORTED_EVALUATION_MODES, required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--benchmark-config", type=Path, default=experiments.curriculum_evaluation.DEFAULT_BENCHMARK_CONFIG_PATH)
    parser.add_argument(
        "--model-scope",
        choices=experiments.curriculum_evaluation.SUPPORTED_MODEL_SCOPES,
        default=experiments.curriculum_evaluation.DEFAULT_MODEL_SCOPE,
    )
    parser.add_argument("--include-baseline-model", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--no-render", action="store_true", help="Disable default simulator GIF rendering.")
    parser.add_argument("--render-fps", type=int, default=experiments.curriculum_evaluation.DEFAULT_RENDER_FPS)
    parser.add_argument("--render-max-steps", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Disable default trajectory plot generation.")
    parser.add_argument(
        "--wandb-mode",
        choices=utils.wandb.WANDB_MODES,
        default=utils.wandb.WANDB_MODE_DISABLED,
        help="Accepted for evaluation CLI symmetry; local artifacts are always written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the curriculum evaluation CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = experiments.curriculum_evaluation.run_curriculum_evaluation(
        summary_path=args.summary,
        mode=args.mode,
        benchmark=args.benchmark,
        benchmark_config_path=args.benchmark_config,
        model_scope=args.model_scope,
        include_baseline_model=args.include_baseline_model,
        baseline_label=args.baseline_label,
        eval_steps=args.eval_steps,
        wandb_mode=args.wandb_mode,
        render=not args.no_render,
        render_fps=args.render_fps,
        render_max_steps=args.render_max_steps,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {"manifest_path": result.manifest_path, "metrics_path": result.metrics_path, "metrics": result.metrics},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
