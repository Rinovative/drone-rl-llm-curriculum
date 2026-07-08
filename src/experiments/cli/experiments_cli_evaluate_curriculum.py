"""
===============================================================================
experiments_cli_evaluate_curriculum.py
===============================================================================
Command-line entry point for curriculum evaluation suites.

Responsibilities:
  - Parse curriculum summary, suite, model-scope, and baseline arguments
  - Run reusable curriculum evaluation helpers
  - Print aggregate metrics and manifest paths for operators

Design principles:
  - Keep CLI behavior thin and report-oriented
  - Leave rollout and artifact details in experiment modules

Boundaries:
  - This CLI does not train policies or render videos directly
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.experiments.curriculum import experiments_curriculum_evaluation as curriculum_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the curriculum evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate trained curriculum PPO models through a canonical evaluation suite.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=curriculum_evaluation.DEFAULT_EVALUATION_SUITE_PATH,
        help="Canonical evaluation suite YAML path.",
    )
    parser.add_argument(
        "--mode",
        choices=curriculum_evaluation.SUPPORTED_EVALUATION_MODES,
        default=curriculum_evaluation.DEFAULT_EVALUATION_MODE,
    )
    parser.add_argument(
        "--model-scope",
        choices=curriculum_evaluation.SUPPORTED_MODEL_SCOPES,
        default=curriculum_evaluation.DEFAULT_MODEL_SCOPE,
    )
    parser.add_argument("--include-baseline-model", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--no-render", action="store_true", help="Disable simulator GIF rendering.")
    parser.add_argument("--render-fps", type=int, default=None, help="Override suite render FPS.")
    parser.add_argument("--render-max-steps", type=int, default=None, help="Override suite render rollout length.")
    parser.add_argument("--no-plots", action="store_true", help="Disable trajectory plot generation.")
    parser.add_argument("--no-traces", action="store_true", help="Disable rollout trace artifacts.")
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
    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=args.summary,
        mode=args.mode,
        suite_path=args.suite,
        model_scope=args.model_scope,
        include_baseline_model=args.include_baseline_model,
        baseline_label=args.baseline_label,
        eval_steps=args.eval_steps,
        wandb_mode=args.wandb_mode,
        render=False if args.no_render else None,
        render_fps=args.render_fps,
        render_max_steps=args.render_max_steps,
        plots=False if args.no_plots else None,
        traces=False if args.no_traces else None,
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
