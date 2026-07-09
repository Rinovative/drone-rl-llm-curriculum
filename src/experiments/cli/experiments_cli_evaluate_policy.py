"""
===============================================================================
experiments_cli_evaluate_policy.py
===============================================================================
Command-line entry point for direct PPO policy evaluation profiles.

Responsibilities:
  - Parse a direct PPO run manifest and optional canonical evaluation suite path
  - Evaluate the trained direct PPO model through the shared policy evaluator
  - Store own-task and suite artifacts under the owning direct PPO run

Design principles:
  - Keep the CLI thin, deterministic, and report-oriented
  - Use run manifests as the source of model and artifact ownership

Boundaries:
  - This CLI does not train policies or compare multiple runs
  - Curriculum evaluation remains in the curriculum CLI
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the direct PPO policy evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a direct PPO run through the standard profile or one explicit suite.",
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("standard", "scenario"),
        default="standard",
        help="Evaluation profile to run when --suite is omitted.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Canonical evaluation suite YAML path. Omit to run the standard profile.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=utils.wandb.WANDB_MODES,
        default=utils.wandb.WANDB_MODE_DISABLED,
        help="Accepted for CLI symmetry; local evaluation artifacts are always written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run direct PPO suite evaluation and return a process status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result: (
        policy_evaluation.PolicyStandardEvaluationResult
        | policy_evaluation.PolicySuiteEvaluationResult
        | policy_evaluation.PolicyScenarioEvaluationResult
    )
    if args.profile == "scenario":
        if args.suite is not None:
            parser.error("--suite cannot be combined with --profile scenario")
        result = policy_evaluation.run_direct_policy_scenario_evaluation(
            run_manifest_path=args.run_manifest,
            wandb_mode=args.wandb_mode,
        )
    elif args.suite is None:
        result = policy_evaluation.run_direct_policy_standard_evaluation(
            run_manifest_path=args.run_manifest,
            wandb_mode=args.wandb_mode,
        )
    else:
        result = policy_evaluation.run_direct_policy_suite_evaluation(
            run_manifest_path=args.run_manifest,
            suite_path=args.suite,
            wandb_mode=args.wandb_mode,
        )
    print(
        json.dumps(
            utils.serialization.to_jsonable({"manifest_path": result.manifest_path, "metrics_path": result.metrics_path, "metrics": result.metrics}),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
