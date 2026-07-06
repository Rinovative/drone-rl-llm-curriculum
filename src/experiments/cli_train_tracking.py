"""
===============================================================================
cli_train_tracking.py
===============================================================================
Command-line entry point for tiny PPO trajectory-tracking smoke training.

Responsibilities:
  - Parse bounded PPO smoke-training command-line arguments
  - Run the reusable PPO tracking smoke helper with safe defaults
  - Print a compact JSON summary for Docker and HPC workflows

Design principles:
  - Keep CLI defaults tiny, deterministic, headless, and reviewable
  - Delegate reusable PPO behavior to experiments_ppo_tracking.py

Boundaries:
  - Shell, queue, and Docker behavior belongs in scripts
  - Curriculum training orchestration belongs in later experiment modules
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import experiments, utils


def build_parser() -> argparse.ArgumentParser:
    """Build the PPO trajectory-tracking smoke CLI parser."""
    parser = argparse.ArgumentParser(description="Run tiny PPO smoke training on TrajectoryTrackingEnv.")
    parser.add_argument("--config", type=Path, default=experiments.ppo_tracking.DEFAULT_PPO_TRACKING_CONFIG_PATH)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument(
        "--task-shape",
        type=str,
        default=None,
        help="Train on the first configured task with this shape; uses task-index when omitted.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=utils.wandb.WANDB_MODES,
        default=None,
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PPO trajectory-tracking smoke CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = experiments.ppo_tracking.run_ppo_tracking_smoke_from_config(
        config_path=args.config,
        task_index=args.task_index,
        task_shape=args.task_shape,
        run_name=args.run_name,
        total_timesteps=args.total_timesteps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_tags=utils.wandb.parse_wandb_tags(args.wandb_tags),
        wandb_dir=args.wandb_dir,
    )
    print(
        json.dumps(
            {
                "metrics": result.metrics,
                "metrics_path": result.metrics_path,
                "model_path": result.model_path,
                "warnings": list(result.warnings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
