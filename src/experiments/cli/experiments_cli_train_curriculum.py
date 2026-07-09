"""
===============================================================================
experiments_cli_train_curriculum.py
===============================================================================
Command-line entry point for manual PPO curriculum training.

Responsibilities:
  - Parse manual curriculum training arguments
  - Run sequential PPO curriculum orchestration through reusable helpers
  - Print compact summary and manifest paths for operators

Design principles:
  - Keep CLI behavior thin and deterministic
  - Leave training, diagnostics, and artifact details in experiment modules

Boundaries:
  - LLM curriculum generation and repair are not invoked here
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.experiments.curriculum import experiments_curriculum_training as curriculum_training


def build_parser() -> argparse.ArgumentParser:
    """Build the manual curriculum training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train PPO line tracking through a fixed manual curriculum and write a compact curriculum summary.",
    )
    parser.add_argument("--config", type=Path, default=curriculum_training.DEFAULT_CURRICULUM_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=utils.wandb.WANDB_MODES,
        default=None,
        help="W&B mode override for all curriculum stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manual curriculum training CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = curriculum_training.run_manual_curriculum_training_from_config(
        config_path=args.config,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
    )
    print(
        json.dumps(
            utils.serialization.to_jsonable(
                {
                    "manifest_path": result.manifest_path,
                    "summary_path": result.summary_path,
                    "summary": result.summary,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
