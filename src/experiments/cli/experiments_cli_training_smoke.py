"""
===============================================================================
experiments_cli_training_smoke.py
===============================================================================
Command-line entry point for tiny deterministic MVP training-smoke runs.

Responsibilities:
  - Parse smoke-training command-line arguments
  - Run the deterministic smoke training helper with bounded defaults
  - Print a compact JSON summary for reviewer-facing reproduction

Design principles:
  - Keep CLI behavior fast, headless, and safe by default
  - Delegate reusable logic to experiments_training_smoke.py

Boundaries:
  - Full training orchestration and long-running jobs belong elsewhere
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.training import experiments_training_smoke as training_smoke


def build_parser() -> argparse.ArgumentParser:
    """Build the training-smoke CLI parser."""
    parser = argparse.ArgumentParser(description="Run a tiny deterministic MVP training smoke loop.")
    parser.add_argument("--config", type=Path, default=training_smoke.DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the training-smoke CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = training_smoke.run_training_smoke_from_config(
        config_path=args.config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        task_index=args.task_index,
    )
    print(json.dumps({"output_path": result.output_path, "metrics": result.metrics, "warnings": list(result.warnings)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
