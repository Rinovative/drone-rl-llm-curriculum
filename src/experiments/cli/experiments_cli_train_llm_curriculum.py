"""
===============================================================================
experiments_cli_train_llm_curriculum.py
===============================================================================
Command-line entry point for local-LLM-guided PPO curriculum training.

Responsibilities:
  - Parse LLM curriculum training and dry-run proposal arguments
  - Apply safe provider and stage-count overrides
  - Print compact summary, manifest, and proposal-log paths for operators

Design principles:
  - Keep CLI behavior thin and deterministic
  - Leave proposal repair, PPO training, and artifacts in reusable modules

Boundaries:
  - Local llama.cpp server startup is external to this repository
  - This CLI does not evaluate or compare trained policies
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import llm, utils
from src.experiments.curriculum import experiments_curriculum_llm_training as llm_curriculum_training


def build_parser() -> argparse.ArgumentParser:
    """Build the LLM curriculum training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train PPO trajectory tracking through a strict JSON local-LLM curriculum, or dry-run proposal validation only.",
    )
    parser.add_argument("--config", type=Path, default=llm_curriculum_training.DEFAULT_LLM_CURRICULUM_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=utils.wandb.WANDB_MODES,
        default=None,
        help="W&B mode override for all curriculum stages.",
    )
    parser.add_argument("--provider", choices=llm.client.SUPPORTED_PROVIDERS, default=None)
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base, for example http://127.0.0.1:18080/v1.")
    parser.add_argument("--model", default=None, help="Model name sent to the OpenAI-compatible provider.")
    parser.add_argument("--max-stages", type=int, default=None, help="Maximum curriculum stages including bootstrap.")
    parser.add_argument("--max-repair-attempts", type=int, default=None, help="Maximum repair completions after an invalid proposal.")
    parser.add_argument(
        "--dry-run-proposals",
        action="store_true",
        help="Exercise proposal generation, parsing, repair, validation, and logging without PPO training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the LLM curriculum training CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = llm_curriculum_training.run_llm_curriculum_training_from_config(
        config_path=args.config,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
        provider=args.provider,
        api_base=args.api_base,
        model=args.model,
        max_stages=args.max_stages,
        max_repair_attempts=args.max_repair_attempts,
        dry_run_proposals=args.dry_run_proposals,
    )
    print(
        json.dumps(
            utils.serialization.to_jsonable(
                {
                    "manifest_path": result.manifest_path,
                    "summary_path": result.summary_path,
                    "proposal_log_path": result.proposal_log_path,
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
