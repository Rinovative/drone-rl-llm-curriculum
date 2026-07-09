"""
===============================================================================
experiments_cli_render_scenario.py
===============================================================================
Command-line entry point for continuous multi-phase scenario rendering.

Responsibilities:
  - Parse scenario render config and bounded override arguments
  - Run one continuous scenario rollout with PPO or scripted-reference control
  - Print a compact JSON summary for review and demo workflows

Design principles:
  - Keep CLI behavior explicit, headless, and artifact-oriented
  - Delegate scenario composition and rendering to experiments_scenario_render.py

Boundaries:
  - PPO training belongs in cli_train_tracking.py
  - Task catalogs are selected by scenario configs, not played implicitly here
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.experiments.rendering import experiments_rendering_policy as policy_render
from src.experiments.rendering import experiments_rendering_scenario as scenario_render


def build_parser() -> argparse.ArgumentParser:
    """Build the continuous scenario render CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render a multi-phase scenario as one continuous evaluation rollout under storage/runs/<run_name>/evaluations/scenario.",
    )
    parser.add_argument("--config", type=Path, default=scenario_render.DEFAULT_SCENARIO_CONFIG_PATH)
    parser.add_argument("--run-name", type=str, default=None, help="Run name under storage/runs; artifacts go in evaluations/scenario.")
    parser.add_argument(
        "--controller",
        choices=policy_render.SUPPORTED_CONTROLLERS,
        default=None,
        help="Override the scenario controller.",
    )
    parser.add_argument(
        "--model-run-name",
        type=str,
        default=None,
        help="Training run name used by PPO scenarios under storage/runs/<model_run_name>/training/models.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--camera-mode", choices=policy_render.SUPPORTED_CAMERA_MODES, default=None)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-yaw", type=float, default=None)
    parser.add_argument("--camera-pitch", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scenario render CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    loaded = scenario_render.load_scenario_render_settings(args.config)
    settings = scenario_render.ScenarioRenderSettings(
        scenario_config_path=loaded.scenario_config_path,
        scenario_name=loaded.scenario_name,
        task_config_path=loaded.task_config_path,
        phases=loaded.phases,
        controller=loaded.controller if args.controller is None else args.controller,
        model_run_name=loaded.model_run_name if args.model_run_name is None else args.model_run_name,
        run_name=loaded.run_name if args.run_name is None else args.run_name,
        max_steps=loaded.max_steps if args.max_steps is None else args.max_steps,
        seed=loaded.seed if args.seed is None else args.seed,
        camera_mode=loaded.camera_mode if args.camera_mode is None else args.camera_mode,
        camera_distance=loaded.camera_distance if args.camera_distance is None else args.camera_distance,
        camera_yaw=loaded.camera_yaw if args.camera_yaw is None else args.camera_yaw,
        camera_pitch=loaded.camera_pitch if args.camera_pitch is None else args.camera_pitch,
        final_hold_sec=loaded.final_hold_sec,
    )
    result = scenario_render.run_scenario_render(settings)
    print(
        json.dumps(
            utils.serialization.to_jsonable(
                {
                    "gif_path": result.gif_path,
                    "manifest_path": result.manifest_path,
                    "metrics": result.metrics,
                    "warnings": list(result.warnings),
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
