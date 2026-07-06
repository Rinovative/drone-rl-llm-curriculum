"""
===============================================================================
cli_render_policy.py
===============================================================================
Command-line entry point for trained PPO policy rollout rendering.

Responsibilities:
  - Parse bounded trained-policy render command-line arguments
  - Run trained PPO rollout rendering with external third-person camera capture
  - Print a compact JSON summary for reviewer and demo workflows

Design principles:
  - Keep CLI behavior headless, deterministic, and fast by default
  - Delegate reusable rollout and artifact logic to experiments_policy_render.py

Boundaries:
  - PPO training and model creation belong in cli_train_tracking.py
  - Simulator wrappers and reward logic belong in envs modules
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import experiments


def build_parser() -> argparse.ArgumentParser:
    """Build the trained-policy render CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render a PPO or scripted-reference evaluation run and write artifacts under storage/evaluation_runs/<run_name>.",
    )
    parser.add_argument("--model-path", type=Path, default=experiments.policy_render.default_model_path())
    parser.add_argument(
        "--model-run-name",
        type=str,
        default=None,
        help="Training run name used to load a model from storage/training_runs/<model_run_name>/models.",
    )
    parser.add_argument("--config", type=Path, default=experiments.policy_render.DEFAULT_PPO_CONFIG_PATH)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument(
        "--task-shape",
        "--render-task-shape",
        dest="render_task_shape",
        type=str,
        default=None,
        help="Render a task with this shape from the configured task list; hover remains the default when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Evaluation run name under storage/evaluation_runs.")
    parser.add_argument(
        "--controller",
        choices=experiments.policy_render.SUPPORTED_CONTROLLERS,
        default=experiments.policy_render.PPO_CONTROLLER,
        help="Choose PPO rendering or a scripted-reference evaluation baseline.",
    )
    parser.add_argument("--max-steps", type=int, default=experiments.policy_render.DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=experiments.policy_render.DEFAULT_SEED)
    parser.add_argument(
        "--camera-mode",
        choices=experiments.policy_render.SUPPORTED_CAMERA_MODES,
        default=experiments.policy_render.DEFAULT_CAMERA_MODE,
    )
    parser.add_argument("--camera-distance", type=float, default=experiments.policy_render.DEFAULT_CAMERA_DISTANCE_M)
    parser.add_argument("--camera-yaw", type=float, default=experiments.policy_render.DEFAULT_CAMERA_YAW_DEG)
    parser.add_argument("--camera-pitch", type=float, default=experiments.policy_render.DEFAULT_CAMERA_PITCH_DEG)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the trained-policy render CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = experiments.policy_render.run_trained_policy_render_from_paths(
        model_path=args.model_path,
        model_run_name=args.model_run_name,
        config_path=args.config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        camera_mode=args.camera_mode,
        task_index=args.task_index,
        render_task_shape=args.render_task_shape,
        controller=args.controller,
        run_name=args.run_name,
        camera_distance=args.camera_distance,
        camera_yaw=args.camera_yaw,
        camera_pitch=args.camera_pitch,
    )
    print(
        json.dumps(
            {
                "gif_path": result.gif_path,
                "manifest_path": result.manifest_path,
                "metrics": result.metrics,
                "warnings": list(result.warnings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
