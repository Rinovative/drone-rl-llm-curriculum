"""
===============================================================================
experiments_cli_render_smoke.py
===============================================================================
Command-line entry point for tiny headless drone render smoke rollouts.

Responsibilities:
  - Parse bounded render-smoke command-line arguments
  - Run the gym-pybullet-drones smoke helper with safe visual defaults
  - Print a compact JSON summary for reproduction through module execution

Design principles:
  - Keep CLI behavior fast, headless, and safe by default
  - Delegate reusable simulator and artifact logic to experiments_render_smoke.py

Boundaries:
  - Full training orchestration and Docker runner internals belong elsewhere
===============================================================================

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.experiments.rendering import experiments_rendering_smoke as render_smoke


def build_parser() -> argparse.ArgumentParser:
    """Build the render-smoke CLI parser."""
    parser = argparse.ArgumentParser(description="Run a tiny headless gym-pybullet-drones render smoke rollout.")
    parser.add_argument("--duration-sec", type=float, default=render_smoke.DEFAULT_DURATION_SEC)
    parser.add_argument("--max-steps", type=int, default=render_smoke.DEFAULT_MAX_STEPS)
    parser.add_argument("--output-dir", type=Path, default=render_smoke.default_output_dir())
    parser.add_argument("--seed", type=int, default=render_smoke.DEFAULT_SEED)
    parser.add_argument("--frame-interval", type=int, default=render_smoke.DEFAULT_FRAME_INTERVAL)
    parser.add_argument("--image-width", type=int, default=render_smoke.DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=render_smoke.DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--camera-mode", choices=render_smoke.SUPPORTED_CAMERA_MODES, default=render_smoke.DEFAULT_CAMERA_MODE)
    parser.add_argument("--task-shape", choices=render_smoke.SUPPORTED_TASK_SHAPES, default=render_smoke.DEFAULT_TASK_SHAPE)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the render-smoke CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    result = render_smoke.run_render_smoke(
        render_smoke.RenderSmokeSettings(
            duration_sec=args.duration_sec,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            seed=args.seed,
            frame_interval=args.frame_interval,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_mode=args.camera_mode,
            task_shape=args.task_shape,
        )
    )
    print(
        json.dumps(
            utils.serialization.to_jsonable({"manifest_path": result.manifest_path, "manifest": result.manifest, "warnings": list(result.warnings)}),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
