"""
Top-level package for the drone RL curriculum project.

Provides:
- envs: drone simulation environment builders and wrappers
- evaluation: evaluation metrics, plots, and result processing utilities
- experiments: experiment orchestration utilities and command-line entry points
- llm: LLM-guided curriculum generation and prompt handling
- trajectories: trajectory generation and reference path utilities
- utils: shared project utilities
- validation: deterministic validation for curriculum tasks
"""

from . import envs, evaluation, experiments, llm, trajectories, utils, validation

__all__ = [
    "envs",
    "evaluation",
    "experiments",
    "llm",
    "trajectories",
    "utils",
    "validation",
]
