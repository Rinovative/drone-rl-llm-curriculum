"""
Evaluation metrics, plots, and result processing utilities.

Provides:
- trajectory_metrics: sampled trajectory tracking error metrics
- rollout: deterministic MVP rollout evaluation helpers
- plots: minimal trajectory comparison output helpers
"""

from . import evaluation_plots as plots
from . import evaluation_rollout as rollout
from . import evaluation_trajectory_metrics as trajectory_metrics

__all__ = [
    "plots",
    "rollout",
    "trajectory_metrics",
]
