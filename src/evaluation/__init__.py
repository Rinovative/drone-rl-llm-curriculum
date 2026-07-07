"""
Evaluation metrics, plots, and result processing utilities.

Provides:
- trajectory_metrics: sampled trajectory tracking error metrics
- rollout: deterministic MVP rollout evaluation helpers
- plots: minimal trajectory comparison output helpers
- diagnostics: trained-policy evaluation failure diagnostics
"""

from . import evaluation_diagnostics as diagnostics
from . import evaluation_plots as plots
from . import evaluation_rollout as rollout
from . import evaluation_trajectory_metrics as trajectory_metrics

__all__ = [
    "diagnostics",
    "plots",
    "rollout",
    "trajectory_metrics",
]
