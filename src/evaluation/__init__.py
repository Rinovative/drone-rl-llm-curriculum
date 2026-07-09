"""
Evaluation metrics, plots, and result processing utilities.

Provides:
- trajectory_metrics: sampled trajectory tracking error metrics
- rollout: deterministic MVP rollout evaluation helpers
- plots: minimal trajectory comparison output helpers
- diagnostics: trained-policy evaluation failure diagnostics
- report: lightweight final-report artifact and matrix table helpers
"""

from . import evaluation_diagnostics as diagnostics
from . import evaluation_plots as plots
from . import evaluation_report as report
from . import evaluation_rollout as rollout
from . import evaluation_trajectory_metrics as trajectory_metrics

__all__ = [
    "diagnostics",
    "plots",
    "report",
    "rollout",
    "trajectory_metrics",
]
