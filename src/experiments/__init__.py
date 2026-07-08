"""
Experiment orchestration utilities and command-line entry points.

Provides:
- cli: executable experiment command modules
- config: minimal experiment configuration loading helpers
- curriculum: curriculum validation, training, and evaluation helpers
- evaluation: report-ready policy evaluation helpers
- rendering: policy, scenario, and render-smoke helpers
- training: PPO configuration, tracking training, and smoke helpers
"""

from . import cli, curriculum, evaluation, rendering, training
from . import experiments_config as config

__all__ = [
    "cli",
    "config",
    "curriculum",
    "evaluation",
    "rendering",
    "training",
]
