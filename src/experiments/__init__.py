"""
Experiment orchestration utilities and command-line entry points.

Provides:
- config: minimal experiment configuration loading helpers
- curriculum: curriculum task summarization helpers
- training_smoke: tiny deterministic MVP training-smoke helpers
"""

from . import experiments_config as config
from . import experiments_curriculum as curriculum
from . import experiments_training_smoke as training_smoke

__all__ = [
    "config",
    "curriculum",
    "training_smoke",
]
