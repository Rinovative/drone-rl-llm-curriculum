"""
Experiment orchestration utilities and command-line entry points.

Provides:
- config: minimal experiment configuration loading helpers
- curriculum: curriculum task summarization helpers
"""

from . import experiments_config as config
from . import experiments_curriculum as curriculum

__all__ = [
    "config",
    "curriculum",
]
