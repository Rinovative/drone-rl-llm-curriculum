"""
Validation utilities for curriculum tasks and generated trajectories.

Provides:
- contracts: shared task vocabulary constants
- tasks: deterministic task and trajectory feasibility checks
"""

from . import validation_contracts as contracts
from . import validation_tasks as tasks

__all__ = [
    "contracts",
    "tasks",
]
