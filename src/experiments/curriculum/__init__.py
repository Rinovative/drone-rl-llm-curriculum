"""
Curriculum orchestration helpers.

Provides:
- validation: curriculum task validation summary helpers
- training: fixed manual curriculum training orchestration
- evaluation: curriculum checkpoint evaluation orchestration
"""

from . import experiments_curriculum_evaluation as evaluation
from . import experiments_curriculum_training as training
from . import experiments_curriculum_validation as validation
from .experiments_curriculum_validation import (
    CurriculumTaskSummary,
    CurriculumValidationSummary,
    summarize_config_path,
    summarize_config_tasks,
    summarize_task_shapes,
)

__all__ = [
    "CurriculumTaskSummary",
    "CurriculumValidationSummary",
    "evaluation",
    "summarize_config_path",
    "summarize_config_tasks",
    "summarize_task_shapes",
    "training",
    "validation",
]
