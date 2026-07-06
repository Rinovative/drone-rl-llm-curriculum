"""
LLM-guided curriculum generation and prompt handling.

Provides:
- task_schema: deterministic schema helpers for proposed trajectory tasks
"""

from . import llm_task_schema as task_schema

__all__ = [
    "task_schema",
]
