"""
LLM-guided curriculum generation and prompt handling.

Provides:
- client: deterministic mock and OpenAI-compatible chat clients
- curriculum: proposal parsing, validation, repair, and accounting
- json: strict JSON-only response parser
- logging: JSONL proposal event helpers
- progression: deterministic duplicate and progression transition helpers
- prompts: bounded curriculum prompt builders
- task_schema: deterministic schema helpers for proposed trajectory tasks
"""

from . import llm_client as client
from . import llm_curriculum as curriculum
from . import llm_json as json
from . import llm_logging as logging
from . import llm_progression as progression
from . import llm_prompts as prompts
from . import llm_task_schema as task_schema

__all__ = [
    "client",
    "curriculum",
    "json",
    "logging",
    "progression",
    "prompts",
    "task_schema",
]
