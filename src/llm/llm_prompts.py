"""
===============================================================================
llm_prompts.py
===============================================================================
Build compact chat prompts for local-LLM curriculum task proposals.

Responsibilities:
  - Package the global task contract and supported schema into bounded prompts
  - Include recent accepted and rejected proposal context only
  - Build repair prompts with concrete parse, schema, or validation failures

Design principles:
  - Make JSON-only output expectations explicit in every prompt
  - Keep prompts provider-agnostic and deterministic

Boundaries:
  - Provider HTTP calls belong in llm_client.py
  - Parsing, validation, and repair-loop control belong in llm_curriculum.py
===============================================================================

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from . import llm_task_schema as task_schema

SYSTEM_PROMPT = (
    "You are a curriculum proposer for a single-drone reinforcement-learning experiment. "
    "You may propose one concrete trajectory task or select one known bounded task distribution. "
    "You never control the drone at runtime. You never generate Python code, shell commands, markdown, or prose. "
    "Your only output is one valid JSON object describing the next task or task distribution."
)
JSON_ONLY_INSTRUCTION = (
    "Return exactly one JSON object. Do not wrap it in markdown. Do not include prose before or after JSON. "
    "Do not include unsupported keys, python_code, command, script, shell, imports, or executable instructions."
)


def build_task_proposal_messages(
    *,
    curriculum_name: str,
    stage_index: int,
    recent_accepted_tasks: Sequence[Mapping[str, Any]],
    recent_rejected_tasks: Sequence[Mapping[str, Any]],
    metrics_summary: Mapping[str, Any] | None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build chat messages for the next curriculum task proposal.

    Parameters
    ----------
    curriculum_name
        Name of the curriculum run being proposed.
    stage_index
        One-based stage index for the proposed task.
    recent_accepted_tasks
        Bounded accepted task history.
    recent_rejected_tasks
        Bounded rejected proposal history with reasons.
    metrics_summary
        Compact metrics from the latest trained stage or dry-run placeholder.
    recent_context_limit
        Maximum number of accepted and rejected entries included in the prompt.
    budget_context
        Optional bounded budget profile context for this proposal.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-style chat messages.

    """
    context = _context_payload(
        curriculum_name=curriculum_name,
        stage_index=stage_index,
        recent_accepted_tasks=recent_accepted_tasks,
        recent_rejected_tasks=recent_rejected_tasks,
        metrics_summary=metrics_summary,
        recent_context_limit=recent_context_limit,
        budget_context=budget_context,
    )
    user_prompt = (
        f"{JSON_ONLY_INSTRUCTION}\n"
        "Propose the next training task using this bounded context. Prefer a small, feasible progression from the latest accepted task. "
        "If adaptive budget profiles are enabled, choose only stage_budget_profile=short, normal, recovery, or extend; never request raw timesteps.\n"
        f"Context JSON:\n{_compact_json(context)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_task_repair_messages(
    *,
    curriculum_name: str,
    stage_index: int,
    recent_accepted_tasks: Sequence[Mapping[str, Any]],
    recent_rejected_tasks: Sequence[Mapping[str, Any]],
    metrics_summary: Mapping[str, Any] | None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None = None,
    previous_response: str = "",
    error_messages: Sequence[str] = (),
) -> list[dict[str, str]]:
    """
    Build chat messages asking the LLM to repair an invalid proposal.

    Parameters
    ----------
    curriculum_name
        Name of the curriculum run being proposed.
    stage_index
        One-based stage index for the proposed task.
    recent_accepted_tasks
        Bounded accepted task history.
    recent_rejected_tasks
        Bounded rejected proposal history with reasons.
    metrics_summary
        Compact metrics from the latest trained stage or dry-run placeholder.
    recent_context_limit
        Maximum number of accepted and rejected entries included in the prompt.
    budget_context
        Optional bounded budget profile context for this repair attempt.
    previous_response
        Raw invalid response returned by the provider.
    error_messages
        Concrete parse, schema, or validation errors to repair.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-style chat messages.

    """
    context = _context_payload(
        curriculum_name=curriculum_name,
        stage_index=stage_index,
        recent_accepted_tasks=recent_accepted_tasks,
        recent_rejected_tasks=recent_rejected_tasks,
        metrics_summary=metrics_summary,
        recent_context_limit=recent_context_limit,
        budget_context=budget_context,
    )
    repair_payload = {
        "context": context,
        "previous_response": previous_response,
        "errors_to_fix": list(error_messages),
    }
    user_prompt = (
        f"{JSON_ONLY_INSTRUCTION}\n"
        "Repair the previous invalid proposal. Address every error. "
        "Use supported shapes, supported task-distribution families, safe numeric ranges, and one allowed budget profile. "
        "If stage_budget_profile is invalid, repair it to short, normal, recovery, or extend. "
        "Return a replacement JSON object only.\n"
        f"Repair JSON:\n{_compact_json(repair_payload)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _context_payload(
    *,
    curriculum_name: str,
    stage_index: int,
    recent_accepted_tasks: Sequence[Mapping[str, Any]],
    recent_rejected_tasks: Sequence[Mapping[str, Any]],
    metrics_summary: Mapping[str, Any] | None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the bounded JSON context embedded in proposal and repair prompts."""
    return {
        "curriculum_name": curriculum_name,
        "next_stage_index": stage_index,
        "task_contract": task_schema.build_task_prompt_contract(),
        "task_schema": task_schema.build_task_schema(),
        "llm_stage_budget": dict(budget_context or {}),
        "recent_accepted_tasks": _tail(recent_accepted_tasks, recent_context_limit),
        "recent_rejected_tasks": _tail(recent_rejected_tasks, recent_context_limit),
        "latest_metrics_summary": dict(metrics_summary or {}),
    }


def _tail(items: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return the last ``limit`` mapping entries as copied dictionaries."""
    if limit <= 0:
        return []
    return [dict(item) for item in items[-limit:]]


def _compact_json(payload: Mapping[str, Any]) -> str:
    """Serialize prompt context compactly and deterministically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "JSON_ONLY_INSTRUCTION",
    "SYSTEM_PROMPT",
    "build_task_proposal_messages",
    "build_task_repair_messages",
]
