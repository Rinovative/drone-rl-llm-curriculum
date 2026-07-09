"""
===============================================================================
llm_prompts.py
===============================================================================
Build compact chat prompts for local-LLM curriculum task proposals.

Responsibilities:
  - Package the global task contract and supported schema into bounded prompts
  - Include compact accepted-stage history, recent rejections, and concrete diagnostics
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
import math
from collections.abc import Mapping, Sequence
from typing import Any

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
PROMPT_COMPACTION_FULL = "full"
PROMPT_COMPACTION_COMPACT = "compact"
PROMPT_COMPACTION_MINIMAL = "minimal"
PROMPT_COMPACTION_MODES = (PROMPT_COMPACTION_FULL, PROMPT_COMPACTION_COMPACT, PROMPT_COMPACTION_MINIMAL)
PROMPT_TOKEN_CHAR_RATIO = 3.5
DEFAULT_PROMPT_CONTEXT_LIMIT_TOKENS = 16384
DEFAULT_PROMPT_RESPONSE_RESERVE_TOKENS = 1800
DEFAULT_PROMPT_BUDGET_TOKENS = 12000
MINIMAL_HISTORY_DETAIL_LIMIT = 3
COMPACT_HISTORY_DETAIL_LIMIT = 10


def build_task_proposal_messages(
    *,
    curriculum_name: str,
    stage_index: int,
    recent_accepted_tasks: Sequence[Mapping[str, Any]],
    recent_rejected_tasks: Sequence[Mapping[str, Any]],
    metrics_summary: Mapping[str, Any] | None,
    curriculum_history: Sequence[Mapping[str, Any]] = (),
    curriculum_summary: Mapping[str, Any] | None = None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None = None,
    compaction_mode: str = PROMPT_COMPACTION_FULL,
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
    curriculum_history
        Compact history for all accepted stages, including stage 1 bootstrap.
    curriculum_summary
        Aggregate family counts, trend summaries, and diagnostic guidance.
    recent_context_limit
        Maximum number of accepted and rejected entries included in the prompt.
    budget_context
        Optional bounded budget profile context for this proposal.
    compaction_mode
        Prompt compaction mode used to bound context size.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-style chat messages.

    """
    mode = _normalize_compaction_mode(compaction_mode)
    context = _context_payload(
        curriculum_name=curriculum_name,
        stage_index=stage_index,
        recent_accepted_tasks=recent_accepted_tasks,
        recent_rejected_tasks=recent_rejected_tasks,
        metrics_summary=metrics_summary,
        curriculum_history=curriculum_history,
        curriculum_summary=curriculum_summary,
        recent_context_limit=recent_context_limit,
        budget_context=budget_context,
        compaction_mode=mode,
    )
    user_prompt = f"{JSON_ONLY_INSTRUCTION}\n{_proposal_instruction_text(mode)}\nContext JSON:\n{_compact_json(context)}"
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
    curriculum_history: Sequence[Mapping[str, Any]] = (),
    curriculum_summary: Mapping[str, Any] | None = None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None = None,
    compaction_mode: str = PROMPT_COMPACTION_FULL,
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
    curriculum_history
        Compact history for all accepted stages, including stage 1 bootstrap.
    curriculum_summary
        Aggregate family counts, trend summaries, and diagnostic guidance.
    recent_context_limit
        Maximum number of accepted and rejected entries included in the prompt.
    budget_context
        Optional bounded budget profile context for this repair attempt.
    compaction_mode
        Prompt compaction mode used to bound context size.
    previous_response
        Raw invalid response returned by the provider.
    error_messages
        Concrete parse, schema, or validation errors to repair.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-style chat messages.

    """
    mode = _normalize_compaction_mode(compaction_mode)
    context = _context_payload(
        curriculum_name=curriculum_name,
        stage_index=stage_index,
        recent_accepted_tasks=recent_accepted_tasks,
        recent_rejected_tasks=recent_rejected_tasks,
        metrics_summary=metrics_summary,
        curriculum_history=curriculum_history,
        curriculum_summary=curriculum_summary,
        recent_context_limit=recent_context_limit,
        budget_context=budget_context,
        compaction_mode=mode,
    )
    repair_payload = {
        "context": context,
        "previous_response_summary": _compact_previous_response(previous_response, mode),
        "errors_to_fix": _limit_strings(error_messages, 6 if mode == PROMPT_COMPACTION_FULL else 3),
    }
    user_prompt = f"{JSON_ONLY_INSTRUCTION}\n{_repair_instruction_text(mode)}\nRepair JSON:\n{_compact_json(repair_payload)}"
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
    curriculum_history: Sequence[Mapping[str, Any]],
    curriculum_summary: Mapping[str, Any] | None,
    recent_context_limit: int,
    budget_context: Mapping[str, Any] | None,
    compaction_mode: str,
) -> dict[str, Any]:
    """Build the bounded JSON context embedded in proposal and repair prompts."""
    mode = _normalize_compaction_mode(compaction_mode)
    summary = dict(curriculum_summary or {})
    metrics = dict(metrics_summary or {})
    context: dict[str, Any] = {
        "curriculum_name": curriculum_name,
        "next_stage_index": stage_index,
        "compaction_mode": mode,
        "task_catalog": _compact_task_catalog(),
        "proposal_rules": _essential_rules(mode),
        "llm_stage_budget": _compact_budget_context(budget_context, mode),
        "immediate_previous_stage_task_shape": _previous_stage_task_shape(recent_accepted_tasks),
        "previous_accepted_family": summary.get("last_accepted_task_family") or summary.get("previous_stage_task_family"),
        "used_family_counts": _dict_of_scalars(summary.get("accepted_task_family_counts")),
        "recent_accepted_tasks": _compact_recent_accepted_tasks(recent_accepted_tasks, recent_context_limit, mode),
        "recent_rejected_tasks": _compact_rejected_tasks(recent_rejected_tasks, recent_context_limit, mode),
        "curriculum_history": _compact_history(curriculum_history, mode),
        "curriculum_summary": _compact_curriculum_summary(summary, metrics, mode),
        "curriculum_feedback": _curriculum_feedback_context(metrics, summary),
        "latest_metrics_summary": _compact_metrics_summary(metrics),
    }
    if mode == PROMPT_COMPACTION_FULL:
        context["task_contract"] = task_schema.build_task_prompt_contract()
        context["task_schema"] = _compact_task_schema(task_schema.build_task_schema())
        context["diagnostic_interpretation_policy"] = _diagnostic_policy()
    elif mode == PROMPT_COMPACTION_COMPACT:
        context["task_schema"] = _minimal_task_schema()
        context["diagnostic_interpretation_policy"] = _diagnostic_policy(compact=True)
    else:
        context["task_schema"] = _minimal_task_schema()
    return context


def _curriculum_feedback_context(
    metrics_summary: Mapping[str, Any] | None,
    curriculum_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return compact structured curriculum-feedback guidance for prompts."""
    metrics = dict(metrics_summary or {})
    summary = dict(curriculum_summary or {})
    strategy = metrics.get("curriculum_strategy")
    strategy_payload = dict(strategy) if isinstance(strategy, Mapping) else {}
    candidates = _compact_candidate_task_families(metrics.get("curriculum_recommended_next_task_families"))
    if not candidates:
        candidates = _compact_candidate_task_families(summary.get("latest_recommended_next_task_families"))
    return {
        "guidance_not_absolute_command": True,
        "llm_instruction_summary": _optional_text(metrics.get("curriculum_feedback_summary") or summary.get("latest_curriculum_feedback_summary")),
        "trend_status": _optional_text(summary.get("trend_status") or summary.get("position_error_trend")),
        "primary_skill_gaps": _limit_strings(metrics.get("curriculum_primary_skill_gaps"), 3),
        "candidate_next_skills": _limit_strings(strategy_payload.get("candidate_next_skills"), 3),
        "candidate_next_task_families": candidates[:5],
        "cautions": _limit_strings(
            metrics.get("curriculum_avoid_next_task_families") or strategy_payload.get("cautions") or summary.get("top_repeated_failure_modes"),
            3,
        ),
        "skill_gap_counts": _dict_of_scalars(summary.get("skill_gap_counts")),
    }


def estimate_messages_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    """Estimate prompt tokens conservatively from chat message characters."""
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("role", ""))) + len(str(message.get("content", ""))) + 8
    return math.ceil(total_chars / PROMPT_TOKEN_CHAR_RATIO)


def prompt_metadata(
    *,
    messages: Sequence[Mapping[str, str]],
    context_limit_tokens: int,
    prompt_budget_tokens: int,
    compaction_mode: str,
) -> dict[str, Any]:
    """Return compact prompt-size metadata without storing prompt text."""
    mode = _normalize_compaction_mode(compaction_mode)
    return {
        "llm_prompt_estimated_tokens": estimate_messages_tokens(messages),
        "llm_prompt_context_limit": int(context_limit_tokens),
        "llm_prompt_budget_tokens": int(prompt_budget_tokens),
        "llm_prompt_compaction_mode": mode,
        "llm_prompt_sections_dropped": _sections_dropped(mode),
        "llm_prompt_sections_summarized": _sections_summarized(mode),
    }


def _normalize_compaction_mode(mode: str) -> str:
    """Return a supported prompt compaction mode."""
    normalized = str(mode or PROMPT_COMPACTION_FULL).strip().lower()
    if normalized not in PROMPT_COMPACTION_MODES:
        message = f"prompt compaction mode must be one of: {', '.join(PROMPT_COMPACTION_MODES)}"
        raise ValueError(message)
    return normalized


def _proposal_instruction_text(mode: str) -> str:
    """Return concise proposal instructions for a compaction mode."""
    prefix = "Propose the next bounded per-episode training task or known task distribution."
    if mode == PROMPT_COMPACTION_MINIMAL:
        return (
            f"{prefix} Use feedback/history/catalog. Avoid immediate duplicate accepted identity. "
            "Choose focused task-family distributions; tracking_small is fallback/consolidation only and tracking_medium is final_broad/late only. "
            "Prefer known safe distributions over raw geometry. Use lower starts around 0.70-0.95m with start_hold_sec=1.2. "
            "Output valid JSON with one allowed stage_budget_profile; never request raw timesteps."
        )
    return (
        f"{prefix} Prefer a small, feasible progression from the latest accepted task. "
        "Do not repeat immediate_previous_stage_task_shape; choose a different task family/shape as a concrete task or known distribution. "
        "Use curriculum_history, curriculum_summary, and curriculum_feedback to avoid looping and to choose concrete, safe progressions. "
        "Use curriculum_feedback as guidance, not as an absolute command. Prefer targeted skill training over returning to hover for every problem. "
        "Use concrete metrics and trends instead of a single readiness_level; readiness_level is intentionally omitted from context. "
        "If z/altitude is weak, consider controlled vertical, takeoff, or altitude-hold tasks. "
        "If XY tracking is weak, consider shorter or slower line tasks. If turns are weak, consider slow polyline or L-shape. "
        "If curvature is weak, consider gentle ellipse or slow circle before figure-eight. "
        "If a reference was too hard, choose an easier or slower same-family variant. "
        "Treat action_saturation, z_instability, and reference_too_fast_or_too_hard as diagnostic context or task-difficulty signals. "
        "Do not choose broad shows, scenarios, or basic_training_show as normal LLM stages. "
        "Choose focused task-family distributions; tracking_small is fallback/consolidation only "
        "and tracking_medium is final_broad/late consolidation only. "
        "Use bounded per-episode distributions and prefer known safe distributions over raw geometry. "
        "Use adjusted lower starts around 0.70-0.95m where safe, start_hold_enabled=true, start_hold_sec=1.2, "
        "and exclude_start_hold_from_tracking_metrics=true. If adaptive budgets are enabled, choose only a listed stage_budget_profile. "
        "Never request raw timesteps."
    )


def _repair_instruction_text(mode: str) -> str:
    """Return concise repair instructions for a compaction mode."""
    base = (
        "Repair the previous invalid proposal. Address every error, including any immediate duplicate task-family rejection. "
        "Return either a concrete task with task_type and shape, or a valid task-distribution reference from the supported list "
        "using task_distribution_id or task_distribution_config_path. Use supported shapes, supported task-distribution families, "
        "concrete safe task values, and curriculum_feedback as guidance, not as an absolute command. "
        "z_instability can be repaired with controlled vertical or takeoff tasks; XY weakness with slower/shorter lines; "
        "turn weakness with slow L-shape/polyline; curvature weakness with gentle ellipse/circle; "
        "repeated crash/divergence metrics are needed to confirm true instability. "
        "Do not choose broad shows, scenarios, or basic_training_show as normal LLM stages; "
        "known distribution ids/paths and one budget profile from llm_stage_budget.allowed_profile_names are required. "
        "If stage_budget_profile is invalid, repair it to an allowed profile. Return a replacement JSON object only."
    )
    if mode == PROMPT_COMPACTION_MINIMAL:
        return base.replace(" Use supported shapes, supported task-distribution families, concrete safe task values,", " Use supported values,")
    return base


def _essential_rules(mode: str) -> list[str]:
    """Return essential proposal rules for prompt context."""
    rules = [
        "choose focused task-family distributions for normal stages",
        "tracking_small is fallback/consolidation only",
        "tracking_medium is final_broad/late consolidation only",
        "avoid immediate duplicate accepted identity",
        "use feedback/history/catalog, but feedback is not an absolute command",
        "use bounded per-episode distributions and prefer known safe distributions over raw geometry",
        "output one valid JSON object",
    ]
    if mode != PROMPT_COMPACTION_MINIMAL:
        rules.extend(
            [
                "adjusted lower-reference starts should usually be 0.70-0.95m, with training start_hold_sec=1.2",
                "do not request PPO hyperparameters, action interfaces, observation flags, reward changes, or raw timesteps",
            ]
        )
    return rules


def _compact_task_catalog() -> list[str]:
    """Return a one-line-per-entry task catalog for prompts."""
    rows = [
        "id | role | family | skills | difficulty | use_when",
        "hover_bootstrap | focused | hover_stabilization | settle,hold | easy | bootstrap/recovery only",
        "vertical_bootstrap | focused | takeoff_stabilization | altitude_hold | easy | basic z control",
        "vertical_up_down_bootstrap | focused | vertical_up_down | climb,descent | easy-medium | altitude_control gap",
        "angled_vertical_bootstrap | focused | angled_vertical | z+xy coupling | medium | angled climb/descent",
        "short_line_bootstrap | focused | start_hold_then_line | xy_tracking | easy | short controlled line",
        "line_bootstrap | focused | line | xy_tracking | medium | longer line after short success",
        "polyline_bootstrap | focused | polyline | turns | medium | turn_following gap",
        "delayed_altitude_polyline_bootstrap | focused | delayed_altitude_polyline | delayed z | medium | lateral first then climb/descent",
        "multi_height_polyline_bootstrap | focused | multi_height_polyline | varied altitude | medium | preserve altitude learning",
        "zigzag_bootstrap | focused | zigzag | repeated turns | medium | turn-following progression",
        "tracking_small | fallback | mixed_small | consolidation | medium | fallback/consolidation only",
        "tracking_medium | final_broad | mixed_medium | broad coverage | hard | late/final broad only",
    ]
    known_ids = set(task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS)
    return [rows[0], *(row for row in rows[1:] if row.split(" | ", maxsplit=1)[0] in known_ids)]


def _compact_task_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return schema fields worth keeping in full prompt mode."""
    return {
        "task_type": schema.get("task_type"),
        "proposal_kinds": schema.get("proposal_kinds"),
        "task_distribution_reference_fields": schema.get("task_distribution_reference_fields"),
        "known_task_distribution_ids": schema.get("known_task_distribution_ids"),
        "allowed_stage_budget_profiles": schema.get("allowed_stage_budget_profiles"),
        "forbidden_example_fields": schema.get("forbidden_example_fields"),
        "shape_required_fields": schema.get("shape_required_fields"),
        "shape_optional_fields": schema.get("shape_optional_fields"),
    }


def _minimal_task_schema() -> dict[str, Any]:
    """Return the minimum schema needed for compact/minimal prompts."""
    return {
        "proposal_kinds": [task_schema.PROPOSAL_KIND_TASK, task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION],
        "task_distribution_reference_fields": [
            task_schema.TASK_DISTRIBUTION_ID_FIELD,
            task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD,
        ],
        "known_task_distribution_ids": dict(task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS),
        "allowed_stage_budget_profiles": list(task_schema.DEFAULT_STAGE_BUDGET_PROFILES),
    }


def _compact_budget_context(budget_context: Mapping[str, Any] | None, mode: str) -> dict[str, Any]:
    """Return prompt-safe adaptive-budget context."""
    budget = dict(budget_context or {})
    allowed_profiles = budget.get("allowed_profiles")
    compact_profiles: dict[str, Any] = {}
    if isinstance(allowed_profiles, Mapping) and mode == PROMPT_COMPACTION_FULL:
        compact_profiles = {str(name): value for name, value in allowed_profiles.items()}
    return {
        "enabled": budget.get("enabled"),
        "allowed_profile_names": list(budget.get("allowed_profile_names") or []),
        "allowed_profiles": compact_profiles,
        "default_profile": budget.get("default_profile"),
        "total_budget_cap_timesteps": budget.get("total_budget_cap_timesteps"),
        "cumulative_llm_budget_timesteps": budget.get("cumulative_llm_budget_timesteps"),
        "next_stage_index": budget.get("next_stage_index"),
        "remaining_stage_slots_including_current": budget.get("remaining_stage_slots_including_current"),
    }


def _compact_recent_accepted_tasks(
    items: Sequence[Mapping[str, Any]],
    limit: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Return recent accepted tasks without full task payloads."""
    max_items = min(limit, 3 if mode == PROMPT_COMPACTION_MINIMAL else limit)
    return [
        {
            "stage_index": item.get("stage_index"),
            "accepted_stage_family": item.get("accepted_task_family"),
            "accepted_distribution_id": item.get("task_distribution_id"),
            "accepted_distribution_role": _distribution_role(item.get("task_distribution_id")),
            "sampled_task_family": _sampled_task_family(item),
            "budget_profile": item.get("selected_stage_budget_profile"),
            "stage_total_timesteps": item.get("stage_total_timesteps"),
            "metrics": _compact_metrics_summary(item.get("metrics") if isinstance(item.get("metrics"), Mapping) else item),
            "skill_gaps": _limit_strings(_feedback_from_item(item).get("primary_skill_gaps"), MINIMAL_HISTORY_DETAIL_LIMIT),
            "feedback_summary": _optional_text(_feedback_from_item(item).get("llm_instruction_summary")),
        }
        for item in _tail(items, max_items)
    ]


def _compact_rejected_tasks(
    items: Sequence[Mapping[str, Any]],
    limit: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Return recent rejected proposal summaries without raw responses."""
    max_items = min(limit, 2 if mode == PROMPT_COMPACTION_MINIMAL else limit)
    return [
        {
            "stage_index": item.get("stage_index"),
            "attempt_index": item.get("attempt_index"),
            "error_type": item.get("error_type"),
            "rejection_reasons": _limit_strings(item.get("rejection_reasons"), MINIMAL_HISTORY_DETAIL_LIMIT),
            "requested_stage_task_shape": item.get("requested_stage_task_shape"),
            "duplicate_task_rejected": item.get("duplicate_task_rejected"),
        }
        for item in _tail(items, max_items)
    ]


def _compact_history(items: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Return compact curriculum history, summarizing older stages when needed."""
    if not items:
        return []
    source = list(items)
    if mode == PROMPT_COMPACTION_MINIMAL and len(source) > MINIMAL_HISTORY_DETAIL_LIMIT:
        source = source[-MINIMAL_HISTORY_DETAIL_LIMIT:]
    elif mode == PROMPT_COMPACTION_COMPACT and len(source) > COMPACT_HISTORY_DETAIL_LIMIT:
        source = source[-COMPACT_HISTORY_DETAIL_LIMIT:]
    return [_compact_history_item(item, mode) for item in source]


def _compact_history_item(item: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Return one compact stage-history item."""
    feedback = _feedback_from_item(item)
    metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else item
    signals = _top_diagnostic_signals(feedback.get("diagnostic_signals"), limit=1 if mode == PROMPT_COMPACTION_MINIMAL else 3)
    return {
        "stage_index": item.get("stage_index"),
        "accepted_stage_family": item.get("accepted_task_family"),
        "accepted_distribution_id": item.get("task_distribution_id"),
        "accepted_distribution_role": _distribution_role(item.get("task_distribution_id")),
        "sampled_task_family": _sampled_task_family(item),
        "budget_profile": item.get("selected_stage_budget_profile"),
        "stage_total_timesteps": item.get("stage_total_timesteps"),
        "metrics": _compact_metrics_summary(metrics if isinstance(metrics, Mapping) else {}),
        "primary_skill_gaps": _limit_strings(feedback.get("primary_skill_gaps"), 3),
        "diagnostic_signals": signals,
        "feedback_summary": _optional_text(feedback.get("llm_instruction_summary")),
    }


def _compact_curriculum_summary(
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Return aggregate context without unbounded metadata."""
    result = {
        "completed_stage_count": summary.get("completed_stage_count"),
        "previous_stage_task_family": summary.get("previous_stage_task_family"),
        "previous_stage_task_shape": summary.get("previous_stage_task_shape"),
        "accepted_task_family_counts": _dict_of_scalars(summary.get("accepted_task_family_counts")),
        "last_accepted_task_family": summary.get("last_accepted_task_family"),
        "trend_status": summary.get("trend_status") or summary.get("position_error_trend"),
        "top_repeated_failure_modes": list(summary.get("top_repeated_failure_modes") or [])[:3],
        "strongest_task_families": _limit_strings(summary.get("strongest_task_families"), 3),
        "weakest_task_families": _limit_strings(summary.get("weakest_task_families"), 3),
        "latest_curriculum_feedback_summary": summary.get("latest_curriculum_feedback_summary") or metrics.get("curriculum_feedback_summary"),
        "latest_recommended_next_task_families": _compact_candidate_task_families(
            summary.get("latest_recommended_next_task_families") or metrics.get("curriculum_recommended_next_task_families")
        )[:5],
        "skill_gap_counts": _dict_of_scalars(summary.get("skill_gap_counts")),
        "recommended_avoid_immediate_duplicate_family": True,
        "readiness_level_omitted_from_llm_context": True,
    }
    if mode == PROMPT_COMPACTION_FULL:
        result["allowed_task_families"] = _limit_strings(summary.get("allowed_task_families"), 20)
    return result


def _compact_metrics_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only metrics useful to choose the next task."""
    metrics = dict(value or {})
    result = {
        "mean_reward": metrics.get("mean_reward"),
        "mean_position_error_tracking_m": metrics.get("mean_position_error_tracking_m"),
        "failure_overall_status": metrics.get("failure_overall_status") or metrics.get("status"),
        "failure_primary_mode": metrics.get("failure_primary_mode"),
        "trend_status": metrics.get("trend_status") or metrics.get("position_error_trend"),
    }
    return {key: val for key, val in result.items() if val is not None}


def _diagnostic_policy(compact: bool = False) -> dict[str, Any]:
    """Return concise diagnostic interpretation policy."""
    policy = {
        "readiness_level_omitted": True,
        "use_concrete_metrics_and_trends": True,
        "do_not_overreact_to_single_failure_mode": True,
        "action_saturation": "difficulty_signal_not_automatic_instability",
        "z_instability": "consider_controlled_vertical_or_altitude_hold",
        "reference_too_fast_or_too_hard": "choose_easier_or_slower_same_family_variant",
    }
    if not compact:
        policy.update(
            {
                "xy_weakness_next_tasks": ["short_line_bootstrap", "line_bootstrap"],
                "turn_weakness_next_tasks": ["polyline_bootstrap", "zigzag_bootstrap"],
                "altitude_weakness_next_tasks": ["vertical_up_down_bootstrap", "angled_vertical_bootstrap"],
            }
        )
    return policy


def _compact_candidate_task_families(value: Any) -> list[Any]:
    """Return compact candidate family records."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    compact: list[Any] = []
    for item in value[:5]:
        if isinstance(item, Mapping):
            compact.append(
                {key: item.get(key) for key in ("task_family", "targeted_skill", "difficulty_hint", "priority") if item.get(key) is not None}
            )
        else:
            compact.append(str(item))
    return compact


def _feedback_from_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return feedback mapping nested inside a history item or the item itself."""
    feedback = item.get("feedback_summary")
    if isinstance(feedback, Mapping):
        return dict(feedback)
    return {
        "llm_instruction_summary": item.get("curriculum_feedback_summary"),
        "primary_skill_gaps": item.get("curriculum_primary_skill_gaps"),
        "diagnostic_signals": item.get("curriculum_diagnostic_signals"),
        "recommended_next_task_families": item.get("curriculum_recommended_next_task_families"),
        "avoid_next_task_families": item.get("curriculum_avoid_next_task_families"),
    }


def _sampled_task_family(item: Mapping[str, Any]) -> Any:
    """Return sampled family from compact metadata if present."""
    metadata = item.get("resolved_task_sample_metadata")
    if isinstance(metadata, Mapping):
        return metadata.get("task_distribution_sampled_family")
    return item.get("sampled_task_family") or item.get("accepted_task_family")


def _distribution_role(distribution_id: Any) -> str | None:
    """Return compact role for known distribution ids."""
    if distribution_id is None:
        return None
    text = str(distribution_id)
    if text == "tracking_medium":
        return "final_broad"
    if text == "tracking_small":
        return "fallback"
    return "focused"


def _top_diagnostic_signals(value: Any, *, limit: int) -> list[str]:
    """Return top diagnostic signal names without evidence payloads."""
    if isinstance(value, Mapping):
        ranked = sorted(value, key=str)
        return [str(key) for key in ranked[:limit]]
    return _limit_strings(value, limit)


def _compact_previous_response(response: str, mode: str) -> str:
    """Return a bounded previous-response summary for repair prompts."""
    if not response:
        return ""
    max_chars = 800 if mode == PROMPT_COMPACTION_FULL else 280
    text = response.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated]"


def _limit_strings(value: Any, limit: int) -> list[str]:
    """Return at most ``limit`` compact strings from a sequence-like value."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        items = list(value)[:limit]
        return [str(item) for item in items]
    if isinstance(value, (str, bytes)):
        return [str(value)] if value else []
    if not isinstance(value, Sequence):
        return [str(value)]
    return [str(item) for item in list(value)[:limit] if item is not None]


def _dict_of_scalars(value: Any) -> dict[str, Any]:
    """Return a JSON-ready mapping containing only scalar values."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): val for key, val in value.items() if isinstance(val, (str, int, float, bool)) or val is None}


def _optional_text(value: Any) -> str | None:
    """Return a non-empty string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sections_dropped(mode: str) -> list[str]:
    """Return high-level prompt sections dropped for a mode."""
    if mode == PROMPT_COMPACTION_FULL:
        return []
    dropped = [
        "full_accepted_task_json",
        "full_resolved_task_json",
        "full_task_distribution_sample_metadata",
        "raw_proposal_attempts",
        "full_feedback_json",
        "full_metrics_json",
    ]
    if mode == PROMPT_COMPACTION_MINIMAL:
        dropped.extend(["full_task_contract", "diagnostic_policy_details", "older_history_detail"])
    return dropped


def _sections_summarized(mode: str) -> list[str]:
    """Return high-level prompt sections summarized for a mode."""
    summarized = [
        "curriculum_history",
        "curriculum_feedback",
        "task_catalog",
        "latest_metrics_summary",
        "recent_rejected_tasks",
    ]
    if mode != PROMPT_COMPACTION_FULL:
        summarized.extend(["task_schema", "budget_context"])
    return summarized


def _previous_stage_task_shape(items: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the latest accepted task shape for duplicate-avoidance prompt context."""
    if not items:
        return None
    latest = dict(items[-1])
    for key in ("accepted_stage_task_shape", "resolved_task_shape", "task_shape"):
        value = latest.get(key)
        if value is not None and str(value).strip():
            return str(value)
    task = latest.get("resolved_task") or latest.get("task")
    if isinstance(task, Mapping):
        shape = task.get("shape")
        if shape is not None and str(shape).strip():
            return str(shape)
    return None


def _tail(items: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return the last ``limit`` mapping entries as copied dictionaries."""
    if limit <= 0:
        return []
    return [dict(item) for item in items[-limit:]]


def _copy_all(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return all mapping entries as copied dictionaries."""
    return [dict(item) for item in items]


def _compact_json(payload: Mapping[str, Any]) -> str:
    """Serialize prompt context compactly and deterministically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "DEFAULT_PROMPT_BUDGET_TOKENS",
    "DEFAULT_PROMPT_CONTEXT_LIMIT_TOKENS",
    "DEFAULT_PROMPT_RESPONSE_RESERVE_TOKENS",
    "JSON_ONLY_INSTRUCTION",
    "PROMPT_COMPACTION_COMPACT",
    "PROMPT_COMPACTION_FULL",
    "PROMPT_COMPACTION_MINIMAL",
    "PROMPT_COMPACTION_MODES",
    "SYSTEM_PROMPT",
    "build_task_proposal_messages",
    "build_task_repair_messages",
    "estimate_messages_tokens",
    "prompt_metadata",
]
