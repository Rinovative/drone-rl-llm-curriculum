"""
===============================================================================
llm_curriculum.py
===============================================================================
Coordinate JSON-only LLM curriculum proposals, validation, repair, and logging.

Responsibilities:
  - Request a proposed trajectory task from a configured chat client
  - Parse exactly one JSON object and normalize through the LLM task schema
  - Validate every accepted task through deterministic validation modules
  - Retry bounded repair attempts and log every proposal event as JSONL-ready data

Design principles:
  - Keep the LLM as a curriculum proposer only
  - Reject code, markdown, unsupported keys, and invalid trajectories before training
  - Make repair accounting explicit and deterministic

Boundaries:
  - HTTP provider details belong in llm_client.py
  - PPO stage orchestration belongs in experiments curriculum modules
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import llm_json as json_parser
from . import llm_prompts as prompts
from . import llm_task_schema as task_schema

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .llm_client import LLMClient
    from .llm_logging import ProposalEventLogger


class LLMCurriculumProposalError(RuntimeError):
    """Raised when an LLM task proposal cannot be accepted after bounded repair."""

    def __init__(
        self,
        message: str,
        *,
        stats: Mapping[str, Any] | None = None,
        rejected_proposals: tuple[dict[str, Any], ...] = (),
    ) -> None:
        """Initialize the proposal error with proposal accounting metadata."""
        super().__init__(message)
        self.stats = dict(stats or {})
        self.rejected_proposals = tuple(rejected_proposals)


@dataclass(frozen=True)
class ProposalSettings:
    """
    Settings controlling LLM proposal repair behavior.

    Parameters
    ----------
    max_repair_attempts
        Maximum number of repair completions after the initial proposal.
    skip_invalid_proposals
        Whether to return without a task after exhausting repair attempts.
    recent_context_limit
        Maximum accepted/rejected context items included in prompts.

    """

    max_repair_attempts: int = 1
    skip_invalid_proposals: bool = False
    recent_context_limit: int = 3

    def __post_init__(self) -> None:
        """Validate proposal settings."""
        if self.max_repair_attempts < 0:
            message = "max_repair_attempts must be nonnegative"
            raise ValueError(message)
        if self.recent_context_limit < 0:
            message = "recent_context_limit must be nonnegative"
            raise ValueError(message)


@dataclass(frozen=True)
class ProposalContext:
    """
    Bounded context used for one curriculum task proposal.

    Parameters
    ----------
    curriculum_name
        Name of the curriculum being proposed.
    stage_index
        One-based stage index for the proposed task.
    recent_accepted_tasks
        Recent accepted task summaries, not full training history.
    recent_rejected_tasks
        Recent rejected proposal summaries with reasons.
    metrics_summary
        Compact latest-stage metrics or dry-run placeholder.
    budget_context
        Optional bounded budget profile context for this stage.

    """

    curriculum_name: str
    stage_index: int
    recent_accepted_tasks: tuple[Mapping[str, Any], ...] = ()
    recent_rejected_tasks: tuple[Mapping[str, Any], ...] = ()
    metrics_summary: Mapping[str, Any] | None = None
    budget_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate prompt context metadata."""
        if not self.curriculum_name.strip():
            message = "curriculum_name must be non-empty"
            raise ValueError(message)
        if self.stage_index < 1:
            message = "stage_index must be positive"
            raise ValueError(message)


@dataclass(frozen=True)
class CurriculumProposalResult:
    """
    Result from one proposal-generation request.

    Parameters
    ----------
    task
        Accepted training task without LLM-only metadata, or ``None`` when skipped.
    task_reason
        Optional reason metadata supplied by the LLM.
    stage_budget_profile
        Optional bounded budget profile selected by the LLM.
    budget_rationale
        Optional budget-profile rationale supplied by the LLM.
    proposal_type
        Accepted proposal type, either a concrete task or task distribution.
    original_proposal
        Raw parsed proposal object returned by the LLM for the accepted attempt.
    normalized_proposal
        Schema-normalized accepted proposal before metadata stripping.
    stats
        Proposal accounting for this request.
    rejected_proposals
        Rejected attempts with concrete reasons.

    """

    task: dict[str, Any] | None
    task_reason: str | None
    stage_budget_profile: str | None
    budget_rationale: str | None
    proposal_type: str | None
    original_proposal: dict[str, Any] | None
    normalized_proposal: dict[str, Any] | None
    stats: dict[str, Any]
    rejected_proposals: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _AttemptOutcome:
    """Internal representation of one parsed and validated proposal attempt."""

    accepted: bool
    error_type: str | None
    rejection_reasons: tuple[str, ...]
    validation_status: str
    parsed_task: dict[str, Any] | None
    normalized_task: dict[str, Any] | None
    task: dict[str, Any] | None
    task_reason: str | None
    stage_budget_profile: str | None
    budget_rationale: str | None


def propose_next_task(
    *,
    client: LLMClient,
    context: ProposalContext,
    settings: ProposalSettings | None = None,
    logger: ProposalEventLogger | None = None,
) -> CurriculumProposalResult:
    """
    Generate, repair if needed, validate, and log one LLM-proposed task.

    Parameters
    ----------
    client
        Configured LLM client.
    context
        Bounded context for this proposal request.
    settings
        Optional proposal repair settings.
    logger
        Optional JSONL event logger.

    Returns
    -------
    CurriculumProposalResult
        Accepted task and proposal accounting. ``task`` is ``None`` only when
        invalid proposals are explicitly configured to be skipped.

    Raises
    ------
    LLMCurriculumProposalError
        If all proposal and repair attempts fail and skipping is disabled.

    """
    active_settings = settings or ProposalSettings()
    stats = empty_proposal_stats()
    rejected_proposals: list[dict[str, Any]] = []
    previous_response = ""
    previous_errors: tuple[str, ...] = ()

    for attempt_index in range(active_settings.max_repair_attempts + 1):
        is_repair_attempt = attempt_index > 0
        if is_repair_attempt:
            stats["repair_attempts"] += 1
        messages = _messages_for_attempt(
            context=context,
            settings=active_settings,
            attempt_index=attempt_index,
            previous_response=previous_response,
            previous_errors=previous_errors,
        )
        response_text = client.complete(messages)
        stats["total_proposals"] += 1
        outcome = _evaluate_response(response_text, context=context)
        event = _proposal_event(
            context=context,
            attempt_index=attempt_index,
            is_repair_attempt=is_repair_attempt,
            response_text=response_text,
            outcome=outcome,
        )
        if logger is not None:
            logger.append(event)

        if outcome.accepted:
            stats["final_accepted_tasks"] += 1
            if is_repair_attempt:
                stats["repair_successes"] += 1
            return CurriculumProposalResult(
                task=outcome.task,
                task_reason=outcome.task_reason,
                stage_budget_profile=outcome.stage_budget_profile,
                budget_rationale=outcome.budget_rationale,
                proposal_type=_proposal_type(outcome.normalized_task),
                original_proposal=outcome.parsed_task,
                normalized_proposal=outcome.normalized_task,
                stats=stats,
                rejected_proposals=tuple(rejected_proposals),
            )

        stats["invalid_proposals"] += 1
        rejection = {
            "stage_index": context.stage_index,
            "attempt_index": attempt_index,
            "is_repair_attempt": is_repair_attempt,
            "error_type": outcome.error_type,
            "rejection_reasons": list(outcome.rejection_reasons),
            "response_text": response_text,
            "parsed_task": outcome.parsed_task,
        }
        rejected_proposals.append(rejection)
        stats["rejected_proposals"].append(rejection)
        previous_response = response_text
        previous_errors = outcome.rejection_reasons

    if active_settings.skip_invalid_proposals:
        return CurriculumProposalResult(
            task=None,
            task_reason=None,
            stage_budget_profile=None,
            budget_rationale=None,
            proposal_type=None,
            original_proposal=None,
            normalized_proposal=None,
            stats=stats,
            rejected_proposals=tuple(rejected_proposals),
        )
    reason_text = "; ".join(previous_errors) if previous_errors else "unknown proposal failure"
    message = f"LLM proposal failed after {active_settings.max_repair_attempts + 1} attempt(s): {reason_text}"
    raise LLMCurriculumProposalError(message, stats=stats, rejected_proposals=tuple(rejected_proposals))


def empty_proposal_stats() -> dict[str, Any]:
    """
    Return an empty proposal-statistics accumulator.

    Returns
    -------
    dict[str, Any]
        JSON-ready proposal counters and rejected proposal entries.

    """
    return {
        "total_proposals": 0,
        "invalid_proposals": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "final_accepted_tasks": 0,
        "fallback_proposals": 0,
        "rejected_proposals": [],
    }


def merge_proposal_stats(accumulator: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge one proposal-statistics mapping into an accumulator.

    Parameters
    ----------
    accumulator
        Mutable accumulator returned by ``empty_proposal_stats``.
    update
        Per-proposal stats returned by ``propose_next_task``.

    Returns
    -------
    dict[str, Any]
        The same accumulator after adding counts and rejected proposals.

    """
    for key in ("total_proposals", "invalid_proposals", "repair_attempts", "repair_successes", "final_accepted_tasks", "fallback_proposals"):
        accumulator[key] = int(accumulator.get(key, 0)) + int(update.get(key, 0))
    rejected = accumulator.setdefault("rejected_proposals", [])
    if isinstance(rejected, list):
        update_rejected = update.get("rejected_proposals", [])
        if isinstance(update_rejected, list):
            rejected.extend(update_rejected)
    return accumulator


def _messages_for_attempt(
    *,
    context: ProposalContext,
    settings: ProposalSettings,
    attempt_index: int,
    previous_response: str,
    previous_errors: tuple[str, ...],
) -> list[dict[str, str]]:
    """Build proposal or repair messages for one attempt."""
    if attempt_index == 0:
        return prompts.build_task_proposal_messages(
            curriculum_name=context.curriculum_name,
            stage_index=context.stage_index,
            recent_accepted_tasks=context.recent_accepted_tasks,
            recent_rejected_tasks=context.recent_rejected_tasks,
            metrics_summary=context.metrics_summary,
            recent_context_limit=settings.recent_context_limit,
            budget_context=context.budget_context,
        )
    return prompts.build_task_repair_messages(
        curriculum_name=context.curriculum_name,
        stage_index=context.stage_index,
        recent_accepted_tasks=context.recent_accepted_tasks,
        recent_rejected_tasks=context.recent_rejected_tasks,
        metrics_summary=context.metrics_summary,
        recent_context_limit=settings.recent_context_limit,
        budget_context=context.budget_context,
        previous_response=previous_response,
        error_messages=previous_errors,
    )


def _evaluate_response(response_text: str, *, context: ProposalContext) -> _AttemptOutcome:
    """Parse, normalize, and validate one raw LLM response."""
    try:
        parsed_task = json_parser.parse_json_object(response_text)
    except json_parser.LLMJsonError as exc:
        return _rejected_outcome(error_type="parse", reasons=(str(exc),), validation_status="not_run", parsed_task=None, normalized_task=None)

    try:
        normalized_task = task_schema.normalize_proposed_task(parsed_task)
    except ValueError as exc:
        return _rejected_outcome(
            error_type="schema",
            reasons=(str(exc),),
            validation_status="not_run",
            parsed_task=parsed_task,
            normalized_task=None,
        )

    reason_value = normalized_task.get(task_schema.REASON_FIELD)
    if reason_value is not None and not isinstance(reason_value, str):
        return _rejected_outcome(
            error_type="schema",
            reasons=("reason metadata must be a string",),
            validation_status="not_run",
            parsed_task=parsed_task,
            normalized_task=normalized_task,
        )

    profile_error = _validate_context_stage_budget_profile(normalized_task, context.budget_context)
    if profile_error is not None:
        return _rejected_outcome(
            error_type="schema",
            reasons=(profile_error,),
            validation_status="not_run",
            parsed_task=parsed_task,
            normalized_task=normalized_task,
        )

    stage_budget_profile_value = normalized_task.get(task_schema.STAGE_BUDGET_PROFILE_FIELD)
    budget_rationale_value = normalized_task.get(task_schema.BUDGET_RATIONALE_FIELD)
    stage_budget_profile = str(stage_budget_profile_value) if stage_budget_profile_value is not None else None
    budget_rationale = str(budget_rationale_value) if budget_rationale_value is not None else None

    validation_result = task_schema.validate_proposed_task(normalized_task)
    if not validation_result.is_valid:
        return _rejected_outcome(
            error_type="validation",
            reasons=validation_result.messages,
            validation_status="invalid",
            parsed_task=parsed_task,
            normalized_task=normalized_task,
        )

    return _AttemptOutcome(
        accepted=True,
        error_type=None,
        rejection_reasons=(),
        validation_status="valid",
        parsed_task=parsed_task,
        normalized_task=normalized_task,
        task=task_schema.task_without_metadata(normalized_task),
        task_reason=reason_value,
        stage_budget_profile=stage_budget_profile,
        budget_rationale=budget_rationale,
    )


def _rejected_outcome(
    *,
    error_type: str,
    reasons: tuple[str, ...],
    validation_status: str,
    parsed_task: dict[str, Any] | None,
    normalized_task: dict[str, Any] | None,
) -> _AttemptOutcome:
    """Build a rejected attempt outcome."""
    return _AttemptOutcome(
        accepted=False,
        error_type=error_type,
        rejection_reasons=reasons,
        validation_status=validation_status,
        parsed_task=parsed_task,
        normalized_task=normalized_task,
        task=None,
        task_reason=None,
        stage_budget_profile=None,
        budget_rationale=None,
    )


def _proposal_type(task: Mapping[str, Any] | None) -> str | None:
    """Return the normalized proposal type for a task-like mapping."""
    if task is None:
        return None
    return str(task.get(task_schema.PROPOSAL_KIND_FIELD, task_schema.PROPOSAL_KIND_TASK))


def _validate_context_stage_budget_profile(task: Mapping[str, Any], budget_context: Mapping[str, Any] | None) -> str | None:
    """Return a profile validation error for this prompt context, if any."""
    profile = task.get(task_schema.STAGE_BUDGET_PROFILE_FIELD)
    if profile is None or not budget_context:
        return None
    allowed_profiles = _context_allowed_budget_profiles(budget_context)
    profile_name = str(profile)
    if allowed_profiles and profile_name not in allowed_profiles:
        available = ", ".join(allowed_profiles)
        return f"stage_budget_profile must be one of: {available}"
    return None


def _context_allowed_budget_profiles(budget_context: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stage-specific allowed budget profile names from prompt context."""
    raw_names = budget_context.get("allowed_profile_names")
    if isinstance(raw_names, (list, tuple)):
        return tuple(str(name) for name in raw_names)
    raw_profiles = budget_context.get("allowed_profiles")
    if isinstance(raw_profiles, dict):
        return tuple(str(name) for name in raw_profiles)
    return ()


def _resolved_concrete_task(task: Mapping[str, Any] | None, proposal_type: str | None) -> dict[str, Any] | None:
    """Return the already-resolved concrete task for direct task proposals."""
    if task is None or proposal_type != task_schema.PROPOSAL_KIND_TASK:
        return None
    return dict(task)


def _resolved_concrete_task_shape(task: Mapping[str, Any] | None) -> str | None:
    """Return the compact shape from a concrete task mapping."""
    if task is None:
        return None
    value = task.get("shape")
    return str(value) if value is not None else None


def _task_distribution_reference(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the constrained distribution reference from a task-like mapping."""
    if _proposal_type(task) != task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION or task is None:
        return None
    return {
        task_schema.TASK_DISTRIBUTION_ID_FIELD: task.get(task_schema.TASK_DISTRIBUTION_ID_FIELD),
        task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: task.get(task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD),
    }


def _proposal_event(
    *,
    context: ProposalContext,
    attempt_index: int,
    is_repair_attempt: bool,
    response_text: str,
    outcome: _AttemptOutcome,
) -> dict[str, Any]:
    """Return one JSON-ready proposal event."""
    proposal_type = _proposal_type(outcome.normalized_task)
    resolved_task = _resolved_concrete_task(outcome.task, proposal_type)
    return {
        "event_type": "llm_proposal_attempt",
        "curriculum_name": context.curriculum_name,
        "stage_index": context.stage_index,
        "attempt_index": attempt_index,
        "is_repair_attempt": is_repair_attempt,
        "response_text": response_text,
        "status": "accepted" if outcome.accepted else "rejected",
        "error_type": outcome.error_type,
        "validation_status": outcome.validation_status,
        "rejection_reasons": list(outcome.rejection_reasons),
        "proposal_failure_reason": "; ".join(outcome.rejection_reasons) if outcome.rejection_reasons else None,
        "original_proposal": outcome.parsed_task,
        "parsed_task": outcome.parsed_task,
        "normalized_task": outcome.normalized_task,
        "accepted_task": outcome.task,
        "proposal_type": proposal_type,
        "resolved_task": resolved_task,
        "resolved_task_shape": _resolved_concrete_task_shape(resolved_task),
        "task_distribution_reference": _task_distribution_reference(outcome.normalized_task),
        "proposal_fallback_used": False,
        "task_reason": outcome.task_reason,
        "stage_budget_profile": outcome.stage_budget_profile,
        "budget_rationale": outcome.budget_rationale,
    }


__all__ = [
    "CurriculumProposalResult",
    "LLMCurriculumProposalError",
    "ProposalContext",
    "ProposalSettings",
    "empty_proposal_stats",
    "merge_proposal_stats",
    "propose_next_task",
]
