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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src import envs, validation

from . import llm_json as json_parser
from . import llm_progression as progression
from . import llm_prompts as prompts
from . import llm_task_schema as task_schema
from .llm_client import LLMClientError

CONTEXT_OVERFLOW_REASON_MAX_CHARS = 500

if TYPE_CHECKING:
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
    prompt_context_limit_tokens
        Local model context-size limit used for prompt budgeting.
    prompt_response_reserve_tokens
        Token reserve held back for the model response.
    prompt_budget_tokens
        Target prompt budget below the context limit and response reserve.

    """

    max_repair_attempts: int = 1
    skip_invalid_proposals: bool = False
    recent_context_limit: int = 3
    prompt_context_limit_tokens: int = prompts.DEFAULT_PROMPT_CONTEXT_LIMIT_TOKENS
    prompt_response_reserve_tokens: int = prompts.DEFAULT_PROMPT_RESPONSE_RESERVE_TOKENS
    prompt_budget_tokens: int = prompts.DEFAULT_PROMPT_BUDGET_TOKENS

    def __post_init__(self) -> None:
        """Validate proposal settings."""
        if self.max_repair_attempts < 0:
            message = "max_repair_attempts must be nonnegative"
            raise ValueError(message)
        if self.recent_context_limit < 0:
            message = "recent_context_limit must be nonnegative"
            raise ValueError(message)
        if self.prompt_context_limit_tokens <= 0:
            message = "prompt_context_limit_tokens must be positive"
            raise ValueError(message)
        if self.prompt_response_reserve_tokens <= 0:
            message = "prompt_response_reserve_tokens must be positive"
            raise ValueError(message)
        max_prompt_budget = self.prompt_context_limit_tokens - self.prompt_response_reserve_tokens
        if max_prompt_budget <= 0:
            message = "prompt response reserve must be smaller than context limit"
            raise ValueError(message)
        if self.prompt_budget_tokens <= 0:
            message = "prompt_budget_tokens must be positive"
            raise ValueError(message)
        if self.prompt_budget_tokens > max_prompt_budget:
            object.__setattr__(self, "prompt_budget_tokens", max_prompt_budget)


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
    curriculum_history
        Compact stage-by-stage accepted history for all completed stages.
    curriculum_summary
        Aggregate curriculum context such as family counts and metric trends.
    budget_context
        Optional bounded budget profile context for this stage.

    """

    curriculum_name: str
    stage_index: int
    recent_accepted_tasks: tuple[Mapping[str, Any], ...] = ()
    recent_rejected_tasks: tuple[Mapping[str, Any], ...] = ()
    metrics_summary: Mapping[str, Any] | None = None
    curriculum_history: tuple[Mapping[str, Any], ...] = ()
    curriculum_summary: Mapping[str, Any] | None = None
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
    previous_stage_task_shape: str | None
    requested_stage_task_shape: str | None
    accepted_stage_task_shape: str | None
    duplicate_task_rejected: bool
    duplicate_task_repair_reason: str | None


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
        response_text, prompt_metadata = _complete_with_prompt_budget(
            client=client,
            context=context,
            settings=active_settings,
            attempt_index=attempt_index,
            previous_response=previous_response,
            previous_errors=previous_errors,
            stats=stats,
        )
        stats["total_proposals"] += 1
        outcome = _evaluate_response(response_text, context=context)
        event = _proposal_event(
            context=context,
            attempt_index=attempt_index,
            is_repair_attempt=is_repair_attempt,
            response_text=response_text,
            outcome=outcome,
            prompt_metadata=prompt_metadata,
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
        if outcome.duplicate_task_rejected:
            stats["duplicate_task_rejections"] += 1
        rejection = {
            "stage_index": context.stage_index,
            "attempt_index": attempt_index,
            "is_repair_attempt": is_repair_attempt,
            "error_type": outcome.error_type,
            "rejection_reasons": list(outcome.rejection_reasons),
            "response_text": response_text,
            "parsed_task": outcome.parsed_task,
            "previous_stage_task_shape": outcome.previous_stage_task_shape,
            "requested_stage_task_shape": outcome.requested_stage_task_shape,
            "accepted_stage_task_shape": outcome.accepted_stage_task_shape,
            "duplicate_task_rejected": outcome.duplicate_task_rejected,
            "duplicate_task_repair_reason": outcome.duplicate_task_repair_reason,
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
        "duplicate_task_rejections": 0,
        "llm_prompt_estimated_tokens": None,
        "llm_prompt_context_limit": prompts.DEFAULT_PROMPT_CONTEXT_LIMIT_TOKENS,
        "llm_prompt_budget_tokens": prompts.DEFAULT_PROMPT_BUDGET_TOKENS,
        "llm_prompt_compaction_mode": prompts.PROMPT_COMPACTION_FULL,
        "llm_prompt_sections_dropped": [],
        "llm_prompt_sections_summarized": [],
        "llm_prompt_attempts": [],
        "llm_request_failed_due_to_context_size": False,
        "llm_context_retry_count": 0,
        "llm_context_retry_modes": [],
        "llm_context_fallback_used": False,
        "llm_context_fallback_reason": None,
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
    for key in (
        "total_proposals",
        "invalid_proposals",
        "repair_attempts",
        "repair_successes",
        "final_accepted_tasks",
        "fallback_proposals",
        "duplicate_task_rejections",
        "llm_context_retry_count",
    ):
        accumulator[key] = int(accumulator.get(key, 0)) + int(update.get(key, 0))
    rejected = accumulator.setdefault("rejected_proposals", [])
    if isinstance(rejected, list):
        update_rejected = update.get("rejected_proposals", [])
        if isinstance(update_rejected, list):
            rejected.extend(update_rejected)
    prompt_attempts = accumulator.setdefault("llm_prompt_attempts", [])
    if isinstance(prompt_attempts, list):
        update_attempts = update.get("llm_prompt_attempts", [])
        if isinstance(update_attempts, list):
            prompt_attempts.extend(update_attempts)
    retry_modes = accumulator.setdefault("llm_context_retry_modes", [])
    if isinstance(retry_modes, list):
        update_modes = update.get("llm_context_retry_modes", [])
        if isinstance(update_modes, list):
            retry_modes.extend(str(mode) for mode in update_modes)
    for key in (
        "llm_prompt_estimated_tokens",
        "llm_prompt_context_limit",
        "llm_prompt_budget_tokens",
        "llm_prompt_compaction_mode",
        "llm_prompt_sections_dropped",
        "llm_prompt_sections_summarized",
        "llm_context_fallback_reason",
    ):
        if key in update and update.get(key) is not None:
            accumulator[key] = update[key]
    for key in ("llm_request_failed_due_to_context_size", "llm_context_fallback_used"):
        accumulator[key] = bool(accumulator.get(key, False) or update.get(key, False))
    return accumulator


def _complete_with_prompt_budget(
    *,
    client: LLMClient,
    context: ProposalContext,
    settings: ProposalSettings,
    attempt_index: int,
    previous_response: str,
    previous_errors: tuple[str, ...],
    stats: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Complete one prompt request, retrying smaller prompt modes on context overflow."""
    initial_mode = prompts.PROMPT_COMPACTION_FULL
    while True:
        messages, metadata = _budgeted_messages_for_attempt(
            context=context,
            settings=settings,
            attempt_index=attempt_index,
            previous_response=previous_response,
            previous_errors=previous_errors,
            initial_mode=initial_mode,
        )
        _record_prompt_metadata(stats, metadata)
        try:
            return client.complete(messages), metadata
        except LLMClientError as exc:
            if not _is_context_size_error(exc):
                raise
            stats["llm_request_failed_due_to_context_size"] = True
            next_mode = _next_prompt_compaction_mode(str(metadata["llm_prompt_compaction_mode"]))
            if next_mode is None:
                stats["llm_context_fallback_used"] = True
                stats["llm_context_fallback_reason"] = _context_overflow_reason(exc)
                message = f"LLM request exceeded context size after prompt compaction: {_context_overflow_reason(exc)}"
                raise LLMCurriculumProposalError(message, stats=stats) from exc
            stats["llm_context_retry_count"] = int(stats.get("llm_context_retry_count", 0)) + 1
            retry_modes = stats.setdefault("llm_context_retry_modes", [])
            if isinstance(retry_modes, list):
                retry_modes.append(next_mode)
            initial_mode = next_mode


def _budgeted_messages_for_attempt(
    *,
    context: ProposalContext,
    settings: ProposalSettings,
    attempt_index: int,
    previous_response: str,
    previous_errors: tuple[str, ...],
    initial_mode: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build messages at the smallest needed compaction mode for the prompt budget."""
    mode = initial_mode
    while True:
        messages = _messages_for_attempt(
            context=context,
            settings=settings,
            attempt_index=attempt_index,
            previous_response=previous_response,
            previous_errors=previous_errors,
            compaction_mode=mode,
        )
        metadata = prompts.prompt_metadata(
            messages=messages,
            context_limit_tokens=settings.prompt_context_limit_tokens,
            prompt_budget_tokens=settings.prompt_budget_tokens,
            compaction_mode=mode,
        )
        if int(metadata["llm_prompt_estimated_tokens"]) <= settings.prompt_budget_tokens or mode == prompts.PROMPT_COMPACTION_MINIMAL:
            return messages, metadata
        next_mode = _next_prompt_compaction_mode(mode)
        if next_mode is None:
            return messages, metadata
        mode = next_mode


def _record_prompt_metadata(stats: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    """Record latest and per-attempt prompt metadata in proposal stats."""
    for key in (
        "llm_prompt_estimated_tokens",
        "llm_prompt_context_limit",
        "llm_prompt_budget_tokens",
        "llm_prompt_compaction_mode",
        "llm_prompt_sections_dropped",
        "llm_prompt_sections_summarized",
    ):
        stats[key] = metadata.get(key)
    attempts = stats.setdefault("llm_prompt_attempts", [])
    if isinstance(attempts, list):
        attempts.append(dict(metadata))


def _next_prompt_compaction_mode(mode: str) -> str | None:
    """Return the next smaller prompt mode after ``mode``."""
    if mode == prompts.PROMPT_COMPACTION_FULL:
        return prompts.PROMPT_COMPACTION_COMPACT
    if mode == prompts.PROMPT_COMPACTION_COMPACT:
        return prompts.PROMPT_COMPACTION_MINIMAL
    return None


def _is_context_size_error(exc: BaseException) -> bool:
    """Return whether an LLM client error reports prompt/context overflow."""
    text = str(exc).lower()
    markers = (
        "exceeds the available context size",
        "exceed_context_size_error",
        "n_prompt_tokens",
        "n_ctx",
        "context size",
        "context length",
    )
    return any(marker in text for marker in markers)


def _context_overflow_reason(exc: BaseException) -> str:
    """Return a bounded context-overflow failure reason."""
    text = str(exc).strip()
    if len(text) <= CONTEXT_OVERFLOW_REASON_MAX_CHARS:
        return text
    return f"{text[:CONTEXT_OVERFLOW_REASON_MAX_CHARS]}... [truncated]"


def _messages_for_attempt(
    *,
    context: ProposalContext,
    settings: ProposalSettings,
    attempt_index: int,
    previous_response: str,
    previous_errors: tuple[str, ...],
    compaction_mode: str,
) -> list[dict[str, str]]:
    """Build proposal or repair messages for one attempt."""
    if attempt_index == 0:
        return prompts.build_task_proposal_messages(
            curriculum_name=context.curriculum_name,
            stage_index=context.stage_index,
            recent_accepted_tasks=context.recent_accepted_tasks,
            recent_rejected_tasks=context.recent_rejected_tasks,
            metrics_summary=context.metrics_summary,
            curriculum_history=context.curriculum_history,
            curriculum_summary=context.curriculum_summary,
            recent_context_limit=settings.recent_context_limit,
            budget_context=context.budget_context,
            compaction_mode=compaction_mode,
        )
    return prompts.build_task_repair_messages(
        curriculum_name=context.curriculum_name,
        stage_index=context.stage_index,
        recent_accepted_tasks=context.recent_accepted_tasks,
        recent_rejected_tasks=context.recent_rejected_tasks,
        metrics_summary=context.metrics_summary,
        curriculum_history=context.curriculum_history,
        curriculum_summary=context.curriculum_summary,
        recent_context_limit=settings.recent_context_limit,
        budget_context=context.budget_context,
        previous_response=previous_response,
        compaction_mode=compaction_mode,
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

    previous_stage_task_shape = _previous_stage_task_shape(context)
    requested_stage_task_shape = _proposal_stage_task_shape(normalized_task)
    validation_result = task_schema.validate_proposed_task(normalized_task)
    if not validation_result.is_valid:
        return _rejected_outcome(
            error_type="validation",
            reasons=validation_result.messages,
            validation_status="invalid",
            parsed_task=parsed_task,
            normalized_task=normalized_task,
            previous_stage_task_shape=previous_stage_task_shape,
            requested_stage_task_shape=requested_stage_task_shape,
            accepted_stage_task_shape=requested_stage_task_shape,
        )

    duplicate_reason = _duplicate_task_rejection_reason(
        previous_stage=_previous_stage_progression_context(context),
        requested_stage=normalized_task,
        previous_stage_task_shape=previous_stage_task_shape,
        requested_stage_task_shape=requested_stage_task_shape,
        stage_index=context.stage_index,
    )
    if duplicate_reason is not None:
        return _rejected_outcome(
            error_type="duplicate_task",
            reasons=(duplicate_reason,),
            validation_status="duplicate",
            parsed_task=parsed_task,
            normalized_task=normalized_task,
            previous_stage_task_shape=previous_stage_task_shape,
            requested_stage_task_shape=requested_stage_task_shape,
            accepted_stage_task_shape=requested_stage_task_shape,
            duplicate_task_rejected=True,
            duplicate_task_repair_reason=duplicate_reason,
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
        previous_stage_task_shape=previous_stage_task_shape,
        requested_stage_task_shape=requested_stage_task_shape,
        accepted_stage_task_shape=requested_stage_task_shape,
        duplicate_task_rejected=False,
        duplicate_task_repair_reason=None,
    )


def _rejected_outcome(
    *,
    error_type: str,
    reasons: tuple[str, ...],
    validation_status: str,
    parsed_task: dict[str, Any] | None,
    normalized_task: dict[str, Any] | None,
    previous_stage_task_shape: str | None = None,
    requested_stage_task_shape: str | None = None,
    accepted_stage_task_shape: str | None = None,
    duplicate_task_rejected: bool = False,
    duplicate_task_repair_reason: str | None = None,
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
        previous_stage_task_shape=previous_stage_task_shape,
        requested_stage_task_shape=requested_stage_task_shape,
        accepted_stage_task_shape=accepted_stage_task_shape,
        duplicate_task_rejected=duplicate_task_rejected,
        duplicate_task_repair_reason=duplicate_task_repair_reason,
    )


def _previous_stage_task_shape(context: ProposalContext) -> str | None:
    """Return the latest accepted task shape from bounded context metadata."""
    previous_stage = _previous_stage_progression_context(context)
    if previous_stage is None:
        return None
    for key in ("accepted_stage_task_shape", "resolved_task_shape", "task_shape"):
        value = previous_stage.get(key)
        if value is not None and str(value).strip():
            return str(value)
    task = previous_stage.get("resolved_task") or previous_stage.get("task")
    if isinstance(task, Mapping):
        shape = task.get(validation.contracts.FIELD_SHAPE)
        if shape is not None and str(shape).strip():
            return str(shape)
    return None


def _previous_stage_progression_context(context: ProposalContext) -> Mapping[str, Any] | None:
    """Return the latest accepted stage context used for progression comparison."""
    if not context.recent_accepted_tasks:
        return None
    return dict(context.recent_accepted_tasks[-1])


def _proposal_stage_task_shape(task: Mapping[str, Any] | None) -> str | None:
    """Return a comparable shape for a concrete or single-family distribution proposal."""
    if task is None:
        return None
    proposal_type = _proposal_type(task)
    if proposal_type == task_schema.PROPOSAL_KIND_TASK:
        shape = task.get(validation.contracts.FIELD_SHAPE)
        return str(shape) if shape is not None and str(shape).strip() else None
    if proposal_type != task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION:
        return None
    try:
        settings = envs.task_distribution.load_task_distribution_settings(str(task[task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD]))
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if len(settings.family_weights) != 1:
        return None
    family = next(iter(settings.family_weights))
    return _task_shape_from_distribution_family(family)


def _task_shape_from_distribution_family(family: str) -> str | None:
    """Map a task-distribution family to its representative validation shape."""
    family_to_shape = {
        envs.task_distribution.FAMILY_HOVER: validation.contracts.SHAPE_HOVER_STABILIZATION,
        envs.task_distribution.FAMILY_TAKEOFF: validation.contracts.SHAPE_VERTICAL,
        envs.task_distribution.FAMILY_VERTICAL_UP_DOWN: validation.contracts.SHAPE_VERTICAL,
        envs.task_distribution.FAMILY_ANGLED_VERTICAL: validation.contracts.SHAPE_LINE,
        envs.task_distribution.FAMILY_LINE: validation.contracts.SHAPE_LINE,
        envs.task_distribution.FAMILY_START_HOLD_LINE: validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
        envs.task_distribution.FAMILY_POLYLINE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_L_SHAPE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_ZIGZAG: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_TRIANGLE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_MULTI_HEIGHT_POLYLINE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_DELAYED_ALTITUDE_POLYLINE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_RECTANGLE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_SQUARE: validation.contracts.SHAPE_POLYLINE,
        envs.task_distribution.FAMILY_CIRCLE: validation.contracts.SHAPE_CIRCLE,
        envs.task_distribution.FAMILY_ELLIPSE: validation.contracts.SHAPE_ELLIPSE,
        envs.task_distribution.FAMILY_FIGURE_EIGHT: validation.contracts.SHAPE_FIGURE_EIGHT,
    }
    return family_to_shape.get(family)


def _duplicate_task_rejection_reason(
    *,
    previous_stage: Mapping[str, Any] | str | None,
    requested_stage: Mapping[str, Any] | str | None,
    previous_stage_task_shape: str | None,
    requested_stage_task_shape: str | None,
    stage_index: int,
) -> str | None:
    """Return a rejection reason when a proposal repeats without valid progression."""
    if stage_index <= 1 or previous_stage is None or requested_stage is None:
        return None
    allowed, transition_reason = progression.is_valid_progression_transition(previous_stage, requested_stage)
    if allowed:
        return None
    previous_label = previous_stage_task_shape or "unknown"
    requested_label = requested_stage_task_shape or "unknown"
    if transition_reason == progression.TRANSITION_EXACT_DUPLICATE:
        return (
            f"proposed task family/shape {requested_label!r} exactly repeats immediately previous "
            f"stage {previous_label!r}; choose a different task family, shape, or difficulty progression"
        )
    return (
        f"proposed task family/shape {requested_label!r} is not a valid immediate progression after "
        f"previous stage {previous_label!r} ({transition_reason}); choose a different task family, shape, or difficulty progression"
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
    prompt_metadata: Mapping[str, Any],
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
        "previous_stage_task_shape": outcome.previous_stage_task_shape,
        "requested_stage_task_shape": outcome.requested_stage_task_shape,
        "accepted_stage_task_shape": outcome.accepted_stage_task_shape,
        "duplicate_task_rejected": outcome.duplicate_task_rejected,
        "duplicate_task_repair_reason": outcome.duplicate_task_repair_reason,
        "proposal_fallback_used": False,
        "task_reason": outcome.task_reason,
        "stage_budget_profile": outcome.stage_budget_profile,
        "budget_rationale": outcome.budget_rationale,
        **dict(prompt_metadata),
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
