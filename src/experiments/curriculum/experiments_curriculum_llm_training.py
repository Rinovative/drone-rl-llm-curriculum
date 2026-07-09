"""
===============================================================================
experiments_curriculum_llm_training.py
===============================================================================
Train PPO trajectory tracking through a local-LLM-guided curriculum.

Responsibilities:
  - Load and validate LLM curriculum training configurations
  - Use strict JSON LLM proposals with deterministic validation and repair
  - Materialize per-stage task configs for the existing PPO tracking helper
  - Write run-scoped manifests, stage summaries, and proposal JSONL logs

Design principles:
  - Reuse PPO training, W&B, diagnostics, and canonical artifact helpers
  - Keep the LLM as a task proposer, never a runtime controller
  - Make dry-run proposal checks deterministic and training-free

Boundaries:
  - Reward logic, action semantics, environment physics, and evaluation metrics stay elsewhere
  - External local-LLM infrastructure and server startup are not managed here
===============================================================================

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src import envs, llm, utils, validation
from src.experiments import experiments_config as config_loader
from src.experiments.curriculum import experiments_curriculum_training as manual_curriculum
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

DEFAULT_LLM_CURRICULUM_CONFIG_PATH = Path("configs/curricula/llm_curriculum_pid_dynprev_m-taskdist_medium.yaml")
LLM_CURRICULUM_KIND = "llm"
LLM_CURRICULUM_MODE = "llm_curriculum"
DEFAULT_RECENT_CONTEXT_LIMIT = 3
DEFAULT_STAGE_TOTAL_TIMESTEPS = ppo_tracking.DEFAULT_TOTAL_TIMESTEPS
DEFAULT_STAGE_EVAL_STEPS = ppo_tracking.DEFAULT_EVAL_STEPS
DEFAULT_LLM_STAGE_BUDGET_PROFILE = "normal"
DEFAULT_PROPOSAL_FALLBACK_DISTRIBUTION_ID = "tracking_medium"
DEFAULT_PROPOSAL_FALLBACK_PROFILE = "short"
DEFAULT_READY_PROPOSAL_FALLBACK_PROFILE = "normal"
GENERATED_TASK_DISTRIBUTION_STRENGTH = 0.35
GENERATED_TASK_START_HOLD_SEC = 1.0
STANDARD_REFERENCE_HEIGHT_POLICY = "standard_reference_1p0m"
STANDARD_REFERENCE_BASE_Z_M = 1.0
STANDARD_REFERENCE_HEIGHT_RANGE_M = (0.9, 1.1)
SUBSTANDARD_REFERENCE_MAX_START_M = 0.95
XYZ_VECTOR_LENGTH = 3
GENERATED_TASK_DISTRIBUTION_LABELS = {
    validation.contracts.SHAPE_HOVER: "hover",
    validation.contracts.SHAPE_HOVER_STABILIZATION: "hover",
    validation.contracts.SHAPE_NEARBY_TARGET_HOVER: "nearby_hover",
    validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE: "short_line",
    validation.contracts.SHAPE_SHORT_SLOW_LINE: "short_line",
    validation.contracts.SHAPE_LINE: "line",
    validation.contracts.SHAPE_VERTICAL: "vertical",
    validation.contracts.SHAPE_POLYLINE: "polyline",
    validation.contracts.SHAPE_CIRCLE: "circle",
    validation.contracts.SHAPE_ELLIPSE: "ellipse",
    validation.contracts.SHAPE_FIGURE_EIGHT: "fig8",
}
RESOLVED_DISTRIBUTION_MAX_ATTEMPTS = 16
MIN_ERRORS_FOR_TREND = 2
PROGRESSION_BUCKET_HOVER = "hover"
PROGRESSION_BUCKET_VERTICAL_ONLY = "vertical_only"
PROGRESSION_BUCKET_XY_LINE = "xy_line"
PROGRESSION_BUCKET_TURN_POLYLINE = "turn_polyline"
PROGRESSION_BUCKET_CURVE = "curve"
PROGRESSION_BUCKET_ALTITUDE_COMBINED = "altitude_combined"
PROGRESSION_BUCKET_BROAD_FALLBACK = "broad_fallback"
PROGRESSION_BUCKET_ORDER = (
    PROGRESSION_BUCKET_HOVER,
    PROGRESSION_BUCKET_VERTICAL_ONLY,
    PROGRESSION_BUCKET_XY_LINE,
    PROGRESSION_BUCKET_ALTITUDE_COMBINED,
    PROGRESSION_BUCKET_TURN_POLYLINE,
    PROGRESSION_BUCKET_CURVE,
    PROGRESSION_BUCKET_BROAD_FALLBACK,
)
PROGRESSION_DISTRIBUTION_BUCKETS = {
    "bootstrap_randomized_hover_target": PROGRESSION_BUCKET_HOVER,
    "hover_bootstrap": PROGRESSION_BUCKET_HOVER,
    "vertical_bootstrap": PROGRESSION_BUCKET_VERTICAL_ONLY,
    "vertical_up_down_bootstrap": PROGRESSION_BUCKET_VERTICAL_ONLY,
    "short_line_bootstrap": PROGRESSION_BUCKET_XY_LINE,
    "line_bootstrap": PROGRESSION_BUCKET_XY_LINE,
    "angled_vertical_bootstrap": PROGRESSION_BUCKET_ALTITUDE_COMBINED,
    "delayed_altitude_polyline_bootstrap": PROGRESSION_BUCKET_ALTITUDE_COMBINED,
    "multi_height_polyline_bootstrap": PROGRESSION_BUCKET_ALTITUDE_COMBINED,
    "polyline_bootstrap": PROGRESSION_BUCKET_TURN_POLYLINE,
    "l_shape_bootstrap": PROGRESSION_BUCKET_TURN_POLYLINE,
    "zigzag_bootstrap": PROGRESSION_BUCKET_TURN_POLYLINE,
    "triangle_bootstrap": PROGRESSION_BUCKET_TURN_POLYLINE,
    "rectangle_bootstrap": PROGRESSION_BUCKET_TURN_POLYLINE,
    "circle_bootstrap": PROGRESSION_BUCKET_CURVE,
    "ellipse_bootstrap": PROGRESSION_BUCKET_CURVE,
    "tracking_small": PROGRESSION_BUCKET_BROAD_FALLBACK,
    "tracking_medium": PROGRESSION_BUCKET_BROAD_FALLBACK,
    "tracking_broad": PROGRESSION_BUCKET_BROAD_FALLBACK,
}
PROGRESSION_DISTRIBUTION_EXPECTED_SHAPES = {
    "bootstrap_randomized_hover_target": validation.contracts.SHAPE_HOVER_STABILIZATION,
    "hover_bootstrap": validation.contracts.SHAPE_HOVER_STABILIZATION,
    "vertical_bootstrap": validation.contracts.SHAPE_VERTICAL,
    "vertical_up_down_bootstrap": validation.contracts.SHAPE_VERTICAL,
    "short_line_bootstrap": validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
    "line_bootstrap": validation.contracts.SHAPE_LINE,
    "angled_vertical_bootstrap": validation.contracts.SHAPE_LINE,
    "multi_height_polyline_bootstrap": validation.contracts.SHAPE_LINE,
    "delayed_altitude_polyline_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "polyline_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "l_shape_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "zigzag_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "triangle_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "rectangle_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "circle_bootstrap": validation.contracts.SHAPE_CIRCLE,
    "ellipse_bootstrap": validation.contracts.SHAPE_ELLIPSE,
}
PURE_HOVER_VERTICAL_BUCKETS = {PROGRESSION_BUCKET_HOVER, PROGRESSION_BUCKET_VERTICAL_ONLY}
PURE_HOVER_VERTICAL_BOOTSTRAP_STAGE_LIMIT = 2
HOVER_VERTICAL_LOOP_WINDOW_SIZE = 3


@dataclass(frozen=True)
class LLMStageBudgetSettings:
    """
    Bounded adaptive budget profiles for LLM-proposed curriculum stages.

    Parameters
    ----------
    enabled
        Whether stage budgets may be selected through LLM proposal metadata.
    total_budget_cap_timesteps
        Optional cumulative cap across all LLM curriculum stages.
    default_profile
        Profile used when a proposal omits ``stage_budget_profile``.
    profiles
        Mapping from profile name to fixed total timestep budget.
    min_stage_timesteps
        Lower bound for any resolved stage budget.
    max_stage_timesteps
        Upper bound for any resolved stage budget.

    """

    enabled: bool
    total_budget_cap_timesteps: int | None
    default_profile: str
    profiles: dict[str, int]
    min_stage_timesteps: int
    max_stage_timesteps: int

    def __post_init__(self) -> None:
        """Validate profile names and timestep bounds."""
        if not self.default_profile.strip():
            message = "llm_stage_budget.default_profile must be non-empty"
            raise ValueError(message)
        normalized_profiles: dict[str, int] = {}
        for name, value in self.profiles.items():
            if not isinstance(name, str) or not name.strip():
                message = "llm_stage_budget profile names must be non-empty strings"
                raise ValueError(message)
            if isinstance(value, bool):
                message = f"llm_stage_budget profile {name!r} total_timesteps must be a positive integer"
                raise TypeError(message)
            timesteps = int(value)
            if timesteps <= 0:
                message = f"llm_stage_budget profile {name!r} total_timesteps must be positive"
                raise ValueError(message)
            normalized_profiles[name] = timesteps
        if not normalized_profiles:
            message = "llm_stage_budget.profiles must not be empty"
            raise ValueError(message)
        if self.default_profile not in normalized_profiles:
            available = ", ".join(sorted(normalized_profiles))
            message = f"llm_stage_budget.default_profile must be one of: {available}"
            raise ValueError(message)
        min_stage = int(self.min_stage_timesteps)
        max_stage = int(self.max_stage_timesteps)
        if min_stage <= 0 or max_stage <= 0 or min_stage > max_stage:
            message = "llm_stage_budget min/max stage timesteps must be positive and ordered"
            raise ValueError(message)
        for name, timesteps in normalized_profiles.items():
            if timesteps < min_stage or timesteps > max_stage:
                message = f"llm_stage_budget profile {name!r} is outside configured min/max stage timesteps"
                raise ValueError(message)
        cap = None if self.total_budget_cap_timesteps is None else int(self.total_budget_cap_timesteps)
        if cap is not None and cap <= 0:
            message = "llm_stage_budget.total_budget_cap_timesteps must be positive when provided"
            raise ValueError(message)
        if self.enabled:
            missing = sorted(set(llm.task_schema.DEFAULT_STAGE_BUDGET_PROFILES) - set(normalized_profiles))
            if missing:
                message = f"enabled llm_stage_budget must define profiles: {', '.join(missing)}"
                raise ValueError(message)
            if cap is None:
                message = "enabled llm_stage_budget requires total_budget_cap_timesteps"
                raise ValueError(message)
        object.__setattr__(self, "profiles", normalized_profiles)
        object.__setattr__(self, "min_stage_timesteps", min_stage)
        object.__setattr__(self, "max_stage_timesteps", max_stage)
        object.__setattr__(self, "total_budget_cap_timesteps", cap)


@dataclass(frozen=True)
class LLMProposalFallbackSettings:
    """
    Opt-in safe fallback settings for overnight task-distribution curricula.

    Parameters
    ----------
    enabled
        Whether exhausted LLM proposal failures should fall back to a known distribution.
    task_distribution_id
        Known task-distribution id used by the fallback proposal.
    default_stage_budget_profile
        Conservative budget profile used when the latest readiness is not ready.
    ready_stage_budget_profile
        Budget profile used when the latest stage is explicitly ready or passed.

    """

    enabled: bool = False
    task_distribution_id: str = DEFAULT_PROPOSAL_FALLBACK_DISTRIBUTION_ID
    default_stage_budget_profile: str = DEFAULT_PROPOSAL_FALLBACK_PROFILE
    ready_stage_budget_profile: str = DEFAULT_READY_PROPOSAL_FALLBACK_PROFILE

    def __post_init__(self) -> None:
        """Validate fallback distribution and budget profile choices."""
        if self.task_distribution_id not in llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS:
            available = ", ".join(sorted(llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS))
            message = f"proposal_fallback.task_distribution_id must be one of: {available}"
            raise ValueError(message)
        for label, profile in (
            ("default_stage_budget_profile", self.default_stage_budget_profile),
            ("ready_stage_budget_profile", self.ready_stage_budget_profile),
        ):
            if profile not in llm.task_schema.DEFAULT_STAGE_BUDGET_PROFILES:
                available = ", ".join(llm.task_schema.DEFAULT_STAGE_BUDGET_PROFILES)
                message = f"proposal_fallback.{label} must be one of: {available}"
                raise ValueError(message)


@dataclass(frozen=True)
class LLMCurriculumStage:
    """
    One stage in an LLM-guided PPO tracking curriculum.

    Parameters
    ----------
    stage_name
        Stable human-readable stage identifier used in run names and artifact paths.
    task_shape
        Expected trajectory task shape for this stage.
    task
        Validated task mapping passed to PPO training.
    total_timesteps
        PPO timestep budget for this stage.
    eval_steps
        Deterministic evaluation steps after this stage trains.
    task_reason
        Optional LLM rationale metadata retained in summaries only.
    notes
        Optional operator notes copied into summaries.
    task_distribution_config_path
        Optional constrained distribution config selected by an LLM proposal.
    task_distribution_id
        Optional known distribution identifier selected by an LLM proposal.
    requested_stage_budget_profile
        Optional raw LLM-selected budget profile before deterministic resolution.
    selected_stage_budget_profile
        Resolved budget profile after fallback or clipping.
    budget_rationale
        Optional LLM rationale for the budget profile.
    previous_stage_task_shape
        Immediate previous accepted task shape used for duplicate prevention.
    requested_stage_task_shape
        Shape or single-family proxy requested by the accepted proposal.
    accepted_stage_task_shape
        Final concrete stage shape accepted after deterministic resolution.
    duplicate_task_rejected
        Whether this stage records a duplicate-repair path before acceptance.
    duplicate_task_repair_reason
        Human-readable duplicate repair reason, when one occurred.
    proposal_repaired
        Whether deterministic progression repair changed the accepted proposal before stage resolution.
    proposal_repair_reason
        Human-readable reason for deterministic progression repair.
    proposal_original_distribution_id
        Original proposed task-distribution identifier before repair, when available.
    proposal_final_distribution_id
        Final task-distribution identifier after repair, when available.
    proposal_progression_rule_applied
        Stable name of the anti-stagnation rule used for repair.
    hover_vertical_loop_detected
        Whether recent accepted buckets plus this proposal formed a hover/vertical loop.
    stage_progression_bucket
        Curriculum progression bucket used for coverage diagnostics.
    fallback_task_shape
        Concrete shape selected by fallback resolution, when fallback was used.
    proposal_type
        Accepted proposal type, either a concrete task or task distribution.
    original_proposal
        Original parsed LLM proposal retained for audit metadata.
    task_distribution_reference
        Constrained distribution reference selected by the LLM, when present.
    resolved_task
        Concrete task resolved from the proposal and validated before PPO starts.
    resolved_task_shape
        Shape of ``resolved_task`` for compact logs.
    resolved_task_sample_metadata
        Deterministic sampler metadata for resolved distribution references.
    proposal_fallback_used
        Whether this stage came from the safe fallback after proposal failure.
    proposal_failure_reason
        Original exhausted proposal failure that triggered fallback, when present.
    budget_was_clipped
        Whether the resolver clipped or fell back to satisfy bounds.
    budget_fallback_reason
        Human-readable explanation for any fallback or clipping.
    cumulative_llm_budget_timesteps
        Cumulative LLM curriculum budget through this stage.
    llm_budget_cap_timesteps
        Total budget cap used for this stage, when enabled.
    bootstrap_stage_source
        Source of a deterministic bootstrap stage, when this stage is config-owned.
    bootstrap_task_shape
        Expected bootstrap task shape copied into manifests for auditability.
    bootstrap_target_sampling_bounds
        Optional per-axis hover target sampling bounds used by the bootstrap task distribution.

    """

    stage_name: str
    task_shape: str
    task: dict[str, Any]
    total_timesteps: int
    eval_steps: int
    task_reason: str | None = None
    notes: str | None = None
    task_distribution_config_path: Path | None = None
    task_distribution_id: str | None = None
    requested_stage_budget_profile: str | None = None
    selected_stage_budget_profile: str | None = None
    budget_rationale: str | None = None
    previous_stage_task_shape: str | None = None
    requested_stage_task_shape: str | None = None
    accepted_stage_task_shape: str | None = None
    duplicate_task_rejected: bool = False
    duplicate_task_repair_reason: str | None = None
    proposal_repaired: bool = False
    proposal_repair_reason: str | None = None
    proposal_original_distribution_id: str | None = None
    proposal_final_distribution_id: str | None = None
    proposal_progression_rule_applied: str | None = None
    hover_vertical_loop_detected: bool = False
    stage_progression_bucket: str | None = None
    fallback_task_shape: str | None = None
    proposal_type: str = llm.task_schema.PROPOSAL_KIND_TASK
    original_proposal: dict[str, Any] | None = None
    task_distribution_reference: dict[str, Any] | None = None
    resolved_task: dict[str, Any] | None = None
    resolved_task_shape: str | None = None
    resolved_task_sample_metadata: dict[str, Any] | None = None
    proposal_fallback_used: bool = False
    proposal_failure_reason: str | None = None
    budget_was_clipped: bool = False
    budget_fallback_reason: str | None = None
    cumulative_llm_budget_timesteps: int = 0
    llm_budget_cap_timesteps: int | None = None
    bootstrap_stage_source: str | None = None
    bootstrap_task_shape: str | None = None
    bootstrap_target_sampling_bounds: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate stage metadata that does not require PPO training."""
        if not self.stage_name.strip():
            message = "stage_name must be non-empty"
            raise ValueError(message)
        if not self.task_shape.strip():
            message = "task_shape must be non-empty"
            raise ValueError(message)
        if self.total_timesteps <= 0:
            message = "stage total_timesteps must be positive"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "stage eval_steps must be positive"
            raise ValueError(message)
        if self.task.get(validation.contracts.FIELD_SHAPE) != self.task_shape:
            message = f"stage {self.stage_name!r} task shape must match task_shape {self.task_shape!r}"
            raise ValueError(message)
        for label, profile in (
            ("requested_stage_budget_profile", self.requested_stage_budget_profile),
            ("selected_stage_budget_profile", self.selected_stage_budget_profile),
        ):
            if profile is not None and profile not in llm.task_schema.DEFAULT_STAGE_BUDGET_PROFILES:
                available = ", ".join(llm.task_schema.DEFAULT_STAGE_BUDGET_PROFILES)
                message = f"{label} must be one of: {available}"
                raise ValueError(message)
        if self.proposal_type not in (llm.task_schema.PROPOSAL_KIND_TASK, llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION):
            message = "proposal_type must be 'task' or 'task_distribution'"
            raise ValueError(message)
        if self.resolved_task is not None and self.resolved_task.get(validation.contracts.FIELD_SHAPE) != self.resolved_task_shape:
            message = "resolved_task_shape must match resolved_task shape"
            raise ValueError(message)
        if self.proposal_fallback_used and not self.proposal_failure_reason:
            message = "fallback stages must include proposal_failure_reason"
            raise ValueError(message)
        if self.accepted_stage_task_shape is not None and self.accepted_stage_task_shape != self.task_shape:
            message = "accepted_stage_task_shape must match task_shape when provided"
            raise ValueError(message)
        if self.cumulative_llm_budget_timesteps < 0:
            message = "cumulative_llm_budget_timesteps must be nonnegative"
            raise ValueError(message)


@dataclass(frozen=True)
class LLMCurriculumSettings:
    """
    Settings for sequential LLM-guided PPO curriculum training.

    Parameters
    ----------
    curriculum_name
        Stable curriculum identifier used in run names and summary artifacts.
    base_training_config
        Existing PPO tracking config used as defaults for stage training.
    seed
        Default deterministic seed for all stages.
    wandb_mode
        W&B mode override for every stage.
    normalize_actions
        Whether every stage should use the normalized PPO action interface.
    max_stages
        Maximum number of curriculum stages including an enabled bootstrap stage.
    stage_total_timesteps
        Default timestep budget for LLM-proposed stages.
    stage_eval_steps
        Default evaluation budget for LLM-proposed stages.
    bootstrap_stage
        Optional first stage supplied by configuration before LLM proposals begin.
    llm_config
        Provider configuration used to construct the LLM client.
    proposal_settings
        Strict JSON repair and prompt-context settings.
    llm_stage_budget
        Optional bounded adaptive stage budget profile settings.
    proposal_fallback
        Optional safe fallback used only by explicitly enabled overnight configs.
    config_path
        Optional source config path included in summary metadata.

    """

    curriculum_name: str
    base_training_config: Path
    seed: int
    wandb_mode: str
    normalize_actions: bool
    max_stages: int
    stage_total_timesteps: int
    stage_eval_steps: int
    bootstrap_stage: LLMCurriculumStage | None
    llm_config: dict[str, Any]
    proposal_settings: llm.curriculum.ProposalSettings
    llm_stage_budget: LLMStageBudgetSettings
    proposal_fallback: LLMProposalFallbackSettings
    reference_medium_config_path: Path | None = None
    reference_medium_timesteps: int | None = None
    stage_budget_multipliers: dict[str, float] | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate curriculum-level metadata."""
        if not self.curriculum_name.strip():
            message = "curriculum_name must be non-empty"
            raise ValueError(message)
        utils.artifacts.get_run_dir(_curriculum_artifact_run_name(self.curriculum_name, self.seed))
        if self.wandb_mode not in utils.wandb.WANDB_MODES:
            message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
            raise ValueError(message)
        if self.max_stages <= 0:
            message = "max_stages must be positive"
            raise ValueError(message)
        if self.stage_total_timesteps <= 0:
            message = "stage_total_timesteps must be positive"
            raise ValueError(message)
        if self.stage_eval_steps <= 0:
            message = "stage_eval_steps must be positive"
            raise ValueError(message)
        if not isinstance(self.llm_config, dict):
            message = "llm_config must be a dictionary"
            raise TypeError(message)
        if not isinstance(self.llm_stage_budget, LLMStageBudgetSettings):
            message = "llm_stage_budget must be an LLMStageBudgetSettings instance"
            raise TypeError(message)
        if not isinstance(self.proposal_fallback, LLMProposalFallbackSettings):
            message = "proposal_fallback must be an LLMProposalFallbackSettings instance"
            raise TypeError(message)
        if self.reference_medium_timesteps is not None and self.reference_medium_timesteps <= 0:
            message = "reference_medium_timesteps must be positive when provided"
            raise ValueError(message)
        if self.stage_budget_multipliers is not None:
            for profile, multiplier in self.stage_budget_multipliers.items():
                if multiplier <= 0.0:
                    message = f"stage_budget_multipliers.{profile} must be positive"
                    raise ValueError(message)
        cap = self.llm_stage_budget.total_budget_cap_timesteps
        if self.llm_stage_budget.enabled and cap is not None and cap < self.max_stages * self.llm_stage_budget.min_stage_timesteps:
            message = "llm_stage_budget total cap is too small to reserve the minimum budget for every stage"
            raise ValueError(message)

    @property
    def llm_provider(self) -> str:
        """Return the configured LLM provider name."""
        return str(self.llm_config.get("provider") or llm.client.PROVIDER_MOCK)

    @property
    def llm_model(self) -> str | None:
        """Return the configured LLM model name when available."""
        model = self.llm_config.get("model")
        return str(model) if model is not None else None


@dataclass(frozen=True)
class LLMCurriculumResult:
    """
    Result returned after an LLM curriculum training or dry-run proposal check.

    Parameters
    ----------
    summary_path
        Path to the written curriculum summary JSON.
    manifest_path
        Path to the written curriculum manifest JSON.
    proposal_log_path
        Path to the run-scoped proposal JSONL log.
    summary
        JSON-serializable curriculum summary payload.

    """

    summary_path: str
    manifest_path: str
    proposal_log_path: str
    summary: dict[str, Any]


def load_llm_curriculum_settings(path: str | Path) -> LLMCurriculumSettings:
    """
    Load LLM curriculum training settings from YAML.

    Parameters
    ----------
    path
        Curriculum YAML path.

    Returns
    -------
    LLMCurriculumSettings
        Validated curriculum settings.

    """
    config_path = Path(path)
    config = config_loader.load_experiment_config(config_path)
    return llm_curriculum_settings_from_mapping(config, config_path=config_path)


def llm_curriculum_settings_from_mapping(
    config: Mapping[str, Any],
    config_path: Path | None = None,
) -> LLMCurriculumSettings:
    """
    Build LLM curriculum settings from a loaded mapping.

    Parameters
    ----------
    config
        Loaded curriculum configuration mapping.
    config_path
        Optional source path copied into metadata.

    Returns
    -------
    LLMCurriculumSettings
        Validated LLM curriculum settings.

    Raises
    ------
    TypeError
        If nested config sections have invalid types.
    ValueError
        If required curriculum fields are missing or malformed.

    """
    stage_defaults = _mapping_or_empty(config.get("stage_defaults"), "stage_defaults")
    stage_total_timesteps = int(stage_defaults.get("total_timesteps", DEFAULT_STAGE_TOTAL_TIMESTEPS))
    stage_eval_steps = int(stage_defaults.get("eval_steps", DEFAULT_STAGE_EVAL_STEPS))
    max_stages = int(config.get("max_stages", 1))
    llm_stage_budget = _llm_stage_budget_settings_from_config(config.get("llm_stage_budget"), stage_total_timesteps)
    proposal_fallback = _proposal_fallback_settings_from_config(config.get("proposal_fallback"))
    llm_config = _llm_config_from_mapping(config)
    proposal_settings = llm.curriculum.ProposalSettings(
        max_repair_attempts=int(llm_config.get("max_repair_attempts", 1)),
        skip_invalid_proposals=bool(llm_config.get("skip_invalid_proposals", False)),
        recent_context_limit=int(llm_config.get("recent_context_limit", DEFAULT_RECENT_CONTEXT_LIMIT)),
        prompt_context_limit_tokens=int(llm_config.get("prompt_context_limit_tokens", llm.prompts.DEFAULT_PROMPT_CONTEXT_LIMIT_TOKENS)),
        prompt_response_reserve_tokens=int(llm_config.get("prompt_response_reserve_tokens", llm.prompts.DEFAULT_PROMPT_RESPONSE_RESERVE_TOKENS)),
        prompt_budget_tokens=int(llm_config.get("prompt_budget_tokens", llm.prompts.DEFAULT_PROMPT_BUDGET_TOKENS)),
    )
    return LLMCurriculumSettings(
        curriculum_name=str(config.get("curriculum_name") or ""),
        base_training_config=Path(str(config.get("base_training_config") or ppo_tracking.DEFAULT_PPO_TRACKING_CONFIG_PATH)),
        seed=int(config.get("seed", ppo_tracking.DEFAULT_SEED)),
        wandb_mode=str(config.get("wandb_mode") or utils.wandb.WANDB_MODE_AUTO),
        normalize_actions=bool(config.get("normalize_actions", ppo_tracking.DEFAULT_NORMALIZE_ACTIONS)),
        max_stages=max_stages,
        stage_total_timesteps=stage_total_timesteps,
        stage_eval_steps=stage_eval_steps,
        bootstrap_stage=_bootstrap_stage_from_config(config.get("bootstrap"), stage_total_timesteps, stage_eval_steps),
        llm_config=llm_config,
        proposal_settings=proposal_settings,
        llm_stage_budget=llm_stage_budget,
        proposal_fallback=proposal_fallback,
        reference_medium_config_path=_optional_path(config.get("reference_medium_config_path")),
        reference_medium_timesteps=_optional_int(config.get("reference_medium_timesteps")),
        stage_budget_multipliers=_optional_float_dict(config.get("stage_budget_multipliers"), "stage_budget_multipliers"),
        config_path=config_path,
    )


def validate_llm_curriculum(settings: LLMCurriculumSettings) -> None:
    """
    Validate bootstrap task and provider configuration before training starts.

    Parameters
    ----------
    settings
        Loaded LLM curriculum settings.

    Raises
    ------
    ValueError
        If configured tasks or provider settings are invalid.

    """
    if settings.bootstrap_stage is not None:
        _validate_stage_task(settings.bootstrap_stage)
    llm.client.client_from_config(settings.llm_config)


def derive_stage_run_name(curriculum_name: str, stage_index: int, stage_name: str, seed: int) -> str:
    """
    Derive the stable run name for one LLM curriculum stage.

    Parameters
    ----------
    curriculum_name
        Curriculum identifier.
    stage_index
        One-based stage index.
    stage_name
        Stage identifier.
    seed
        Deterministic seed.

    Returns
    -------
    str
        Stage run name used in metadata and W&B tracking.

    """
    return f"{curriculum_name}_stage{stage_index:02d}_{stage_name}_seed{seed}"


def run_llm_curriculum_training(settings: LLMCurriculumSettings, dry_run_proposals: bool = False) -> LLMCurriculumResult:
    """
    Train PPO tracking sequentially through an LLM-guided curriculum.

    Parameters
    ----------
    settings
        Loaded and validated LLM curriculum settings.
    dry_run_proposals
        If true, exercise proposal parsing, repair, validation, and logging
        without launching PPO training.

    Returns
    -------
    LLMCurriculumResult
        Summary, manifest, proposal-log paths, and summary payload.

    """
    validate_llm_curriculum(settings)
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    utils.artifacts.get_run_dir(curriculum_run_name).mkdir(parents=True, exist_ok=True)
    utils.artifacts.get_run_config_dir(curriculum_run_name).mkdir(parents=True, exist_ok=True)
    llm_logs_dir = utils.artifacts.ensure_run_llm_logs_dir(curriculum_run_name)
    proposal_log_path = llm_logs_dir / utils.artifacts.LLM_PROPOSALS_FILENAME
    proposal_log_path.write_text("", encoding="utf-8")
    proposal_logger = llm.logging.ProposalEventLogger(proposal_log_path)
    client = llm.client.client_from_config(settings.llm_config)

    stage_entries: list[dict[str, Any]] = []
    recent_accepted_tasks: list[dict[str, Any]] = []
    recent_rejected_tasks: list[dict[str, Any]] = []
    proposal_stats = llm.curriculum.empty_proposal_stats()
    previous_model_path: str | None = None
    cumulative_llm_budget_timesteps = 0
    latest_metrics_summary: dict[str, Any] = {"status": "not_started", "dry_run_proposals": dry_run_proposals}

    if settings.bootstrap_stage is not None and len(stage_entries) < settings.max_stages:
        bootstrap_stage = _resolve_llm_stage_budget(
            settings=settings,
            stage=settings.bootstrap_stage,
            stage_index=1,
            cumulative_timesteps=cumulative_llm_budget_timesteps,
        )
        cumulative_llm_budget_timesteps = bootstrap_stage.cumulative_llm_budget_timesteps
        _log_stage_budget_decision(proposal_logger, settings, bootstrap_stage, stage_index=1)
        entry, previous_model_path, latest_metrics_summary = _run_or_dry_stage(
            settings=settings,
            stage=bootstrap_stage,
            stage_index=1,
            previous_model_path=previous_model_path,
            dry_run_proposals=dry_run_proposals,
        )
        stage_entries.append(entry)
        recent_accepted_tasks.append(_accepted_task_context(entry))

    while len(stage_entries) < settings.max_stages:
        next_stage_index = len(stage_entries) + 1
        previous_stage_task_shape = _previous_accepted_stage_shape(recent_accepted_tasks)
        context = llm.curriculum.ProposalContext(
            curriculum_name=settings.curriculum_name,
            stage_index=next_stage_index,
            recent_accepted_tasks=tuple(recent_accepted_tasks),
            recent_rejected_tasks=tuple(recent_rejected_tasks),
            metrics_summary=latest_metrics_summary,
            curriculum_history=_llm_context_history(stage_entries),
            curriculum_summary=_llm_context_summary(stage_entries, latest_metrics_summary),
            budget_context=_budget_prompt_context(settings, next_stage_index, cumulative_llm_budget_timesteps),
        )
        try:
            proposal = llm.curriculum.propose_next_task(
                client=client,
                context=context,
                settings=settings.proposal_settings,
                logger=proposal_logger,
            )
        except llm.curriculum.LLMCurriculumProposalError as exc:
            context_overflow = bool(exc.stats.get("llm_request_failed_due_to_context_size"))
            if not settings.proposal_fallback.enabled and not context_overflow:
                raise
            llm.curriculum.merge_proposal_stats(proposal_stats, exc.stats)
            recent_rejected_tasks.extend(exc.rejected_proposals)
            proposal_stats["fallback_proposals"] = int(proposal_stats.get("fallback_proposals", 0)) + 1
            fallback_resolution = _stage_from_fallback_failure(
                settings=settings,
                stage_index=next_stage_index,
                metrics_summary=latest_metrics_summary,
                failure_reason=str(exc),
                previous_stage_task_shape=previous_stage_task_shape,
                context_overflow=context_overflow,
                stage_entries=stage_entries,
                latest_metrics_summary=latest_metrics_summary,
            )
            fallback_proposal = fallback_resolution["fallback_proposal"]
            fallback_repair = fallback_resolution["fallback_repair"]
            if fallback_repair["proposal_repaired"]:
                _log_proposal_progression_repair(
                    logger=proposal_logger,
                    settings=settings,
                    stage_index=next_stage_index,
                    repair=fallback_repair,
                    proposal_fallback_used=True,
                )
            stage = fallback_resolution["stage"]
            _log_proposal_fallback(
                logger=proposal_logger,
                settings=settings,
                stage_index=next_stage_index,
                fallback_proposal=fallback_proposal,
                stage=stage,
            )
        else:
            llm.curriculum.merge_proposal_stats(proposal_stats, proposal.stats)
            recent_rejected_tasks.extend(proposal.rejected_proposals)
            if proposal.task is None:
                break
            proposal_repair = _repair_proposal_for_progression(
                settings=settings,
                stage_index=next_stage_index,
                task=proposal.task,
                proposal_type=proposal.proposal_type,
                stage_entries=stage_entries,
                latest_metrics_summary=latest_metrics_summary,
            )
            if proposal_repair["proposal_repaired"]:
                _log_proposal_progression_repair(
                    logger=proposal_logger,
                    settings=settings,
                    stage_index=next_stage_index,
                    repair=proposal_repair,
                    proposal_fallback_used=False,
                )
            stage = _stage_from_proposal(
                settings=settings,
                stage_index=next_stage_index,
                task=proposal_repair["task"],
                task_reason=proposal.task_reason,
                stage_budget_profile=proposal.stage_budget_profile,
                budget_rationale=proposal.budget_rationale,
                proposal_type=proposal_repair["proposal_type"],
                original_proposal=proposal.original_proposal,
                previous_stage_task_shape=previous_stage_task_shape,
                proposal_fallback_used=False,
                proposal_failure_reason=None,
                progression_repair=proposal_repair,
            )
        stage = _resolve_llm_stage_budget(
            settings=settings,
            stage=stage,
            stage_index=next_stage_index,
            cumulative_timesteps=cumulative_llm_budget_timesteps,
        )
        cumulative_llm_budget_timesteps = stage.cumulative_llm_budget_timesteps
        _log_stage_budget_decision(proposal_logger, settings, stage, stage_index=next_stage_index)
        _validate_stage_task(stage)
        entry, previous_model_path, latest_metrics_summary = _run_or_dry_stage(
            settings=settings,
            stage=stage,
            stage_index=next_stage_index,
            previous_model_path=previous_model_path,
            dry_run_proposals=dry_run_proposals,
        )
        stage_entries.append(entry)
        recent_accepted_tasks.append(_accepted_task_context(entry))

    summary = utils.serialization.to_jsonable(
        _build_curriculum_summary(
            settings=settings,
            stage_entries=stage_entries,
            proposal_log_path=proposal_log_path,
            proposal_stats=proposal_stats,
            dry_run_proposals=dry_run_proposals,
        )
    )
    if not isinstance(summary, dict):
        message = "LLM curriculum summary must serialize to a JSON object"
        raise TypeError(message)
    summary_path, manifest_path = _write_curriculum_artifacts(settings=settings, summary=summary)
    return LLMCurriculumResult(
        summary_path=str(summary_path),
        manifest_path=str(manifest_path),
        proposal_log_path=str(proposal_log_path),
        summary=summary,
    )


def run_llm_curriculum_training_from_config(
    config_path: str | Path = DEFAULT_LLM_CURRICULUM_CONFIG_PATH,
    seed: int | None = None,
    wandb_mode: str | None = None,
    provider: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    max_stages: int | None = None,
    max_repair_attempts: int | None = None,
    dry_run_proposals: bool = False,
) -> LLMCurriculumResult:
    """
    Load LLM curriculum settings with CLI-style overrides and run the workflow.

    Parameters
    ----------
    config_path
        Curriculum YAML path.
    seed
        Optional deterministic seed override.
    wandb_mode
        Optional W&B mode override.
    provider
        Optional LLM provider override.
    api_base
        Optional OpenAI-compatible API base override.
    model
        Optional LLM model override.
    max_stages
        Optional maximum stage-count override.
    max_repair_attempts
        Optional repair-attempt override.
    dry_run_proposals
        If true, do not launch PPO training.

    Returns
    -------
    LLMCurriculumResult
        Summary, manifest, and proposal-log metadata for the run.

    """
    settings = load_llm_curriculum_settings(config_path)
    overridden = _settings_with_overrides(
        settings=settings,
        seed=seed,
        wandb_mode=wandb_mode,
        provider=provider,
        api_base=api_base,
        model=model,
        max_stages=max_stages,
        max_repair_attempts=max_repair_attempts,
    )
    return run_llm_curriculum_training(overridden, dry_run_proposals=dry_run_proposals)


def _run_or_dry_stage(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    previous_model_path: str | None,
    dry_run_proposals: bool,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Run a PPO stage or produce a dry-run stage summary."""
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    stage_dirs = utils.artifacts.ensure_curriculum_stage_training_dirs(curriculum_run_name, stage_index, stage.stage_name)
    stage_training_dir = stage_dirs[utils.artifacts.TRAINING_DIRNAME]
    task_config_path = _write_stage_task_config(
        settings=settings,
        stage=stage,
        stage_index=stage_index,
        stage_training_dir=stage_training_dir,
    )
    run_name = derive_stage_run_name(settings.curriculum_name, stage_index, stage.stage_name, settings.seed)
    if dry_run_proposals:
        entry = _dry_stage_summary_entry(
            settings=settings,
            stage=stage,
            stage_index=stage_index,
            run_name=run_name,
            training_dir=stage_training_dir,
            previous_model_path=previous_model_path,
            task_config_path=task_config_path,
        )
        return entry, previous_model_path, _metrics_summary_from_entry(entry)

    result = ppo_tracking.run_ppo_tracking_smoke_from_config(
        config_path=settings.base_training_config,
        task_config_path=task_config_path,
        task_index=0,
        task_shape=stage.task_shape,
        run_name=run_name,
        total_timesteps=stage.total_timesteps,
        artifact_root=stage_training_dir,
        eval_steps=stage.eval_steps,
        seed=settings.seed,
        wandb_mode=settings.wandb_mode,
        normalize_actions=settings.normalize_actions,
        task_distribution_config_path=stage.task_distribution_config_path,
        wandb_group=_curriculum_wandb_group(curriculum_run_name),
        wandb_tags=_stage_wandb_tags(settings, stage_index, stage),
        initial_model_path=previous_model_path,
        run_metadata=_stage_training_run_metadata(
            settings=settings,
            stage=stage,
            stage_index=stage_index,
            run_name=run_name,
            curriculum_run_name=curriculum_run_name,
            previous_model_path=previous_model_path,
        ),
    )
    entry = _stage_summary_entry(
        settings=settings,
        stage=stage,
        stage_index=stage_index,
        run_name=run_name,
        result=result,
        training_dir=stage_training_dir,
        previous_model_path=previous_model_path,
        initial_model_path=previous_model_path,
        task_config_path=task_config_path,
    )
    return entry, _preferred_result_model_path(result), _metrics_summary_from_entry(entry)


def _stage_summary_entry(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    run_name: str,
    result: ppo_tracking.PPOTrackingSmokeResult,
    training_dir: Path,
    previous_model_path: str | None,
    initial_model_path: str | None,
    task_config_path: Path,
) -> dict[str, Any]:
    """Build one compact LLM curriculum stage summary from PPO metrics."""
    metrics = result.metrics
    run_root = training_dir.parent.parent.parent
    entry: dict[str, Any] = {
        "stage_index": stage_index,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "task": stage.task,
        "task_reason": stage.task_reason,
        "notes": stage.notes,
        "task_distribution_config_path": None if stage.task_distribution_config_path is None else str(stage.task_distribution_config_path),
        "task_distribution_id": stage.task_distribution_id,
        "curriculum_run_name": _curriculum_artifact_run_name(settings.curriculum_name, settings.seed),
        "run_name": run_name,
        "curriculum_stage_run_name": run_name,
        "stage_dir": str(training_dir.parent),
        "stage_dir_relative": utils.artifacts.path_relative_to(training_dir.parent, run_root),
        "training_dir": str(training_dir),
        "training_dir_relative": utils.artifacts.path_relative_to(training_dir, run_root),
        "task_config_path": str(task_config_path),
        "task_config_path_relative": utils.artifacts.path_relative_to(task_config_path, run_root),
        "model_path": result.model_path,
        "model_path_relative": utils.artifacts.path_relative_to(result.model_path, run_root),
        "last_model_path": _result_last_model_path(result),
        "last_model_path_relative": utils.artifacts.path_relative_to(_result_last_model_path(result), run_root),
        "best_model_path": result.best_model_path,
        "best_model_path_relative": utils.artifacts.path_relative_to(result.best_model_path, run_root),
        "best_model_metric": result.best_model_metric,
        "best_model_step": result.best_model_step,
        "best_model_source": result.best_model_source,
        "selected_transfer_model_path": _preferred_result_model_path(result),
        "selected_transfer_model_source": _preferred_result_model_source(result),
        "metrics_path": result.metrics_path,
        "metrics_path_relative": utils.artifacts.path_relative_to(result.metrics_path, run_root),
        "manifest_path": result.manifest_path,
        "manifest_path_relative": utils.artifacts.path_relative_to(result.manifest_path, run_root),
        "diagnostics_dir": metrics.get("diagnostics_dir"),
        "diagnostics_dir_relative": utils.artifacts.path_relative_to(metrics.get("diagnostics_dir"), run_root),
        "total_timesteps": stage.total_timesteps,
        **_stage_run_metadata(stage),
        "eval_steps": stage.eval_steps,
        "seed": metrics.get("seed"),
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": initial_model_path,
        "initial_model_path_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
        "previous_model_path": previous_model_path,
        "previous_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "model_transfer_enabled": initial_model_path is not None,
        "model_transfer_source": initial_model_path,
        "model_transfer_source_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
        "dry_run_proposals": False,
    }
    for key in manual_curriculum.SUMMARY_METRIC_KEYS:
        entry[key] = metrics.get(key)
    entry["task_distribution_config_path"] = None if stage.task_distribution_config_path is None else str(stage.task_distribution_config_path)
    entry["task_distribution_id"] = stage.task_distribution_id
    entry["task_distribution_reference"] = stage.task_distribution_reference
    return entry


def _dry_stage_summary_entry(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    run_name: str,
    training_dir: Path,
    previous_model_path: str | None,
    task_config_path: Path,
) -> dict[str, Any]:
    """Build one stage summary without launching PPO training."""
    run_root = training_dir.parent.parent.parent
    validation_result = validation.tasks.validate_task(stage.task)
    entry: dict[str, Any] = {
        "stage_index": stage_index,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "task": stage.task,
        "task_reason": stage.task_reason,
        "notes": stage.notes,
        "task_distribution_config_path": None if stage.task_distribution_config_path is None else str(stage.task_distribution_config_path),
        "task_distribution_id": stage.task_distribution_id,
        "curriculum_run_name": _curriculum_artifact_run_name(settings.curriculum_name, settings.seed),
        "run_name": run_name,
        "curriculum_stage_run_name": run_name,
        "stage_dir": str(training_dir.parent),
        "stage_dir_relative": utils.artifacts.path_relative_to(training_dir.parent, run_root),
        "training_dir": str(training_dir),
        "training_dir_relative": utils.artifacts.path_relative_to(training_dir, run_root),
        "task_config_path": str(task_config_path),
        "task_config_path_relative": utils.artifacts.path_relative_to(task_config_path, run_root),
        "model_path": None,
        "model_path_relative": None,
        "last_model_path": None,
        "last_model_path_relative": None,
        "best_model_path": None,
        "best_model_path_relative": None,
        "best_model_metric": None,
        "best_model_step": None,
        "best_model_source": None,
        "selected_transfer_model_path": None,
        "selected_transfer_model_source": None,
        "metrics_path": None,
        "metrics_path_relative": None,
        "manifest_path": None,
        "manifest_path_relative": None,
        "diagnostics_dir": None,
        "diagnostics_dir_relative": None,
        "total_timesteps": stage.total_timesteps,
        **_stage_run_metadata(stage),
        "eval_steps": stage.eval_steps,
        "seed": settings.seed,
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": previous_model_path,
        "initial_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "previous_model_path": previous_model_path,
        "previous_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "model_transfer_enabled": False,
        "model_transfer_source": None,
        "model_transfer_source_relative": None,
        "dry_run_proposals": True,
        "validation_status": "valid" if validation_result.is_valid else "invalid",
        "validation_messages": list(validation_result.messages),
    }
    for key in manual_curriculum.SUMMARY_METRIC_KEYS:
        entry[key] = None
    entry.update(_dry_stage_task_distribution_metadata(stage))
    return entry


def _dry_stage_task_distribution_metadata(stage: LLMCurriculumStage) -> dict[str, Any]:
    """Return task-distribution metadata for dry-run stage summaries."""
    if stage.task_distribution_config_path is None:
        return {}
    distribution_settings = envs.task_distribution.load_task_distribution_settings(stage.task_distribution_config_path)
    return distribution_settings.to_metadata()


def _build_curriculum_summary(
    *,
    settings: LLMCurriculumSettings,
    stage_entries: Sequence[dict[str, Any]],
    proposal_log_path: Path,
    proposal_stats: Mapping[str, Any],
    dry_run_proposals: bool,
) -> dict[str, Any]:
    """Build the curriculum-level JSON summary payload."""
    run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    run_manifest_path = utils.artifacts.get_run_manifest_path(run_name)
    final_stage = stage_entries[-1] if stage_entries else None
    final_model_path = _stage_selected_model_path(final_stage) if final_stage is not None else None
    coverage_summary = _curriculum_coverage_summary(stage_entries)
    return {
        "run_type": "training",
        "run_kind": "curriculum",
        "curriculum_kind": LLM_CURRICULUM_KIND,
        "mode": LLM_CURRICULUM_MODE,
        "curriculum_name": settings.curriculum_name,
        "run_name": run_name,
        "run_manifest_path": str(run_manifest_path),
        "config_path": str(settings.config_path) if settings.config_path is not None else None,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "stage_count": len(stage_entries),
        "stage_run_names": [str(stage["run_name"]) for stage in stage_entries],
        "max_stages": settings.max_stages,
        **_llm_reference_budget_metadata(settings),
        "llm_stage_budget": _llm_stage_budget_summary(settings.llm_stage_budget),
        "proposal_fallback": _proposal_fallback_summary(settings.proposal_fallback),
        "proposal_fallback_used": any(bool(stage.get("proposal_fallback_used")) for stage in stage_entries),
        "fallback_count": sum(1 for stage in stage_entries if bool(stage.get("proposal_fallback_used"))),
        "curriculum_coverage": coverage_summary,
        "stage_progression_bucket_counts": coverage_summary["bucket_counts"],
        "longest_repeated_category_run": coverage_summary["longest_repeated_category_run"],
        "hover_vertical_loop_detected": coverage_summary["hover_vertical_loop_detected"],
        "progression_score": coverage_summary["progression_score"],
        "proposal_progression_repair_count": sum(1 for stage in stage_entries if bool(stage.get("proposal_repaired"))),
        "repair_count": int(proposal_stats.get("repair_attempts", 0)),
        "repair_success_count": int(proposal_stats.get("repair_successes", 0)),
        "budget_profile_counts": _budget_profile_counts(stage_entries),
        "llm_budget_cap_timesteps": settings.llm_stage_budget.total_budget_cap_timesteps,
        "cumulative_llm_budget_timesteps": final_stage.get("cumulative_llm_budget_timesteps") if final_stage is not None else 0,
        "cumulative_budget_timesteps": final_stage.get("cumulative_llm_budget_timesteps") if final_stage is not None else 0,
        "total_configured_timesteps": sum(int(stage.get("stage_total_timesteps", stage.get("total_timesteps", 0))) for stage in stage_entries),
        "total_actual_timesteps": sum(int(stage.get("stage_total_timesteps", stage.get("total_timesteps", 0))) for stage in stage_entries),
        "selected_stage_budget_profiles": [stage.get("selected_stage_budget_profile") for stage in stage_entries],
        "model_transfer_enabled": any(bool(stage.get("model_transfer_enabled")) for stage in stage_entries),
        "action_interface": final_stage.get("action_interface") if final_stage is not None else None,
        "ppo_action_dim": final_stage.get("ppo_action_dim") if final_stage is not None else None,
        "real_action_type": final_stage.get("real_action_type") if final_stage is not None else None,
        "include_dynamics_observation": final_stage.get("include_dynamics_observation") if final_stage is not None else None,
        "include_previous_action": final_stage.get("include_previous_action") if final_stage is not None else None,
        "observation_dim": final_stage.get("observation_dim") if final_stage is not None else None,
        "observation_components": final_stage.get("observation_components") if final_stage is not None else None,
        "policy_kwargs": final_stage.get("policy_kwargs") if final_stage is not None else None,
        "termination_limits_mode": final_stage.get("termination_limits_mode") if final_stage is not None else None,
        "termination_limits": final_stage.get("termination_limits") if final_stage is not None else None,
        "diagnostic_limits": final_stage.get("diagnostic_limits") if final_stage is not None else None,
        "base_truncation_policy": final_stage.get("base_truncation_policy") if final_stage is not None else None,
        "terminate_on_base_truncation": final_stage.get("terminate_on_base_truncation") if final_stage is not None else None,
        "evaluation_termination_limits_mode": final_stage.get("evaluation_termination_limits_mode") if final_stage is not None else None,
        "evaluation_termination_limits": final_stage.get("evaluation_termination_limits") if final_stage is not None else None,
        "evaluation_diagnostic_limits": final_stage.get("evaluation_diagnostic_limits") if final_stage is not None else None,
        "task_distribution_enabled": final_stage.get("task_distribution_enabled") if final_stage is not None else None,
        "task_distribution_mode": final_stage.get("task_distribution_mode") if final_stage is not None else None,
        "task_distribution_strength": final_stage.get("task_distribution_strength") if final_stage is not None else None,
        "task_distribution_sample_on_reset": final_stage.get("task_distribution_sample_on_reset") if final_stage is not None else None,
        "task_distribution_seed": final_stage.get("task_distribution_seed") if final_stage is not None else None,
        "task_distribution_config_path": final_stage.get("task_distribution_config_path") if final_stage is not None else None,
        "task_distribution_supported_families": final_stage.get("task_distribution_supported_families") if final_stage is not None else None,
        "task_distribution_family_weights": final_stage.get("task_distribution_family_weights") if final_stage is not None else None,
        "task_distribution_name": final_stage.get("task_distribution_name") if final_stage is not None else None,
        "final_stage_run_name": final_stage.get("run_name") if final_stage is not None else None,
        "final_model_path": final_model_path,
        "final_model_source": _stage_selected_model_source(final_stage) if final_stage is not None else None,
        "final_last_model_path": (final_stage.get("last_model_path") or final_stage.get("model_path")) if final_stage is not None else None,
        "final_last_model_path_relative": (final_stage.get("last_model_path_relative") or final_stage.get("model_path_relative"))
        if final_stage is not None
        else None,
        "final_best_model_path": final_stage.get("best_model_path") if final_stage is not None else None,
        "final_best_model_path_relative": final_stage.get("best_model_path_relative") if final_stage is not None else None,
        "final_best_model_metric": final_stage.get("best_model_metric") if final_stage is not None else None,
        "final_best_model_step": final_stage.get("best_model_step") if final_stage is not None else None,
        "final_best_model_source": final_stage.get("best_model_source") if final_stage is not None else None,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "proposal_log_path": str(proposal_log_path),
        "proposal_log_path_relative": utils.artifacts.path_relative_to_run(proposal_log_path, run_name),
        "proposal_stats": dict(proposal_stats),
        "dry_run_proposals": dry_run_proposals,
        "final_stage": _final_stage_summary(final_stage),
        "stages": list(stage_entries),
    }


def _write_curriculum_artifacts(settings: LLMCurriculumSettings, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the canonical LLM curriculum run manifest JSON."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    curriculum_root = utils.artifacts.get_run_dir(artifact_run_name)
    manifest_path = utils.artifacts.get_run_manifest_path(artifact_run_name)
    curriculum_root.mkdir(parents=True, exist_ok=True)
    utils.artifacts.get_run_config_dir(artifact_run_name).mkdir(parents=True, exist_ok=True)
    config_snapshot_path = _write_curriculum_config_snapshot(settings)
    manifest = {
        **summary,
        "artifact_root": str(curriculum_root),
        "artifact_root_relative": ".",
        "summary_path": str(manifest_path),
        "summary_path_relative": utils.artifacts.path_relative_to(manifest_path, curriculum_root),
        "manifest_path": str(manifest_path),
        "manifest_path_relative": utils.artifacts.path_relative_to(manifest_path, curriculum_root),
        "curriculum_config_snapshot_path": str(config_snapshot_path),
        "curriculum_config_snapshot_path_relative": utils.artifacts.path_relative_to(config_snapshot_path, curriculum_root),
        "config": {
            "curriculum_config_path": str(settings.config_path) if settings.config_path is not None else None,
            "curriculum_config_snapshot_path": str(config_snapshot_path),
            "curriculum_config_snapshot_path_relative": utils.artifacts.path_relative_to(config_snapshot_path, curriculum_root),
            "base_training_config": str(settings.base_training_config),
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            **_llm_reference_budget_metadata(settings),
            "llm_stage_budget": _llm_stage_budget_summary(settings.llm_stage_budget),
            "proposal_fallback": _proposal_fallback_summary(settings.proposal_fallback),
        },
        "evaluation_index": _evaluation_index_manifest(artifact_run_name),
    }
    safe_manifest = utils.serialization.to_jsonable(manifest)
    if not isinstance(safe_manifest, dict):
        message = "LLM curriculum manifest must serialize to a JSON object"
        raise TypeError(message)
    utils.serialization.assert_json_serializable(safe_manifest, "LLM curriculum manifest")
    manifest_path.write_text(json.dumps(safe_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path, manifest_path


def _write_curriculum_config_snapshot(settings: LLMCurriculumSettings) -> Path:
    """Materialize a sanitized curriculum config snapshot for a run."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    snapshot_path = utils.artifacts.get_run_curriculum_config_snapshot_path(artifact_run_name)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "curriculum_name": settings.curriculum_name,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "wandb_mode": settings.wandb_mode,
        "normalize_actions": settings.normalize_actions,
        "max_stages": settings.max_stages,
        **_llm_reference_budget_metadata(settings),
        "stage_defaults": {
            "total_timesteps": settings.stage_total_timesteps,
            "eval_steps": settings.stage_eval_steps,
        },
        "bootstrap": _bootstrap_snapshot(settings.bootstrap_stage),
        "llm_stage_budget": _llm_stage_budget_summary(settings.llm_stage_budget),
        "proposal_fallback": _proposal_fallback_summary(settings.proposal_fallback),
        "llm": _sanitized_llm_config(settings.llm_config),
    }
    snapshot_path.write_text(_to_yaml(payload), encoding="utf-8")
    return snapshot_path


def _settings_with_overrides(
    *,
    settings: LLMCurriculumSettings,
    seed: int | None,
    wandb_mode: str | None,
    provider: str | None,
    api_base: str | None,
    model: str | None,
    max_stages: int | None,
    max_repair_attempts: int | None,
) -> LLMCurriculumSettings:
    """Return settings with CLI-style overrides applied."""
    llm_config = dict(settings.llm_config)
    if provider is not None:
        llm_config["provider"] = provider
    if api_base is not None:
        llm_config["api_base"] = api_base
    if model is not None:
        llm_config["model"] = model
    proposal_settings = settings.proposal_settings
    if max_repair_attempts is not None:
        llm_config["max_repair_attempts"] = max_repair_attempts
        proposal_settings = replace(proposal_settings, max_repair_attempts=max_repair_attempts)
    return LLMCurriculumSettings(
        curriculum_name=settings.curriculum_name,
        base_training_config=settings.base_training_config,
        seed=settings.seed if seed is None else seed,
        wandb_mode=settings.wandb_mode if wandb_mode is None else wandb_mode,
        normalize_actions=settings.normalize_actions,
        max_stages=settings.max_stages if max_stages is None else max_stages,
        stage_total_timesteps=settings.stage_total_timesteps,
        stage_eval_steps=settings.stage_eval_steps,
        bootstrap_stage=settings.bootstrap_stage,
        llm_config=llm_config,
        proposal_settings=proposal_settings,
        llm_stage_budget=settings.llm_stage_budget,
        proposal_fallback=settings.proposal_fallback,
        reference_medium_config_path=settings.reference_medium_config_path,
        reference_medium_timesteps=settings.reference_medium_timesteps,
        stage_budget_multipliers=settings.stage_budget_multipliers,
        config_path=settings.config_path,
    )


def _optional_int(value: Any) -> int | None:
    """Return an optional integer metadata value from config."""
    return None if value is None else int(value)


def _optional_path(value: Any) -> Path | None:
    """Return an optional path metadata value from config."""
    return None if value is None else Path(str(value))


def _optional_float_dict(value: Any, field_name: str) -> dict[str, float] | None:
    """Return an optional mapping of profile multipliers from config."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    return {str(key): float(raw_value) for key, raw_value in value.items()}


def _llm_reference_budget_metadata(settings: LLMCurriculumSettings) -> dict[str, Any]:
    """Return reference-medium budget metadata for LLM curriculum summaries."""
    return {
        "reference_medium_config_path": None if settings.reference_medium_config_path is None else str(settings.reference_medium_config_path),
        "reference_medium_timesteps": settings.reference_medium_timesteps,
        "stage_budget_multipliers": None if settings.stage_budget_multipliers is None else dict(settings.stage_budget_multipliers),
    }


def _llm_stage_budget_settings_from_config(raw_budget: Any, default_total_timesteps: int) -> LLMStageBudgetSettings:
    """Return bounded adaptive LLM stage budget settings from config."""
    if raw_budget is None:
        return LLMStageBudgetSettings(
            enabled=False,
            total_budget_cap_timesteps=None,
            default_profile=DEFAULT_LLM_STAGE_BUDGET_PROFILE,
            profiles={DEFAULT_LLM_STAGE_BUDGET_PROFILE: default_total_timesteps},
            min_stage_timesteps=default_total_timesteps,
            max_stage_timesteps=default_total_timesteps,
        )
    if not isinstance(raw_budget, Mapping):
        message = "llm_stage_budget must be a mapping"
        raise TypeError(message)
    enabled = bool(raw_budget.get("enabled", False))
    raw_profiles = _mapping_or_empty(raw_budget.get("profiles"), "llm_stage_budget.profiles")
    profiles: dict[str, int] = {}
    for name, raw_profile in raw_profiles.items():
        profile_mapping = _mapping_or_empty(raw_profile, f"llm_stage_budget.profiles.{name}")
        profiles[str(name)] = int(profile_mapping.get("total_timesteps", 0))
    if not profiles and not enabled:
        profiles[DEFAULT_LLM_STAGE_BUDGET_PROFILE] = default_total_timesteps
    if not profiles:
        message = "enabled llm_stage_budget requires profile definitions"
        raise ValueError(message)
    default_profile = str(raw_budget.get("default_profile") or DEFAULT_LLM_STAGE_BUDGET_PROFILE)
    min_stage = int(raw_budget.get("min_stage_timesteps", min(profiles.values())))
    max_stage = int(raw_budget.get("max_stage_timesteps", max(profiles.values())))
    cap_raw = raw_budget.get("total_budget_cap_timesteps")
    cap = None if cap_raw is None else int(cap_raw)
    return LLMStageBudgetSettings(
        enabled=enabled,
        total_budget_cap_timesteps=cap,
        default_profile=default_profile,
        profiles=profiles,
        min_stage_timesteps=min_stage,
        max_stage_timesteps=max_stage,
    )


def _proposal_fallback_settings_from_config(raw_fallback: Any) -> LLMProposalFallbackSettings:
    """Return opt-in safe proposal fallback settings from config."""
    if raw_fallback is None:
        return LLMProposalFallbackSettings()
    if not isinstance(raw_fallback, Mapping):
        message = "proposal_fallback must be a mapping"
        raise TypeError(message)
    return LLMProposalFallbackSettings(
        enabled=bool(raw_fallback.get("enabled", False)),
        task_distribution_id=str(raw_fallback.get("task_distribution_id") or DEFAULT_PROPOSAL_FALLBACK_DISTRIBUTION_ID),
        default_stage_budget_profile=str(raw_fallback.get("default_stage_budget_profile") or DEFAULT_PROPOSAL_FALLBACK_PROFILE),
        ready_stage_budget_profile=str(raw_fallback.get("ready_stage_budget_profile") or DEFAULT_READY_PROPOSAL_FALLBACK_PROFILE),
    )


def _llm_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied LLM provider configuration mapping."""
    raw_llm_config = config.get("llm")
    if not isinstance(raw_llm_config, Mapping):
        message = "LLM curriculum config must contain a top-level 'llm' mapping"
        raise TypeError(message)
    llm_config = dict(raw_llm_config)
    provider = str(llm_config.get("provider") or llm.client.PROVIDER_MOCK)
    if provider not in llm.client.SUPPORTED_PROVIDERS:
        message = f"llm.provider must be one of: {', '.join(llm.client.SUPPORTED_PROVIDERS)}"
        raise ValueError(message)
    llm_config["provider"] = provider
    return llm_config


def _bootstrap_stage_from_config(raw_bootstrap: Any, default_total_timesteps: int, default_eval_steps: int) -> LLMCurriculumStage | None:
    """Return the optional configured bootstrap stage."""
    if raw_bootstrap is None:
        message = "LLM curriculum config must include bootstrap settings or explicitly disable bootstrap"
        raise ValueError(message)
    if raw_bootstrap is False:
        return None
    if not isinstance(raw_bootstrap, Mapping):
        message = "bootstrap must be a mapping or false"
        raise TypeError(message)
    if not bool(raw_bootstrap.get("enabled", True)):
        return None
    return _stage_from_mapping(raw_bootstrap, default_total_timesteps=default_total_timesteps, default_eval_steps=default_eval_steps)


def _stage_from_mapping(raw_stage: Mapping[str, Any], default_total_timesteps: int, default_eval_steps: int) -> LLMCurriculumStage:
    """Return a validated stage from a raw YAML mapping."""
    task = raw_stage.get("task")
    if not isinstance(task, Mapping):
        message = "stage must contain an explicit task mapping"
        raise TypeError(message)
    task_without_metadata = llm.task_schema.task_without_metadata(dict(task))
    task_shape = str(raw_stage.get("task_shape") or task_without_metadata.get(validation.contracts.FIELD_SHAPE) or "")
    stage_name = _stage_display_name(
        resolved_task=task_without_metadata,
        task_distribution_metadata=None,
        fallback=str(raw_stage.get("stage_name") or task_shape),
    )
    return LLMCurriculumStage(
        stage_name=stage_name,
        task_shape=task_shape,
        task=task_without_metadata,
        total_timesteps=int(raw_stage.get("total_timesteps", default_total_timesteps)),
        eval_steps=int(raw_stage.get("eval_steps", default_eval_steps)),
        task_reason=str(raw_stage[llm.task_schema.REASON_FIELD]) if raw_stage.get(llm.task_schema.REASON_FIELD) is not None else None,
        notes=str(raw_stage["notes"]) if raw_stage.get("notes") is not None else None,
        task_distribution_config_path=Path(str(raw_stage["task_distribution_config_path"]))
        if raw_stage.get("task_distribution_config_path") is not None
        else None,
        task_distribution_id=str(raw_stage["task_distribution_id"]) if raw_stage.get("task_distribution_id") is not None else None,
        requested_stage_budget_profile=str(raw_stage[llm.task_schema.STAGE_BUDGET_PROFILE_FIELD])
        if raw_stage.get(llm.task_schema.STAGE_BUDGET_PROFILE_FIELD) is not None
        else None,
        budget_rationale=str(raw_stage[llm.task_schema.BUDGET_RATIONALE_FIELD])
        if raw_stage.get(llm.task_schema.BUDGET_RATIONALE_FIELD) is not None
        else None,
        requested_stage_task_shape=task_shape,
        accepted_stage_task_shape=task_shape,
        bootstrap_stage_source=str(raw_stage["bootstrap_stage_source"]) if raw_stage.get("bootstrap_stage_source") is not None else None,
        bootstrap_task_shape=str(raw_stage["bootstrap_task_shape"]) if raw_stage.get("bootstrap_task_shape") is not None else None,
        bootstrap_target_sampling_bounds=_optional_json_mapping(raw_stage.get("bootstrap_target_sampling_bounds")),
    )


def _repair_proposal_for_progression(
    *,
    settings: LLMCurriculumSettings,
    stage_index: int,
    task: Mapping[str, Any],
    proposal_type: str | None,
    stage_entries: Sequence[Mapping[str, Any]],
    latest_metrics_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Repair hover/vertical loops and early broad proposals into focused progression distributions."""
    active_type = proposal_type or str(task.get(llm.task_schema.PROPOSAL_KIND_FIELD, llm.task_schema.PROPOSAL_KIND_TASK))
    task_payload = dict(task)
    original_distribution_id = _proposal_distribution_id(task_payload, active_type)
    original_bucket = _progression_bucket_for_proposal(task_payload, active_type)
    recent_buckets = [_stage_progression_bucket_from_entry(entry) for entry in stage_entries]
    recent_buckets = [bucket for bucket in recent_buckets if bucket]
    previous_stage_task_shape = _latest_stage_task_shape(stage_entries)
    hover_vertical_loop = _hover_vertical_loop_detected((*recent_buckets, original_bucket))
    target_distribution_id: str | None = None
    repair_reason: str | None = None
    rule: str | None = None

    recovery_needed = _latest_metrics_need_recovery(latest_metrics_summary)
    if hover_vertical_loop:
        target_distribution_id = _progression_repair_target_distribution_id(
            stage_entries=stage_entries,
            latest_metrics_summary=latest_metrics_summary,
            previous_stage_task_shape=previous_stage_task_shape,
            preferred_bucket=PROGRESSION_BUCKET_XY_LINE,
        )
        repair_reason = "recent stages would continue a hover/vertical loop"
        rule = "hover_vertical_loop_repaired_to_progression"
    elif original_bucket in PURE_HOVER_VERTICAL_BUCKETS and stage_index > PURE_HOVER_VERTICAL_BOOTSTRAP_STAGE_LIMIT and not recovery_needed:
        target_distribution_id = _progression_repair_target_distribution_id(
            stage_entries=stage_entries,
            latest_metrics_summary=latest_metrics_summary,
            previous_stage_task_shape=previous_stage_task_shape,
            preferred_bucket=PROGRESSION_BUCKET_XY_LINE,
        )
        repair_reason = "hover and pure vertical tasks are reserved for early bootstrap or explicit recovery"
        rule = "hover_vertical_reserved_for_early_or_recovery"
    elif original_bucket == PROGRESSION_BUCKET_BROAD_FALLBACK and stage_index < max(settings.max_stages - 1, 3):
        target_distribution_id = _progression_repair_target_distribution_id(
            stage_entries=stage_entries,
            latest_metrics_summary=latest_metrics_summary,
            previous_stage_task_shape=previous_stage_task_shape,
            preferred_bucket=None,
        )
        repair_reason = "broad tracking distribution is deferred until late curriculum coverage exists"
        rule = "early_broad_distribution_repaired_to_focused_progression"

    if target_distribution_id is None or target_distribution_id == original_distribution_id:
        return {
            "task": task_payload,
            "proposal_type": active_type,
            "proposal_repaired": False,
            "proposal_repair_reason": None,
            "proposal_original_distribution_id": original_distribution_id,
            "proposal_final_distribution_id": original_distribution_id,
            "proposal_progression_rule_applied": None,
            "hover_vertical_loop_detected": hover_vertical_loop,
            "stage_progression_bucket": original_bucket,
        }

    repaired_task = _task_distribution_reference(target_distribution_id)
    return {
        "task": repaired_task,
        "proposal_type": llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
        "proposal_repaired": True,
        "proposal_repair_reason": repair_reason,
        "proposal_original_distribution_id": original_distribution_id,
        "proposal_final_distribution_id": target_distribution_id,
        "proposal_progression_rule_applied": rule,
        "hover_vertical_loop_detected": hover_vertical_loop,
        "stage_progression_bucket": _progression_bucket_for_distribution_id(target_distribution_id),
    }


def _stage_progression_repair_fields(
    progression_repair: Mapping[str, Any] | None,
    *,
    task: Mapping[str, Any],
    proposal_type: str,
) -> dict[str, Any]:
    """Return stage dataclass fields for deterministic progression repair metadata."""
    if progression_repair is None:
        distribution_id = _proposal_distribution_id(task, proposal_type)
        return {
            "proposal_repaired": False,
            "proposal_repair_reason": None,
            "proposal_original_distribution_id": distribution_id,
            "proposal_final_distribution_id": distribution_id,
            "proposal_progression_rule_applied": None,
            "hover_vertical_loop_detected": False,
            "stage_progression_bucket": _progression_bucket_for_proposal(task, proposal_type),
        }
    return {
        "proposal_repaired": bool(progression_repair.get("proposal_repaired", False)),
        "proposal_repair_reason": progression_repair.get("proposal_repair_reason"),
        "proposal_original_distribution_id": progression_repair.get("proposal_original_distribution_id"),
        "proposal_final_distribution_id": progression_repair.get("proposal_final_distribution_id"),
        "proposal_progression_rule_applied": progression_repair.get("proposal_progression_rule_applied"),
        "hover_vertical_loop_detected": bool(progression_repair.get("hover_vertical_loop_detected", False)),
        "stage_progression_bucket": progression_repair.get("stage_progression_bucket"),
    }


def _proposal_distribution_id(task: Mapping[str, Any], proposal_type: str | None) -> str | None:
    """Return a proposed distribution id when the proposal references one."""
    if proposal_type == llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION or llm.task_schema.TASK_DISTRIBUTION_ID_FIELD in task:
        value = task.get(llm.task_schema.TASK_DISTRIBUTION_ID_FIELD)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _task_distribution_reference(distribution_id: str) -> dict[str, Any]:
    """Return a known task-distribution proposal reference."""
    return {
        llm.task_schema.PROPOSAL_KIND_FIELD: llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
        llm.task_schema.TASK_DISTRIBUTION_ID_FIELD: distribution_id,
        llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS[distribution_id],
    }


def _progression_repair_target_distribution_id(
    *,
    stage_entries: Sequence[Mapping[str, Any]],
    latest_metrics_summary: Mapping[str, Any],
    previous_stage_task_shape: str | None,
    preferred_bucket: str | None,
) -> str:
    """Return the next focused distribution id for anti-stagnation repair."""
    gaps = set(_latest_skill_gaps(latest_metrics_summary))
    if "curvature_following" in gaps:
        return _first_known_non_duplicate_distribution(
            ("ellipse_bootstrap", "circle_bootstrap", "polyline_bootstrap"),
            fallback="polyline_bootstrap",
            previous_stage_task_shape=previous_stage_task_shape,
        )
    if "turn_following" in gaps or "multi_segment_tracking" in gaps:
        return _first_known_non_duplicate_distribution(
            ("polyline_bootstrap", "zigzag_bootstrap", "triangle_bootstrap", "ellipse_bootstrap"),
            fallback="polyline_bootstrap",
            previous_stage_task_shape=previous_stage_task_shape,
        )
    if "altitude_control" in gaps:
        return _first_known_non_duplicate_distribution(
            ("angled_vertical_bootstrap", "delayed_altitude_polyline_bootstrap", "multi_height_polyline_bootstrap", "polyline_bootstrap"),
            fallback="angled_vertical_bootstrap",
            previous_stage_task_shape=previous_stage_task_shape,
        )
    if "xy_tracking" in gaps or preferred_bucket == PROGRESSION_BUCKET_XY_LINE:
        return _first_known_non_duplicate_distribution(
            ("short_line_bootstrap", "line_bootstrap", "polyline_bootstrap"),
            fallback="short_line_bootstrap",
            previous_stage_task_shape=previous_stage_task_shape,
        )

    seen_buckets = {_stage_progression_bucket_from_entry(entry) for entry in stage_entries}
    ordered_candidates = (
        "short_line_bootstrap",
        "line_bootstrap",
        "angled_vertical_bootstrap",
        "delayed_altitude_polyline_bootstrap",
        "polyline_bootstrap",
        "zigzag_bootstrap",
        "triangle_bootstrap",
        "ellipse_bootstrap",
        "circle_bootstrap",
    )
    for distribution_id in ordered_candidates:
        bucket = _progression_bucket_for_distribution_id(distribution_id)
        if (
            distribution_id in llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS
            and bucket not in seen_buckets
            and not _distribution_duplicates_previous_shape(distribution_id, previous_stage_task_shape)
        ):
            return distribution_id
    return _first_known_non_duplicate_distribution(
        ("polyline_bootstrap", "short_line_bootstrap", "ellipse_bootstrap"),
        fallback="short_line_bootstrap",
        previous_stage_task_shape=previous_stage_task_shape,
    )


def _first_known_non_duplicate_distribution(
    candidate_ids: Sequence[str],
    *,
    fallback: str,
    previous_stage_task_shape: str | None,
) -> str:
    """Return the first known distribution that does not repeat the previous shape family."""
    for distribution_id in candidate_ids:
        if distribution_id in llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS and not _distribution_duplicates_previous_shape(
            distribution_id, previous_stage_task_shape
        ):
            return distribution_id
    return _first_known_distribution(candidate_ids, fallback=fallback)


def _first_known_distribution(candidate_ids: Sequence[str], *, fallback: str) -> str:
    """Return the first known distribution id from candidates."""
    for distribution_id in candidate_ids:
        if distribution_id in llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS:
            return distribution_id
    return fallback


def _distribution_duplicates_previous_shape(distribution_id: str, previous_stage_task_shape: str | None) -> bool:
    """Return whether a distribution is not a valid immediate progression."""
    allowed, _ = llm.progression.is_valid_progression_transition(
        previous_stage_task_shape,
        _progression_stage_context(distribution_id=distribution_id),
    )
    return not allowed


def _latest_stage_task_shape(stage_entries: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the latest accepted stage task shape from summary entries."""
    if not stage_entries:
        return None
    latest = stage_entries[-1]
    for key in ("accepted_stage_task_shape", "resolved_task_shape", "task_shape", "stage_name"):
        value = latest.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _progression_bucket_for_proposal(task: Mapping[str, Any], proposal_type: str | None) -> str:
    """Return the progression bucket for a proposal task or distribution."""
    distribution_id = _proposal_distribution_id(task, proposal_type)
    if distribution_id is not None:
        return _progression_bucket_for_distribution_id(distribution_id)
    return _progression_bucket_for_shape(str(task.get(validation.contracts.FIELD_SHAPE) or ""))


def _progression_bucket_for_distribution_id(distribution_id: str | None) -> str:
    """Return the progression bucket for a known distribution id."""
    if distribution_id is None:
        return PROGRESSION_BUCKET_BROAD_FALLBACK
    return PROGRESSION_DISTRIBUTION_BUCKETS.get(distribution_id, PROGRESSION_BUCKET_BROAD_FALLBACK)


def _progression_bucket_for_shape(shape: str | None) -> str:
    """Return the progression bucket for a concrete task shape."""
    if shape in {validation.contracts.SHAPE_HOVER, validation.contracts.SHAPE_HOVER_STABILIZATION, validation.contracts.SHAPE_NEARBY_TARGET_HOVER}:
        return PROGRESSION_BUCKET_HOVER
    if shape == validation.contracts.SHAPE_VERTICAL:
        return PROGRESSION_BUCKET_VERTICAL_ONLY
    if shape in {validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE, validation.contracts.SHAPE_LINE}:
        return PROGRESSION_BUCKET_XY_LINE
    if shape in {validation.contracts.SHAPE_POLYLINE, "l_shape", "zigzag", "triangle", "rectangle"}:
        return PROGRESSION_BUCKET_TURN_POLYLINE
    if shape in {validation.contracts.SHAPE_CIRCLE, validation.contracts.SHAPE_ELLIPSE, validation.contracts.SHAPE_FIGURE_EIGHT}:
        return PROGRESSION_BUCKET_CURVE
    return PROGRESSION_BUCKET_BROAD_FALLBACK


def _stage_progression_bucket_from_entry(entry: Mapping[str, Any]) -> str:
    """Return the progression bucket recorded or inferred for a stage summary entry."""
    bucket = entry.get("stage_progression_bucket")
    if isinstance(bucket, str) and bucket.strip():
        return bucket
    distribution_id = entry.get("task_distribution_id")
    if isinstance(distribution_id, str) and distribution_id.strip():
        return _progression_bucket_for_distribution_id(distribution_id)
    shape = entry.get("resolved_task_shape") or entry.get("task_shape")
    return _progression_bucket_for_shape(str(shape) if shape is not None else None)


def _hover_vertical_loop_detected(buckets: Sequence[str]) -> bool:
    """Return whether recent progression buckets form a hover/vertical loop."""
    recent = [bucket for bucket in buckets if bucket]
    if len(recent) < HOVER_VERTICAL_LOOP_WINDOW_SIZE:
        return False
    window = recent[-HOVER_VERTICAL_LOOP_WINDOW_SIZE:]
    return all(bucket in PURE_HOVER_VERTICAL_BUCKETS for bucket in window) and len(set(window)) > 1


def _latest_metrics_need_recovery(metrics_summary: Mapping[str, Any]) -> bool:
    """Return whether latest metrics justify simple hover/vertical recovery tasks."""
    status = str(metrics_summary.get("failure_overall_status") or metrics_summary.get("status") or "")
    primary_mode = str(metrics_summary.get("failure_primary_mode") or "")
    if status in {"unstable", "blocked", "safety_failed"}:
        return True
    return primary_mode in {"attitude_instability", "early_termination", "repeated_truncation", "safety_limit_violation"}


def _curriculum_coverage_summary(stage_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return progression coverage diagnostics for accepted curriculum stages."""
    buckets = [_stage_progression_bucket_from_entry(entry) for entry in stage_entries]
    counts = dict.fromkeys(PROGRESSION_BUCKET_ORDER, 0)
    for bucket in buckets:
        counts[bucket] = counts.get(bucket, 0) + 1
    repeated_run = _longest_repeated_bucket_run(buckets)
    hover_vertical_loop = any(_hover_vertical_loop_detected(buckets[index : index + 3]) for index in range(max(len(buckets) - 2, 0)))
    focused_count = sum(
        counts.get(bucket, 0)
        for bucket in (PROGRESSION_BUCKET_XY_LINE, PROGRESSION_BUCKET_TURN_POLYLINE, PROGRESSION_BUCKET_CURVE, PROGRESSION_BUCKET_ALTITUDE_COMBINED)
    )
    progression_score = focused_count + len({bucket for bucket in buckets if bucket not in PURE_HOVER_VERTICAL_BUCKETS})
    return {
        "bucket_counts": counts,
        "hover_count": counts.get(PROGRESSION_BUCKET_HOVER, 0),
        "vertical_only_count": counts.get(PROGRESSION_BUCKET_VERTICAL_ONLY, 0),
        "xy_line_count": counts.get(PROGRESSION_BUCKET_XY_LINE, 0),
        "turn_polyline_count": counts.get(PROGRESSION_BUCKET_TURN_POLYLINE, 0),
        "curve_count": counts.get(PROGRESSION_BUCKET_CURVE, 0),
        "altitude_combined_count": counts.get(PROGRESSION_BUCKET_ALTITUDE_COMBINED, 0),
        "broad_fallback_count": counts.get(PROGRESSION_BUCKET_BROAD_FALLBACK, 0),
        "longest_repeated_category_run": repeated_run,
        "hover_vertical_loop_detected": hover_vertical_loop,
        "progression_score": int(progression_score),
        "bucket_sequence": buckets,
    }


def _longest_repeated_bucket_run(buckets: Sequence[str]) -> dict[str, Any]:
    """Return the longest repeated progression bucket run."""
    best_bucket = None
    best_length = 0
    current_bucket = None
    current_length = 0
    for bucket in buckets:
        if bucket == current_bucket:
            current_length += 1
        else:
            current_bucket = bucket
            current_length = 1
        if current_length > best_length:
            best_bucket = current_bucket
            best_length = current_length
    return {"bucket": best_bucket, "length": int(best_length)}


def _log_proposal_progression_repair(
    *,
    logger: llm.logging.ProposalEventLogger,
    settings: LLMCurriculumSettings,
    stage_index: int,
    repair: Mapping[str, Any],
    proposal_fallback_used: bool,
) -> None:
    """Append one deterministic progression-repair event to the proposal log."""
    logger.append(
        {
            "event_type": "llm_proposal_progression_repair",
            "curriculum_name": settings.curriculum_name,
            "stage_index": stage_index,
            "status": "accepted",
            "proposal_fallback_used": bool(proposal_fallback_used),
            "proposal_repaired": bool(repair.get("proposal_repaired", False)),
            "proposal_repair_reason": repair.get("proposal_repair_reason"),
            "proposal_original_distribution_id": repair.get("proposal_original_distribution_id"),
            "proposal_final_distribution_id": repair.get("proposal_final_distribution_id"),
            "proposal_progression_rule_applied": repair.get("proposal_progression_rule_applied"),
            "hover_vertical_loop_detected": bool(repair.get("hover_vertical_loop_detected", False)),
            "stage_progression_bucket": repair.get("stage_progression_bucket"),
            "task_distribution_reference": dict(repair.get("task") or {}),
        }
    )


def _stage_from_proposal(
    settings: LLMCurriculumSettings,
    stage_index: int,
    task: dict[str, Any],
    task_reason: str | None,
    stage_budget_profile: str | None,
    budget_rationale: str | None,
    proposal_type: str | None,
    original_proposal: dict[str, Any] | None,
    previous_stage_task_shape: str | None,
    proposal_fallback_used: bool,
    proposal_failure_reason: str | None,
    progression_repair: Mapping[str, Any] | None = None,
) -> LLMCurriculumStage:
    """Return a stage from one accepted LLM proposal task or distribution reference."""
    active_proposal_type = proposal_type or str(task.get(llm.task_schema.PROPOSAL_KIND_FIELD, llm.task_schema.PROPOSAL_KIND_TASK))
    original = _json_mapping_copy(original_proposal) or dict(task)
    repair_fields = _stage_progression_repair_fields(progression_repair, task=task, proposal_type=active_proposal_type)
    if active_proposal_type == llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION:
        resolved = _resolve_task_distribution_stage_task(
            settings=settings,
            task=task,
            stage_index=stage_index,
            previous_stage_task_shape=previous_stage_task_shape,
        )
        repair_fields = _merge_distribution_resolution_repair_fields(
            repair_fields,
            resolution_repair=_optional_json_mapping(resolved.get("distribution_resolution_repair")),
        )
        stage_name = _stage_display_name(
            resolved_task=resolved["resolved_task"],
            task_distribution_metadata=resolved["resolved_task_sample_metadata"],
            fallback=resolved["task_distribution_id"],
        )
        return LLMCurriculumStage(
            stage_name=stage_name,
            task_shape=resolved["resolved_task_shape"],
            task=resolved["resolved_task"],
            total_timesteps=settings.stage_total_timesteps,
            eval_steps=settings.stage_eval_steps,
            task_reason=task_reason,
            task_distribution_config_path=resolved["task_distribution_config_path"],
            task_distribution_id=resolved["task_distribution_id"],
            requested_stage_budget_profile=stage_budget_profile,
            budget_rationale=budget_rationale,
            previous_stage_task_shape=previous_stage_task_shape,
            requested_stage_task_shape=_proposal_requested_stage_task_shape(task),
            accepted_stage_task_shape=resolved["resolved_task_shape"],
            **repair_fields,
            fallback_task_shape=resolved["resolved_task_shape"] if proposal_fallback_used else None,
            proposal_type=llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
            original_proposal=original,
            task_distribution_reference=resolved["task_distribution_reference"],
            resolved_task=resolved["resolved_task"],
            resolved_task_shape=resolved["resolved_task_shape"],
            resolved_task_sample_metadata=resolved["resolved_task_sample_metadata"],
            proposal_fallback_used=proposal_fallback_used,
            proposal_failure_reason=proposal_failure_reason,
        )
    task_payload = dict(task)
    task_shape = str(task_payload.get(validation.contracts.FIELD_SHAPE) or "")
    stage_name = _stage_display_name(resolved_task=task_payload, task_distribution_metadata=None, fallback=task_shape)
    generated_distribution = _materialize_concrete_task_distribution(
        settings=settings,
        stage_index=stage_index,
        stage_name=stage_name,
        task=task_payload,
    )
    return LLMCurriculumStage(
        stage_name=stage_name,
        task_shape=task_shape,
        task=task_payload,
        total_timesteps=settings.stage_total_timesteps,
        eval_steps=settings.stage_eval_steps,
        task_reason=task_reason,
        task_distribution_config_path=generated_distribution["task_distribution_config_path"],
        task_distribution_id=generated_distribution["task_distribution_id"],
        requested_stage_budget_profile=stage_budget_profile,
        budget_rationale=budget_rationale,
        previous_stage_task_shape=previous_stage_task_shape,
        requested_stage_task_shape=task_shape,
        accepted_stage_task_shape=task_shape,
        **repair_fields,
        proposal_type=llm.task_schema.PROPOSAL_KIND_TASK,
        original_proposal=original,
        task_distribution_reference=generated_distribution["task_distribution_reference"],
        resolved_task=dict(task_payload),
        resolved_task_shape=task_shape,
        resolved_task_sample_metadata=generated_distribution["resolved_task_sample_metadata"],
        proposal_fallback_used=proposal_fallback_used,
        proposal_failure_reason=proposal_failure_reason,
    )


def _progression_stage_context(
    *,
    distribution_id: str | None = None,
    task_shape: str | None = None,
    sample_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stage-like mapping for progression transition checks."""
    context: dict[str, Any] = {}
    if distribution_id is not None:
        context[llm.task_schema.TASK_DISTRIBUTION_ID_FIELD] = distribution_id
    if task_shape is not None:
        context[validation.contracts.FIELD_SHAPE] = task_shape
    if sample_metadata is not None:
        context["resolved_task_sample_metadata"] = dict(sample_metadata)
    return context


def _merge_distribution_resolution_repair_fields(
    repair_fields: Mapping[str, Any],
    *,
    resolution_repair: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge post-sampling distribution repair metadata into stage repair fields."""
    merged = dict(repair_fields)
    if not resolution_repair:
        return merged
    original_distribution_id = merged.get("proposal_original_distribution_id") or resolution_repair.get("proposal_original_distribution_id")
    merged.update(
        {
            "proposal_repaired": True,
            "proposal_repair_reason": _combined_metadata_text(
                merged.get("proposal_repair_reason"),
                resolution_repair.get("proposal_repair_reason"),
            ),
            "proposal_original_distribution_id": original_distribution_id,
            "proposal_final_distribution_id": resolution_repair.get("proposal_final_distribution_id"),
            "proposal_progression_rule_applied": _combined_metadata_text(
                merged.get("proposal_progression_rule_applied"),
                resolution_repair.get("proposal_progression_rule_applied"),
            ),
            "hover_vertical_loop_detected": bool(
                merged.get("hover_vertical_loop_detected", False) or resolution_repair.get("hover_vertical_loop_detected", False)
            ),
            "stage_progression_bucket": resolution_repair.get("stage_progression_bucket") or merged.get("stage_progression_bucket"),
        }
    )
    return merged


def _combined_metadata_text(first: Any, second: Any) -> str | None:
    """Return a stable semicolon-joined metadata string without duplicate fragments."""
    values: list[str] = []
    for value in (first, second):
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    if not values:
        return None
    return "; ".join(values)


def _resolve_task_distribution_stage_task(
    *,
    settings: LLMCurriculumSettings,
    task: Mapping[str, Any],
    stage_index: int,
    previous_stage_task_shape: str | None,
) -> dict[str, Any]:
    """Resolve a constrained distribution reference into one concrete valid stage task."""
    config_path = Path(str(task[llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD]))
    distribution_id = str(task.get(llm.task_schema.TASK_DISTRIBUTION_ID_FIELD) or config_path.stem)
    primary_resolution = _sample_valid_progression_task_from_distribution(
        settings=settings,
        distribution_id=distribution_id,
        config_path=config_path,
        stage_index=stage_index,
        previous_stage_task_shape=previous_stage_task_shape,
    )
    if primary_resolution["resolved"]:
        return _resolved_distribution_stage_payload(
            config_path=config_path,
            distribution_id=distribution_id,
            resolved_task=primary_resolution["resolved_task"],
            resolved_task_shape=primary_resolution["resolved_task_shape"],
            resolved_sample_metadata=primary_resolution["resolved_task_sample_metadata"],
            resolution_repair=None,
        )

    rejected_alternatives: list[str] = []
    for candidate_index, candidate_distribution_id in enumerate(
        _alternative_distribution_candidates_for_progression(
            distribution_id=distribution_id,
            previous_stage_task_shape=previous_stage_task_shape,
            stage_index=stage_index,
            max_stages=settings.max_stages,
        ),
        start=1,
    ):
        allowed, transition_reason = llm.progression.is_valid_progression_transition(
            previous_stage_task_shape,
            _progression_stage_context(distribution_id=candidate_distribution_id),
            diagnostics={"stage_index": stage_index, "max_stages": settings.max_stages},
        )
        if not allowed:
            rejected_alternatives.append(f"{candidate_distribution_id}: {transition_reason}")
            continue
        candidate_config_path = Path(llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS[candidate_distribution_id])
        candidate_resolution = _sample_valid_progression_task_from_distribution(
            settings=settings,
            distribution_id=candidate_distribution_id,
            config_path=candidate_config_path,
            stage_index=stage_index,
            previous_stage_task_shape=previous_stage_task_shape,
        )
        if not candidate_resolution["resolved"]:
            rejected_alternatives.append(
                _distribution_resolution_rejection_summary(
                    distribution_id=candidate_distribution_id,
                    rejected_samples=candidate_resolution["rejected_samples"],
                )
            )
            continue
        resolved_sample_metadata = dict(candidate_resolution["resolved_task_sample_metadata"])
        resolution_repair = _distribution_resolution_repair_metadata(
            original_distribution_id=distribution_id,
            final_distribution_id=candidate_distribution_id,
            original_rejected_samples=primary_resolution["rejected_samples"],
            rejected_alternatives=rejected_alternatives,
            candidate_index=candidate_index,
        )
        resolved_sample_metadata.update(resolution_repair["resolved_task_sample_metadata"])
        return _resolved_distribution_stage_payload(
            config_path=candidate_config_path,
            distribution_id=candidate_distribution_id,
            resolved_task=candidate_resolution["resolved_task"],
            resolved_task_shape=candidate_resolution["resolved_task_shape"],
            resolved_sample_metadata=resolved_sample_metadata,
            resolution_repair=resolution_repair["stage_repair_fields"],
        )

    primary_details = _sample_rejection_details(primary_resolution["rejected_samples"], previous_stage_task_shape=previous_stage_task_shape)
    alternative_details = "; alternatives rejected: " + "; ".join(rejected_alternatives) if rejected_alternatives else ""
    message = (
        f"resolved task-distribution proposal {distribution_id!r} only produced invalid consecutive progression sample(s): "
        f"{primary_details}{alternative_details}"
    )
    raise ValueError(message)


def _sample_valid_progression_task_from_distribution(
    *,
    settings: LLMCurriculumSettings,
    distribution_id: str,
    config_path: Path,
    stage_index: int,
    previous_stage_task_shape: str | None,
) -> dict[str, Any]:
    """Sample one distribution until a valid non-repeating progression task is found."""
    distribution_settings = envs.task_distribution.load_task_distribution_settings(config_path)
    stage_seed = _stage_task_distribution_seed(settings=settings, distribution_settings=distribution_settings, stage_index=stage_index)
    sampler_settings = replace(distribution_settings, seed=stage_seed)
    sampler = envs.task_distribution.TaskDistributionSampler(sampler_settings, env_rank=0)
    rejected_samples: list[str] = []
    for resolution_attempt in range(1, RESOLVED_DISTRIBUTION_MAX_ATTEMPTS + 1):
        resolved_task = sampler.sample_task()
        resolved_sample_metadata = dict(sampler.sample_metadata())
        validation_result = validation.tasks.validate_task(resolved_task, limits=sampler_settings.validation_limits)
        if not validation_result.is_valid:
            details = "; ".join(validation_result.messages)
            message = f"resolved task-distribution proposal {distribution_id!r} produced an invalid task: {details}"
            raise ValueError(message)
        resolved_task_shape = str(resolved_task.get(validation.contracts.FIELD_SHAPE) or "")
        candidate_stage = _progression_stage_context(
            distribution_id=distribution_id,
            task_shape=resolved_task_shape,
            sample_metadata=resolved_sample_metadata,
        )
        allowed, transition_reason = llm.progression.is_valid_progression_transition(previous_stage_task_shape, candidate_stage)
        if allowed:
            resolved_sample_metadata.update(
                {
                    "distribution_resolution_attempts": resolution_attempt,
                    "duplicate_samples_rejected": len(rejected_samples),
                    "candidate_distribution_rejected": False,
                    "candidate_distribution_rejection_reason": None,
                    "progression_transition_reason": transition_reason,
                }
            )
            return {
                "resolved": True,
                "resolved_task": dict(resolved_task),
                "resolved_task_shape": resolved_task_shape,
                "resolved_task_sample_metadata": resolved_sample_metadata,
                "rejected_samples": rejected_samples,
            }
        rejected_samples.append(f"{resolved_task_shape} ({transition_reason})")
    return {
        "resolved": False,
        "resolved_task": None,
        "resolved_task_shape": None,
        "resolved_task_sample_metadata": None,
        "rejected_samples": rejected_samples,
    }


def _alternative_distribution_candidates_for_progression(
    *,
    distribution_id: str,
    previous_stage_task_shape: str | None,
    stage_index: int,
    max_stages: int,
) -> tuple[str, ...]:
    """Return deterministic repair candidates after a distribution samples only repeats."""
    previous_bucket = _progression_bucket_for_shape(previous_stage_task_shape)
    original_bucket = _progression_bucket_for_distribution_id(distribution_id)
    if previous_stage_task_shape in {validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE}:
        candidates: tuple[str, ...] = (
            "line_bootstrap",
            "angled_vertical_bootstrap",
            "polyline_bootstrap",
            "l_shape_bootstrap",
            "delayed_altitude_polyline_bootstrap",
            "zigzag_bootstrap",
            "triangle_bootstrap",
        )
    elif PROGRESSION_BUCKET_XY_LINE in {previous_bucket, original_bucket}:
        candidates = (
            "polyline_bootstrap",
            "l_shape_bootstrap",
            "angled_vertical_bootstrap",
            "delayed_altitude_polyline_bootstrap",
            "multi_height_polyline_bootstrap",
            "zigzag_bootstrap",
            "triangle_bootstrap",
            "ellipse_bootstrap",
        )
    elif previous_bucket == PROGRESSION_BUCKET_TURN_POLYLINE:
        candidates = (
            "l_shape_bootstrap",
            "zigzag_bootstrap",
            "triangle_bootstrap",
            "rectangle_bootstrap",
            "ellipse_bootstrap",
            "circle_bootstrap",
            "angled_vertical_bootstrap",
        )
    elif previous_bucket == PROGRESSION_BUCKET_CURVE:
        candidates = (
            "ellipse_bootstrap",
            "polyline_bootstrap",
            "zigzag_bootstrap",
            "circle_bootstrap",
            "angled_vertical_bootstrap",
        )
    elif previous_bucket in PURE_HOVER_VERTICAL_BUCKETS:
        candidates = (
            "short_line_bootstrap",
            "line_bootstrap",
            "angled_vertical_bootstrap",
            "polyline_bootstrap",
            "l_shape_bootstrap",
        )
    else:
        candidates = (
            "short_line_bootstrap",
            "line_bootstrap",
            "polyline_bootstrap",
            "l_shape_bootstrap",
            "angled_vertical_bootstrap",
            "ellipse_bootstrap",
        )
    if stage_index >= max(max_stages - 1, 1):
        candidates = (*candidates, "tracking_small", "tracking_medium")
    return tuple(candidate for candidate in _known_unique_distribution_ids(candidates) if candidate != distribution_id)


def _distribution_resolution_repair_metadata(
    *,
    original_distribution_id: str,
    final_distribution_id: str,
    original_rejected_samples: Sequence[str],
    rejected_alternatives: Sequence[str],
    candidate_index: int,
) -> dict[str, Any]:
    """Return stage and sample metadata for a repaired distribution resolution."""
    reason = "resolved distribution samples repeated the immediate progression state"
    rule = "resolved_distribution_repeated_samples_repaired_to_progression"
    return {
        "stage_repair_fields": {
            "proposal_repaired": True,
            "proposal_repair_reason": reason,
            "proposal_original_distribution_id": original_distribution_id,
            "proposal_final_distribution_id": final_distribution_id,
            "proposal_progression_rule_applied": rule,
            "hover_vertical_loop_detected": False,
            "stage_progression_bucket": _progression_bucket_for_distribution_id(final_distribution_id),
        },
        "resolved_task_sample_metadata": {
            "distribution_resolution_repaired": True,
            "distribution_resolution_repair_reason": reason,
            "distribution_resolution_original_distribution_id": original_distribution_id,
            "distribution_resolution_final_distribution_id": final_distribution_id,
            "distribution_resolution_rule_applied": rule,
            "distribution_resolution_original_rejected_samples": list(original_rejected_samples),
            "distribution_resolution_rejected_alternatives": list(rejected_alternatives),
            "distribution_resolution_candidate_attempts": candidate_index,
        },
    }


def _resolved_distribution_stage_payload(
    *,
    config_path: Path,
    distribution_id: str,
    resolved_task: Mapping[str, Any],
    resolved_task_shape: str,
    resolved_sample_metadata: Mapping[str, Any],
    resolution_repair: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the resolved distribution payload consumed by stage construction."""
    return {
        "task_distribution_config_path": config_path,
        "task_distribution_id": distribution_id,
        "task_distribution_reference": {
            llm.task_schema.TASK_DISTRIBUTION_ID_FIELD: distribution_id,
            llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: str(config_path),
        },
        "resolved_task": dict(resolved_task),
        "resolved_task_shape": resolved_task_shape,
        "resolved_task_sample_metadata": dict(resolved_sample_metadata),
        "distribution_resolution_repair": dict(resolution_repair) if resolution_repair else None,
    }


def _distribution_resolution_rejection_summary(*, distribution_id: str, rejected_samples: Sequence[str]) -> str:
    """Return a compact human-readable rejection summary for one candidate distribution."""
    details = _sample_rejection_details(rejected_samples, previous_stage_task_shape=None)
    return f"{distribution_id}: {details}"


def _sample_rejection_details(rejected_samples: Sequence[str], *, previous_stage_task_shape: str | None) -> str:
    """Return the tail of rejected sample reasons for an exhausted distribution."""
    return ", ".join(rejected_samples[-3:]) or str(previous_stage_task_shape)


def _generated_task_distribution_id(*, stage_index: int, stage_name: str, task: Mapping[str, Any]) -> str:
    """Return a compact deterministic ID for a materialized concrete LLM task."""
    raw_shape = task.get(validation.contracts.FIELD_SHAPE)
    source_label = str(raw_shape or stage_name).strip()
    compact_label = GENERATED_TASK_DISTRIBUTION_LABELS.get(source_label) or _compact_generated_identifier(source_label)
    return f"gen_stage{stage_index:02d}_{compact_label}"


def _compact_generated_identifier(value: str) -> str:
    """Return a short ASCII identifier segment for an unknown generated task shape."""
    normalized = "".join(character.lower() if character.isascii() and character.isalnum() else "_" for character in value)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    compact = normalized.strip("_")[:24].rstrip("_")
    return compact or "task"


def _materialize_concrete_task_distribution(
    *,
    settings: LLMCurriculumSettings,
    stage_index: int,
    stage_name: str,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a run-scoped bounded distribution config for one concrete LLM task."""
    task_payload = _with_generated_task_start_hold(dict(task))
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    config_dir = utils.artifacts.get_run_config_dir(curriculum_run_name)
    config_dir.mkdir(parents=True, exist_ok=True)
    distribution_id = _generated_task_distribution_id(stage_index=stage_index, stage_name=stage_name, task=task_payload)
    config_path = config_dir / f"stage{stage_index:02d}_{stage_name}_task_distribution.yaml"
    fixed_settings = envs.task_distribution.normalize_fixed_task_to_distribution(
        task_payload,
        seed=_stage_task_distribution_seed(
            settings=settings,
            distribution_settings=envs.task_distribution.normalize_fixed_task_to_distribution(task_payload),
            stage_index=stage_index,
        ),
        name=distribution_id,
        config_path=config_path,
    )
    family = next(iter(fixed_settings.family_weights))
    payload = {
        envs.task_distribution.DISTRIBUTION_CONFIG_KEY: {
            "name": distribution_id,
            "enabled": True,
            "mode": envs.task_distribution.MODE_RANDOMIZED,
            "seed": fixed_settings.seed,
            "strength": GENERATED_TASK_DISTRIBUTION_STRENGTH,
            "sample_on_reset": True,
            "base_task": task_payload,
            "family_weights": {family: 1.0},
            "variations": {family: _bounded_variation_for_concrete_task(family=family, task=task_payload)},
        }
    }
    config_path.write_text(_to_yaml(payload), encoding="utf-8")
    distribution_settings = envs.task_distribution.load_task_distribution_settings(config_path)
    return {
        "task_distribution_config_path": config_path,
        "task_distribution_id": distribution_id,
        "task_distribution_reference": {
            llm.task_schema.TASK_DISTRIBUTION_ID_FIELD: distribution_id,
            llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: str(config_path),
            "generated_from_concrete_task": True,
            "source_stage_name": stage_name,
            "source_task_shape": task_payload.get(validation.contracts.FIELD_SHAPE),
        },
        "resolved_task_sample_metadata": distribution_settings.to_metadata(),
    }


def _with_generated_task_start_hold(task: dict[str, Any]) -> dict[str, Any]:
    """Apply the uniform LLM curriculum start-hold policy to a materialized concrete task."""
    task = _with_standard_generated_reference_height(task)
    task[validation.contracts.FIELD_START_HOLD_ENABLED] = True
    task[validation.contracts.FIELD_START_HOLD_SEC] = max(
        float(task.get(validation.contracts.FIELD_START_HOLD_SEC, GENERATED_TASK_START_HOLD_SEC)),
        GENERATED_TASK_START_HOLD_SEC,
    )
    task[validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS] = True
    task.pop("lower_start_height_enabled", None)
    task.pop("base_z_offset_range_m", None)
    task["standard_reference_height_enabled"] = True
    task["start_height_policy"] = STANDARD_REFERENCE_HEIGHT_POLICY
    task["start_hold_policy"] = envs.task_distribution.STANDARD_START_HOLD_POLICY
    task["start_hold_reward_policy"] = envs.task_distribution.STANDARD_START_HOLD_REWARD_POLICY
    task["tracking_reward_starts_after_start_hold"] = False
    if task.get(validation.contracts.FIELD_SHAPE) == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE:
        task[validation.contracts.FIELD_HOLD_DURATION_SEC] = max(
            float(task.get(validation.contracts.FIELD_HOLD_DURATION_SEC, GENERATED_TASK_START_HOLD_SEC)),
            GENERATED_TASK_START_HOLD_SEC,
        )
    return task


def _with_standard_generated_reference_height(task: dict[str, Any]) -> dict[str, Any]:
    """Shift sub-standard concrete proposals up to the standard 1.0m reference height."""
    offset = _generated_reference_standard_height_offset(task)
    if offset > 0.0:
        if validation.contracts.FIELD_POSITION in task:
            task[validation.contracts.FIELD_POSITION] = _raised_xyz(task[validation.contracts.FIELD_POSITION], offset)
        if validation.contracts.FIELD_START in task:
            task[validation.contracts.FIELD_START] = _raised_xyz(task[validation.contracts.FIELD_START], offset)
        if validation.contracts.FIELD_END in task:
            task[validation.contracts.FIELD_END] = _raised_xyz(task[validation.contracts.FIELD_END], offset)
        if validation.contracts.FIELD_POINTS in task:
            task[validation.contracts.FIELD_POINTS] = [_raised_xyz(point, offset) for point in task[validation.contracts.FIELD_POINTS]]
        if validation.contracts.FIELD_START_HEIGHT in task:
            task[validation.contracts.FIELD_START_HEIGHT] = _round_height(float(task[validation.contracts.FIELD_START_HEIGHT]) + offset)
        if validation.contracts.FIELD_END_HEIGHT in task:
            task[validation.contracts.FIELD_END_HEIGHT] = _round_height(float(task[validation.contracts.FIELD_END_HEIGHT]) + offset)
        if validation.contracts.FIELD_HEIGHT in task:
            task[validation.contracts.FIELD_HEIGHT] = _round_height(float(task[validation.contracts.FIELD_HEIGHT]) + offset)
    task["base_z_m"] = _round_height(_task_z_anchor(task) or STANDARD_REFERENCE_BASE_Z_M)
    task["sampled_start_height_m"] = task["base_z_m"]
    task["base_z_range_m"] = [float(STANDARD_REFERENCE_HEIGHT_RANGE_M[0]), float(STANDARD_REFERENCE_HEIGHT_RANGE_M[1])]
    task["height_variation_enabled"] = True
    return task


def _generated_reference_standard_height_offset(task: Mapping[str, Any]) -> float:
    """Return the upward shift needed for concrete proposals below the standard active band."""
    anchor = _task_z_anchor(task)
    if anchor is None or anchor >= STANDARD_REFERENCE_BASE_Z_M or anchor > SUBSTANDARD_REFERENCE_MAX_START_M:
        return 0.0
    return max(0.0, STANDARD_REFERENCE_BASE_Z_M - anchor)


def _raised_xyz(value: Any, offset: float) -> list[float]:
    """Return an XYZ vector with z shifted upward by ``offset``."""
    vector = [float(component) for component in value]
    if len(vector) != XYZ_VECTOR_LENGTH:
        message = "reference point must contain exactly three values"
        raise ValueError(message)
    vector[2] = _round_height(vector[2] + offset)
    return [float(round(vector[0], 6)), float(round(vector[1], 6)), vector[2]]


def _round_height(value: float) -> float:
    """Return a compact rounded height value."""
    return float(round(float(value), 6))


def _bounded_variation_for_concrete_task(*, family: str, task: Mapping[str, Any]) -> dict[str, Any]:
    """Return conservative variation bounds around one validated concrete task."""
    duration = _task_duration_sec(task)
    variation: dict[str, Any] = {
        "duration_range_sec": _positive_range(duration, max(0.5, duration * 0.2), lower=1.0, upper=20.0),
        "start_hold_range_sec": [GENERATED_TASK_START_HOLD_SEC, GENERATED_TASK_START_HOLD_SEC],
        "final_hold_range_sec": [0.75, 1.0],
    }
    z_anchor = _task_z_anchor(task)
    if z_anchor is not None:
        variation["base_z_range_m"] = [float(STANDARD_REFERENCE_HEIGHT_RANGE_M[0]), float(STANDARD_REFERENCE_HEIGHT_RANGE_M[1])]
        variation["z_range_m"] = _bounded_range(z_anchor, 0.10, lower=0.9, upper=1.1)
    if family == envs.task_distribution.FAMILY_HOVER:
        variation.update({"xy_radius_m": 0.12})
    elif family == envs.task_distribution.FAMILY_TAKEOFF:
        start_height = float(task.get(validation.contracts.FIELD_START_HEIGHT, 0.4))
        end_height = float(task.get(validation.contracts.FIELD_END_HEIGHT, z_anchor or 1.0))
        variation.update(
            {
                "xy_radius_m": 0.08,
                "start_z_range_m": _bounded_range(start_height, 0.08, lower=0.9, upper=1.1),
                "z_range_m": _bounded_range(end_height, 0.12, lower=0.75, upper=1.35),
            }
        )
    elif family in {envs.task_distribution.FAMILY_LINE, envs.task_distribution.FAMILY_START_HOLD_LINE}:
        length = _task_line_length(task, default=0.4)
        variation.update(
            {
                "start_xy_radius_m": 0.08,
                "length_range_m": _positive_range(length, max(0.08, length * 0.25), lower=0.15, upper=0.9),
                "heading_jitter_deg": 12.0,
                "start_hold_range_sec": [GENERATED_TASK_START_HOLD_SEC, GENERATED_TASK_START_HOLD_SEC],
            }
        )
    elif family in {
        envs.task_distribution.FAMILY_POLYLINE,
        envs.task_distribution.FAMILY_L_SHAPE,
        envs.task_distribution.FAMILY_ZIGZAG,
        envs.task_distribution.FAMILY_TRIANGLE,
        envs.task_distribution.FAMILY_MULTI_HEIGHT_POLYLINE,
        envs.task_distribution.FAMILY_RECTANGLE,
        envs.task_distribution.FAMILY_SQUARE,
    }:
        length = _task_line_length(task, default=0.5)
        variation.update(
            {
                "start_xy_radius_m": 0.08,
                "length_range_m": _positive_range(length, max(0.1, length * 0.25), lower=0.2, upper=1.0),
                "heading_jitter_deg": 15.0,
            }
        )
    elif family == envs.task_distribution.FAMILY_CIRCLE:
        radius = float(task.get(validation.contracts.FIELD_RADIUS, 0.3))
        variation.update({"center_xy_radius_m": 0.06, "radius_range_m": _positive_range(radius, 0.06, lower=0.12, upper=0.6)})
    elif family in {envs.task_distribution.FAMILY_ELLIPSE, envs.task_distribution.FAMILY_FIGURE_EIGHT}:
        radius_x = float(task.get(validation.contracts.FIELD_RADIUS_X, 0.3))
        radius_y = float(task.get(validation.contracts.FIELD_RADIUS_Y, 0.18))
        variation.update(
            {
                "center_xy_radius_m": 0.06,
                "radius_x_range_m": _positive_range(radius_x, 0.06, lower=0.12, upper=0.6),
                "radius_y_range_m": _positive_range(radius_y, 0.04, lower=0.08, upper=0.45),
            }
        )
    return variation


def _task_duration_sec(task: Mapping[str, Any]) -> float:
    """Return the representative duration for a task mapping."""
    if validation.contracts.FIELD_DURATION_SEC in task:
        return float(task[validation.contracts.FIELD_DURATION_SEC])
    if validation.contracts.FIELD_MOVE_DURATION_SEC in task:
        return float(task[validation.contracts.FIELD_MOVE_DURATION_SEC])
    return 3.0


def _task_z_anchor(task: Mapping[str, Any]) -> float | None:
    """Return a representative task height for variation bounds."""
    if validation.contracts.FIELD_POSITION in task:
        return float(task[validation.contracts.FIELD_POSITION][2])
    if validation.contracts.FIELD_START in task:
        return float(task[validation.contracts.FIELD_START][2])
    if validation.contracts.FIELD_POINTS in task:
        return float(task[validation.contracts.FIELD_POINTS][0][2])
    if validation.contracts.FIELD_START_HEIGHT in task:
        return float(task[validation.contracts.FIELD_START_HEIGHT])
    if validation.contracts.FIELD_END_HEIGHT in task:
        return float(task[validation.contracts.FIELD_END_HEIGHT])
    if validation.contracts.FIELD_HEIGHT in task:
        return float(task[validation.contracts.FIELD_HEIGHT])
    return None


def _task_line_length(task: Mapping[str, Any], *, default: float) -> float:
    """Return the XY distance between the representative start and end points."""
    if validation.contracts.FIELD_START in task and validation.contracts.FIELD_END in task:
        start = task[validation.contracts.FIELD_START]
        end = task[validation.contracts.FIELD_END]
    elif validation.contracts.FIELD_POINTS in task:
        points = task[validation.contracts.FIELD_POINTS]
        start = points[0]
        end = points[-1]
    else:
        return float(default)
    return max(float(default), math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1])))


def _positive_range(value: float, radius: float, *, lower: float, upper: float) -> list[float]:
    """Return a rounded positive numeric range around a value."""
    return _bounded_range(value, radius, lower=lower, upper=upper)


def _bounded_range(value: float, radius: float, *, lower: float, upper: float) -> list[float]:
    """Return a rounded numeric range clipped to finite safety bounds."""
    center = float(value)
    low = max(float(lower), center - float(radius))
    high = min(float(upper), center + float(radius))
    high = max(high, low)
    return [float(round(low, 6)), float(round(high, 6))]


def _proposal_requested_stage_task_shape(task: Mapping[str, Any]) -> str | None:
    """Return requested shape metadata for a proposal task or distribution reference."""
    if task.get(llm.task_schema.PROPOSAL_KIND_FIELD) == llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION:
        try:
            distribution_settings = envs.task_distribution.load_task_distribution_settings(
                str(task[llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD])
            )
        except (KeyError, OSError, TypeError, ValueError):
            return str(task.get(llm.task_schema.TASK_DISTRIBUTION_ID_FIELD) or "task_distribution")
        if len(distribution_settings.family_weights) == 1:
            family = next(iter(distribution_settings.family_weights))
            return _shape_from_distribution_family(family)
        return str(task.get(llm.task_schema.TASK_DISTRIBUTION_ID_FIELD) or distribution_settings.name or "task_distribution")
    shape = task.get(validation.contracts.FIELD_SHAPE)
    return str(shape) if shape is not None else None


def _shape_from_distribution_family(family: str) -> str:
    """Return representative task shape for a task-distribution family."""
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
    return family_to_shape.get(family, family)


def _is_consecutive_duplicate_stage_shape(previous_shape: str | None, next_shape: str | None) -> bool:
    """Return whether two concrete stage shapes are not a valid immediate progression."""
    allowed, _ = llm.progression.is_valid_progression_transition(previous_shape, next_shape)
    return not allowed


def _previous_accepted_stage_shape(recent_accepted_tasks: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the most recent accepted concrete stage shape."""
    if not recent_accepted_tasks:
        return None
    latest = dict(recent_accepted_tasks[-1])
    for key in ("accepted_stage_task_shape", "resolved_task_shape", "task_shape"):
        value = latest.get(key)
        if value is not None and str(value).strip():
            return str(value)
    task = latest.get("resolved_task") or latest.get("task")
    if isinstance(task, Mapping):
        shape = task.get(validation.contracts.FIELD_SHAPE)
        if shape is not None and str(shape).strip():
            return str(shape)
    return None


def _stage_task_distribution_seed(
    *,
    settings: LLMCurriculumSettings,
    distribution_settings: envs.task_distribution.TaskDistributionSettings,
    stage_index: int,
) -> int:
    """Return the deterministic seed used to resolve a stage distribution reference."""
    return int(distribution_settings.seed) + int(settings.seed) + int(stage_index)


def _json_mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a plain dictionary copy for JSON-ready metadata fields."""
    if value is None:
        return None
    return dict(value)


def _optional_json_mapping(value: Any) -> dict[str, Any] | None:
    """Return an optional plain mapping for JSON-ready metadata fields."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        message = "bootstrap_target_sampling_bounds must be a mapping"
        raise TypeError(message)
    return dict(value)


def _stage_display_name(
    *,
    resolved_task: Mapping[str, Any] | None,
    task_distribution_metadata: Mapping[str, Any] | None,
    fallback: str,
) -> str:
    """Return the display name used for stage run names and metadata."""
    if resolved_task is not None:
        shape = resolved_task.get(validation.contracts.FIELD_SHAPE)
        if isinstance(shape, str) and shape.strip():
            return shape.strip()
    if task_distribution_metadata is not None:
        sampled_family = task_distribution_metadata.get("task_distribution_sampled_family")
        if isinstance(sampled_family, str) and sampled_family.strip():
            return sampled_family.strip()
    fallback_text = str(fallback).strip()
    return fallback_text or "stage"


def _resolve_llm_stage_budget(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    cumulative_timesteps: int,
) -> LLMCurriculumStage:
    """Resolve one stage budget through bounded deterministic profile rules."""
    budget_settings = settings.llm_stage_budget
    requested_profile = stage.requested_stage_budget_profile or _default_stage_budget_profile(settings, stage_index)
    if requested_profile not in budget_settings.profiles:
        available = ", ".join(sorted(budget_settings.profiles))
        message = f"stage_budget_profile must be one of: {available}"
        raise ValueError(message)
    selected_profile = requested_profile
    resolved_timesteps = stage.total_timesteps
    budget_was_clipped = False
    fallback_reasons: list[str] = []

    if budget_settings.enabled:
        if stage_index > 1 and selected_profile == llm.task_schema.BUDGET_PROFILE_BOOTSTRAP:
            selected_profile = _post_bootstrap_default_profile(budget_settings)
            budget_was_clipped = True
            fallback_reasons.append(f"requested profile {requested_profile!r} is reserved for stage 1; fell back to {selected_profile!r}")
        resolved_timesteps = budget_settings.profiles[selected_profile]
        cap = budget_settings.total_budget_cap_timesteps
        if cap is None:
            message = "enabled llm_stage_budget requires a total budget cap"
            raise ValueError(message)
        remaining_stage_slots = max(settings.max_stages - stage_index, 0)
        reserve_profile = _reserve_budget_profile(budget_settings)
        reserved_future_timesteps = remaining_stage_slots * budget_settings.profiles[reserve_profile]
        max_allowed_this_stage = cap - cumulative_timesteps - reserved_future_timesteps
        if max_allowed_this_stage < budget_settings.min_stage_timesteps:
            message = "llm_stage_budget cap leaves no valid budget for the current stage while reserving remaining stages"
            raise ValueError(message)
        if resolved_timesteps > max_allowed_this_stage:
            budget_was_clipped = True
            safe_profile = _largest_safe_stage_budget_profile(
                budget_settings=budget_settings,
                stage_index=stage_index,
                max_allowed_timesteps=max_allowed_this_stage,
            )
            fallback_reasons.append(
                f"selected profile {selected_profile!r} would exceed the total budget cap after reserving "
                f"{remaining_stage_slots} remaining {reserve_profile!r} stage(s); fell back to {safe_profile!r}"
            )
            selected_profile = safe_profile
            resolved_timesteps = budget_settings.profiles[selected_profile]
            if resolved_timesteps > max_allowed_this_stage:
                fallback_reasons.append("fallback profile did not fit; clipped to remaining reserved budget")
                resolved_timesteps = max_allowed_this_stage
        if resolved_timesteps < budget_settings.min_stage_timesteps or resolved_timesteps > budget_settings.max_stage_timesteps:
            message = "resolved LLM stage budget is outside configured min/max bounds"
            raise ValueError(message)
        cumulative_after = cumulative_timesteps + resolved_timesteps
        if cumulative_after > cap:
            message = "resolved LLM stage budget exceeds total budget cap"
            raise ValueError(message)
    else:
        cumulative_after = cumulative_timesteps + resolved_timesteps

    return replace(
        stage,
        total_timesteps=resolved_timesteps,
        requested_stage_budget_profile=requested_profile,
        selected_stage_budget_profile=selected_profile,
        budget_was_clipped=budget_was_clipped,
        budget_fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
        cumulative_llm_budget_timesteps=cumulative_after,
        llm_budget_cap_timesteps=budget_settings.total_budget_cap_timesteps,
    )


def _default_stage_budget_profile(settings: LLMCurriculumSettings, stage_index: int) -> str:
    """Return the default budget profile for a stage index."""
    budget_settings = settings.llm_stage_budget
    if budget_settings.enabled and stage_index == 1 and llm.task_schema.BUDGET_PROFILE_BOOTSTRAP in budget_settings.profiles:
        return llm.task_schema.BUDGET_PROFILE_BOOTSTRAP
    return budget_settings.default_profile


def _post_bootstrap_default_profile(budget_settings: LLMStageBudgetSettings) -> str:
    """Return the default profile used if bootstrap is requested after stage 1."""
    if budget_settings.default_profile != llm.task_schema.BUDGET_PROFILE_BOOTSTRAP:
        return budget_settings.default_profile
    return _reserve_budget_profile(budget_settings)


def _reserve_budget_profile(budget_settings: LLMStageBudgetSettings) -> str:
    """Return the profile reserved for future stages under the total cap."""
    if "short" in budget_settings.profiles:
        return "short"
    return min(budget_settings.profiles, key=budget_settings.profiles.__getitem__)


def _allowed_budget_profile_names(budget_settings: LLMStageBudgetSettings, stage_index: int) -> tuple[str, ...]:
    """Return budget profiles allowed for this stage index."""
    names = tuple(budget_settings.profiles)
    if stage_index <= 1:
        return names
    non_bootstrap = tuple(name for name in names if name != llm.task_schema.BUDGET_PROFILE_BOOTSTRAP)
    return non_bootstrap or names


def _largest_safe_stage_budget_profile(
    *,
    budget_settings: LLMStageBudgetSettings,
    stage_index: int,
    max_allowed_timesteps: int,
) -> str:
    """Return the largest allowed profile that fits, falling back to short."""
    allowed_profiles = _allowed_budget_profile_names(budget_settings, stage_index)
    safe_profiles = [name for name in allowed_profiles if budget_settings.profiles[name] <= max_allowed_timesteps]
    if safe_profiles:
        return max(safe_profiles, key=lambda name: (budget_settings.profiles[name], name))
    return _reserve_budget_profile(budget_settings)


def _validate_stage_task(stage: LLMCurriculumStage) -> None:
    """Validate a stage task before PPO training or dry-run acceptance."""
    result = validation.tasks.validate_task(stage.task)
    if not result.is_valid:
        details = "; ".join(result.messages)
        message = f"invalid LLM curriculum stage {stage.stage_name!r}: {details}"
        raise ValueError(message)


def _write_stage_task_config(
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    stage_training_dir: Path,
) -> Path:
    """Write the one-task config consumed by the existing PPO helper."""
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    config_dir = utils.artifacts.get_run_config_dir(curriculum_run_name)
    config_dir.mkdir(parents=True, exist_ok=True)
    task_config_path = config_dir / f"stage{stage_index:02d}_{stage.stage_name}_task.yaml"
    stage_training_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": f"{settings.curriculum_name}_stage{stage_index:02d}",
        "seed": settings.seed,
        "tasks": [stage.task],
    }
    task_config_path.write_text(_to_yaml(payload), encoding="utf-8")
    return task_config_path


def _stage_from_fallback_failure(
    *,
    settings: LLMCurriculumSettings,
    stage_index: int,
    metrics_summary: Mapping[str, Any],
    failure_reason: str,
    previous_stage_task_shape: str | None,
    context_overflow: bool,
    stage_entries: Sequence[Mapping[str, Any]],
    latest_metrics_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a deterministic fallback by trying valid candidate distributions in order."""
    rejected_candidates: list[str] = []
    candidate_ids = _fallback_distribution_candidates(
        settings=settings,
        previous_stage_task_shape=previous_stage_task_shape,
        metrics_summary=metrics_summary,
        stage_index=stage_index,
        context_overflow=context_overflow,
    )
    for distribution_id in candidate_ids:
        allowed, transition_reason = llm.progression.is_valid_progression_transition(
            previous_stage_task_shape,
            _progression_stage_context(distribution_id=distribution_id),
            history=stage_entries,
            diagnostics={"stage_index": stage_index, "max_stages": settings.max_stages},
        )
        if not allowed:
            rejected_candidates.append(f"{distribution_id}: {transition_reason}")
            continue
        fallback_proposal = _fallback_proposal_from_failure(
            settings=settings,
            stage_index=stage_index,
            metrics_summary=metrics_summary,
            failure_reason=failure_reason,
            previous_stage_task_shape=previous_stage_task_shape,
            context_overflow=context_overflow,
            distribution_id=distribution_id,
        )
        fallback_repair = _repair_proposal_for_progression(
            settings=settings,
            stage_index=stage_index,
            task=fallback_proposal["task"],
            proposal_type=llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
            stage_entries=stage_entries,
            latest_metrics_summary=latest_metrics_summary,
        )
        final_distribution_id = fallback_repair.get("proposal_final_distribution_id")
        if isinstance(final_distribution_id, str) and final_distribution_id.strip():
            final_allowed, final_reason = llm.progression.is_valid_progression_transition(
                previous_stage_task_shape,
                _progression_stage_context(distribution_id=final_distribution_id),
                history=stage_entries,
                diagnostics={"stage_index": stage_index, "max_stages": settings.max_stages},
            )
            if not final_allowed:
                rejected_candidates.append(f"{distribution_id}->{final_distribution_id}: {final_reason}")
                continue
        try:
            stage = _stage_from_proposal(
                settings=settings,
                stage_index=stage_index,
                task=fallback_repair["task"],
                task_reason=fallback_proposal["task_reason"],
                stage_budget_profile=fallback_proposal["stage_budget_profile"],
                budget_rationale=fallback_proposal["budget_rationale"],
                proposal_type=llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
                original_proposal=fallback_proposal["original_proposal"],
                previous_stage_task_shape=previous_stage_task_shape,
                proposal_fallback_used=True,
                proposal_failure_reason=failure_reason,
                progression_repair=fallback_repair,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            rejected_candidates.append(f"{distribution_id}: {exc}")
            continue
        fallback_attempts = len(rejected_candidates) + 1
        if stage.resolved_task_sample_metadata is not None:
            stage = replace(
                stage,
                resolved_task_sample_metadata={
                    **stage.resolved_task_sample_metadata,
                    "fallback_distribution_attempts": fallback_attempts,
                },
            )
        fallback_proposal["fallback_distribution_attempts"] = fallback_attempts
        fallback_proposal["fallback_original_distribution_id"] = distribution_id
        fallback_proposal["fallback_final_distribution_id"] = stage.task_distribution_id
        fallback_proposal["original_proposal"]["fallback_distribution_attempts"] = fallback_attempts
        fallback_proposal["original_proposal"]["fallback_original_distribution_id"] = distribution_id
        fallback_proposal["original_proposal"]["fallback_final_distribution_id"] = stage.task_distribution_id
        if rejected_candidates:
            fallback_proposal["fallback_resolution_rejections"] = list(rejected_candidates)
            fallback_proposal["original_proposal"]["fallback_resolution_rejections"] = list(rejected_candidates)
        return {
            "fallback_proposal": fallback_proposal,
            "fallback_repair": fallback_repair,
            "stage": stage,
        }
    details = "; ".join(rejected_candidates) or "no known fallback candidates"
    message = f"deterministic fallback could not resolve a valid progression at stage {stage_index}: {details}"
    raise ValueError(message)


def _fallback_proposal_from_failure(
    *,
    settings: LLMCurriculumSettings,
    stage_index: int,
    metrics_summary: Mapping[str, Any],
    failure_reason: str,
    previous_stage_task_shape: str | None,
    context_overflow: bool = False,
    distribution_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic safe proposal after exhausted LLM proposal attempts."""
    if distribution_id is None:
        distribution_id = _fallback_distribution_id(
            settings=settings,
            previous_stage_task_shape=previous_stage_task_shape,
            metrics_summary=metrics_summary,
            stage_index=stage_index,
            context_overflow=context_overflow,
        )
    config_path = llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS[distribution_id]
    stage_budget_profile = _fallback_stage_budget_profile(settings=settings, metrics_summary=metrics_summary)
    status = metrics_summary.get("failure_overall_status") or metrics_summary.get("status")
    if context_overflow:
        task_reason = f"Safe focused fallback after LLM context overflow at stage {stage_index}; using known distribution {distribution_id}."
        budget_rationale = f"Context-overflow fallback selected {stage_budget_profile!r} from latest concrete status={status!r}."
    else:
        task_reason = f"Safe fallback after exhausted LLM proposal repair at stage {stage_index}; using known distribution {distribution_id}."
        budget_rationale = f"Fallback selected {stage_budget_profile!r} from latest concrete status={status!r}."
    task = {
        llm.task_schema.PROPOSAL_KIND_FIELD: llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
        llm.task_schema.TASK_DISTRIBUTION_ID_FIELD: distribution_id,
        llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: config_path,
    }
    original_proposal = {
        **task,
        llm.task_schema.REASON_FIELD: task_reason,
        llm.task_schema.STAGE_BUDGET_PROFILE_FIELD: stage_budget_profile,
        llm.task_schema.BUDGET_RATIONALE_FIELD: budget_rationale,
        "proposal_fallback_used": True,
        "proposal_failure_reason": failure_reason,
        "llm_request_failed_due_to_context_size": context_overflow,
        "llm_context_fallback_used": context_overflow,
        "llm_context_fallback_reason": failure_reason if context_overflow else None,
    }
    return {
        "task": task,
        "task_reason": task_reason,
        "stage_budget_profile": stage_budget_profile,
        "budget_rationale": budget_rationale,
        "original_proposal": original_proposal,
        "proposal_failure_reason": failure_reason,
        "llm_request_failed_due_to_context_size": context_overflow,
        "llm_context_fallback_used": context_overflow,
        "llm_context_fallback_reason": failure_reason if context_overflow else None,
    }


def _fallback_distribution_id(
    *,
    settings: LLMCurriculumSettings,
    previous_stage_task_shape: str | None,
    metrics_summary: Mapping[str, Any] | None = None,
    stage_index: int = 1,
    context_overflow: bool = False,
) -> str:
    """Return a safe fallback distribution that avoids invalid immediate repeats when possible."""
    candidates = _fallback_distribution_candidates(
        settings=settings,
        previous_stage_task_shape=previous_stage_task_shape,
        metrics_summary=metrics_summary or {},
        stage_index=stage_index,
        context_overflow=context_overflow,
    )
    return _first_non_duplicate_distribution(
        candidates,
        configured_id=settings.proposal_fallback.task_distribution_id,
        previous_stage_task_shape=previous_stage_task_shape,
    )


def _fallback_distribution_candidates(
    *,
    settings: LLMCurriculumSettings,
    previous_stage_task_shape: str | None,
    metrics_summary: Mapping[str, Any],
    stage_index: int,
    context_overflow: bool,
) -> tuple[str, ...]:
    """Return ordered deterministic fallback distribution candidates for one stage."""
    configured_id = settings.proposal_fallback.task_distribution_id
    if not context_overflow:
        if previous_stage_task_shape in {validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE}:
            candidates: tuple[str, ...] = (
                "line_bootstrap",
                "angled_vertical_bootstrap",
                "delayed_altitude_polyline_bootstrap",
                "polyline_bootstrap",
                "zigzag_bootstrap",
                "short_line_bootstrap",
                "triangle_bootstrap",
            )
        elif previous_stage_task_shape == validation.contracts.SHAPE_LINE:
            candidates = (
                "angled_vertical_bootstrap",
                "delayed_altitude_polyline_bootstrap",
                "polyline_bootstrap",
                "zigzag_bootstrap",
                "triangle_bootstrap",
                "line_bootstrap",
                "short_line_bootstrap",
            )
        else:
            candidates = (
                "short_line_bootstrap",
                "line_bootstrap",
                "angled_vertical_bootstrap",
                "delayed_altitude_polyline_bootstrap",
                "polyline_bootstrap",
                "zigzag_bootstrap",
                "triangle_bootstrap",
            )
        if stage_index >= max(settings.max_stages - 1, 1):
            candidates = (*candidates, "tracking_small", configured_id)
        return _known_unique_distribution_ids(candidates)

    candidates = _context_overflow_fallback_candidates(metrics_summary, previous_stage_task_shape=previous_stage_task_shape)
    if stage_index >= max(settings.max_stages - 1, 1):
        candidates = (*candidates, "tracking_small", "tracking_medium", configured_id)
    else:
        candidates = (*candidates, "tracking_small")
    return _known_unique_distribution_ids(candidates)


def _known_unique_distribution_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    """Return known distribution ids while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for distribution_id in candidate_ids:
        if distribution_id in seen or distribution_id not in llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS:
            continue
        seen.add(distribution_id)
        ordered.append(distribution_id)
    return tuple(ordered)


def _context_overflow_fallback_candidates(
    metrics_summary: Mapping[str, Any],
    *,
    previous_stage_task_shape: str | None,
) -> tuple[str, ...]:
    """Return focused fallback candidates from latest compact skill gaps."""
    gaps = set(_latest_skill_gaps(metrics_summary))
    if "altitude_control" in gaps:
        return ("angled_vertical_bootstrap", "delayed_altitude_polyline_bootstrap", "multi_height_polyline_bootstrap", "polyline_bootstrap")
    if "xy_tracking" in gaps:
        return ("short_line_bootstrap", "line_bootstrap", "polyline_bootstrap")
    if "turn_following" in gaps:
        return ("polyline_bootstrap", "zigzag_bootstrap", "triangle_bootstrap", "short_line_bootstrap")
    if "curvature_following" in gaps:
        return ("ellipse_bootstrap", "circle_bootstrap", "polyline_bootstrap", "zigzag_bootstrap", "short_line_bootstrap")
    if previous_stage_task_shape in {validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE}:
        return ("line_bootstrap", "angled_vertical_bootstrap", "delayed_altitude_polyline_bootstrap", "polyline_bootstrap")
    if previous_stage_task_shape == validation.contracts.SHAPE_LINE:
        return ("angled_vertical_bootstrap", "delayed_altitude_polyline_bootstrap", "polyline_bootstrap", "short_line_bootstrap")
    return ("short_line_bootstrap", "vertical_bootstrap", "polyline_bootstrap", "hover_bootstrap")


def _latest_skill_gaps(metrics_summary: Mapping[str, Any]) -> tuple[str, ...]:
    """Return normalized latest skill-gap labels from metrics and feedback summaries."""
    raw = metrics_summary.get("curriculum_primary_skill_gaps")
    if not raw:
        strategy = metrics_summary.get("curriculum_strategy")
        if isinstance(strategy, Mapping):
            raw = strategy.get("primary_skill_gaps") or strategy.get("candidate_next_skills")
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, Sequence):
        return ()
    return tuple(str(item) for item in raw if item is not None)


def _first_non_duplicate_distribution(
    candidate_ids: Sequence[str],
    *,
    configured_id: str,
    previous_stage_task_shape: str | None,
) -> str:
    """Return the first known candidate that is a valid immediate progression."""
    return _first_known_non_duplicate_distribution(
        candidate_ids,
        fallback=configured_id,
        previous_stage_task_shape=previous_stage_task_shape,
    )


def _fallback_stage_budget_profile(*, settings: LLMCurriculumSettings, metrics_summary: Mapping[str, Any]) -> str:
    """Return a conservative fallback budget profile from latest concrete status."""
    status = str(metrics_summary.get("failure_overall_status") or metrics_summary.get("status") or "")
    if status == "passed":
        return settings.proposal_fallback.ready_stage_budget_profile
    return settings.proposal_fallback.default_stage_budget_profile


def _log_proposal_fallback(
    *,
    logger: llm.logging.ProposalEventLogger,
    settings: LLMCurriculumSettings,
    stage_index: int,
    fallback_proposal: Mapping[str, Any],
    stage: LLMCurriculumStage,
) -> None:
    """Append one explicit safe-fallback event to the proposal log."""
    task = dict(fallback_proposal["task"])
    logger.append(
        {
            "event_type": "llm_proposal_fallback",
            "curriculum_name": settings.curriculum_name,
            "stage_index": stage_index,
            "status": "accepted",
            "proposal_type": llm.task_schema.PROPOSAL_KIND_TASK_DISTRIBUTION,
            "task_distribution_reference": {
                llm.task_schema.TASK_DISTRIBUTION_ID_FIELD: task.get(llm.task_schema.TASK_DISTRIBUTION_ID_FIELD),
                llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD: task.get(llm.task_schema.TASK_DISTRIBUTION_CONFIG_PATH_FIELD),
            },
            "stage_name": stage.stage_name,
            "original_proposal": fallback_proposal.get("original_proposal"),
            "accepted_task": dict(stage.task),
            "task_reason": fallback_proposal.get("task_reason"),
            "stage_budget_profile": fallback_proposal.get("stage_budget_profile"),
            "budget_rationale": fallback_proposal.get("budget_rationale"),
            "proposal_fallback_used": True,
            "proposal_failure_reason": fallback_proposal.get("proposal_failure_reason"),
            "fallback_resolution_rejections": fallback_proposal.get("fallback_resolution_rejections"),
            "fallback_distribution_attempts": fallback_proposal.get("fallback_distribution_attempts"),
            "fallback_original_distribution_id": fallback_proposal.get("fallback_original_distribution_id"),
            "fallback_final_distribution_id": fallback_proposal.get("fallback_final_distribution_id"),
            "llm_request_failed_due_to_context_size": fallback_proposal.get("llm_request_failed_due_to_context_size", False),
            "llm_context_fallback_used": fallback_proposal.get("llm_context_fallback_used", False),
            "llm_context_fallback_reason": fallback_proposal.get("llm_context_fallback_reason"),
            **_stage_proposal_metadata(stage),
        }
    )


def _budget_prompt_context(settings: LLMCurriculumSettings, stage_index: int, cumulative_timesteps: int) -> dict[str, Any]:
    """Return bounded budget context embedded in LLM proposal prompts."""
    budget_settings = settings.llm_stage_budget
    allowed_profile_names = _allowed_budget_profile_names(budget_settings, stage_index)
    return {
        "enabled": budget_settings.enabled,
        "allowed_profile_names": list(allowed_profile_names),
        "allowed_profiles": {name: {"total_timesteps": budget_settings.profiles[name]} for name in allowed_profile_names},
        "default_profile": _default_stage_budget_profile(settings, stage_index),
        "min_stage_timesteps": budget_settings.min_stage_timesteps,
        "max_stage_timesteps": budget_settings.max_stage_timesteps,
        "total_budget_cap_timesteps": budget_settings.total_budget_cap_timesteps,
        "cumulative_llm_budget_timesteps": cumulative_timesteps,
        "next_stage_index": stage_index,
        "remaining_stage_slots_including_current": max(settings.max_stages - stage_index + 1, 0),
        "guidance": {
            "bootstrap": "stage 1 policy warmup only",
            "short": "easy confirmation stage",
            "normal": "ordinary progression",
            "recovery": "previous stage unstable but promising",
            "extend": "use sparingly when appropriate but undertrained",
            "forbidden": ["raw timesteps", "num_envs", "PPO hyperparameters", "action interface", "reward logic"],
        },
    }


def _log_stage_budget_decision(
    logger: llm.logging.ProposalEventLogger,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
) -> None:
    """Append one deterministic stage budget decision event to the proposal log."""
    logger.append(
        {
            "event_type": "llm_stage_budget_decision",
            "curriculum_name": settings.curriculum_name,
            "stage_index": stage_index,
            "stage_name": stage.stage_name,
            **_stage_run_metadata(stage),
        }
    )


def _stage_run_metadata(stage: LLMCurriculumStage) -> dict[str, Any]:
    """Return JSON-ready LLM proposal and budget metadata for a stage."""
    return {
        **_stage_budget_metadata(stage),
        **_stage_proposal_metadata(stage),
    }


def _stage_budget_metadata(stage: LLMCurriculumStage) -> dict[str, Any]:
    """Return JSON-ready budget metadata for a resolved LLM curriculum stage."""
    return {
        "stage_budget_profile": stage.selected_stage_budget_profile,
        "requested_stage_budget_profile": stage.requested_stage_budget_profile,
        "selected_stage_budget_profile": stage.selected_stage_budget_profile,
        "stage_total_timesteps": stage.total_timesteps,
        "cumulative_llm_budget_timesteps": stage.cumulative_llm_budget_timesteps,
        "llm_budget_cap_timesteps": stage.llm_budget_cap_timesteps,
        "budget_was_clipped": stage.budget_was_clipped,
        "budget_fallback_reason": stage.budget_fallback_reason,
        "budget_rationale": stage.budget_rationale,
        "bootstrap_stage_source": stage.bootstrap_stage_source,
        "bootstrap_task_shape": stage.bootstrap_task_shape,
        "bootstrap_target_sampling_bounds": stage.bootstrap_target_sampling_bounds,
    }


def _stage_proposal_metadata(stage: LLMCurriculumStage) -> dict[str, Any]:
    """Return JSON-ready proposal audit metadata for a resolved stage."""
    return {
        "proposal_type": stage.proposal_type,
        "original_proposal": stage.original_proposal,
        "accepted_task": dict(stage.task),
        "stage_display_name": stage.stage_name,
        "task_distribution_reference": stage.task_distribution_reference,
        "resolved_task": stage.resolved_task or dict(stage.task),
        "resolved_task_shape": stage.resolved_task_shape or stage.task_shape,
        "resolved_task_sample_metadata": stage.resolved_task_sample_metadata,
        "proposal_fallback_used": stage.proposal_fallback_used,
        "proposal_failure_reason": stage.proposal_failure_reason,
        "previous_stage_task_shape": stage.previous_stage_task_shape,
        "requested_stage_task_shape": stage.requested_stage_task_shape,
        "accepted_stage_task_shape": stage.accepted_stage_task_shape,
        "duplicate_task_rejected": stage.duplicate_task_rejected,
        "duplicate_task_repair_reason": stage.duplicate_task_repair_reason,
        "proposal_repaired": stage.proposal_repaired,
        "proposal_repair_reason": stage.proposal_repair_reason,
        "proposal_original_distribution_id": stage.proposal_original_distribution_id,
        "proposal_final_distribution_id": stage.proposal_final_distribution_id,
        "proposal_progression_rule_applied": stage.proposal_progression_rule_applied,
        "hover_vertical_loop_detected": stage.hover_vertical_loop_detected,
        "stage_progression_bucket": stage.stage_progression_bucket,
        "fallback_task_shape": stage.fallback_task_shape,
    }


def _stage_wandb_tags(settings: LLMCurriculumSettings, stage_index: int, stage: LLMCurriculumStage) -> tuple[str, ...]:
    """Return caller-owned W&B tags for one LLM curriculum stage."""
    tags = (
        f"stage_index:{stage_index}",
        f"stage:{stage.stage_name}",
        f"task:{stage.task_shape}",
        f"llm_provider:{settings.llm_provider}",
        f"llm_fallback:{str(stage.proposal_fallback_used).lower()}",
    )
    if stage.selected_stage_budget_profile is None:
        return tags
    return (*tags, f"llm_budget_profile:{stage.selected_stage_budget_profile}")


def _stage_training_run_metadata(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    run_name: str,
    curriculum_run_name: str,
    previous_model_path: str | None,
) -> dict[str, Any]:
    """Return identity metadata copied into stage metrics, manifests, and W&B config."""
    return {
        "run_type": "training",
        "run_kind": "curriculum_stage",
        "curriculum_kind": LLM_CURRICULUM_KIND,
        "curriculum_run_name": curriculum_run_name,
        "curriculum_stage_index": stage_index,
        "curriculum_stage_name": stage.stage_name,
        "curriculum_stage_count": settings.max_stages,
        "curriculum_stage_run_name": run_name,
        "source_config_path": str(settings.base_training_config),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "model_transfer_enabled": previous_model_path is not None,
        "previous_stage_model_path": previous_model_path,
        **_stage_run_metadata(stage),
    }


def _budget_profile_counts(stage_entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count selected LLM budget profiles in accepted stage entries."""
    counts: dict[str, int] = {}
    for stage in stage_entries:
        profile = stage.get("selected_stage_budget_profile")
        if profile is None:
            continue
        profile_name = str(profile)
        counts[profile_name] = counts.get(profile_name, 0) + 1
    return counts


def _proposal_fallback_summary(settings: LLMProposalFallbackSettings) -> dict[str, Any]:
    """Return sanitized fallback settings for summaries and manifests."""
    return {
        "enabled": settings.enabled,
        "task_distribution_id": settings.task_distribution_id,
        "task_distribution_config_path": llm.task_schema.KNOWN_TASK_DISTRIBUTION_CONFIGS[settings.task_distribution_id],
        "default_stage_budget_profile": settings.default_stage_budget_profile,
        "ready_stage_budget_profile": settings.ready_stage_budget_profile,
    }


def _llm_stage_budget_summary(settings: LLMStageBudgetSettings) -> dict[str, Any]:
    """Return sanitized budget profile settings for summaries and manifests."""
    return {
        "enabled": settings.enabled,
        "total_budget_cap_timesteps": settings.total_budget_cap_timesteps,
        "default_profile": settings.default_profile,
        "min_stage_timesteps": settings.min_stage_timesteps,
        "max_stage_timesteps": settings.max_stage_timesteps,
        "profiles": {name: {"total_timesteps": timesteps} for name, timesteps in settings.profiles.items()},
    }


def _accepted_task_context(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact accepted-task context for future LLM prompts."""
    return {
        "stage_index": entry.get("stage_index"),
        "stage_name": entry.get("stage_name"),
        "task_shape": entry.get("task_shape"),
        "accepted_task_family": _accepted_stage_family(entry),
        "task": entry.get("task"),
        "task_reason": entry.get("task_reason"),
        "task_distribution_config_path": entry.get("task_distribution_config_path"),
        "task_distribution_id": entry.get("task_distribution_id"),
        "proposal_type": entry.get("proposal_type"),
        "task_distribution_reference": entry.get("task_distribution_reference"),
        "resolved_task_shape": entry.get("resolved_task_shape"),
        "resolved_task_sample_metadata": _compact_sample_metadata(entry.get("resolved_task_sample_metadata")),
        "previous_stage_task_shape": entry.get("previous_stage_task_shape"),
        "requested_stage_task_shape": entry.get("requested_stage_task_shape"),
        "accepted_stage_task_shape": entry.get("accepted_stage_task_shape"),
        "duplicate_task_rejected": entry.get("duplicate_task_rejected"),
        "duplicate_task_repair_reason": entry.get("duplicate_task_repair_reason"),
        "fallback_task_shape": entry.get("fallback_task_shape"),
        "proposal_fallback_used": entry.get("proposal_fallback_used"),
        "selected_stage_budget_profile": entry.get("selected_stage_budget_profile"),
        "stage_total_timesteps": entry.get("stage_total_timesteps"),
        "cumulative_llm_budget_timesteps": entry.get("cumulative_llm_budget_timesteps"),
        "budget_was_clipped": entry.get("budget_was_clipped"),
        "bootstrap_stage_source": entry.get("bootstrap_stage_source"),
        "bootstrap_task_shape": entry.get("bootstrap_task_shape"),
        "bootstrap_target_sampling_bounds": entry.get("bootstrap_target_sampling_bounds"),
        "metrics": _metrics_summary_from_entry(entry),
    }


def _llm_context_history(stage_entries: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Return compact full-history items for LLM proposal context."""
    return tuple(_llm_context_history_item(entry) for entry in stage_entries)


def _llm_context_history_item(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return one compact stage-history item for prompts."""
    sample_metadata = _compact_sample_metadata(entry.get("resolved_task_sample_metadata"))
    return {
        "stage_index": entry.get("stage_index"),
        "stage_name": entry.get("stage_name"),
        "accepted_task_family": _accepted_stage_family(entry),
        "task_shape": entry.get("task_shape"),
        "resolved_task_shape": entry.get("resolved_task_shape"),
        "task_distribution_id": entry.get("task_distribution_id"),
        "task_distribution_config_path": entry.get("task_distribution_config_path"),
        "proposal_type": entry.get("proposal_type"),
        "selected_stage_budget_profile": entry.get("selected_stage_budget_profile"),
        "stage_total_timesteps": entry.get("stage_total_timesteps"),
        "variation_strength": sample_metadata.get("task_distribution_strength"),
        "sample_on_reset": sample_metadata.get("task_distribution_sample_on_reset"),
        "family_weights": sample_metadata.get("task_distribution_family_weights"),
        "metrics": _metrics_summary_from_entry(entry),
        "feedback_summary": _feedback_summary_from_entry(entry),
    }


def _llm_context_summary(stage_entries: Sequence[Mapping[str, Any]], latest_metrics_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate LLM context that avoids readiness labels."""
    family_counts: dict[str, int] = {}
    family_errors: dict[str, list[float]] = {}
    position_errors: list[float] = []
    repeated_failure_modes: dict[str, int] = {}
    skill_gap_counts: dict[str, int] = {}
    feedback_summaries: list[dict[str, Any]] = []
    for entry in stage_entries:
        family = _accepted_stage_family(entry)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        error = entry.get("mean_position_error_tracking_m", entry.get("mean_position_error_m"))
        if isinstance(error, (int, float)):
            error_value = float(error)
            position_errors.append(error_value)
            if family:
                family_errors.setdefault(family, []).append(error_value)
        failure_mode = entry.get("failure_primary_mode")
        if failure_mode is not None and str(failure_mode).strip():
            key = str(failure_mode)
            repeated_failure_modes[key] = repeated_failure_modes.get(key, 0) + 1
        feedback_summary = _feedback_summary_from_entry(entry)
        if feedback_summary:
            feedback_summaries.append(feedback_summary)
            for skill_gap in feedback_summary.get("primary_skill_gaps", []):
                key = str(skill_gap)
                skill_gap_counts[key] = skill_gap_counts.get(key, 0) + 1

    ranked_families = _rank_task_families_by_error(family_errors)
    trend = _position_error_trend(position_errors)
    allowed_families = [
        family for family in envs.task_distribution.supported_task_families() if family != envs.task_distribution.FAMILY_BASIC_TRAINING_SHOW
    ]
    coverage_summary = _curriculum_coverage_summary(stage_entries)
    return {
        "completed_stage_count": len(stage_entries),
        "previous_stage_task_family": _accepted_stage_family(stage_entries[-1]) if stage_entries else None,
        "previous_stage_task_shape": stage_entries[-1].get("task_shape") if stage_entries else None,
        "accepted_task_family_counts": family_counts,
        "progression_coverage_summary": coverage_summary,
        "stage_progression_bucket_counts": coverage_summary["bucket_counts"],
        "hover_vertical_loop_detected": coverage_summary["hover_vertical_loop_detected"],
        "progression_score": coverage_summary["progression_score"],
        "last_accepted_task_family": _accepted_stage_family(stage_entries[-1]) if stage_entries else None,
        "position_error_trend": trend,
        "trend_status": trend,
        "recent_improvements": ["mean_position_error_tracking_m"] if trend == "improving" else [],
        "recent_regressions": ["mean_position_error_tracking_m"] if trend == "worsening" else [],
        "repeated_failure_modes": repeated_failure_modes,
        "top_repeated_failure_modes": _top_repeated_failure_modes(repeated_failure_modes),
        "strongest_task_families": ranked_families[:3],
        "weakest_task_families": list(reversed(ranked_families[-3:])),
        "latest_metrics_summary": dict(latest_metrics_summary),
        "latest_curriculum_feedback_summary": latest_metrics_summary.get("curriculum_feedback_summary"),
        "latest_recommended_next_task_families": list(latest_metrics_summary.get("curriculum_recommended_next_task_families") or []),
        "previous_feedback_summaries": feedback_summaries[-3:],
        "skill_gap_counts": skill_gap_counts,
        "recommended_avoid_immediate_duplicate_family": True,
        "allowed_task_families": allowed_families,
        "task_families_with_bounded_variation_support": allowed_families,
        "task_families_without_bounded_variation_support": [],
        "own_task_status": "tracked_separately_from_llm_prompt" if latest_metrics_summary else "unknown",
        "generalization_status": "tracked_separately_from_llm_prompt",
        "scenario_status": "stress_test_not_readiness_gate",
        "readiness_level_omitted_from_llm_context": True,
        "diagnostic_guidance": {
            "prefer_metrics_over_readiness_label": True,
            "do_not_overreact_to_single_failure_mode": True,
            "prefer_targeted_skill_training_over_default_hover": True,
            "hover_and_vertical_are_early_or_recovery_tools_only": True,
            "after_vertical_altitude_gaps_prefer_angled_delayed_or_multi_height_paths": True,
            "avoid_hover_vertical_loops": True,
            "action_saturation": "treat_as_task_difficulty_signal_unless_crash_or_divergence_confirms_instability",
            "z_instability": "pure_vertical_only_for_early_or_recovery_then_angled_delayed_or_multi_height",
            "xy_tracking": "consider_shorter_slower_line_or_start_hold_then_line",
            "turn_following": "consider_slow_l_shape_or_polyline",
            "curvature_following": "consider_gentle_ellipse_or_slow_circle_before_figure_eight",
            "reference_too_fast": "treat_as_task_difficulty_signal_unless_crash_or_divergence_confirms_instability",
            "reference_too_fast_or_too_hard": "choose_easier_or_slower_same_family_variant",
            "accepted_concrete_tasks_are_materialized_as_bounded_distributions": True,
            "hard_scenarios_are_stress_tests_not_whole_policy_readiness_labels": True,
        },
    }


def _feedback_summary_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact structured feedback fields for LLM prompt history."""
    strategy = entry.get("curriculum_strategy")
    strategy_payload = dict(strategy) if isinstance(strategy, Mapping) else {}
    feedback_summary = {
        "readiness_level_omitted": True,
        "llm_instruction_summary": entry.get("curriculum_feedback_summary"),
        "current_task_family": entry.get("curriculum_current_task_family"),
        "current_difficulty_level": entry.get("curriculum_current_difficulty_level"),
        "primary_skill_gaps": list(entry.get("curriculum_primary_skill_gaps") or []),
        "diagnostic_signals": dict(entry.get("curriculum_diagnostic_signals") or {})
        if isinstance(entry.get("curriculum_diagnostic_signals"), Mapping)
        else {},
        "strategy": strategy_payload,
        "recommended_next_task_families": list(entry.get("curriculum_recommended_next_task_families") or []),
        "avoid_next_task_families": list(entry.get("curriculum_avoid_next_task_families") or []),
    }
    has_feedback = any(
        bool(feedback_summary[key])
        for key in (
            "llm_instruction_summary",
            "current_task_family",
            "primary_skill_gaps",
            "diagnostic_signals",
            "strategy",
            "recommended_next_task_families",
            "avoid_next_task_families",
        )
    )
    return feedback_summary if has_feedback else {}


def _rank_task_families_by_error(family_errors: Mapping[str, Sequence[float]]) -> list[str]:
    """Return task families ordered from lowest to highest recent tracking error."""
    means = []
    for family, errors in family_errors.items():
        if errors:
            means.append((sum(float(error) for error in errors) / len(errors), family))
    return [family for _, family in sorted(means)]


def _top_repeated_failure_modes(failure_modes: Mapping[str, int]) -> list[dict[str, Any]]:
    """Return the top repeated failure modes as compact prompt records."""
    ranked = sorted(((int(count), str(mode)) for mode, count in failure_modes.items()), reverse=True)
    return [{"failure_mode": mode, "count": count} for count, mode in ranked[:3]]


def _position_error_trend(errors: Sequence[float]) -> str:
    """Return a compact trend label from recent tracking errors."""
    if len(errors) < MIN_ERRORS_FOR_TREND:
        return "unknown"
    previous = errors[-2]
    current = errors[-1]
    tolerance = max(1.0e-6, abs(previous) * 0.05)
    if current < previous - tolerance:
        return "improving"
    if current > previous + tolerance:
        return "worsening"
    return "flat"


def _accepted_stage_family(entry: Mapping[str, Any]) -> str | None:
    """Return the best available task-family label for an accepted stage."""
    metadata = entry.get("resolved_task_sample_metadata")
    if isinstance(metadata, Mapping):
        family = metadata.get("task_distribution_sampled_family") or metadata.get("task_distribution_name")
        if family is not None and str(family).strip():
            return str(family)
    task = entry.get("resolved_task") or entry.get("task")
    if isinstance(task, Mapping):
        task_family = task.get("task_family") or task.get(validation.contracts.FIELD_SHAPE)
        if task_family is not None and str(task_family).strip():
            return str(task_family)
    shape = entry.get("resolved_task_shape") or entry.get("task_shape") or entry.get("accepted_stage_task_shape")
    return str(shape) if shape is not None and str(shape).strip() else None


def _compact_sample_metadata(value: Any) -> dict[str, Any]:
    """Return prompt-safe sample metadata without embedding full sampled task payloads."""
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "task_distribution_name",
        "task_distribution_mode",
        "task_distribution_strength",
        "task_distribution_sample_on_reset",
        "task_distribution_config_path",
        "task_distribution_family_weights",
        "task_distribution_sampled_family",
        "task_distribution_sampled_task_shape",
        "task_distribution_sample_index",
    )
    return {key: value.get(key) for key in keys if key in value}


def _result_last_model_path(result: ppo_tracking.PPOTrackingSmokeResult) -> str:
    """Return the last saved model path from a PPO result."""
    return result.last_model_path or result.model_path


def _preferred_result_model_path(result: ppo_tracking.PPOTrackingSmokeResult) -> str:
    """Return the preferred model path for LLM curriculum transfer."""
    return result.best_model_path or result.last_model_path or result.model_path


def _preferred_result_model_source(result: ppo_tracking.PPOTrackingSmokeResult) -> str:
    """Return whether a PPO result selected best or last for transfer."""
    return "best" if result.best_model_path else "last"


def _stage_selected_model_path(stage: Mapping[str, Any] | None) -> Any:
    """Return the preferred model path from a stage summary."""
    if stage is None:
        return None
    return stage.get("best_model_path") or stage.get("last_model_path") or stage.get("model_path")


def _stage_selected_model_path_relative(stage: Mapping[str, Any] | None) -> Any:
    """Return the preferred relative model path from a stage summary."""
    if stage is None:
        return None
    return stage.get("best_model_path_relative") or stage.get("last_model_path_relative") or stage.get("model_path_relative")


def _stage_selected_model_source(stage: Mapping[str, Any] | None) -> str | None:
    """Return whether a stage exposes a best model or falls back to last."""
    if stage is None or not _stage_selected_model_path(stage):
        return None
    return "best" if stage.get("best_model_path") else "last"


def _metrics_summary_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact metrics used in subsequent proposal prompts."""
    keys = (
        "stage_index",
        "stage_name",
        "task_shape",
        "task_distribution_config_path",
        "task_distribution_id",
        "proposal_type",
        "task_distribution_reference",
        "resolved_task_shape",
        "proposal_fallback_used",
        "proposal_failure_reason",
        "previous_stage_task_shape",
        "requested_stage_task_shape",
        "accepted_stage_task_shape",
        "duplicate_task_rejected",
        "duplicate_task_repair_reason",
        "proposal_repaired",
        "proposal_repair_reason",
        "proposal_original_distribution_id",
        "proposal_final_distribution_id",
        "proposal_progression_rule_applied",
        "hover_vertical_loop_detected",
        "stage_progression_bucket",
        "fallback_task_shape",
        "task_distribution_mode",
        "task_distribution_strength",
        "selected_stage_budget_profile",
        "stage_total_timesteps",
        "cumulative_llm_budget_timesteps",
        "llm_budget_cap_timesteps",
        "budget_was_clipped",
        "budget_fallback_reason",
        "bootstrap_stage_source",
        "bootstrap_task_shape",
        "bootstrap_target_sampling_bounds",
        "dry_run_proposals",
        "validation_status",
        "mean_position_error_m",
        "mean_position_error_tracking_m",
        "final_position_error_m",
        "max_position_error_m",
        "xy_tracking_ratio",
        "failure_overall_status",
        "failure_primary_mode",
        "curriculum_recommended_next_tasks",
        "curriculum_avoid_next_tasks",
        "curriculum_feedback_version",
        "curriculum_feedback_summary",
        "curriculum_current_task_family",
        "curriculum_current_difficulty_level",
        "curriculum_primary_skill_gaps",
        "curriculum_diagnostic_signals",
        "curriculum_strategy",
        "curriculum_recommended_next_task_families",
        "curriculum_avoid_next_task_families",
        "curriculum_constraints_for_next",
    )
    summary = {key: entry.get(key) for key in keys if key in entry}
    summary["readiness_level_omitted_from_llm_context"] = True
    summary["diagnostic_interpretation"] = _diagnostic_interpretation_from_entry(entry)
    return summary


def _diagnostic_interpretation_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return concrete diagnostic interpretation hints without readiness labels."""
    failure_mode = str(entry.get("failure_primary_mode") or "")
    feedback_summary = _feedback_summary_from_entry(entry)
    return {
        "failure_primary_mode": failure_mode or None,
        "action_saturation_is_difficulty_signal": "action_saturation" in failure_mode,
        "reference_too_fast_is_difficulty_signal": "reference_too_fast" in failure_mode,
        "do_not_treat_difficulty_signals_as_automatic_instability": True,
        "primary_skill_gaps": list(feedback_summary.get("primary_skill_gaps", [])),
        "recommended_next_task_families": list(feedback_summary.get("recommended_next_task_families", [])),
        "curriculum_feedback_summary": feedback_summary.get("llm_instruction_summary"),
    }


def _final_stage_summary(final_stage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the compact final-stage manifest entry."""
    if final_stage is None:
        return None
    return {
        "stage_index": final_stage.get("stage_index"),
        "stage_name": final_stage.get("stage_name"),
        "run_name": final_stage.get("run_name"),
        "model_path": _stage_selected_model_path(final_stage),
        "model_path_relative": _stage_selected_model_path_relative(final_stage),
        "last_model_path": final_stage.get("last_model_path") or final_stage.get("model_path"),
        "last_model_path_relative": final_stage.get("last_model_path_relative") or final_stage.get("model_path_relative"),
        "best_model_path": final_stage.get("best_model_path"),
        "best_model_path_relative": final_stage.get("best_model_path_relative"),
        "best_model_metric": final_stage.get("best_model_metric"),
        "best_model_step": final_stage.get("best_model_step"),
        "best_model_source": final_stage.get("best_model_source"),
        "selected_model_source": _stage_selected_model_source(final_stage),
        "manifest_path": final_stage.get("manifest_path"),
        "manifest_path_relative": final_stage.get("manifest_path_relative"),
        "task_distribution_config_path": final_stage.get("task_distribution_config_path"),
        "task_distribution_id": final_stage.get("task_distribution_id"),
        **{key: final_stage.get(key) for key in _stage_run_metadata_keys()},
    }


def _stage_run_metadata_keys() -> tuple[str, ...]:
    """Return stable stage metadata keys used in compact summaries."""
    return (
        *_stage_budget_metadata_keys(),
        *_stage_proposal_metadata_keys(),
    )


def _stage_budget_metadata_keys() -> tuple[str, ...]:
    """Return stable stage budget metadata keys used in compact summaries."""
    return (
        "stage_budget_profile",
        "requested_stage_budget_profile",
        "selected_stage_budget_profile",
        "stage_total_timesteps",
        "cumulative_llm_budget_timesteps",
        "llm_budget_cap_timesteps",
        "budget_was_clipped",
        "budget_fallback_reason",
        "budget_rationale",
        "bootstrap_stage_source",
        "bootstrap_task_shape",
        "bootstrap_target_sampling_bounds",
    )


def _stage_proposal_metadata_keys() -> tuple[str, ...]:
    """Return stable proposal audit metadata keys used in compact summaries."""
    return (
        "proposal_type",
        "original_proposal",
        "accepted_task",
        "stage_display_name",
        "task_distribution_reference",
        "resolved_task",
        "resolved_task_shape",
        "resolved_task_sample_metadata",
        "proposal_fallback_used",
        "proposal_failure_reason",
        "previous_stage_task_shape",
        "requested_stage_task_shape",
        "accepted_stage_task_shape",
        "duplicate_task_rejected",
        "duplicate_task_repair_reason",
        "fallback_task_shape",
    )


def _evaluation_index_manifest(run_name: str) -> dict[str, Any]:
    """Return the run-manifest link to the deterministic evaluation index."""
    index_path = utils.artifacts.get_run_evaluation_index_path(run_name)
    entries: list[Any] = []
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        raw_entries = payload.get("evaluations") if isinstance(payload, dict) else None
        entries = raw_entries if isinstance(raw_entries, list) else []
    return {
        "path": str(index_path),
        "path_relative": utils.artifacts.path_relative_to_run(index_path, run_name),
        "entry_count": len(entries),
        "evaluations": entries,
    }


def _bootstrap_snapshot(stage: LLMCurriculumStage | None) -> dict[str, Any]:
    """Return sanitized bootstrap settings for config snapshots."""
    if stage is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "total_timesteps": stage.total_timesteps,
        "eval_steps": stage.eval_steps,
        "reason": stage.task_reason,
        "notes": stage.notes,
        "task": stage.task,
        "task_distribution_config_path": None if stage.task_distribution_config_path is None else str(stage.task_distribution_config_path),
        "task_distribution_id": stage.task_distribution_id,
        **_stage_run_metadata(stage),
    }


def _sanitized_llm_config(llm_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return provider config safe for manifests and config snapshots."""
    sanitized = dict(llm_config)
    if sanitized.get("api_key"):
        sanitized["api_key"] = "[REDACTED]"
    return sanitized


def _mapping_or_empty(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping config section or an empty mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping"
        raise TypeError(message)
    return value


def _curriculum_artifact_run_name(curriculum_name: str, seed: int, curriculum_kind: str = LLM_CURRICULUM_KIND) -> str:
    """Return the self-describing storage run name used for curriculum-level artifacts."""
    if curriculum_kind == LLM_CURRICULUM_KIND and curriculum_name.startswith("llm_curriculum_"):
        return f"{curriculum_name}_seed{seed}"
    topic = _curriculum_run_topic(curriculum_name, curriculum_kind)
    return f"curriculum_{curriculum_kind}_{topic}_seed{seed}"


def _curriculum_run_topic(curriculum_name: str, curriculum_kind: str) -> str:
    """Return the curriculum topic without a duplicated curriculum/kind prefix."""
    canonical_prefix = f"curriculum_{curriculum_kind}_"
    if curriculum_name.startswith(canonical_prefix):
        return curriculum_name[len(canonical_prefix) :]
    kind_prefix = f"{curriculum_kind}_"
    if curriculum_name.startswith(kind_prefix):
        return curriculum_name[len(kind_prefix) :]
    return curriculum_name


def _curriculum_wandb_group(curriculum_run_name: str) -> str:
    """Return the W&B group used for all stages in one LLM curriculum."""
    if curriculum_run_name.startswith("llm_curriculum_"):
        return curriculum_run_name
    return f"curriculum/llm/{curriculum_run_name}"


def _to_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize a small config payload to YAML with the project dependency."""
    return yaml.safe_dump(dict(payload), sort_keys=False)


__all__ = [
    "DEFAULT_LLM_CURRICULUM_CONFIG_PATH",
    "LLMCurriculumResult",
    "LLMCurriculumSettings",
    "LLMCurriculumStage",
    "LLMProposalFallbackSettings",
    "LLMStageBudgetSettings",
    "derive_stage_run_name",
    "llm_curriculum_settings_from_mapping",
    "load_llm_curriculum_settings",
    "run_llm_curriculum_training",
    "run_llm_curriculum_training_from_config",
    "validate_llm_curriculum",
]
