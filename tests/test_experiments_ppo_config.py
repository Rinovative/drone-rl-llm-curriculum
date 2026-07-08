"""Tests for explicit PPO hyperparameter config validation."""

# ruff: noqa: S101

from __future__ import annotations

import pytest

from src.experiments.training import experiments_training_ppo_config as ppo_config

LEGACY_LEARNING_RATE = 0.0005
LEGACY_N_STEPS = 16
LEGACY_BATCH_SIZE = 8
CONFIGURED_N_STEPS = 12
CONFIGURED_BATCH_SIZE = 6

VALID_PPO_CONFIG = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "learning_rate": 0.0007,
    "gamma": 0.91,
    "gae_lambda": 0.82,
    "n_steps": 12,
    "batch_size": 6,
    "n_epochs": 3,
    "clip_range": 0.17,
    "ent_coef": 0.004,
    "vf_coef": 0.42,
    "max_grad_norm": 0.9,
    "target_kl": 0.07,
}


def test_ppo_config_loads_nested_config_values() -> None:
    """Verify nested ppo config values are resolved without hidden defaults."""
    config = ppo_config.load_ppo_config_from_mapping({"ppo": VALID_PPO_CONFIG})

    assert config.to_dict() == VALID_PPO_CONFIG
    assert config.to_sb3_kwargs() == VALID_PPO_CONFIG


def test_ppo_config_preserves_legacy_flat_keys() -> None:
    """Verify existing flat PPO keys remain a low-risk compatibility path."""
    config = ppo_config.load_ppo_config_from_mapping(
        {
            "learning_rate": LEGACY_LEARNING_RATE,
            "n_steps": LEGACY_N_STEPS,
            "batch_size": LEGACY_BATCH_SIZE,
        }
    )

    assert config.learning_rate == LEGACY_LEARNING_RATE
    assert config.n_steps == LEGACY_N_STEPS
    assert config.batch_size == LEGACY_BATCH_SIZE
    assert config.policy == "MlpPolicy"


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"policy": ""}, "ppo.policy"),
        ({"device": ""}, "ppo.device"),
        ({"learning_rate": 0.0}, "ppo.learning_rate"),
        ({"gamma": 1.1}, "ppo.gamma"),
        ({"gae_lambda": 0.0}, "ppo.gae_lambda"),
        ({"n_steps": 0}, "ppo.n_steps"),
        ({"batch_size": 0}, "ppo.batch_size"),
        ({"n_steps": 8, "batch_size": 16}, "ppo.batch_size"),
        ({"n_epochs": 0}, "ppo.n_epochs"),
        ({"clip_range": 0.0}, "ppo.clip_range"),
        ({"ent_coef": -0.1}, "ppo.ent_coef"),
        ({"vf_coef": -0.1}, "ppo.vf_coef"),
        ({"max_grad_norm": 0.0}, "ppo.max_grad_norm"),
        ({"target_kl": 0.0}, "ppo.target_kl"),
    ],
)
def test_ppo_config_rejects_invalid_values(updates: dict[str, object], match: str) -> None:
    """Verify invalid PPO hyperparameters fail with clear field names."""
    values = {**VALID_PPO_CONFIG, **updates}

    with pytest.raises(ValueError, match=match):
        ppo_config.PPOConfig(**values)


def test_ppo_config_rejects_non_mapping_section() -> None:
    """Verify ppo config sections must be YAML mappings."""
    with pytest.raises(ValueError, match="ppo config section must be a mapping"):
        ppo_config.load_ppo_config_from_mapping({"ppo": ["not", "a", "mapping"]})


def test_ppo_config_rejects_unknown_nested_keys() -> None:
    """Verify misspelled nested PPO keys fail instead of being ignored."""
    with pytest.raises(ValueError, match="unsupported keys: typo_learning_rate"):
        ppo_config.load_ppo_config_from_mapping({"ppo": {"typo_learning_rate": 0.001}})


def test_ppo_config_validates_total_timesteps_against_configured_rollout() -> None:
    """Verify tiny training budgets must make rollout size explicit."""
    config = ppo_config.PPOConfig(n_steps=CONFIGURED_N_STEPS, batch_size=CONFIGURED_BATCH_SIZE)

    config.validate_total_timesteps(CONFIGURED_N_STEPS)
    with pytest.raises(ValueError, match=r"total_timesteps must be greater than or equal to ppo\.n_steps"):
        config.validate_total_timesteps(CONFIGURED_N_STEPS - 1)
