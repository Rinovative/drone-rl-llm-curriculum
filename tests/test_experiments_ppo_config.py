"""Tests for explicit PPO hyperparameter config validation."""

# ruff: noqa: S101

from __future__ import annotations

import pytest

from src.experiments.training import experiments_training_ppo_config as ppo_config

CONFIGURED_N_STEPS = 12
CONFIGURED_BATCH_SIZE = 6
VECTOR_TEST_N_STEPS = 8
VECTOR_TEST_NUM_ENVS = 2
VECTOR_TEST_BATCH_SIZE = 16
VECTOR_TEST_OVERSIZED_BATCH_SIZE = 17
VECTOR_TEST_EFFECTIVE_ROLLOUT_STEPS = 16
POLICY_NET_ARCH = [128, 128]
POLICY_PI_NET_ARCH = [128, 64]
POLICY_VF_NET_ARCH = [64, 64]

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
    """Verify nested ppo config values include the project default net256 architecture."""
    config = ppo_config.load_ppo_config_from_mapping({"ppo": VALID_PPO_CONFIG})
    expected = {**VALID_PPO_CONFIG, "policy_kwargs": ppo_config.default_policy_kwargs()}

    assert config.to_dict() == expected
    assert config.to_sb3_kwargs() == expected


def test_ppo_config_uses_net256_policy_kwargs_by_default() -> None:
    """Verify omitted policy kwargs use the project net256 pi/vf architecture."""
    config = ppo_config.PPOConfig()

    assert config.policy_kwargs == ppo_config.default_policy_kwargs()
    assert config.to_dict()["policy_kwargs"] == ppo_config.default_policy_kwargs()
    assert config.to_sb3_kwargs()["policy_kwargs"] == ppo_config.default_policy_kwargs()


def test_ppo_config_accepts_list_net_arch_policy_kwargs() -> None:
    """Verify a shared MLP net_arch list is accepted and passed to SB3."""
    config = ppo_config.PPOConfig(policy_kwargs={"net_arch": POLICY_NET_ARCH})

    assert config.policy_kwargs == {"net_arch": POLICY_NET_ARCH}
    assert config.to_dict()["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert config.to_sb3_kwargs()["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}


def test_ppo_config_accepts_pi_vf_net_arch_policy_kwargs() -> None:
    """Verify separate policy and value network architectures are accepted."""
    config = ppo_config.PPOConfig(
        policy_kwargs={"net_arch": {"pi": POLICY_PI_NET_ARCH, "vf": POLICY_VF_NET_ARCH}},
    )

    assert config.policy_kwargs == {"net_arch": {"pi": POLICY_PI_NET_ARCH, "vf": POLICY_VF_NET_ARCH}}
    assert config.to_sb3_kwargs()["policy_kwargs"] == {"net_arch": {"pi": POLICY_PI_NET_ARCH, "vf": POLICY_VF_NET_ARCH}}


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, "ppo config section is required"),
        ({"learning_rate": 0.0005, "n_steps": 16, "batch_size": 8}, "top-level PPO keys are not supported"),
        ({"ppo": VALID_PPO_CONFIG, "learning_rate": 0.0005}, "top-level PPO keys are not supported"),
    ],
)
def test_ppo_config_rejects_missing_or_flat_keys(config: dict[str, object], match: str) -> None:
    """Verify PPO settings must be provided only through the nested ppo section."""
    with pytest.raises(ValueError, match=match):
        ppo_config.load_ppo_config_from_mapping(config)


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


@pytest.mark.parametrize(
    ("policy_kwargs", "match"),
    [
        (["not", "a", "mapping"], "ppo.policy_kwargs must be a mapping"),
        ({}, "ppo.policy_kwargs must define net_arch"),
        ({"activation_fn": "relu"}, "unsupported keys: activation_fn"),
        ({"net_arch": []}, "ppo.policy_kwargs.net_arch must be a non-empty list"),
        ({"net_arch": [128, 0]}, r"ppo.policy_kwargs.net_arch\[\]"),
        ({"net_arch": [True]}, r"ppo.policy_kwargs.net_arch\[\]"),
        ({"net_arch": {"pi": [64]}}, "must define pi and vf lists"),
        ({"net_arch": {"pi": [64], "vf": [64], "qf": [64]}}, "unsupported keys: qf"),
        ({"net_arch": {"pi": [64], "vf": []}}, r"ppo.policy_kwargs.net_arch.vf"),
    ],
)
def test_ppo_config_rejects_invalid_policy_kwargs(policy_kwargs: object, match: str) -> None:
    """Verify unsupported or malformed policy kwargs fail with clear errors."""
    with pytest.raises(ValueError, match=match):
        ppo_config.PPOConfig(policy_kwargs=policy_kwargs)  # type: ignore[arg-type]


def test_ppo_config_validates_total_timesteps_against_configured_rollout() -> None:
    """Verify tiny training budgets must make rollout size explicit."""
    config = ppo_config.PPOConfig(n_steps=CONFIGURED_N_STEPS, batch_size=CONFIGURED_BATCH_SIZE)

    config.validate_total_timesteps(CONFIGURED_N_STEPS)
    with pytest.raises(ValueError, match=r"total_timesteps must be greater than or equal to ppo\.n_steps"):
        config.validate_total_timesteps(CONFIGURED_N_STEPS - 1)


def test_ppo_config_accepts_vectorized_rollout_batch_size() -> None:
    """Verify batch_size may exceed per-env n_steps when num_envs supplies enough rollout samples."""
    config = ppo_config.PPOConfig(n_steps=VECTOR_TEST_N_STEPS, batch_size=VECTOR_TEST_BATCH_SIZE)

    assert config.effective_rollout_steps(num_envs=VECTOR_TEST_NUM_ENVS) == VECTOR_TEST_EFFECTIVE_ROLLOUT_STEPS
    config.validate_rollout_consistency(num_envs=VECTOR_TEST_NUM_ENVS)


def test_ppo_config_rejects_batch_size_larger_than_effective_rollout() -> None:
    """Verify batch_size cannot exceed n_steps multiplied by num_envs."""
    config = ppo_config.PPOConfig(n_steps=VECTOR_TEST_N_STEPS, batch_size=VECTOR_TEST_OVERSIZED_BATCH_SIZE)

    with pytest.raises(ValueError, match=r"ppo\.batch_size must be less than or equal to ppo\.n_steps \* num_envs"):
        config.validate_rollout_consistency(num_envs=VECTOR_TEST_NUM_ENVS)
