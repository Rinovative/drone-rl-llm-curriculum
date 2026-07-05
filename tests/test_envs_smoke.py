"""Smoke tests for upstream drone environment construction."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np

from src import envs


def test_hover_aviary_resets_and_steps_once() -> None:
    """Verify the installed PyBullet drone environment can reset and step once."""
    env = envs.builders.make_hover_aviary_env(gui=False, record=False)
    try:
        obs, info = env.reset(seed=0)
        assert env.action_space.shape is not None
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        assert isinstance(info, dict)
        assert isinstance(next_info, dict)
        assert np.all(np.isfinite(np.asarray(obs, dtype=float)))
        assert np.all(np.isfinite(np.asarray(next_obs, dtype=float)))
        assert np.isfinite(float(reward))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
