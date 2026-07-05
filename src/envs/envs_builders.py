"""
===============================================================================
envs_builders.py
===============================================================================
Construct minimal drone simulation environments for smoke tests and early MVPs.

Responsibilities:
  - Keep PyBullet drone environment setup in one importable module
  - Provide a headless hover environment constructor for short deterministic tests
  - Avoid training loops, reward shaping, or experiment orchestration

Design principles:
  - Prefer the upstream gym-pybullet-drones API for the first integration check
  - Keep heavy PyBullet imports lazy so pure modules remain cheap to import

Boundaries:
  - Trajectory-tracking wrappers belong in later environment modules
  - Stable-Baselines3 training code belongs in experiments modules
===============================================================================

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym


def make_hover_aviary_env(gui: bool = False, record: bool = False) -> gym.Env:
    """
    Build the upstream single-drone hover environment for smoke testing.

    Parameters
    ----------
    gui
        Whether to open the PyBullet GUI.
    record
        Whether the upstream environment should record frames.

    Returns
    -------
    gym.Env
        Gymnasium-compatible hover environment using kinematic observations and one-dimensional RPM actions.

    """
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary  # noqa: PLC0415
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType  # noqa: PLC0415

    return HoverAviary(gui=gui, record=record, obs=ObservationType.KIN, act=ActionType.ONE_D_RPM)
