"""
Drone environment builders and wrappers.

Provides:
- actions: action-interface contracts for trajectory tracking
- builders: minimal constructors for PyBullet drone environments
- initial_state: initial drone spawn-position configuration helpers
- task_adapter: validated trajectory task reference packaging helpers
- termination: termination and diagnostic safety limit configuration
- task_distribution: fixed and randomized task-distribution sampling helpers
- tracking_env: Gymnasium-compatible trajectory tracking environment wrapper
- tracking_reward: deterministic MVP trajectory-tracking reward helpers
"""

from . import envs_actions as actions
from . import envs_builders as builders
from . import envs_initial_state as initial_state
from . import envs_task_adapter as task_adapter
from . import envs_task_distribution as task_distribution
from . import envs_termination as termination
from . import envs_tracking_env as tracking_env
from . import envs_tracking_reward as tracking_reward

__all__ = [
    "actions",
    "builders",
    "initial_state",
    "task_adapter",
    "task_distribution",
    "termination",
    "tracking_env",
    "tracking_reward",
]
