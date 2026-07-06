"""
Experiment orchestration utilities and command-line entry points.

Provides:
- config: minimal experiment configuration loading helpers
- curriculum: curriculum task summarization helpers
- training_smoke: tiny deterministic MVP training-smoke helpers
- render_smoke: tiny headless drone render-smoke helpers
- ppo_tracking: tiny Stable-Baselines3 PPO trajectory-tracking smoke helpers
- policy_render: trained PPO rollout rendering helpers with external camera capture
- scenario_render: continuous multi-phase scenario rendering helpers
"""

from . import experiments_config as config
from . import experiments_curriculum as curriculum
from . import experiments_policy_render as policy_render
from . import experiments_ppo_tracking as ppo_tracking
from . import experiments_render_smoke as render_smoke
from . import experiments_scenario_render as scenario_render
from . import experiments_training_smoke as training_smoke

__all__ = [
    "config",
    "curriculum",
    "policy_render",
    "ppo_tracking",
    "render_smoke",
    "scenario_render",
    "training_smoke",
]
