"""
Experiment command-line entry points.

Executable modules:
- experiments_cli_train_tracking: Train a PPO trajectory-tracking policy
- experiments_cli_train_curriculum: Train a fixed manual PPO curriculum
- experiments_cli_evaluate_curriculum: Evaluate curriculum checkpoints
- experiments_cli_render_policy: Render a trained policy or scripted baseline
- experiments_cli_render_scenario: Render a continuous multi-phase scenario
- experiments_cli_render_smoke: Run a tiny render integration smoke
- experiments_cli_training_smoke: Run a deterministic training smoke loop
- experiments_cli_mvp: Run or print the legacy MVP smoke sequence
"""

__all__: list[str] = []
