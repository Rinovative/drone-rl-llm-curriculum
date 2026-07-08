# Drone RL with LLM-Guided Curriculum Learning  
### *Deep Reinforcement Learning Project – MSE Data Science, Autumn 2026*

**Master of Science in Engineering – Major Data Science**  
**Eastern Switzerland University of Applied Sciences (OST)**  
**Author:** Rino M. Albertin  

---

## 📌 Project Overview

This project investigates whether a large language model can act as an adaptive curriculum generator for reinforcement-learning-based quadrotor trajectory tracking.

The central idea is not to let an LLM directly control a drone. Instead, the LLM proposes structured training tasks, which are validated by deterministic safety and feasibility checks before they are used for reinforcement learning.

The core research question is:

> Can an LLM propose valid and useful training tasks that improve the learning process of a drone RL agent compared to a fixed manual curriculum or direct training on difficult tasks?

The repository provides a modular research pipeline covering:

<details>
<summary><strong>🧩 Drone simulation</strong></summary>

A simulated quadrotor environment based on `gym-pybullet-drones` and PyBullet.

The environment is used to train and evaluate a reinforcement-learning agent for trajectory tracking. The main task is single-drone reference tracking, while optional multi-drone scenes are treated as visualization and showcase extensions.

</details>

<details>
<summary><strong>🧭 Trajectory generation</strong></summary>

Parametric trajectory generators are used to create progressively more difficult reference paths.

Planned trajectory families include:

- hover targets
- straight lines
- circles
- figure-eight paths
- stars
- spirals
- optional formation-style paths for visual demonstrations

These trajectories define the tracking targets used during training and evaluation.

</details>

<details>
<summary><strong>🧪 Task validation</strong></summary>

LLM-generated tasks are not used directly. Each proposed task is checked by deterministic validation logic before it can enter the reinforcement-learning workflow.

The validation layer checks constraints such as:

- arena bounds
- maximum speed
- maximum acceleration
- minimum duration
- discontinuities or jumps
- valid parameter ranges
- optional minimum separation for multi-drone showcase scenes

</details>

<details>
<summary><strong>🧠 LLM-guided curriculum generation</strong></summary>

The LLM acts as an adaptive curriculum manager.

Based on recent learning performance and a compact task history, it proposes the next training task in a strict structured format. Invalid tasks are rejected or repaired before they are used.

The LLM is therefore not a controller. It is a task proposer that shapes the training curriculum.

</details>

<details>
<summary><strong>🎮 Reinforcement learning</strong></summary>

The agent is trained with Stable-Baselines3, with the initial focus on PPO.

The learned policy should track reference trajectories while minimizing tracking error, avoiding crashes, and keeping actions reasonably smooth.

</details>

<details>
<summary><strong>📊 Evaluation</strong></summary>

The project compares different training strategies using quantitative metrics and visualizations.

The main comparison is:

- direct training on difficult tasks
- fixed manual curriculum
- LLM-guided adaptive curriculum

Evaluation criteria include learning curves, tracking error, success rate, crash rate, action-cost proxies, curriculum progression, and invalid proposal statistics.

</details>

<details>
<summary><strong>📓 Notebook report</strong></summary>

The final submission artifact is an executable Jupyter notebook.

The notebook explains the motivation, environment, state/action/reward design, curriculum strategy, training setup, evaluation protocol, results, limitations, and outlook.

</details>

---

## 🎯 Research Scope

The main focus is **single-drone trajectory tracking with adaptive curriculum learning**.

Optional multi-drone scenes are treated as **visual showcase extensions**, not as the main learning problem. A trained policy may be applied to multiple drones with different reference trajectories, but the project does not aim to implement full multi-agent reinforcement learning.

The LLM is used only as a **curriculum manager**. It must not generate arbitrary executable Python code and must not directly control the drone during simulation.

---

## 🧠 Methodology

The project compares three training setups:

1. **Direct training**  
   The agent is trained directly on difficult target trajectories.

2. **Manual curriculum**  
   The agent follows a predefined progression from simple to harder tasks.

3. **LLM-guided adaptive curriculum**  
   The LLM proposes the next training task based on recent evaluation metrics and a compact training history.

Only tasks that pass deterministic validation are accepted for training or evaluation.

---

## ⚙️ Local / HPC Execution

<details>
<summary><strong>Docker workflow</strong></summary>

1. Clone the repository:

```bash
git clone https://github.com/Rinovative/drone-rl-llm-curriculum.git
cd drone-rl-llm-curriculum
```

2. Create the external storage directory next to the repository if it does not already exist:

```bash
mkdir -p ../storage
```

3. Build the Docker image:

```bash
bash scripts/docker_build.sh
```

4. Start the development container:

```bash
bash scripts/docker_dev.sh
```

Docker and job helpers default `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `TORCH_NUM_THREADS` to `1` to avoid CPU thread-pool oversubscription when PPO uses `num_envs > 1`. Override them from the host before launching when needed:

```bash
OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2 bash scripts/docker_dev.sh
```

5. Attach with Visual Studio Code:

```text
Remote Explorer -> Containers -> drone-rl-llm-curriculum-dev
```

6. Open the project notebook:

```text
Drone_RL_LLM_Curriculum.ipynb
```

</details>

### Real Training Commands

The report-ready training configs are split into smoke, medium, and final tiers. Smoke commands are for correctness only; medium and final commands start meaningful training and should be run intentionally.

Direct PPO training uses top-level `num_envs` to control training-time sample collection parallelism. `num_envs: 1` is the normal single-environment path, while higher values use parallel environments for PPO rollouts. PPO still treats `ppo.n_steps` as the per-environment rollout length, so the effective rollout size per update is `ppo.n_steps * num_envs`. Evaluation remains deterministic and single-rollout based, separate from training parallelism.

Direct PPO trajectory tracking supports an explicit `action_interface` setting. The default is `action_interface: pid_position`, where the policy outputs normalized target-position commands and the upstream PID controller stabilizes the drone by converting those targets into motor RPMs. An experimental `action_interface: direct_rpm` option lets the policy output four normalized motor commands, mapped around hover RPM into clipped motor RPMs before PyBullet physics. Direct RPM is a harder low-level control problem and should usually be paired with richer observations such as `include_dynamics_observation: true` and, for oversteering/crash experiments, `include_previous_action: true`. The previous-action option appends the previous PPO-facing action to the observation after reset/step, which can help policies reason about recent control input but does not guarantee better performance. `VEL` may be a future intermediate action interface; it is not implemented in this repository workflow yet.

PPO network architecture can be configured experimentally under `ppo.policy_kwargs.net_arch`, either as a shared list such as `[128, 128]` or as separate `{pi: [128, 128], vf: [128, 128]}` policy/value networks. Larger networks may train slower and should be treated as experiments; existing configs omit `policy_kwargs` so the Stable-Baselines3 default architecture is unchanged.

```bash
python -m src.experiments.cli.experiments_cli_train_tracking --config configs/training/ppo_tracking_smoke.yaml --run-name direct_ppo_line_smoke_seed0 --seed 0 --wandb-mode disabled
python -m src.experiments.cli.experiments_cli_train_curriculum --config configs/curricula/curriculum_manual_line_smoke.yaml --seed 0 --wandb-mode disabled
python -m src.experiments.cli.experiments_cli_train_tracking --config configs/training/ppo_tracking_direct_rpm_smoke.yaml --wandb-mode disabled
python -m src.experiments.cli.experiments_cli_train_tracking --config configs/training/ppo_tracking_dynamics_smoke.yaml --wandb-mode disabled

python -m src.experiments.cli.experiments_cli_train_tracking --config configs/training/ppo_tracking_medium.yaml --run-name direct_ppo_line_medium_seed0 --seed 0 --wandb-mode offline
python -m src.experiments.cli.experiments_cli_train_curriculum --config configs/curricula/curriculum_manual_line_medium.yaml --seed 0 --wandb-mode offline

python -m src.experiments.cli.experiments_cli_train_tracking --config configs/training/ppo_tracking_final.yaml --run-name direct_ppo_line_final_seed0 --seed 0 --wandb-mode auto
python -m src.experiments.cli.experiments_cli_train_curriculum --config configs/curricula/curriculum_manual_line_final.yaml --seed 0 --wandb-mode auto
```

### Phase 6 LLM Curriculum Commands

The default LLM curriculum smoke config uses a deterministic mock provider, so it does not require a running LLM server:

```bash
python -m src.experiments.cli.experiments_cli_train_llm_curriculum --config configs/curricula/curriculum_llm_smoke.yaml --seed 0 --wandb-mode disabled
```

To check a local OpenAI-compatible server without launching PPO training, start the server outside this repository and run:

```bash
python -m src.experiments.cli.experiments_cli_train_llm_curriculum \
  --config configs/curricula/curriculum_llm_local_smoke.yaml \
  --provider openai_compatible \
  --api-base http://127.0.0.1:18080/v1 \
  --model qwen2.5-coder-32b-instruct-q4_k_m.gguf \
  --max-stages 2 \
  --dry-run-proposals \
  --wandb-mode disabled
```

Remove `--dry-run-proposals` only when you want the accepted tasks to launch PPO stage training. Proposal events are written under the run-scoped `storage/runs/<run_name>/llm_logs/proposals.jsonl` path.

Evaluate direct PPO and manual-curriculum checkpoints as separate owned runs. When `--suite` is omitted, both evaluation CLIs run the standard profile: `own_task`, `line_eval`, `final_benchmark`, and `generalization` when `configs/evaluation/generalization_eval_suite.yaml` exists. Direct PPO evaluation artifacts stay under the direct run root. Curriculum evaluation runs every default evaluation for every stage, with artifacts written under the owning curriculum stage. The curriculum run root keeps only `evaluation_index.json`, optional `evaluation_summary.json`, and config snapshots; `config/evaluation_suites/` contains reproducibility snapshots, not duplicate result data.

```bash
python -m src.experiments.cli.experiments_cli_evaluate_policy \
  --run-manifest storage/runs/direct_ppo_line_final_seed0/run_manifest.json \
  --wandb-mode auto

python -m src.experiments.cli.experiments_cli_evaluate_curriculum \
  --summary storage/runs/curriculum_manual_line_final_seed0/run_manifest.json \
  --wandb-mode auto
```

Pass `--suite configs/evaluation/final_benchmark_eval_suite.yaml` for a single explicit suite instead of the full standard profile. Pass `--model-scope final-stage` to evaluate only the final curriculum stage.

All generated artifacts stay under `storage/runs/<self_describing_run_id>/`. The direct PPO task source for real training is `configs/training/ppo_tracking_tasks.yaml`; `configs/smoke/*` remains correctness-only.

---

## 📂 Project Structure

<details>
<summary><strong>Project structure anzeigen</strong></summary>

```text
.
├── .github/
│   └── workflows/
│       └── quality-report.yml          # Non-blocking quality report
│
├── .vscode/
│   └── settings.json                   # VS Code development settings
│
├── configs/                            # Experiment configurations
│
├── docs/                               # Curated figures and media for documentation
│   ├── figures/
│   └── media/
│
├── scripts/                            # Docker, HPC, and job helper scripts
│   ├── docker_build.sh                 # Build the Docker image
│   ├── docker_dev.sh                   # Start the development container
│   ├── docker_job.sh                   # Launch a GPU job
│   └── _docker_run.sh                  # Internal Docker job helper
│
├── src/                                # Modular Python source code
│   ├── __init__.py
│   ├── envs/                           # Environment wrappers, observations, rewards
│   ├── trajectories/                   # Trajectory and formation generators
│   ├── llm/                            # LLM prompts, schemas, curriculum logic
│   ├── validation/                     # Feasibility checks for generated tasks
│   ├── evaluation/                     # Metrics, plots, result aggregation
│   ├── experiments/                    # Experiment orchestration and entry points
│   │   ├── cli/                        # python -m experiment CLIs
│   │   ├── training/                   # PPO config, tracking training, training smoke
│   │   ├── evaluation/                 # Policy evaluation orchestration
│   │   ├── curriculum/                 # Manual curriculum training/evaluation helpers
│   │   └── rendering/                  # Policy, scenario, and render smoke workflows
│   └── utils/                          # Paths, seeds, logging, serialization
│
├── tests/                              # Unit and smoke tests
│
├── storage -> ../storage               # Optional symlink to external storage
│   └── runs/<run_id>/                  # Canonical generated run container
│
├── AGENTS.md                           # Instructions for coding agents
├── PROJECT_BRIEF.md                    # Detailed project brief
├── Drone_RL_LLM_Curriculum.ipynb       # Main project notebook
├── Dockerfile                          # Docker image definition
├── pyproject.toml                      # Python project configuration
├── uv.lock                             # Locked Python environment
└── README.md                           # Project overview
```

</details>

Generated artifacts use a run-scoped layout under `storage/runs/<self_describing_run_id>/`. Direct PPO training writes to `training/` and direct evaluations write to `evaluations/<evaluation_name>/`. Curriculum training and evaluation artifacts are stage-centric: each stage writes under `stages/stageNN_<stage_name>/`, while the curriculum run root keeps only `evaluation_index.json` and optional `evaluation_summary.json` links to stage-owned evaluation manifests.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📚 References

- J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, and A. P. Schoellig, **“Learning to Fly — A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control”**, 2021.  
- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, **“Stable-Baselines3: Reliable Reinforcement Learning Implementations”**, Journal of Machine Learning Research, 2021.  
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, **“Proximal Policy Optimization Algorithms”**, 2017.  
- E. Coumans and Y. Bai, **PyBullet: A Python Module for Physics Simulation for Games, Robotics and Machine Learning**, 2016–2021.  
- Farama Foundation, **Gymnasium Documentation**, maintained successor of OpenAI Gym.  
