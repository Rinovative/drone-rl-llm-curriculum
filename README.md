[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovative/drone-rl-llm-curriculum/blob/main/Drone_RL_LLM_Curriculum.ipynb)  
_Open Interactive Jupyter Notebook directly in your browser (via Colab)_

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


---

## Task Distributions and Overnight Runs

Training can use fixed or randomized task distributions. A fixed task is treated as a degenerate distribution (`mode: fixed`, `strength: 0.0`), while randomized distributions sample bounded validated tasks per reset when `sample_on_reset: true`. Evaluation remains fixed and reproducible through `configs/evaluation/evaluation_task_suite_variation.yaml` and `configs/evaluation/evaluation_task_suite_broad.yaml`.

The local LLM curriculum can choose constrained known task distributions such as `tracking_small`, `tracking_medium`, and experimental `tracking_broad`; it does not freely invent arbitrary trajectory randomization configs.

Medium-screening overnight entry points:

```bash
export LANE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
bash scripts/run_lane_1.sh
bash scripts/run_lane_2.sh
bash scripts/run_lane_3.sh
bash scripts/run_lane_4.sh
```

See `docs/experiments/overnight_lane_assignment.tsv` and `docs/experiments/overnight_runner_usage.md` for the exact matrix, logs, resume markers, LLM skip behavior, and evaluation/render outputs.


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
