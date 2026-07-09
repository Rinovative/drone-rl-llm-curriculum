#!/usr/bin/env bash
# Final overnight experiment matrix. Source this file from lane runners.

experiment_kind() {
  case "$1" in
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "llm_curriculum" ;;
    direct_ppo_pid_dynprev_basic_show_seed0) echo "direct_ppo" ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "manual_curriculum" ;;
    direct_ppo_directrpm_dynprev_basic_show_seed0) echo "direct_ppo" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "llm_curriculum" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "manual_curriculum" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_gamma095_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_clip010_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_targetkl015_seed0) echo "direct_ppo" ;;
    *) return 1 ;;
  esac
}

experiment_config() {
  case "$1" in
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/llm_curriculum_pid_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_basic_show_seed0) echo "configs/training/ppo_tracking_pid_dynprev_basic_show.yaml" ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/curriculum_pid_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_basic_show_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_basic_show.yaml" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/llm_curriculum_directrpm_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml" ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/curriculum_directrpm_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net256_m-taskdist_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_net256_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_low_lr.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_low_lr.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_gamma095.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_gamma095_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_gamma095.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_ent005.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_ent005.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_clip010.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_clip010_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_clip010.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_targetkl015.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_targetkl015_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_targetkl015.yaml" ;;
    *) return 1 ;;
  esac
}

experiment_run_name() { echo "$1"; }

experiment_units() {
  case "$1" in
    curriculum_manual_*) echo "5" ;;
    llm_curriculum_*) echo "5" ;;
    direct_ppo_*) echo "1" ;;
    *) return 1 ;;
  esac
}

experiment_priority() {
  case "$1" in
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_basic_show_seed0) echo "must-have" ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_basic_show_seed0) echo "must-have" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_gamma095_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_clip010_seed0) echo "must-have" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "must-have" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_targetkl015_seed0) echo "must-have" ;;
    *) return 1 ;;
  esac
}

experiment_notes() {
  case "$1" in
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "Local LLM PID dynprev curriculum, 10 adaptive stages capped at 5 units, net128 default." ;;
    direct_ppo_pid_dynprev_basic_show_seed0) echo "PID dynprev basic-training-show distribution, net128 default." ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "Manual PID dynprev curriculum, 5 fixed-budget stages, net128 default." ;;
    direct_ppo_directrpm_dynprev_basic_show_seed0) echo "Direct-RPM dynprev basic-training-show distribution, net128 default." ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "Local LLM direct-RPM dynprev curriculum, 10 adaptive stages capped at 5 units, net128 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "PID dynprev tracking-medium task distribution, net128 default." ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "Manual direct-RPM dynprev curriculum, 5 fixed-budget stages, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "Direct-RPM dynprev tracking-medium task distribution, net128 default." ;;
    direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0) echo "PID dynprev tracking-medium net256 architecture comparison." ;;
    direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0) echo "Direct-RPM dynprev tracking-medium net256 architecture comparison." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "PID dynprev tracking-medium low learning-rate PPO profile, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "Direct-RPM dynprev tracking-medium low learning-rate PPO profile, net128 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0) echo "PID dynprev tracking-medium gamma 0.95 PPO profile, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_gamma095_seed0) echo "Direct-RPM dynprev tracking-medium gamma 0.95 PPO profile, net128 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "PID dynprev tracking-medium entropy 0.005 PPO profile, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "Direct-RPM dynprev tracking-medium entropy 0.005 PPO profile, net128 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "PID dynprev tracking-medium clip range 0.10 PPO profile, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_clip010_seed0) echo "Direct-RPM dynprev tracking-medium clip range 0.10 PPO profile, net128 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "PID dynprev tracking-medium target KL 0.015 PPO profile, net128 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_targetkl015_seed0) echo "Direct-RPM dynprev tracking-medium target KL 0.015 PPO profile, net128 default." ;;
    *) return 1 ;;
  esac
}

lane_experiments() {
  case "$1" in
    1) echo "llm_curriculum_pid_dynprev_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_basic_show_seed0" ;;
    2) echo "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0 direct_ppo_directrpm_dynprev_basic_show_seed0" ;;
    3) echo "llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_seed0" ;;
    4) echo "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0" ;;
    5) echo "direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0 direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0" ;;
    6) echo "direct_ppo_directrpm_dynprev_m-taskdist_medium_gamma095_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_clip010_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_targetkl015_seed0" ;;
    *) return 1 ;;
  esac
}
