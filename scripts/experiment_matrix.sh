#!/usr/bin/env bash
# Medium-screening overnight experiment matrix. Source this file from lane runners.

experiment_kind() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net128_small_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net512_large_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_net512_large_m-taskdist_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "direct_ppo" ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "manual_curriculum" ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "manual_curriculum" ;;
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "llm_curriculum" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "llm_curriculum" ;;
    *) return 1 ;;
  esac
}

experiment_config() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "configs/training/ppo_tracking_pid_baseline_medium.yaml" ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_medium.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_net128_small_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net128_small_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_net512_large_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net512_large_m-taskdist_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_net512_large_m-taskdist_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_net512_large_m-taskdist_medium.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_low_lr.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_ent005.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_clip010.yaml" ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium_targetkl015.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_low_lr.yaml" ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium_ent005.yaml" ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/curriculum_pid_dynprev_m-taskdist_medium.yaml" ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/curriculum_directrpm_dynprev_m-taskdist_medium.yaml" ;;
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/llm_curriculum_pid_dynprev_m-taskdist_medium.yaml" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "configs/curricula/llm_curriculum_directrpm_dynprev_m-taskdist_medium.yaml" ;;
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
    direct_ppo_directrpm_dynprev_net512_large_m-taskdist_medium_seed0|direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0|direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0|curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "should-have" ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "optional" ;;
    *) echo "must-have" ;;
  esac
}

experiment_notes() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "Fixed basic-training-show PID baseline, no dynamics or previous action, net256 default." ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "Fixed basic-training-show PID with dynamics and previous-action observations, net256 default." ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "Experimental fixed basic-training-show direct-RPM dynprev comparison, net256 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_seed0) echo "PID dynprev with medium task distribution, net256 default." ;;
    direct_ppo_pid_dynprev_net128_small_m-taskdist_medium_seed0) echo "PID dynprev medium task distribution with net128 small architecture comparison." ;;
    direct_ppo_pid_dynprev_net512_large_m-taskdist_medium_seed0) echo "PID dynprev medium task distribution with net512 large architecture comparison." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0) echo "Experimental direct-RPM dynprev with medium task distribution, net256 default." ;;
    direct_ppo_directrpm_dynprev_net512_large_m-taskdist_medium_seed0) echo "Experimental direct-RPM dynprev medium task distribution with net512 large architecture comparison." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0) echo "PID taskdist low learning-rate micro-HPO, net256 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0) echo "PID taskdist entropy 0.005 micro-HPO, net256 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0) echo "PID taskdist clip range 0.10 one-parameter comparison, net256 default." ;;
    direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0) echo "PID taskdist target KL 0.015 one-parameter comparison, net256 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0) echo "Experimental direct-RPM taskdist low learning-rate micro-HPO, net256 default." ;;
    direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0) echo "Experimental direct-RPM taskdist entropy 0.005 micro-HPO, net256 default." ;;
    curriculum_manual_pid_dynprev_m-taskdist_medium_seed0) echo "Manual PID curriculum, 5 fixed-budget stages, medium task distribution base, net256 default." ;;
    curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0) echo "Experimental manual direct-RPM curriculum, 5 fixed-budget stages, net256 default." ;;
    llm_curriculum_pid_dynprev_m-taskdist_medium_seed0) echo "Local LLM PID curriculum, 10 adaptive stages capped at 5 units, bounded task distributions, net256 default." ;;
    llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0) echo "Optional experimental local LLM direct-RPM curriculum, 10 conservative adaptive stages capped at 5 units, net256 default." ;;
    *) return 1 ;;
  esac
}

lane_experiments() {
  case "$1" in
    1) echo "llm_curriculum_pid_dynprev_m-taskdist_medium_seed0 direct_ppo_pid_baseline_medium_seed0" ;;
    2) echo "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_medium_seed0" ;;
    3) echo "llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0 direct_ppo_directrpm_dynprev_medium_seed0" ;;
    4) echo "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0" ;;
    5) echo "direct_ppo_pid_dynprev_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_net128_small_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_clip010_seed0 direct_ppo_directrpm_dynprev_net512_large_m-taskdist_medium_seed0" ;;
    6) echo "direct_ppo_pid_dynprev_net512_large_m-taskdist_medium_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0 direct_ppo_pid_dynprev_m-taskdist_medium_targetkl015_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0 direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0" ;;
    *) return 1 ;;
  esac
}
