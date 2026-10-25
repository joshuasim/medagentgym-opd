#!/bin/bash
# On-policy distillation with VeRL's built-in distillation (teacher scores the student's
# own multi-turn rollouts; env observation tokens are masked out). DRAFT, not yet GPU-tested.
#   TEACHER_PATH  e.g. MedAgentGym/MedCopilot-14B or Qwen/Qwen2.5-14B-Instruct (same tokenizer as student)
# Loss choice to confirm on GPU: k1 + use_policy_gradient treats -(log p_student - log p_teacher) as a
# per-token reward (the OPD recipe); task rewards off = pure OPD. Set use_task_rewards=true to mix in GRPO.
source "$(dirname "$0")/common.sh"
python -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" \
  algorithm.adv_estimator=grpo \
  distillation.enabled=true \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_PATH" \
  distillation.teacher_models.teacher_model.num_replicas=1 \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.n_gpus_per_node="${TEACHER_GPUS:-1}" \
  distillation.nnodes=1 \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=true \
  distillation.distillation_loss.use_task_rewards=false \
  trainer.project_name=medagentgym-opd trainer.experiment_name="opd_seed${SEED:-0}" \
  "$@"
