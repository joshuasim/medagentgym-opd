#!/bin/bash
#SBATCH --job-name=opd-smoke
#SBATCH --account=torch_pr_69_tandon_priority   # fallback: torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_public   # skip 48 GB L40S: 7B actor + vLLM won't fit
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# OPD smoke run: 2 training steps on 16 training tasks, to check the pipeline before real runs.
# GPU 0: student (FSDP actor + its vLLM rollout, colocated). GPU 1: teacher (vLLM, scores tokens).
# Plumbing test only, so it does not need the SFT model or the final teacher choice:
#   sbatch train/opd_smoke.sbatch
#   STUDENT_PATH=$SCRATCH/capstone/models/qwen7b-sft-full TEACHER_PATH=MedAgentGym/MedCopilot-14B sbatch train/opd_smoke.sbatch
# Pass if: 2 steps finish, distillation/* metrics are finite, a checkpoint is saved, and
# scripts/check_rollouts.py shows multi-turn episodes with env output masked out.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
"$TRAIN_PY" -c "import flash_attn" || { echo "flash_attn missing: sbatch torch/install_flash_attn.sbatch first"; exit 1; }
export STUDENT_PATH="${STUDENT_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export TEACHER_PATH="${TEACHER_PATH:-Qwen/Qwen2.5-14B-Instruct}"
export N_GPUS=1 TEACHER_GPUS=1
OUT="$SCRATCH/capstone/runs/opd_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "student=$STUDENT_PATH teacher=$TEACHER_PATH out=$OUT"

bash train/opd.sh \
  data.train_max_samples=16 \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
  trainer.total_training_steps=2 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=2 \
  trainer.resume_mode=disable \
  trainer.rollout_data_dir="$OUT/rollouts" \
  trainer.default_local_dir="$OUT/ckpt" \
  trainer.experiment_name=opd_smoke 2>&1 | tee "$OUT/train.log"

echo "=== rollout check ==="
"$TRAIN_PY" scripts/check_rollouts.py "$OUT/rollouts"
echo "=== distillation metrics ==="
grep -oE "distillation/[a-z_]+:[-0-9.e]+|step:[0-9]+|critic/score/mean:[-0-9.e]+" "$OUT/train.log" | tr '\n' ' '; echo
ls "$OUT/ckpt"/global_step_* -d 2>/dev/null || echo "NO CHECKPOINT SAVED"
