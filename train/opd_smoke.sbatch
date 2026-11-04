#!/bin/bash
#SBATCH --job-name=opd-smoke
#SBATCH --account=torch_pr_69_tandon_priority   # fallback: torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_public   # skip 48 GB L40S: 7B actor + vLLM won't fit
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00   # 2 h was not enough to rule out slow rollouts; the watchdog ends crashed runs early
#SBATCH --output=logs/%x-%j.out
# OPD smoke run: 2 training steps on 16 training tasks, to check the pipeline before real runs.
# GPUs 0-1: student (FSDP actor + its vLLM rollout, colocated). GPU 2: teacher (vLLM, scores tokens).
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
# Student on 2 GPUs: on 1 GPU the training process held 61 GB of real allocations at the first weight
# sync (Nov 4 run: OOM in merged_lora_context's base-weight backup, 0.07 GB fragmentation), leaving no
# room next to vLLM. FSDP shards the student across 2 GPUs. GPU 3 is the teacher.
export N_GPUS=2 TEACHER_GPUS=1
OUT="$SCRATCH/capstone/runs/opd_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Student GPU holds the FSDP actor (staged on GPU for weight sync) + vLLM at 0.4; teacher GPU holds a
# 14B model at 0.85. Both need 80 GB cards; fail in seconds instead of OOM-ing mid-run.
MIN_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "$MIN_MB" -ge 75000 ] || { echo "GPU has ${MIN_MB} MiB; this run needs 80 GB GPUs"; exit 1; }
echo "student=$STUDENT_PATH teacher=$TEACHER_PATH out=$OUT"
source train/watchdog.sh
start_watchdog "$OUT/train.log"

bash train/opd.sh \
  data.train_max_samples=16 \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.7 \
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
