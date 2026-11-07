#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --account=torch_pr_69_tandon_priority   # fallback: torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_public
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Full OPD or GRPO run. Resubmitting with the same RUN_NAME resumes from its latest checkpoint.
#   OPD, 80 GB GPUs (student 2 + Coder-32B teacher 2):
#     METHOD=opd RUN_NAME=opd_coder32b_s0 sbatch train/run_rl.sbatch
#   OPD, H200 (student 1 + teacher 1, Coder-32B fits on one 141 GB GPU):
#     METHOD=opd RUN_NAME=opd_coder32b_s0 N_GPUS=1 TEACHER_GPUS=1 TEACHER_TP=1 \
#       sbatch --partition=h200_public --gres=gpu:2 train/run_rl.sbatch
#   GRPO (no teacher):
#     METHOD=grpo RUN_NAME=grpo_s0 sbatch --gres=gpu:2 train/run_rl.sbatch
# Checkpoints: $SCRATCH/capstone/checkpoints/$RUN_NAME  (evaluate with train/eval_ckpt.sbatch)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
"$TRAIN_PY" -c "import flash_attn, transfer_queue" || { echo "trainenv incomplete (flash_attn / transfer_queue)"; exit 1; }
METHOD="${METHOD:?METHOD=opd or METHOD=grpo}"
export RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. opd_coder32b_s0}"
export STUDENT_PATH="${STUDENT_PATH:-$SCRATCH/capstone/models/qwen7b-sft-full}"   # SFT 3 epochs (chosen Nov 3)
export TEACHER_PATH="${TEACHER_PATH:-Qwen/Qwen2.5-Coder-32B-Instruct}"
export N_GPUS="${N_GPUS:-2}" TEACHER_GPUS="${TEACHER_GPUS:-2}" TEACHER_TP="${TEACHER_TP:-2}"
[ -d "$STUDENT_PATH" ] || { echo "student model not found: $STUDENT_PATH"; exit 1; }
OUT="$SCRATCH/capstone/runs/$RUN_NAME"
mkdir -p "$OUT"
LOG="$OUT/train_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=name,memory.total --format=csv
MIN_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "$MIN_MB" -ge 75000 ] || { echo "GPU has ${MIN_MB} MiB; this run needs 80 GB GPUs"; exit 1; }
echo "method=$METHOD run=$RUN_NAME student=$STUDENT_PATH teacher=$TEACHER_PATH gpus: student $N_GPUS, teacher $TEACHER_GPUS (tp $TEACHER_TP)"
source train/watchdog.sh
start_watchdog "$LOG"
case "$METHOD" in
  opd)  bash train/opd.sh  trainer.rollout_data_dir="$OUT/rollouts" "$@" 2>&1 | tee -a "$LOG" ;;
  grpo) bash train/grpo.sh trainer.rollout_data_dir="$OUT/rollouts" "$@" 2>&1 | tee -a "$LOG" ;;
  *) echo "unknown METHOD=$METHOD"; exit 1 ;;
esac
ls -d "$SCRATCH/capstone/checkpoints/$RUN_NAME"/global_step_* 2>/dev/null | tail -3
