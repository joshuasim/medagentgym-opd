#!/bin/bash
#SBATCH --job-name=rl-extdbg
#SBATCH --account=torch_pr_69_tandon_priority
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# GRPO with an external debugger (Nov 25 experiment). GPU A: the usual 1-GPU GRPO run (train/run_rl.sbatch,
# unchanged). GPU B: a vLLM server with the debugger model; the agent loop sends debug requests there
# (configs/agent_loop_extdbg.yaml -> opd/agent_loop_extdbg.py). Everything else = grpo_multi_lr3e5_s0.
#   RUN_NAME=grpo_multi_lr3e5_extdbg_s0 sbatch train/run_grpo_extdbg.sbatch
# Resubmit the same command after a cancellation: it resumes from the last checkpoint, like run_rl.sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
umask 077
export RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. grpo_multi_lr3e5_extdbg_s0}"
DEBUGGER_MODEL="${DEBUGGER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
IFS=, read -r GPU_A GPU_B _ <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ -n "${GPU_B:-}" ] || { echo "needs 2 GPUs (--gres=gpu:2)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
MIN_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
[ "$MIN_MB" -ge 140000 ] || { echo "GPU has ${MIN_MB} MiB; 1-GPU GRPO needs an H200 (80 GB GPUs ran out of memory on Nov 24)"; exit 1; }

# --- debugger server on GPU B (same start/health logic as torch/eval_baseline.sbatch)
VLLM_BIN="$(dirname "$TRAIN_PY")/vllm"
PORT=$("$TRAIN_PY" -c "
import socket, sys
for p in range(int(sys.argv[1]), int(sys.argv[1]) + 500):
    s = socket.socket()
    try:
        s.bind(('', p)); print(p); break
    except OSError:
        pass
    finally:
        s.close()
else:
    sys.exit('no free port')" $(( 30000 + SLURM_JOB_ID % 20000 )))
mkdir -p "$SCRATCH/capstone/runs/$RUN_NAME"
DBG_LOG="$SCRATCH/capstone/runs/$RUN_NAME/vllm_debugger_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=$GPU_B "$VLLM_BIN" serve "$DEBUGGER_MODEL" --port "$PORT" --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --dtype bfloat16 --served-model-name "$DEBUGGER_MODEL" > "$DBG_LOG" 2>&1 &
DBG_PID=$!
trap 'kill $DBG_PID 2>/dev/null' EXIT
for k in $(seq 1 120); do
  kill -0 $DBG_PID 2>/dev/null || { echo "debugger server exited"; tail -40 "$DBG_LOG"; exit 1; }
  curl -sf "localhost:$PORT/v1/models" | grep -qF "\"$DEBUGGER_MODEL\"" && break
  [ "$k" = 120 ] && { echo "debugger server did not start in 20 min"; tail -40 "$DBG_LOG"; exit 1; }
  sleep 10
done
echo "[external debugger] $DEBUGGER_MODEL up on port $PORT (GPU $GPU_B)"

# --- the usual GRPO run on GPU A, unchanged script, external-debugger agent loop
export DEBUGGER_BASE_URL="http://localhost:$PORT/v1" DEBUGGER_MODEL
export CUDA_VISIBLE_DEVICES=$GPU_A
export METHOD=grpo N_GPUS=1 SAVE_FREQ="${SAVE_FREQ:-2}" LR="${LR:-3e-5}"
export STUDENT_PATH="${STUDENT_PATH:-$SCRATCH/capstone/models/qwen7b-sft_multi_e2}"
export AGENT_LOOP_CONFIG="$REPO/configs/agent_loop_extdbg.yaml"
bash train/run_rl.sbatch
