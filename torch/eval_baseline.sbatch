#!/bin/bash
#SBATCH --job-name=biocoder-eval
#SBATCH --account=torch_pr_69_tandon_priority   # approved by Prof. Tan (Oct 27); fallback: torch_pr_69_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Baseline eval on the 157 BioCoder test tasks. Submit from the repo folder:
#   sbatch torch/eval_baseline.sbatch                                   (base Qwen2.5-7B-Instruct)
#   MODEL=MedAgentGym/MedCopilot-7B TAG=medcopilot7b sbatch torch/eval_baseline.sbatch
#   MODEL=Qwen/Qwen2.5-32B-Instruct TAG=base_qwen32b TP=2 sbatch --gres=gpu:2 --time=04:00:00 torch/eval_baseline.sbatch
# Runs twice: MedAgentGym's vLLM prompt format (system merged into user, like the paper's
# open-model runs) and the separate-system format our SFT/RL training uses.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-base_qwen7b}"
TP="${TP:-1}"   # tensor parallel GPUs for vLLM; 32B needs TP=2 plus: sbatch --gres=gpu:2
OUT="$REPO/results/$TAG"
mkdir -p "$OUT"
# Run from a throwaway folder: the agent's terminal commands and the task code write files into the
# working directory (on Nov 2 they left venvs, CNVkit source and csv files in MedAgentGym/). An empty
# folder also matches how scripts/check_tasks.py audited the tasks. validate_code writes ./cache here.
WORK="$SCRATCH/capstone/work/${SLURM_JOB_ID:-local}_$TAG"
mkdir -p "$WORK/cache" && cd "$WORK"

nvidia-smi --query-gpu=name,memory.total --format=csv
# Ports from the job id: two eval jobs on one node both used 8000 (Nov 13, jobs 23083894/5 on gl034),
# so one job's eval talked to the other job's model. PORT+1 is the debugger.
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 )); DBG_PORT=$(( PORT + 1 ))
echo "vLLM port $PORT (debugger $DBG_PORT)"
# LOCAL_DEBUGGER=<model>: serve a second model on its own GPU (submit with --gres=gpu:2) and let it
# answer debug requests, e.g. LOCAL_DEBUGGER=Qwen/Qwen2.5-Coder-32B-Instruct. Student model on GPU 0.
# Use the GPU ids Slurm assigned (not literal 0/1, which may be other jobs' GPUs on some nodes).
IFS=, read -r GPU_A GPU_B _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
STUDENT_GPUS=""; [ -n "${LOCAL_DEBUGGER:-}" ] && STUDENT_GPUS="CUDA_VISIBLE_DEVICES=$GPU_A"
env $STUDENT_GPUS "$(dirname "$TRAIN_PY")/vllm" serve "$MODEL" --port "$PORT" --max-model-len 32768 --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.9 --dtype bfloat16 --served-model-name "$MODEL" \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
DBG_PID=""
if [ -n "${LOCAL_DEBUGGER:-}" ]; then
  [ -n "${GPU_B:-}" ] || { echo "LOCAL_DEBUGGER needs 2 GPUs: sbatch --gres=gpu:2"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPU_B "$(dirname "$TRAIN_PY")/vllm" serve "$LOCAL_DEBUGGER" --port "$DBG_PORT" --max-model-len 16384 \
    --gpu-memory-utilization 0.9 --dtype bfloat16 --served-model-name "$LOCAL_DEBUGGER" \
    > "$OUT/vllm_debugger.log" 2>&1 &
  DBG_PID=$!
  export DEBUGGER_API_KEY=EMPTY
  EVAL_ARGS="${EVAL_ARGS:-} --debugger api --debugger_model $LOCAL_DEBUGGER --debugger_base_url http://localhost:$DBG_PORT/v1"
fi
trap 'kill $VLLM_PID $DBG_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf localhost:$PORT/health >/dev/null && break
  sleep 10
done
curl -sf localhost:$PORT/health >/dev/null || { echo "vLLM failed to start"; tail -50 "$OUT/vllm.log"; exit 1; }
echo "vLLM up after $((i*10))s"
# fail if the server on this port is not our model (or our server process died)
kill -0 $VLLM_PID 2>/dev/null || { echo "our vLLM process exited"; tail -50 "$OUT/vllm.log"; exit 1; }
curl -sf localhost:$PORT/v1/models | grep -qF "\"$MODEL\"" || { echo "port $PORT is not serving $MODEL"; exit 1; }
EVAL_ARGS="${EVAL_ARGS:-} --base_url http://localhost:$PORT/v1"
if [ -n "${LOCAL_DEBUGGER:-}" ]; then
  for i in $(seq 1 120); do curl -sf localhost:$DBG_PORT/health >/dev/null && break; sleep 10; done
  curl -sf localhost:$DBG_PORT/health >/dev/null || { echo "debugger vLLM failed to start"; tail -50 "$OUT/vllm_debugger.log"; exit 1; }
  echo "debugger ($LOCAL_DEBUGGER) up"
fi

# EVAL_ARGS: extra eval_biocoder.py args, e.g. "--mode train --idx_file $REPO/data/checks/dev_sft_select.json"
# SYSTEM_ONLY=1: skip the merged-prompt (paper format) run
if [ -z "${SYSTEM_ONLY:-}" ]; then
  "$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/merged_prompt" --merge_system --save_history ${EVAL_ARGS:-}
  echo "=== merged prompt (paper format) ===";  cat "$OUT/merged_prompt/summary.json"
fi
"$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/system_prompt" --save_history ${EVAL_ARGS:-}
echo "=== system prompt (training format) ==="; cat "$OUT/system_prompt/summary.json"
