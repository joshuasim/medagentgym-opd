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
# Ports: on Nov 13 two eval jobs on one node (23083894/5, gl034) both used port 8000, so one job's eval
# talked to the other job's model. Now each server starts at a job-specific port (20000 + job id mod 20000),
# skips ports already in use, restarts on another port if it still loses the port to someone else, and the
# job checks that the port really serves our model before evaluating. Student and debugger start together.
VLLM_BIN="$(dirname "$TRAIN_PY")/vllm"
NEXT_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
PIDS=""
trap 'kill $PIDS 2>/dev/null' EXIT
free_port() {  # first port >= $1 that nothing is listening on
  "$TRAIN_PY" -c "
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
    sys.exit('no free port')" "$1"
}
wait_up() {  # $1 pid, $2 port -> 0 up, 1 process exited, 2 timeout (20 min)
  local k
  for k in $(seq 1 120); do
    kill -0 "$1" 2>/dev/null || return 1
    curl -sf "localhost:$2/health" >/dev/null && return 0
    sleep 10
  done
  return 2
}
# server i: LABEL[i] ENVP[i] ("" or CUDA_VISIBLE_DEVICES=..) MDL[i] LEN[i] LOG[i] EXTRA[i]; start_srv fills PORT[i] PID[i]
declare -a LABEL ENVP MDL LEN LOG EXTRA PORT PID
start_srv() {
  local i=$1
  PORT[$i]=$(free_port "$NEXT_PORT"); NEXT_PORT=$(( PORT[$i] + 1 ))
  env ${ENVP[$i]} "$VLLM_BIN" serve "${MDL[$i]}" --port "${PORT[$i]}" --max-model-len "${LEN[$i]}" \
    --gpu-memory-utilization 0.9 --dtype bfloat16 --served-model-name "${MDL[$i]}" ${EXTRA[$i]} > "${LOG[$i]}" 2>&1 &
  PID[$i]=$!; PIDS="$PIDS ${PID[$i]}"
}
ensure_srv() {  # wait until server i is up on a port that serves its model; restart elsewhere if the port was lost
  local i=$1 rc
  for attempt in 1 2 3 4 5; do
    if wait_up "${PID[$i]}" "${PORT[$i]}"; then
      if curl -sf "localhost:${PORT[$i]}/v1/models" | grep -qF "\"${MDL[$i]}\""; then
        echo "${LABEL[$i]} up on port ${PORT[$i]} (attempt $attempt)"; return 0
      fi
      echo "port ${PORT[$i]} answers but is not serving ${MDL[$i]}; restarting on another port"
      kill "${PID[$i]}" 2>/dev/null || true
    else
      rc=$?
      [ "$rc" = 2 ] && { echo "${LABEL[$i]} did not start within 20 min"; tail -50 "${LOG[$i]}"; exit 1; }
      grep -qi "address already in use" "${LOG[$i]}" || { echo "${LABEL[$i]} exited"; tail -50 "${LOG[$i]}"; exit 1; }
      echo "port ${PORT[$i]} was taken while starting; restarting on another port"
    fi
    start_srv "$i"
  done
  echo "${LABEL[$i]}: no usable port after 5 attempts"; exit 1
}

# LOCAL_DEBUGGER=<model>: serve a second model on its own GPU (submit with --gres=gpu:2) and let it
# answer debug requests, e.g. LOCAL_DEBUGGER=Qwen/Qwen2.5-Coder-32B-Instruct. Student model on GPU 0.
# Use the GPU ids Slurm assigned (not literal 0/1, which may be other jobs' GPUs on some nodes).
IFS=, read -r GPU_A GPU_B _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
[ -z "${LOCAL_DEBUGGER:-}" ] || [ -n "${GPU_B:-}" ] || { echo "LOCAL_DEBUGGER needs 2 GPUs: sbatch --gres=gpu:2"; exit 1; }
LABEL[0]="vLLM ($MODEL)"; ENVP[0]=""; MDL[0]="$MODEL"; LEN[0]=32768; LOG[0]="$OUT/vllm.log"; EXTRA[0]="--tensor-parallel-size $TP"
[ -n "${LOCAL_DEBUGGER:-}" ] && ENVP[0]="CUDA_VISIBLE_DEVICES=$GPU_A"
start_srv 0
if [ -n "${LOCAL_DEBUGGER:-}" ]; then
  LABEL[1]="debugger ($LOCAL_DEBUGGER)"; ENVP[1]="CUDA_VISIBLE_DEVICES=$GPU_B"; MDL[1]="$LOCAL_DEBUGGER"
  LEN[1]=16384; LOG[1]="$OUT/vllm_debugger.log"; EXTRA[1]=""
  start_srv 1   # starts now, in parallel with the student server
fi
ensure_srv 0
EVAL_ARGS="${EVAL_ARGS:-} --base_url http://localhost:${PORT[0]}/v1"
if [ -n "${LOCAL_DEBUGGER:-}" ]; then
  ensure_srv 1
  export DEBUGGER_API_KEY=EMPTY
  EVAL_ARGS="$EVAL_ARGS --debugger api --debugger_model $LOCAL_DEBUGGER --debugger_base_url http://localhost:${PORT[1]}/v1"
fi

# EVAL_ARGS: extra eval_biocoder.py args, e.g. "--mode train --idx_file $REPO/data/checks/dev_sft_select.json"
# SYSTEM_ONLY=1: skip the merged-prompt (paper format) run
if [ -z "${SYSTEM_ONLY:-}" ]; then
  "$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/merged_prompt" --merge_system --save_history ${EVAL_ARGS:-}
  echo "=== merged prompt (paper format) ===";  cat "$OUT/merged_prompt/summary.json"
fi
"$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/system_prompt" --save_history ${EVAL_ARGS:-}
echo "=== system prompt (training format) ==="; cat "$OUT/system_prompt/summary.json"
