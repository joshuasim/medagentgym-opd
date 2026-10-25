#!/bin/bash
# One-time setup on NYU Torch. Run on the login node from anywhere:
#   bash torch/setup.sh            (takes ~20-40 min, mostly downloads)
#
# Creates, under $SCRATCH/capstone:
#   MedAgentGym/   official repo (task files + ehr_gym package)
#   taskenv/       Python 3.11 env that EXECUTES task code. Mirrors MedAgentGym's Docker image:
#                  their requirements.txt (numpy 2.2.4, transformers 4.51 ...) + pysam + centrosome.
#   trainenv/      Python 3.11 env that RUNS training/eval: verl 0.9.0 + vLLM + what ehr_gym imports.
#   hf_cache/      Hugging Face models
# Two envs because verl needs transformers 5.x but MedAgentGym pins 4.51; task code runs as
# `python file.py` via PATH, so taskenv's python must come first on PATH (see torch/env.sh).
set -euo pipefail

BASE="$SCRATCH/capstone"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$BASE" && cd "$BASE"

# --- uv (fast Python/package manager; installs its own Python 3.11, no module needed)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$BASE/uv_cache"   # keep caches off the 50GB home quota

# --- MedAgentGym (pinned to the commit we audited)
if [ ! -d MedAgentGym ]; then
  git clone https://github.com/wshi83/MedAgentGym.git
fi
git -C MedAgentGym checkout -q 81737ba
mkdir -p MedAgentGym/cache

# --- taskenv: executes BioCoder task code (same packages as ehr_gym:capstone)
if [ ! -x taskenv/bin/python ]; then
  uv venv -q -p 3.11 taskenv
fi
uv pip install -q -p taskenv/bin/python -r MedAgentGym/requirements.txt pysam centrosome torch

# --- trainenv: verl + vLLM for training and fast evaluation
if [ ! -x trainenv/bin/python ]; then
  uv venv -q -p 3.11 trainenv
fi
uv pip install -q -p trainenv/bin/python "verl[vllm]==0.9.0" \
  "langchain==0.3.23" "langchain_community==0.3.21" openai Levenshtein gymnasium tiktoken toml jsonlines \
  cachetools fastapi uvicorn pandas pyarrow huggingface_hub

# --- model weights
export HF_HOME="$BASE/hf_cache"
trainenv/bin/hf download Qwen/Qwen2.5-7B-Instruct --quiet

echo
echo "taskenv : $(taskenv/bin/python -c 'import numpy, pysam, centrosome; print("numpy", numpy.__version__, "pysam ok, centrosome ok")')"
echo "trainenv: $(trainenv/bin/python -c 'import verl, vllm, torch; print("verl", verl.__version__, "vllm", vllm.__version__, "torch", torch.__version__)')"
echo "repo    : $REPO"
echo "Setup done."
