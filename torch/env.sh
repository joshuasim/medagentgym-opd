# Source this in every Torch job:  source torch/env.sh
BASE="$SCRATCH/capstone"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEDAGENTGYM_ROOT="$BASE/MedAgentGym"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"
export TRAIN_PY="$BASE/trainenv/bin/python"
# task code is executed as `python <file>` -> must resolve to taskenv (MedAgentGym's packages)
export PATH="$BASE/taskenv/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO:$MEDAGENTGYM_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED=0   # some task outputs print sets; fix iteration order so answers are stable
export TOKENIZERS_PARALLELISM=false
