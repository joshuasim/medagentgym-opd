#!/bin/bash
# Read-only: which VeRL config keys hold seeds, and what the data shuffle uses.
# Usage: bash scripts/check_seeds.sh 2>&1 | tee logs/seed_check.txt
source torch/env.sh
V=$("$TRAIN_PY" -c "import verl,os;print(os.path.dirname(verl.__file__))")
echo "=== dotted paths of every *seed* key in the FSDP trainer config"
"$TRAIN_PY" - "$V" <<'PY'
import sys, glob, os, yaml
cfg_dir = os.path.join(sys.argv[1], "trainer", "config")
files = [f for f in glob.glob(os.path.join(cfg_dir, "*ppo_trainer*.yaml")) if "megatron" not in f and "torchtitan" not in f]
def walk(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            q = f"{p}.{k}" if p else str(k)
            if "seed" in str(k): print(f"  {q} = {v}")
            walk(v, q)
for f in files:
    print("--", os.path.basename(f)); walk(yaml.safe_load(open(f)))
PY
echo "=== where the dataloader sampler is built, and its seed lines"
for f in $(grep -rl "def create_rl_sampler\|RandomSampler" "$V" --include=*.py); do
  echo "-- ${f#$V/}"; grep -n "seed\|RandomSampler\|Generator\|shuffle" "$f" | head -12
done
echo "=== rl_dataset.py around the seed use"
sed -n 150,160p "$V/utils/dataset/rl_dataset.py"; sed -n 180,192p "$V/utils/dataset/rl_dataset.py"
echo "=== how vLLM rollout uses rollout.seed"
grep -rn "seed" "$V/workers/rollout/vllm_rollout/" --include=*.py | head -8
