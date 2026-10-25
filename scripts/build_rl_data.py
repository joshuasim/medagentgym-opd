"""Build VeRL RL/eval parquet files for BioCoder (used by GRPO and OPD runs).

Each row is one task. VeRL's RLHFDataset passes `prompt` to the agent loop as `raw_prompt`;
`agent_name` routes the row to opd.agent_loop.MedAgentGymAgentLoop; `extra_info` tells the
loop which task to load.

Outputs (in --out_dir):
  train_valid.parquet   the 429 training tasks with a meaningful reward (data/checks/train_valid.json)
  train_all.parquet     all 981 training tasks (for reference; don't train GRPO on these)
  test_all.parquet      all 157 test tasks (score full / valid-78 / valid-clean-30 afterwards)

Usage:
  MEDAGENTGYM_ROOT=../MedAgentGym python scripts/build_rl_data.py --out_dir data/rl
  (add --merge_system to reproduce MedAgentGym's vLLM prompt format)
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opd.biocoder_env import build_initial_messages, load_tasks  # noqa: E402

DATA_SOURCE = "medagentgym_biocoder"


def rows(tasks, mode, keep_idx=None, merge_system=False):
    out = []
    for i, t in enumerate(tasks):
        if keep_idx is not None and t["idx"] not in keep_idx:
            continue
        out.append({
            "data_source": DATA_SOURCE,
            "prompt": build_initial_messages(t, merge_system_into_user=merge_system),
            "agent_name": "medagentgym",
            "ability": "biocoder",
            "reward_model": {"style": "rule", "ground_truth": ""},  # reward computed in the agent loop
            "extra_info": {"mode": mode, "task_index": i, "idx": t["idx"], "index": len(out)},
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("MEDAGENTGYM_ROOT", "../MedAgentGym"), "data", "biocoder"))
    ap.add_argument("--valid_train", default="data/checks/train_valid.json")
    ap.add_argument("--out_dir", default="data/rl")
    ap.add_argument("--merge_system", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = load_tasks(args.data_path, "train")
    test = load_tasks(args.data_path, "test")
    valid = set(json.load(open(args.valid_train))["valid_idx"])

    frames = {
        "train_valid": rows(train, "train", valid, args.merge_system),
        "train_all": rows(train, "train", None, args.merge_system),
        "test_all": rows(test, "test", None, args.merge_system),
    }
    for name, df in frames.items():
        df.to_parquet(os.path.join(args.out_dir, f"{name}.parquet"), index=False)
        print(name, len(df))


if __name__ == "__main__":
    main()
