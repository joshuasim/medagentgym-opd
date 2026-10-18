"""Convert MedAgentGym SampledTrajs (BioCoder) into VeRL multi-turn SFT parquet files.

Input:  SampledTrajs/biocoder/*.jsonl  (each file is ONE trajectory: a JSON list of
        {role, content} messages, ending with {"result": ..., "score": ...})
Output: parquet files with a single `messages` column (list of {role, content}),
        which is what verl.utils.dataset.multiturn_sft_dataset expects. VeRL only
        puts loss on assistant turns; system/user (task + env feedback) are context.

Two versions are written:
  full  - every successful trajectory (matches what the paper could have trained on)
  clean - drops trajectories whose function signature also appears in the BioCoder
          TEST set (MedAgentGym's train/test split shares 69 of 157 test functions)

Also writes test_overlap.json: the test-task idx values that overlap with training
functions, so evaluation can be reported on the full and the clean (non-overlap) test set.

Usage:
  python scripts/convert_sft.py \
      --traj_dir path/to/SampledTrajs/biocoder \
      --test_tasks path/to/MedAgentGym/data/biocoder/test_tasks.jsonl \
      --train_tasks path/to/MedAgentGym/data/biocoder/train_tasks.jsonl \
      --out_dir data/sft
"""

import argparse
import glob
import json
import os
import random
import re

import pandas as pd

# signature block may include decorators, ends at the blank lines before the action list
SIG_RE = re.compile(r"signature:\n(.*?)\n\n", re.S)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_traj(path):
    with open(path) as f:
        data = json.load(f)  # despite the .jsonl extension, each file is one JSON list
    msgs = [{"role": m["role"], "content": m["content"]} for m in data if "role" in m]
    outcome = next((m for m in data if "result" in m), {})
    return msgs, outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True)
    ap.add_argument("--test_tasks", required=True)
    ap.add_argument("--train_tasks", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test = load_jsonl(args.test_tasks)
    train = load_jsonl(args.train_tasks)
    test_sigs = {norm(t["signature"]) for t in test}
    train_sigs = {norm(t["signature"]) for t in train}

    rows, skipped = [], {"not_success": 0, "no_signature": 0, "bad_roles": 0}
    for path in sorted(glob.glob(os.path.join(args.traj_dir, "*.jsonl"))):
        msgs, outcome = load_traj(path)
        if outcome.get("score") != 1:
            skipped["not_success"] += 1
            continue
        if not msgs or msgs[0]["role"] != "system" or not any(m["role"] == "assistant" for m in msgs):
            skipped["bad_roles"] += 1
            continue
        m = SIG_RE.search(msgs[0]["content"])
        if not m:
            skipped["no_signature"] += 1
            continue
        sig = norm(m.group(1))
        rows.append({
            "messages": msgs,
            "source_file": os.path.basename(path),
            "signature": m.group(1),
            "in_test": sig in test_sigs,
            "in_train": sig in train_sigs,
            "n_assistant_turns": sum(x["role"] == "assistant" for x in msgs),
        })

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)

    def split_and_write(frame, name):
        idx = list(frame.index)
        random.Random(args.seed).shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_frac))
        val, tr = frame.loc[idx[:n_val]], frame.loc[idx[n_val:]]
        tr.to_parquet(os.path.join(args.out_dir, f"sft_{name}_train.parquet"), index=False)
        val.to_parquet(os.path.join(args.out_dir, f"sft_{name}_val.parquet"), index=False)
        return len(tr), len(val)

    full = split_and_write(df, "full")
    clean = split_and_write(df[~df["in_test"]], "clean")

    overlap = [t["idx"] for t in test if norm(t["signature"]) in train_sigs]
    with open(os.path.join(args.out_dir, "test_overlap.json"), "w") as f:
        json.dump({"overlapping_test_idx": overlap,
                   "n_test": len(test), "n_overlap": len(overlap),
                   "n_clean": len(test) - len(overlap)}, f, indent=2)

    stats = {
        "trajectories_kept": len(df),
        "skipped": skipped,
        "trajectories_on_test_functions": int(df["in_test"].sum()),
        "trajectories_not_in_train_tasks": int((~df["in_train"]).sum()),
        "assistant_turns_mean": round(float(df["n_assistant_turns"].mean()), 2),
        "single_turn_fraction": round(float((df["n_assistant_turns"] == 1).mean()), 3),
        "full_train_val": full,
        "clean_train_val": clean,
        "test_overlap": f"{len(overlap)}/{len(test)}",
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
