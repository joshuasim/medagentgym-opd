"""Read-only: are GRPO's all-pass groups the tasks the SFT model was trained on, and the all-fail groups the unseen ones?
    $TRAIN_PY scripts/groups_by_sft.py [run_name ...]      (default: grpo_2gpu_s0 opd_coder32b_v2_s0)
Labels each RL task 'seen' if its first user message appears in the SFT data (train or val), else 'unseen',
then counts all-fail / all-pass / mixed groups per label. Also prints SFT row counts (which model saw how many)."""
import collections, glob, json, os, sys
import pandas as pd

S = os.environ["SCRATCH"]
runs = sys.argv[1:] or ["grpo_2gpu_s0", "opd_coder32b_v2_s0"]

def user_text(msgs):
    msgs = list(msgs)
    return next(m["content"] for m in msgs if m["role"] == "user").strip()

for name in ["sft_full", "sft_full_e2"]:
    for split in ["train", "val"]:
        p = f"{S}/capstone/checkpoints/{name}/data/sft_full_{split}.parquet"
        if os.path.exists(p):
            print(f"{name} {split}: {len(pd.read_parquet(p))} trajectories")

seen = set()
for p in glob.glob("data/sft/sft_full_*.parquet"):
    for m in pd.read_parquet(p)["messages"]:
        seen.add(user_text(m)[:400])
rl = pd.read_parquet("data/rl/train_valid.parquet")
keys = [(user_text(p)[:400], str(e["idx"])) for p, e in zip(rl["prompt"], rl["extra_info"])]
n_seen = sum(k in seen for k, _ in keys)
print(f"RL tasks: {len(keys)} | with an SFT trajectory: {n_seen} | without: {len(keys) - n_seen}")

for r in runs:
    tot = collections.defaultdict(collections.Counter)
    unmatched = 0
    for f in glob.glob(f"{S}/capstone/runs/{r}/rollouts/*.jsonl"):
        groups = collections.defaultdict(list)
        for line in open(f):
            if not line.strip(): continue
            x = json.loads(line)
            groups[str(x.get("uid", "")).rsplit("_", 2)[0]].append(x)
        for g in groups.values():
            key = next((k for k, _ in keys if k[:200] in g[0]["input"]), None)
            if key is None: unmatched += 1; continue
            lab = "seen" if key in seen else "unseen"
            ok = [x["score"] > 0 for x in g]
            tot[lab]["all_fail" if not any(ok) else "all_pass" if all(ok) else "mixed"] += 1
    print(f"== {r} (unmatched groups: {unmatched})")
    for lab, c in tot.items():
        n = sum(c.values())
        print(f"  {lab:<7} {n:>4} groups | all-fail {c['all_fail']/n:.0%} | all-pass {c['all_pass']/n:.0%} | mixed {c['mixed']/n:.0%}")
