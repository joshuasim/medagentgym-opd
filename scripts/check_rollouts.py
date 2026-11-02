"""Summarize VeRL rollout dumps (trainer.rollout_data_dir) from a GRPO/OPD run.

Per step: rollouts, mean task reward, solved fraction, assistant turns per episode, and empty
outputs. Turns are counted from the decoded text: each environment observation is a user turn,
so assistant turns = user turns in the response + 1.

Usage: python scripts/check_rollouts.py <rollout_dir> [--show 1]
"""
import argparse
import glob
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout_dir")
    ap.add_argument("--show", type=int, default=0, help="print the first N episodes in full")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.rollout_dir, "*.jsonl")), key=lambda p: int(os.path.basename(p)[:-6]))
    if not files:
        raise SystemExit(f"no rollout dumps in {args.rollout_dir}")
    shown = 0
    for path in files:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        turns = [r["output"].count("\nuser\n") + 1 for r in rows]
        scores = [r["score"] for r in rows]
        empty = sum(not r["output"].strip() for r in rows)
        print(f"step {os.path.basename(path)[:-6]}: {len(rows)} rollouts | reward mean {statistics.mean(scores):.3f} | "
              f"solved {sum(s > 0 for s in scores)}/{len(rows)} | assistant turns mean {statistics.mean(turns):.2f} "
              f"max {max(turns)} | multi-turn {sum(t > 1 for t in turns)}/{len(rows)} | empty {empty}")
        for r in rows:
            if shown < args.show:
                print("-" * 80, f"\nscore={r['score']}\n", r["output"][:4000])
                shown += 1


if __name__ == "__main__":
    main()
