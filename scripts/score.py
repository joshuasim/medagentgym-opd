"""Score per-task results three ways: all 157, valid 78, valid & non-overlapping 30.

Usage: python scripts/score.py results/<run>/per_task.jsonl
"""
import json
import os
import sys


def score_results(results, checks_dir="data/checks"):
    subsets = json.load(open(os.path.join(checks_dir, "test_valid_subsets.json")))
    valid, clean = set(subsets["valid_idx"]), set(subsets["valid_clean_idx"])
    solved = {r["idx"] for r in results if r["solved"]}
    n_all = len(results)
    n_valid = sum(r["idx"] in valid for r in results)
    n_clean = sum(r["idx"] in clean for r in results)

    def pct(k, n):
        return round(100 * k / n, 2) if n else None

    return {
        "all":         {"solved": len(solved), "n": n_all, "pct": pct(len(solved), n_all)},
        "valid":       {"solved": len(solved & valid), "n": n_valid, "pct": pct(len(solved & valid), n_valid)},
        "valid_clean": {"solved": len(solved & clean), "n": n_clean, "pct": pct(len(solved & clean), n_clean)},
        "mean_turns": round(sum(r["turns"] for r in results) / max(n_all, 1), 2),
    }


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(score_results(rows, os.path.join(here, "data", "checks")), indent=2))
