"""Build a multi-dataset SFT set from MedAgentGym SampledTrajs (closer to the paper's SFT warm-up).

Prints counts and token-length statistics ONLY. It never prints trajectory text: the mimic_iii, eicu and
treqs trajectories are derived from PhysioNet data, so their content stays on Torch (PhysioNet DUA;
no sharing, no pasting into online tools). Write the output OUTSIDE the git repo (--out_dir under $SCRATCH).

BioCoder rows are taken unchanged from data/sft/sft_full_{train,val}.parquet, so the BioCoder part and its
17-trajectory validation split are identical to the current SFT model. Other datasets: every successful
trajectory (score == 1), 5% per dataset held out for validation (seed 42), trajectories longer than
--max_tokens dropped (not truncated, same rule as filter_sft_length.py).

Usage (Torch login node):
  $TRAIN_PY scripts/convert_sft_multi.py --traj_root $SCRATCH/capstone/data/SampledTrajs \
      --out_dir $SCRATCH/capstone/data/sft_multi --max_tokens 10240
"""
import argparse, glob, json, os, random, stat

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

OPEN = ["biodsbench", "medcalcbench", "medagentbench"]  # public sources (MedAgentBench: MIT repo, FHIR data in a public Docker image, no agreement)
PHYSIONET = ["mimic_iii", "eicu", "treqs"]     # need PhysioNet credentialed access (TREQS is built on MIMIC-III)
# not included by default: ehrshot (needs the EHRSHOT Credentialed Health Data License via Redivis; add with --datasets once signed)


def load_traj(path):
    with open(path) as f:
        data = json.load(f)  # each .jsonl file is one JSON list
    msgs = [{"role": m["role"], "content": str(m["content"])} for m in data if m.get("role")]
    outcome = next((m for m in data if "score" in m), {})
    return msgs, outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--datasets", default=",".join(OPEN + PHYSIONET))
    ap.add_argument("--biocoder_dir", default="data/sft")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=10240)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if os.path.abspath(args.out_dir).startswith(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))):
        raise SystemExit("--out_dir is inside the git repo; put it under $SCRATCH so it can't be committed")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ntok = lambda msgs: len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"])

    train, val, report = [], [], []
    for split, sink in [("train", train), ("val", val)]:
        df = pd.read_parquet(os.path.join(args.biocoder_dir, f"sft_full_{split}.parquet"))
        for m, f in zip(df["messages"], df["source_file"]):
            msgs = [{"role": x["role"], "content": x["content"]} for x in m]
            if ntok(msgs) <= args.max_tokens:
                sink.append({"messages": msgs, "source": "biocoder", "source_file": f"biocoder/{f}"})

    for ds in [d for d in args.datasets.split(",") if d]:
        files = sorted(glob.glob(os.path.join(args.traj_root, ds, "*.jsonl")))
        rows, n_fail, n_bad, lens = [], 0, 0, []
        for p in files:
            msgs, outcome = load_traj(p)
            if outcome.get("score") != 1:
                n_fail += 1; continue
            if not msgs or msgs[0]["role"] != "system" or not any(m["role"] == "assistant" for m in msgs):
                n_bad += 1; continue
            n = ntok(msgs); lens.append(n)
            if n <= args.max_tokens:
                rows.append({"messages": msgs, "source": ds, "source_file": f"{ds}/{os.path.basename(p)}",
                             "turns": sum(m["role"] == "assistant" for m in msgs)})
        random.Random(args.seed).shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac)) if rows else 0
        val += rows[:n_val]; train += rows[n_val:]
        L = np.array(lens) if lens else np.array([0])
        turns = np.array([r["turns"] for r in rows]) if rows else np.array([0])
        report.append(f"{ds:<14} files {len(files):>5} | not successful {n_fail:>4} | bad format {n_bad:>3} | "
                      f"over {args.max_tokens} tokens {int((L > args.max_tokens).sum()):>4} | kept {len(rows):>5} "
                      f"(val {n_val}) | tokens p50 {int(np.median(L))} p90 {int(np.percentile(L, 90))} max {int(L.max())} | "
                      f"assistant turns mean {turns.mean():.1f}, single-turn {np.mean(turns == 1):.0%}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.chmod(args.out_dir, stat.S_IRWXU)  # 700: only you can read it
    for name, rows in [("train", train), ("val", val)]:
        out = os.path.join(args.out_dir, f"sft_multi_{name}.parquet")
        pd.DataFrame([{k: r[k] for k in ("messages", "source", "source_file")} for r in rows]).to_parquet(out, index=False)
        os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
    print("\n".join(report))
    by = pd.Series([r["source"] for r in train]).value_counts().to_dict()
    print(f"TOTAL train {len(train)} / val {len(val)} | train by source {by}")
    print(f"written to {args.out_dir} (mode 700)")


if __name__ == "__main__":
    main()
