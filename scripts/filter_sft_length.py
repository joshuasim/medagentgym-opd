"""Drop SFT trajectories longer than --max_tokens (Qwen chat template), instead of truncating them.

A truncated trajectory loses its final turns, including the answer that made it a success,
so we would be training on an unfinished episode. In sft_full_train, 2 of 334 are over 8,192:
347.jsonl (33,442 tokens, mostly long environment output) and 97.jsonl (8,597).

Usage: python scripts/filter_sft_length.py --tokenizer Qwen/Qwen2.5-7B-Instruct --max_tokens 8192 \
           --out_dir <dir> data/sft/sft_full_train.parquet data/sft/sft_full_val.parquet
"""
import argparse
import os

import pandas as pd
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.files:
        df = pd.read_parquet(path)
        n_tok = df["messages"].apply(lambda m: len(tok.apply_chat_template(
            [{"role": x["role"], "content": x["content"]} for x in m], tokenize=True, return_dict=True)["input_ids"]))
        keep = n_tok <= args.max_tokens
        for f, n in zip(df.loc[~keep, "source_file"], n_tok[~keep]):
            print(f"  drop {os.path.basename(path)}: {f} ({n} tokens)")
        out = os.path.join(args.out_dir, os.path.basename(path))
        df[keep].to_parquet(out, index=False)
        print(f"{os.path.basename(path)}: kept {keep.sum()}/{len(df)} -> {out}")


if __name__ == "__main__":
    main()
