"""Merge a VeRL LoRA SFT checkpoint into the base model, so vLLM and the RL trainer can load it.

VeRL's own `verl.model_merger` writes the adapter to <target>/lora_adapter but leaves the base
weights unmerged, so this script runs it and then merges with PEFT.

Usage:
  python scripts/merge_lora.py --ckpt_dir <sft run dir> --base Qwen/Qwen2.5-7B-Instruct --out <merged dir>
  (uses the latest global_step_* in ckpt_dir)
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def latest_step(ckpt_dir):
    steps = glob.glob(os.path.join(ckpt_dir, "global_step_*"))
    if not steps:
        sys.exit(f"no global_step_* in {ckpt_dir}")
    return max(steps, key=lambda p: int(re.search(r"global_step_(\d+)", p).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    step_dir = latest_step(args.ckpt_dir)
    print("checkpoint:", step_dir)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(args.out)), prefix="merge_tmp_")
    subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                    "--local_dir", step_dir, "--target_dir", tmp], check=True)
    adapter = os.path.join(tmp, "lora_adapter")
    if not os.path.isfile(os.path.join(adapter, "adapter_model.safetensors")):
        sys.exit(f"verl.model_merger wrote no LoRA adapter in {adapter}")

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    ref = base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    model = PeftModel.from_pretrained(base, adapter)

    # lora_B starts at zero, so an all-zero lora_B means the adapter keys did not load
    lora_B = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    zero = [n for n, p in lora_B.items() if p.abs().max().item() == 0]
    print(f"LoRA B matrices: {len(lora_B)} loaded, {len(zero)} all-zero")
    if not lora_B or zero:
        sys.exit("adapter did not load correctly (zero lora_B) - check key names in adapter_model.safetensors")

    model = model.merge_and_unload()
    new = model.model.layers[0].self_attn.q_proj.weight.detach()
    print(f"layer-0 q_proj relative change: {((new - ref).norm() / ref.norm()).item():.4f}")

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("merged model saved to", args.out)
    subprocess.run(["rm", "-rf", tmp])


if __name__ == "__main__":
    main()
