"""Evaluate a model on BioCoder test tasks the way MedAgentGym does, against a vLLM
OpenAI-compatible server (`vllm serve <model>`), then score full / valid / valid-clean.

Episode logic comes from opd.biocoder_env (MedAgentGym's prompt, parser, validate_code,
task validation; up to 15 turns; ends when the answer is correct). For evaluation we copy
MedAgentGym's history handling exactly: if a reply fails to parse, the assistant message
stored in the history is replaced by "Error: <e>. Please regenerate the action.".

Usage (inside a GPU job, after `vllm serve` is up):
  python scripts/eval_biocoder.py --model Qwen/Qwen2.5-7B-Instruct --out results/base_qwen7b
Options: --merge_system (MedAgentGym's vLLM prompt format), --debugger policy|none,
         --mode test|train, --limit N, --concurrency 32
"""

import argparse
import asyncio
import logging
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import AsyncOpenAI  # noqa: E402

from ehr_gym.agent.parser import parse_llm_response  # noqa: E402
from opd.biocoder_env import BioCoderEpisode, build_initial_messages, load_tasks  # noqa: E402
from scripts.score import score_results  # noqa: E402


async def chat(client, model, messages, max_tokens, temperature):
    r = await client.chat.completions.create(model=model, messages=messages,
                                             max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content or ""


async def run_episode(args, client, data_path, index, sem):
    async with sem:
        loop = asyncio.get_running_loop()
        ep = await loop.run_in_executor(None, BioCoderEpisode, data_path, args.mode, index)
        history = build_initial_messages(ep.task_data, merge_system_into_user=args.merge_system)
        record = {"index": index, "idx": ep.task_data["idx"], "solved": False, "turns": 0,
                  "actions": [], "stop": "max_turns"}
        for _ in range(args.max_turns):
            try:
                reply = await chat(client, args.model, history, args.max_tokens, args.temperature)
            except Exception as e:  # context too long etc.
                record["stop"] = f"llm_error: {type(e).__name__}"
                break
            record["turns"] += 1
            try:
                parse_llm_response(reply)
                history.append({"role": "assistant", "content": reply})
            except Exception as e:  # MedAgentGym: store a regenerate request instead of the reply
                history.append({"role": "assistant", "content": f"Error: {e}. Please regenerate the action."})
            result = await loop.run_in_executor(None, ep.step, reply)
            if result is None:  # debug action
                if args.debugger == "none":
                    text = "The debug action is not available. Please fix the code yourself."
                else:
                    summary_prompt, make_debug = ep.debug_prompts(ep.pending_params)
                    summary = await chat(client, args.model, [{"role": "user", "content": summary_prompt}], 1024, args.temperature)
                    text = await chat(client, args.model, [{"role": "user", "content": make_debug(summary)}], 1024, args.temperature)
                result = ep.finish_debug(text)
            record["actions"].append(result.action if result.parsed else "parse_error")
            if result.done:
                record["solved"] = result.reward > 0
                record["stop"] = "solved"
                break
            history.append({"role": "user", "content": result.observation})
        if args.save_history:
            record["history"] = history
        return record


async def main_async(args):
    data_path = os.path.join(os.environ["MEDAGENTGYM_ROOT"], "data", "biocoder")
    tasks = load_tasks(data_path, args.mode)
    indices = list(range(len(tasks)))[: args.limit or None]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    results = []
    for coro in asyncio.as_completed([run_episode(args, client, data_path, i, sem) for i in indices]):
        r = await coro
        results.append(r)
        if len(results) % 10 == 0:
            print(f"{len(results)}/{len(indices)} done, solved so far {sum(x['solved'] for x in results)}", flush=True)
    results.sort(key=lambda r: r["index"])
    with open(os.path.join(args.out, "per_task.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = score_results(results, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checks"))
    summary.update({"model": args.model, "merge_system": args.merge_system, "debugger": args.debugger,
                    "temperature": args.temperature, "minutes": round((time.time() - t0) / 60, 1)})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    for name in ("httpx", "httpx2", "ehr_gym", "ehr_gym.env.action.function"):
        logging.getLogger(name).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--mode", default="test")
    ap.add_argument("--merge_system", action="store_true")
    ap.add_argument("--debugger", default="policy", choices=["policy", "none"])
    ap.add_argument("--max_turns", type=int, default=15)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_history", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
