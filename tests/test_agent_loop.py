"""CPU tests for opd.agent_loop.MedAgentGymAgentLoop (no GPU, no vLLM).

A scripted fake LLM server replaces vLLM: it returns pre-written assistant replies as token ids.
The episode itself is real: replies are parsed by MedAgentGym's parser, code runs through
MedAgentGym's validate_code inside the task context, and correctness comes from
BiocoderTask.validate.

Run:
  MEDAGENTGYM_ROOT=../MedAgentGym PATH=<python-with-task-deps>/bin:$PATH \
  python tests/test_agent_loop.py
(`python` on PATH executes task code, so it needs MedAgentGym's deps + pysam/centrosome.)
"""

import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from opd.agent_loop import MedAgentGymAgentLoop  # noqa: E402
from opd.biocoder_env import build_initial_messages, load_tasks  # noqa: E402

ROOT = os.environ.get("MEDAGENTGYM_ROOT", "../MedAgentGym")
DATA = os.path.join(ROOT, "data", "biocoder")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "Qwen/Qwen2.5-7B-Instruct"))
IM_END = TOK.convert_tokens_to_ids("<|im_end|>")


class ScriptedServer:
    """Stands in for VeRL's LLM server. Replies in order; debugger side-calls get `debug_reply`."""

    def __init__(self, replies, debug_reply="Use a list comprehension."):
        self.replies = list(replies)
        self.debug_reply = debug_reply
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        prompt_text = TOK.decode(prompt_ids)
        is_debug_call = "You are a Python debugging expert" in prompt_text or "You are a bioinformatics expert. Your task is to summarize" in prompt_text
        text = self.debug_reply if is_debug_call else self.replies.pop(0)
        self.calls.append(("debug" if is_debug_call else "policy", sampling_params.get("max_tokens")))
        return TokenOutput(token_ids=TOK.encode(text, add_special_tokens=False) + [IM_END])


def make_loop(server, max_turns=15, response_length=8192, debugger="policy"):
    cfg = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"prompt_length": 4096, "response_length": response_length}},
        "data": {"apply_chat_template_kwargs": {}, "mm_processor_kwargs": {},
                 "continuous_token": {"enable": False}},
    })
    return MedAgentGymAgentLoop(
        trainer_config=DictConfigWrap(cfg), server_manager=server, tokenizer=TOK, processor=None,
        dataset_cls=None, data_config=DictConfigWrap(cfg.data),
        data_path=DATA, max_turns=max_turns, debugger=debugger, cache_dir=os.path.join(REPO, "cache"),
    )


def action(code):
    return json.dumps({"action": "validate_code", "params": {"code": code}}, indent=2)


def run(server, mode, task_index, merge=False, **loop_kwargs):
    """Build the agent loop inside the running event loop (as VeRL's worker does) and run one episode."""
    task = load_tasks(DATA, mode)[task_index]
    kwargs = {"raw_prompt": build_initial_messages(task, merge), "extra_info": {"mode": mode, "task_index": task_index}}

    async def go():
        return await make_loop(server, **loop_kwargs).run({"temperature": 0.0}, **kwargs)

    return asyncio.run(go())


def pick_tasks():
    valid = json.load(open(os.path.join(REPO, "data/checks/test_valid_subsets.json")))["valid_idx"]
    free = {r[0] for r in json.load(open(os.path.join(REPO, "data/checks/test_free_credit.json")))}
    test = load_tasks(DATA, "test")
    by_idx = {t["idx"]: i for i, t in enumerate(test)}
    return [by_idx[i] for i in valid[:3]], by_idx[sorted(free)[0]]


def dummy(task):
    name = re.search(r"def\s+(\w+)", task["signature"]).group(1)
    return f"def {name}(*args, **kwargs):\n    return None"


def check_alignment(out, messages, replies_and_obs):
    """Token ids built turn by turn must equal the chat template of the whole conversation."""
    convo = list(messages)
    for role, text in replies_and_obs:
        convo.append({"role": role, "content": text})
    full = TOK.apply_chat_template(convo, tokenize=False)
    built = TOK.decode(out.prompt_ids + out.response_ids)
    assert full.startswith(built), "incremental tokens diverge from chat template"
    # trained tokens are exactly the assistant replies (+ their <|im_end|>)
    trained = TOK.decode([t for t, m in zip(out.response_ids, out.response_mask) if m])
    expected = "".join(t + "<|im_end|>" for r, t in replies_and_obs if r == "assistant")
    assert trained == expected, f"mask mismatch:\n{trained[:200]}\n!=\n{expected[:200]}"


def main():
    tests = load_tasks(DATA, "test")
    (t1, t2, t3), free_t = pick_tasks()
    results = {}

    # 1. reference solution -> solved in one turn, reward 1
    task = tests[t1]
    server = ScriptedServer([action(task["solution"])])
    out = run(server, "test", t1)
    assert out.reward_score == 1.0 and out.extra_fields["assistant_turns"] == 1, out.extra_fields
    check_alignment(out, build_initial_messages(task), [("assistant", action(task["solution"]))])
    results["reference solution solves in 1 turn"] = "pass"

    # 2. wrong -> parse error -> debug -> right: 4 assistant turns, observations masked out
    task = tests[t2]
    replies = [action(dummy(task)), "```python\nprint('not json')\n```",
               json.dumps({"action": "debug", "params": {"code": "x", "error_msg": "y"}}),
               action(task["solution"])]
    server = ScriptedServer(replies)
    out = run(server, "test", t2)
    fx = out.extra_fields
    assert out.reward_score == 1.0 and fx["assistant_turns"] == 4, fx
    assert fx["actions"] == ["validate_code", "parse_error", "debug", "validate_code"], fx["actions"]
    assert sum(1 for c in server.calls if c[0] == "debug") == 2  # summary + debug side calls
    # rebuild the conversation the env produced to check token alignment
    from opd.biocoder_env import BioCoderEpisode
    ep = BioCoderEpisode(DATA, "test", t2)
    o1 = ep.step(replies[0]).observation
    o2 = ep.step(replies[1]).observation
    ep.step(replies[2]); o3 = ep.finish_debug(server.debug_reply).observation
    convo = [("assistant", replies[0]), ("user", o1), ("assistant", replies[1]), ("user", o2),
             ("assistant", replies[2]), ("user", o3), ("assistant", replies[3])]
    check_alignment(out, build_initial_messages(task), convo)
    assert o2.startswith("error: str("), o2
    results["wrong -> parse error -> debug -> right (4 turns, mask + alignment)"] = "pass"

    # 3. always wrong -> stops at max_turns with reward 0
    task = tests[t3]
    server = ScriptedServer([action(dummy(task))] * 3)
    out = run(server, "test", t3, max_turns=3)
    assert out.reward_score == 0.0 and out.extra_fields["assistant_turns"] == 3, out.extra_fields
    assert out.extra_fields["stop_reason"] == "max_turns"
    results["wrong answers -> reward 0 after max_turns"] = "pass"

    # 4. token budget is respected
    server = ScriptedServer([action(dummy(task))] * 15)
    out = run(server, "test", t3, response_length=600)
    assert len(out.response_ids) <= 600 and out.extra_fields["stop_reason"] == "token_budget", out.extra_fields
    results["response_length budget respected"] = "pass"

    # 5. free-credit task: a dummy function is scored correct (why we filter tasks)
    task = tests[free_t]
    server = ScriptedServer([action(dummy(task))])
    out = run(server, "test", free_t)
    assert out.reward_score == 1.0, "expected free credit"
    results["free-credit task gives reward to a dummy (expected)"] = "pass"

    # 6. merged-system prompt format (MedAgentGym's vLLM path) also works
    task = tests[t1]
    server = ScriptedServer([action(task["solution"])])
    out = run(server, "test", t1, merge=True)
    assert out.reward_score == 1.0
    results["merged system prompt format"] = "pass"

    for k, v in results.items():
        print(f"[{v}] {k}")
    print(f"ALL {len(results)} TESTS PASSED")


if __name__ == "__main__":
    main()
