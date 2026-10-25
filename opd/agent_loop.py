"""VeRL agent loop that runs MedAgentGym BioCoder episodes (multi-turn).

One trajectory = one BioCoder episode:
    prompt (system + task) -> [assistant reply -> environment observation] x up to max_turns
The episode ends when the answer is correct (MedAgentGym's `done`), when max_turns is reached,
or when the token budget (rollout.response_length) runs out.

Token layout follows VeRL's ToolAgentLoop: assistant tokens get response_mask=1 (trained),
observation tokens get response_mask=0 (context only). The same trajectories feed GRPO
(reward_score) and VeRL's built-in on-policy distillation (teacher log-probs are computed by
VeRL on prompt+response after this loop returns; the mask keeps env tokens out of the loss).

Register via an agent-loop config file (configs/agent_loop.yaml) and set
`actor_rollout_ref.rollout.agent.agent_loop_config_path` to it. Each dataset row needs
`agent_name: medagentgym` and `extra_info: {mode, task_index}` (see scripts/build_rl_data.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from opd.biocoder_env import BioCoderEpisode

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("medagentgym")
class MedAgentGymAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        data_path: str | None = None,
        max_turns: int = 15,
        debugger: str = "policy",  # "policy" (the model being trained) | "none"
        debugger_max_tokens: int = 1024,
        format_reward_weight: float = 0.0,
        terminal_timeout: int = 120,
        cache_dir: str | None = None,
        tools=None,  # passed by AgentLoopWorker; unused
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_path = data_path or os.path.join(os.environ.get("MEDAGENTGYM_ROOT", "."), "data", "biocoder")
        self.max_turns = max_turns
        self.debugger = debugger
        self.debugger_max_tokens = debugger_max_tokens
        self.format_reward_weight = format_reward_weight
        self.terminal_timeout = terminal_timeout
        self.response_length = self.rollout_config.response_length
        # validate_code writes ./cache/validation_code_*.py relative to the working directory
        os.makedirs(cache_dir or "cache", exist_ok=True)

    # ------------------------------------------------------------------ helpers

    async def _generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str):
        return await self.server_manager.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
        )

    async def _run_debugger(self, episode: BioCoderEpisode, params: dict, sampling_params: dict) -> str:
        """MedAgentGym's debug(): summary call then debug call, answered by the policy model.
        These calls are side queries: their tokens are NOT part of the trajectory."""
        if self.debugger == "none":
            return "The debug action is not available. Please fix the code yourself."
        summary_prompt, make_debug_prompt = episode.debug_prompts(params)
        side_params = {**sampling_params, "max_tokens": self.debugger_max_tokens, "logprobs": False}
        texts = []
        for content in (summary_prompt, None):
            if content is None:
                content = make_debug_prompt(texts[0])
            ids = await self.apply_chat_template([{"role": "user", "content": content}])
            out = await self._generate(ids, side_params, uuid4().hex)
            texts.append(self.tokenizer.decode(out.token_ids, skip_special_tokens=True))
        return texts[1]

    async def _observation_ids(self, text: str) -> list[int]:
        ids = await self.apply_chat_template([{"role": "user", "content": text}], remove_system_prompt=True)
        # model stopped at <|im_end|>; restore the template's turn separator (see ToolAgentLoop)
        return self.turn_separator + ids

    # ------------------------------------------------------------------ main loop

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        mode = extra.get("mode", "train")
        task_index = int(extra["task_index"])
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        metrics: dict[str, float] = {}

        loop = asyncio.get_running_loop()
        episode = await loop.run_in_executor(
            None, lambda: BioCoderEpisode(self.data_path, mode, task_index, self.terminal_timeout)
        )

        prompt_ids = await self.apply_chat_template(messages)
        all_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        solved, turns, parsed_turns = False, 0, 0
        actions: list[str] = []
        stop_reason = "max_turns"

        while turns < self.max_turns:
            remaining = self.response_length - len(response_mask)
            if remaining <= 0:
                stop_reason = "token_budget"
                break
            gen_params = dict(sampling_params)
            if gen_params.get("max_tokens") is None or gen_params["max_tokens"] > remaining:
                gen_params["max_tokens"] = remaining
            with simple_timer("generate_sequences", metrics):
                out = await self._generate(all_ids, gen_params, request_id)
            turns += 1
            all_ids += out.token_ids
            response_mask += [1] * len(out.token_ids)
            if out.log_probs:
                response_logprobs += out.log_probs

            text = self.tokenizer.decode(out.token_ids, skip_special_tokens=True)
            with simple_timer("tool_calls", metrics):
                result = await loop.run_in_executor(None, episode.step, text)
                if result is None:  # debug action
                    debug_text = await self._run_debugger(episode, episode.pending_params, sampling_params)
                    result = episode.finish_debug(debug_text)
            actions.append(result.action if result.parsed else "parse_error")
            parsed_turns += int(result.parsed)

            if result.done:
                solved = result.reward > 0
                stop_reason = "solved"
                break
            if turns >= self.max_turns:
                break

            obs_ids = await self._observation_ids(result.observation)
            if len(response_mask) + len(obs_ids) >= self.response_length:
                stop_reason = "token_budget"
                break
            all_ids += obs_ids
            response_mask += [0] * len(obs_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(obs_ids)

        response_ids = all_ids[len(prompt_ids):]
        format_score = parsed_turns / max(turns, 1)
        reward = float(solved) + self.format_reward_weight * format_score

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            reward_score=reward,
            num_turns=2 * turns,
            metrics=metrics,
            extra_fields={
                "task_index": task_index,
                "mode": mode,
                "solved": float(solved),
                "assistant_turns": turns,
                "format_score": format_score,
                "stop_reason": stop_reason,
                "actions": actions,
            },
        )
