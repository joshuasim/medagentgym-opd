"""Agent loop with an external debugger (experiment added Nov 25; all earlier runs use opd/agent_loop.py).

Identical to MedAgentGymAgentLoop except for who answers the agent's `debug` action. Earlier runs let the
policy answer its own debug requests. Here they go to a separate OpenAI-compatible server
(env DEBUGGER_BASE_URL / DEBUGGER_MODEL, started by train/run_grpo_extdbg.sbatch on its own GPU), the way
MedAgentGym's evaluation lets GPT-4.1-mini answer them. Same two-call protocol (summary, then debug
explanation), same 1,024-token limit as our evals. The debugger's answer enters the trajectory only as an
environment observation (loss mask 0), exactly as before, so the training objective is unchanged.
"""
from __future__ import annotations

import os

from openai import AsyncOpenAI

from opd.agent_loop import MedAgentGymAgentLoop

_WARNED = False


class MedAgentGymExtDebugAgentLoop(MedAgentGymAgentLoop):
    def __init__(self, *args, debugger_temperature: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugger_temperature = debugger_temperature
        self._dbg_client = None

    async def _run_debugger(self, episode, params: dict, sampling_params: dict) -> str:
        if self.debugger != "external":
            return await super()._run_debugger(episode, params, sampling_params)
        if self._dbg_client is None:
            self._dbg_client = AsyncOpenAI(base_url=os.environ["DEBUGGER_BASE_URL"], api_key="EMPTY", timeout=600)
        model = os.environ["DEBUGGER_MODEL"]
        summary_prompt, make_debug_prompt = episode.debug_prompts(params)
        try:
            texts = []
            for content in (summary_prompt, None):
                if content is None:
                    content = make_debug_prompt(texts[0])
                r = await self._dbg_client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}],
                    max_tokens=self.debugger_max_tokens, temperature=self.debugger_temperature)
                texts.append(r.choices[0].message.content or "")
            return texts[1]
        except Exception as e:  # server hiccup: don't kill the episode, but make it visible in the log
            global _WARNED
            if not _WARNED:
                _WARNED = True
                print(f"[external debugger] request failed: {type(e).__name__}: {e}", flush=True)
            return f"The debugger is unavailable ({type(e).__name__}). Please fix the code yourself."
