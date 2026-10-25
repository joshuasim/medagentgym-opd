"""MedAgentGym BioCoder episode logic, reused for VeRL rollouts.

This mirrors what MedAgentGym does in `main.py` + `EHREnv.step` + `EHRAgent.act`, but as a
small object that VeRL's agent loop can drive turn by turn. Wherever possible it calls
MedAgentGym's own functions (prompt template, JSON parser, validate_code, task validation)
so behaviour matches the original environment.

Deviations from MedAgentGym (all deliberate, all documented in the README):
  1. Parse failure: MedAgentGym replaces the model's reply with "Error: ... Please
     regenerate the action." in the history. On-policy training needs the model's real
     tokens, so we keep them; the next observation is the same error string MedAgentGym's
     env returns (`"error: str(<exception>)"`).
  2. `terminal` runs with a timeout (MedAgentGym's has none, and its failure branch raises
     NameError because `execution_time` is unset). Messages are otherwise identical.
  3. `debug` needs a second LLM. MedAgentGym builds an Azure gpt-4.1-mini client inside
     EHREnv.__init__. Here the agent loop supplies the debugger (the policy itself by
     default), so no Azure credentials are needed.
  4. Reference answers are computed once per task and cached, instead of re-running the
     reference code on every reset (same function, same output for deterministic tasks).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

_MEDAGENTGYM_ROOT = os.environ.get("MEDAGENTGYM_ROOT")
if _MEDAGENTGYM_ROOT and _MEDAGENTGYM_ROOT not in sys.path:
    sys.path.insert(0, _MEDAGENTGYM_ROOT)

from ehr_gym.agent.parser import parse_llm_response  # noqa: E402
from ehr_gym.agent.prompt import DynamicPrompt  # noqa: E402
from ehr_gym.env.action.function import validate_code  # noqa: E402
from ehr_gym.env.task.biocoder import BiocoderTask, instruction, signature_information  # noqa: E402

PERMITTED_ACTIONS = BiocoderTask.permitted_actions  # ['validate_code', 'debug', 'terminal']
CONTEXT_PATTERN = "<<insert solution here>>"
DEBUG_SUFFIX = "\nPlease then use validate_code action to validate the debugged code."

_SUMMARY_PROMPT = """You are a bioinformatics expert. Your task is to summarize the following requested information:
<requested_information>
{requested_information}
</requested_information>
Please summarize the information in a concise and clear manner:
"""

_DEBUG_PROMPT = """You are a Python debugging expert. Your task is to help the user debug their code.
The user is solving the problem:
{problem}

The user has gotten the following knowledge:
{summary}

The user has provided the following code:
{code}

The user has encountered the following error:
{error_msg}

Please provide an explanation of the error and suggest a solution to fix it.
"""


# --------------------------------------------------------------------------- tasks / prompts

_TASK_CACHE: dict[str, list[dict]] = {}
_ANSWER_CACHE: dict[tuple[str, str], str] = {}
_LOCK = threading.Lock()


def task_file(data_path: str, mode: str) -> str:
    return os.path.join(data_path, "test_tasks.jsonl" if mode == "test" else "train_tasks.jsonl")


def load_tasks(data_path: str, mode: str) -> list[dict]:
    path = task_file(data_path, mode)
    with _LOCK:
        if path not in _TASK_CACHE:
            with open(path) as f:
                _TASK_CACHE[path] = [json.loads(line) for line in f if line.strip()]
        return _TASK_CACHE[path]


def reference_answer(task: dict) -> str:
    """Same as BiocoderTask.setup: the printed output of the task's reference code."""
    key = (task["idx"], task["code"][:64])
    with _LOCK:
        if key in _ANSWER_CACHE:
            return _ANSWER_CACHE[key]
    answer = validate_code(task["code"])["env_message"]
    with _LOCK:
        _ANSWER_CACHE[key] = answer
    return answer


def build_initial_messages(task: dict, merge_system_into_user: bool = False) -> list[dict]:
    """Exactly the first turn EHRAgent.act builds for a BioCoder task.

    merge_system_into_user=True reproduces MedAgentGym's vLLM path (system text prepended to
    the user message); False reproduces its OpenAI/Azure path (separate system message), which
    is also the format of the released SFT trajectories.
    """
    prompt = DynamicPrompt()
    task_instruction = instruction.format(
        signature_info=signature_information.format(signature=task["signature"])
    )
    action_definitions = "\n".join(prompt.action_definition[a] for a in PERMITTED_ACTIONS)
    action_formats = "\nor\n".join(prompt.action_format[a] for a in PERMITTED_ACTIONS)
    system_msg = prompt.prompt_template.format(
        instruction=task_instruction, action_definition=action_definitions, format_output=action_formats
    )
    user_msg = task["problem"]
    if merge_system_into_user:
        return [{"role": "user", "content": system_msg + "\n" + user_msg}]
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


# --------------------------------------------------------------------------- episode

@dataclass
class StepResult:
    observation: str  # text shown to the agent next turn (obs['env_message'])
    reward: float
    done: bool
    action: str
    parsed: bool
    obs_type: str
    info: dict = field(default_factory=dict)


class BioCoderEpisode:
    """One BioCoder episode. Call `step(text)` with each assistant reply.

    For `debug`, call `debug_prompts(params)` to get the two prompts, run them through any LLM,
    then `finish_debug(response)` to get the StepResult.
    """

    def __init__(self, data_path: str, mode: str, task_index: int, terminal_timeout: int = 120):
        self.tasks = load_tasks(data_path, mode)
        self.task_data = self.tasks[task_index]
        self.task_index = task_index
        self.terminal_timeout = terminal_timeout

        # Build a BiocoderTask without calling setup() (which would re-run the reference code),
        # then fill in exactly the fields setup() sets, so we can reuse its validate().
        task = BiocoderTask(task_id=task_index, data_path=data_path, debugger_config=None, mode=mode)
        task.task_list = self.tasks
        task.question = self.task_data["problem"]
        task.answer = reference_answer(self.task_data)
        task.code_id = self.task_data["idx"]
        task.context = self.task_data["context"]
        task.signature = self.task_data["signature"]
        task.context_pattern = CONTEXT_PATTERN
        task.setup_goal()
        self.task = task
        self.env_history: list[dict] = [task._get_obs()]  # initial_observation, used by debug()

    # ---- actions ---------------------------------------------------------------------------

    def parse(self, text: str) -> tuple[str, dict, bool]:
        """MedAgentGym's EHRAgent parsing: returns (action, params, parsed_ok)."""
        try:
            action, params = parse_llm_response(text)
            return action, params, True
        except Exception as e:  # same string MedAgentGym builds on parse failure
            return f"error: str({e})", {}, False

    def _terminal(self, cmd: str) -> dict:
        start = time.time()
        try:
            r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True,
                               timeout=self.terminal_timeout)
            return {"type": "cmd_result", "status": "SUCCESS", "env_message": r.stdout,
                    "execution_time": f"{time.time() - start:.2f}s"}
        except subprocess.CalledProcessError as e:
            return {"type": "cmd_result", "status": "FAILED",
                    "env_message": f"Command '{cmd}' failed with error: {e.stderr}",
                    "execution_time": f"{time.time() - start:.2f}s"}
        except subprocess.TimeoutExpired:
            return {"type": "cmd_result", "status": "FAILED",
                    "env_message": f"Command '{cmd}' failed with error: timed out after {self.terminal_timeout}s",
                    "execution_time": f"{time.time() - start:.2f}s"}

    def _finish(self, action: str, parsed: bool, results: Any) -> StepResult:
        # EHREnv._get_obs normalisation
        if isinstance(results, str):
            obs = {"type": "requested_info", "env_message": results}
        else:
            obs = dict(results)
            obs.setdefault("type", "environment_feedback")
            obs.setdefault("env_message", json.dumps(results, indent=4))
        self.env_history.append(obs)
        reward, done, _user_message, task_info = self.task.validate(None, obs)
        return StepResult(observation=obs["env_message"], reward=float(reward), done=bool(done),
                          action=action, parsed=parsed, obs_type=obs["type"], info=task_info)

    def step(self, text: str) -> Optional[StepResult]:
        """Run one assistant reply. Returns None for a valid `debug` action (caller must run
        the debugger and call finish_debug)."""
        action, params, parsed = self.parse(text)
        self.pending_params = params
        if "error" in action:  # EHREnv.step: parse errors go straight back as the observation
            obs = {"type": "compile_error", "env_message": action}
            self.env_history.append(obs)
            return StepResult(observation=action, reward=0.0, done=False, action=action,
                              parsed=False, obs_type="compile_error")
        try:
            if action == "validate_code":
                code = self.task.context.replace(self.task.context_pattern, "\n" + params["code"] + "\n")
                results = validate_code(code)
            elif action == "terminal":
                results = self._terminal(params["cmd"])
            elif action == "debug":
                return None
            else:  # e.g. request_info: not permitted for BioCoder; MedAgentGym errors out on it
                raise ValueError(f"Action '{action}' is not available for this task")
        except Exception as e:
            results = {"type": "error", "env_message": f"Error: {type(e).__name__}: {e}"}
        return self._finish(action, parsed, results)

    def debug_prompts(self, params: dict) -> tuple[str, callable]:
        """Prompts for MedAgentGym's debug(): a summary call, then the debug call."""
        requested = [o["env_message"] for o in self.env_history if o.get("type") == "requested_info"]
        summary_prompt = _SUMMARY_PROMPT.format(requested_information="\n".join(requested))

        def make_debug_prompt(summary: str) -> str:
            return _DEBUG_PROMPT.format(problem=self.task.goal, summary=summary,
                                        code=params.get("code", ""), error_msg=params.get("error_msg", ""))

        return summary_prompt, make_debug_prompt

    def finish_debug(self, debugger_response: str) -> StepResult:
        results = {"type": "debugging_info", "env_message": debugger_response + DEBUG_SUFFIX}
        return self._finish("debug", True, results)
