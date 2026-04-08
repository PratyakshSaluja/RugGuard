"""
inference.py — RugGuard baseline inference runner.

Runs an LLM agent against the RugGuard OpenEnv environment, classifying
crypto tokens across three tasks (contract / transaction / liquidity
analysis) and emitting structured logs in the exact format required by
the OpenEnv Round 1 evaluator.

Environment variables (per Round 1 submission spec):
    API_BASE_URL      OpenAI-compatible LLM endpoint injected by validator
                      (default: https://router.huggingface.co/v1 for local dev)
    MODEL_NAME        Model identifier
                      (default: Qwen/Qwen2.5-72B-Instruct)
    API_KEY           API key injected by the validator's LiteLLM proxy.
                      Falls back to HF_TOKEN for local dev runs.
    LOCAL_IMAGE_NAME  Docker image tag to spin up via from_docker_image()
                      (optional; if unset, falls back to RUGGUARD_URL or
                      a default local image "rugguard-env:latest")
    RUGGUARD_URL      Direct base URL override for an already-running env
                      (optional; only used when LOCAL_IMAGE_NAME is unset)

STDOUT format (one line per event, exact key=value layout):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

One [START] / [STEP]* / [END] block is emitted per task type (3 total),
so each task is graded independently in the [0, 1] range as the spec
requires.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Make the sibling `rugguard_env` package importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    from rugguard_env import RugGuardAction, RugGuardEnv
except ImportError:
    # Running from inside rugguard_env/ itself
    from client import RugGuardEnv  # type: ignore
    from models import RugGuardAction  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — per Round 1 submission checklist:
#   * Defaults are set ONLY for API_BASE_URL and MODEL_NAME
#   * HF_TOKEN has NO default (must be supplied via env var)
#   * LOCAL_IMAGE_NAME is optional and used with from_docker_image()
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Validator injects API_KEY for their LiteLLM proxy; HF_TOKEN is a local-dev fallback.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
RUGGUARD_URL = os.getenv("RUGGUARD_URL")

BENCHMARK = "rugguard_env"
TEMPERATURE = 0.2
MAX_TOKENS = 512
DEFAULT_LOCAL_IMAGE = "rugguard-env:latest"
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_ORDER = ["contract_analysis", "transaction_analysis", "liquidity_analysis"]


# ---------------------------------------------------------------------------
# Structured log helpers — exact key=value format required by evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_str = "null" if error is None else error.replace("\n", " ").replace("\r", " ")
    action_clean = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={err_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call (OpenAI client — mandated by spec)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a blockchain security expert specialising in crypto token scam "
    "detection. Always respond with a single valid JSON object and nothing else."
)


def build_user_prompt(obs: Dict[str, Any]) -> str:
    task_desc = {
        "contract_analysis": "Analyse this Solidity smart contract code for red flags.",
        "transaction_analysis": "Analyse this on-chain transaction pattern for scam signals.",
        "liquidity_analysis": "Analyse this liquidity pool data for exit risk or manipulation.",
    }.get(obs["task_type"], "Analyse the token data.")

    return (
        f"Task: {task_desc}\n\n"
        f"Token: {obs['token_name']}\n\n"
        f"Data:\n{obs['token_data']}\n\n"
        "Classify this token. Respond ONLY with this JSON:\n"
        "{\n"
        '  "verdict": "<rug_pull|honeypot|wash_trading|safe>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning": "<concise explanation>"\n'
        "}\n\n"
        "Definitions:\n"
        "- rug_pull: developer drains liquidity/funds and abandons project\n"
        "- honeypot: users can buy but cannot sell (trapped funds)\n"
        "- wash_trading: artificial volume via coordinated self-trades\n"
        "- safe: no significant scam signals detected\n"
    )


def get_verdict(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.startswith("```")
            ).strip()
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(f"LLM call failed: {exc}")
        parsed = {"verdict": "safe", "confidence": 0.5, "reasoning": f"error: {exc}"}

    valid_verdicts = {"rug_pull", "honeypot", "wash_trading", "safe"}
    if parsed.get("verdict") not in valid_verdicts:
        parsed["verdict"] = "safe"
    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.5
    if not isinstance(parsed.get("reasoning"), str):
        parsed["reasoning"] = "no reasoning"

    return parsed


# ---------------------------------------------------------------------------
# Per-task log buffer
# ---------------------------------------------------------------------------

class TaskRunner:
    """Collects steps for one task and emits a single START/STEP*/END block."""

    def __init__(self, task_name: str, model_name: str):
        self.task_name = task_name
        self.model_name = model_name
        self.rewards: List[float] = []
        self.steps: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None

    def record_step(
        self,
        action_str: str,
        reward: float,
        done: bool,
        error: Optional[str],
    ) -> None:
        self.rewards.append(reward)
        self.steps.append(
            {"action": action_str, "reward": reward, "done": done, "error": error}
        )
        if error:
            self.last_error = error

    def emit(self) -> Dict[str, Any]:
        log_start(task=self.task_name, env_name=BENCHMARK, model=self.model_name)
        for i, s in enumerate(self.steps, start=1):
            log_step(
                step=i,
                action=s["action"],
                reward=s["reward"],
                done=s["done"],
                error=s["error"],
            )
        max_reward = float(len(self.steps)) if self.steps else 1.0
        score = sum(self.rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=success,
            steps=len(self.steps),
            score=score,
            rewards=self.rewards,
        )
        return {
            "task": self.task_name,
            "score": score,
            "success": success,
            "steps": len(self.steps),
        }


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------

async def create_env() -> RugGuardEnv:
    """
    Create and connect a RugGuardEnv client.

    Preference order:
      1. LOCAL_IMAGE_NAME / IMAGE_NAME → spin up Docker container
      2. RUGGUARD_URL → connect to an already-running server
      3. Fall back to DEFAULT_LOCAL_IMAGE
    """
    image = LOCAL_IMAGE_NAME
    url = RUGGUARD_URL

    if image:
        logger.info(f"Starting env container from image: {image}")
        return await RugGuardEnv.from_docker_image(image)

    if url:
        logger.info(f"Connecting to existing env at: {url}")
        env = RugGuardEnv(base_url=url)
        await env.connect()
        return env

    logger.info(f"No LOCAL_IMAGE_NAME/RUGGUARD_URL set; using default image: {DEFAULT_LOCAL_IMAGE}")
    return await RugGuardEnv.from_docker_image(DEFAULT_LOCAL_IMAGE)


def obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert a Pydantic observation to a plain dict for prompt building."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs) if obs else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run() -> None:
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    runners: Dict[str, TaskRunner] = {
        t: TaskRunner(t, MODEL_NAME) for t in TASK_ORDER
    }

    fatal_error: Optional[str] = None
    env: Optional[RugGuardEnv] = None

    try:
        env = await create_env()

        reset_result = await env.reset()
        obs = reset_result.observation
        logger.info(f"Episode started — first token: {getattr(obs, 'token_name', '?')}")

        max_steps = int(getattr(obs, "total_steps", 45) or 45)
        done = bool(getattr(reset_result, "done", False))

        for _ in range(max_steps):
            if done:
                break

            obs_dict = obs_to_dict(obs)
            current_task = obs_dict.get("task_type", "contract_analysis")
            runner = runners.get(current_task)
            if runner is None:
                logger.warning(f"Unknown task type: {current_task}")
                break

            verdict_data = get_verdict(openai_client, obs_dict)
            action_str = (
                f"{verdict_data['verdict']}|conf={verdict_data['confidence']:.2f}|"
                f"{verdict_data['reasoning'][:100]}"
            )

            step_error: Optional[str] = None
            try:
                action = RugGuardAction(
                    verdict=verdict_data["verdict"],
                    confidence=verdict_data["confidence"],
                    reasoning=verdict_data["reasoning"],
                )
                step_result = await env.step(action)
                reward = float(getattr(step_result, "reward", 0.0) or 0.0)
                done = bool(getattr(step_result, "done", False))
                obs = step_result.observation
            except Exception as exc:
                step_error = str(exc)
                reward = 0.0
                done = True
                logger.error(f"Step error: {exc}")

            runner.record_step(
                action_str=action_str,
                reward=reward,
                done=done,
                error=step_error,
            )
            logger.info(
                f"{current_task} | token={obs_dict.get('token_name','?')} | "
                f"verdict={verdict_data['verdict']} | reward={reward:.4f}"
            )

    except Exception as exc:
        fatal_error = str(exc)
        logger.error(f"Episode error: {exc}")

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                logger.warning(f"env.close() failed: {exc}")

    # Emit one [START]/[STEP]*/[END] block per task, in order
    summaries = []
    for task in TASK_ORDER:
        runner = runners[task]
        if not runner.steps:
            # Still emit a minimal block so the evaluator sees all 3 tasks
            log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)
            err_msg = fatal_error or "no steps recorded"
            log_step(step=1, action="error", reward=0.0, done=True, error=err_msg)
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
            summaries.append({"task": task, "score": 0.0, "success": False})
        else:
            summaries.append(runner.emit())

    logger.info("=" * 60)
    for s in summaries:
        logger.info(
            f"{s['task']:>22}: score={s['score']:.2f} success={s['success']}"
        )
    logger.info("=" * 60)


def main() -> None:
    try:
        asyncio.run(run())
    except Exception as exc:
        # Fallback — ensure we never exit with an unhandled exception
        logger.error(f"Fatal inference error: {exc}")
        for task in TASK_ORDER:
            log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)
            log_step(
                step=1,
                action="fatal_error",
                reward=0.0,
                done=True,
                error=str(exc),
            )
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        sys.exit(0)  # exit 0 so the evaluator still reads the logs


if __name__ == "__main__":
    main()
