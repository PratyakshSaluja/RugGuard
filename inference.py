"""
inference.py — RugGuard baseline inference runner.

This is the main entry point that the hackathon validator calls. It spins up
the RugGuard environment (via Docker or a remote URL), then loops through all
45 tokens in an episode using an LLM to classify each one.

For each token the agent does two things:
  1. Investigate — picks the most useful tool for the current task type
     (e.g. contract_functions for contract_analysis) to get extra context
  2. Classify — sends the token data + investigation results to the LLM
     and asks it to return a verdict (rug_pull/honeypot/wash_trading/safe)

The LLM prompt gives clear definitions of each scam type and asks for
structured JSON output with verdict, confidence, and reasoning.

Environment variables the validator injects:
    API_BASE_URL      OpenAI-compatible LLM endpoint (we default to HF router)
    MODEL_NAME        Which model to use (default: Qwen2.5-72B-Instruct)
    API_KEY           API key for the LLM endpoint
    LOCAL_IMAGE_NAME  Docker image tag — set this to run the env in a container
    RUGGUARD_URL      Or set this to point at an already-running env server

Structured logging on STDOUT (validator parses these):
    [START] task=<name> env=rugguard_env model=<model>
    [STEP]  step=<n> action=<verdict|conf=X|...> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<comma-separated>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    from rugguard_env import RugGuardAction, RugGuardEnv
except ImportError:
    from client import RugGuardEnv  # type: ignore
    from models import RugGuardAction  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — mostly from env vars, with sensible defaults for local testing
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
RUGGUARD_URL = os.getenv("RUGGUARD_URL")

BENCHMARK = "rugguard_env"
TEMPERATURE = 0.2       # low temp for consistent classifications
MAX_TOKENS = 512        # enough for JSON verdict + reasoning
DEFAULT_LOCAL_IMAGE = "rugguard-env:latest"
SUCCESS_SCORE_THRESHOLD = 0.5  # score >= 0.5 counts as success

TASK_ORDER = ["contract_analysis", "transaction_analysis", "liquidity_analysis"]

# We only do 1 investigation per token to keep things fast. The reward function
# actually gives a bonus for fewer investigations when you're correct, so being
# selective pays off more than being thorough.
INVESTIGATIONS_PER_TOKEN = 1

# For each task type, these are the tools most likely to give useful signal.
# Ordered by relevance — we pick the first one that's still available.
PREFERRED_TOOLS = {
    "contract_analysis": ["contract_functions", "similar_contracts", "deployer_history"],
    "transaction_analysis": ["holder_distribution", "deployer_history", "price_history"],
    "liquidity_analysis": ["holder_distribution", "price_history", "similar_contracts"],
}


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_str = "null" if error is None else error.replace("\n", " ").replace("\r", " ")
    action_clean = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompting — this is where the actual "agent logic" lives.
# We give the model a security analyst persona and ask for structured JSON.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a blockchain security expert specialising in crypto token scam "
    "detection. You analyse smart contracts, transaction patterns, and liquidity "
    "pool data to identify rug pulls, honeypots, and wash trading.\n\n"
    "Always respond with a single valid JSON object and nothing else."
)


def build_classify_prompt(obs: Dict[str, Any]) -> str:
    task_desc = {
        "contract_analysis": "Analyse this Solidity smart contract for hidden scam mechanisms.",
        "transaction_analysis": "Analyse these on-chain transaction patterns for scam signals.",
        "liquidity_analysis": "Analyse this liquidity pool data for manipulation or exit risk.",
    }.get(obs.get("task_type", ""), "Analyse the token data.")

    prompt = (
        f"Task: {task_desc}\n\n"
        f"Token: {obs.get('token_name', 'Unknown')}\n\n"
        f"Base Data:\n{obs.get('token_data', '')}\n\n"
    )

    # Include investigation results if any
    inv_results = obs.get("investigation_results", {})
    if inv_results:
        prompt += "=== Investigation Results ===\n"
        for tool, result in inv_results.items():
            prompt += f"\n[{tool}]\n{result}\n"
        prompt += "\n"

    prompt += (
        "Classify this token. Respond ONLY with this JSON:\n"
        "{\n"
        '  "verdict": "<rug_pull|honeypot|wash_trading|safe>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning": "<concise explanation referencing specific evidence>"\n'
        "}\n\n"
        "Definitions:\n"
        "- rug_pull: developer can drain liquidity/funds (migration backdoor, "
        "unstaked fund withdrawal, LP removal without lock)\n"
        "- honeypot: users can buy but cannot sell (hidden sell tax, blacklist, "
        "transfer restriction, cooldown manipulation)\n"
        "- wash_trading: artificial volume via coordinated wallets (batch transfers, "
        "circular flows, market maker self-dealing)\n"
        "- safe: legitimate token with proper access control and no scam mechanisms\n"
    )
    return prompt


def get_verdict(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_classify_prompt(obs)},
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
# Per-task log buffer — accumulates steps, then emits the [START]/[STEP]/[END]
# block for each task at the end. The validator parses these to compute scores.
# ---------------------------------------------------------------------------

class TaskRunner:
    def __init__(self, task_name: str, model_name: str):
        self.task_name = task_name
        self.model_name = model_name
        self.rewards: List[float] = []
        self.steps: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None

    def record_step(self, action_str: str, reward: float, done: bool,
                    error: Optional[str]) -> None:
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
        log_end(success=success, steps=len(self.steps), score=score, rewards=self.rewards)
        return {"task": self.task_name, "score": score, "success": success, "steps": len(self.steps)}


# ---------------------------------------------------------------------------
# Environment lifecycle — figures out how to connect to the env server.
# Priority: LOCAL_IMAGE_NAME (docker) > RUGGUARD_URL (remote) > default docker
# ---------------------------------------------------------------------------

async def create_env() -> RugGuardEnv:
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
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs) if obs else {}


# ---------------------------------------------------------------------------
# Main loop — the core agent logic.
#
# For each of the 45 tokens in an episode:
#   1. Pick the best investigation tool for this task type and use it
#   2. Send token data + investigation results to the LLM
#   3. Parse the LLM's JSON response and submit it as a classification
#   4. Record the reward and move on
# ---------------------------------------------------------------------------

async def run() -> None:
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    runners: Dict[str, TaskRunner] = {t: TaskRunner(t, MODEL_NAME) for t in TASK_ORDER}
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
                break

            # Phase 1: grab extra context with the most relevant tool for this task
            available = obs_dict.get("available_tools", [])
            remaining = obs_dict.get("investigations_remaining", 0)
            preferred = PREFERRED_TOOLS.get(current_task, [])

            investigations_done = 0
            while investigations_done < INVESTIGATIONS_PER_TOKEN and remaining > 0 and available:
                # Pick best available tool
                tool = None
                for t in preferred:
                    if t in available:
                        tool = t
                        break
                if tool is None:
                    tool = available[0]

                try:
                    inv_action = RugGuardAction(
                        action_type="investigate",
                        tool=tool,
                    )
                    step_result = await env.step(inv_action)
                    obs = step_result.observation
                    obs_dict = obs_to_dict(obs)
                    available = obs_dict.get("available_tools", [])
                    remaining = obs_dict.get("investigations_remaining", 0)
                    investigations_done += 1
                    logger.info(f"  Investigated: {tool} for {obs_dict.get('token_name', '?')}")
                except Exception as exc:
                    logger.warning(f"Investigation failed: {exc}")
                    break

            # Phase 2: ask the LLM to classify based on everything we've gathered
            verdict_data = get_verdict(openai_client, obs_dict)
            action_str = (
                f"{verdict_data['verdict']}|conf={verdict_data['confidence']:.2f}|"
                f"inv={investigations_done}|{verdict_data['reasoning'][:80]}"
            )

            step_error: Optional[str] = None
            try:
                action = RugGuardAction(
                    action_type="classify",
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

    # Emit logs
    summaries = []
    for task in TASK_ORDER:
        runner = runners[task]
        if not runner.steps:
            log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)
            err_msg = fatal_error or "no steps recorded"
            log_step(step=1, action="error", reward=0.0, done=True, error=err_msg)
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
            summaries.append({"task": task, "score": 0.0, "success": False})
        else:
            summaries.append(runner.emit())

    logger.info("=" * 60)
    for s in summaries:
        logger.info(f"{s['task']:>22}: score={s['score']:.2f} success={s['success']}")
    logger.info("=" * 60)


def main() -> None:
    try:
        asyncio.run(run())
    except Exception as exc:
        logger.error(f"Fatal inference error: {exc}")
        for task in TASK_ORDER:
            log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="fatal_error", reward=0.0, done=True, error=str(exc))
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        sys.exit(0)


if __name__ == "__main__":
    main()
