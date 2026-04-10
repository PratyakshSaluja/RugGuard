"""
inference.py — RugGuard inference agent.

This is what the hackathon validator runs. It connects to the RugGuard env
(Docker container or remote HF Space), then works through 45 tokens using
an LLM with a two-phase approach per token:

  Phase 1 — Investigate: We pick the single most informative tool for the
  current task type. Contract analysis gets contract_functions, transaction
  analysis gets holder_distribution, liquidity analysis gets price_history.
  One investigation keeps us fast and earns the efficiency bonus (+0.05).

  Phase 2 — Classify: LLM call with base data + investigation results.
  Task-specific prompts tell the model exactly what red flags to look for
  (e.g. "migrateV2 = rug pull", "sell tax >50% = honeypot"). Confidence
  calibration guidelines help the model give honest scores instead of
  defaulting to 0.85 on everything.

The reward function gives +0.05 for fewer investigations and +0.20 for
getting the specific scam type right. So we invest in one good investigation
and spend the rest of our budget on prompt quality.

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
TEMPERATURE = 0.15       # low temp for consistent classifications
MAX_TOKENS = 768         # enough for step-by-step reasoning in JSON
DEFAULT_LOCAL_IMAGE = "rugguard-env:latest"
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_ORDER = ["contract_analysis", "transaction_analysis", "liquidity_analysis"]

# One investigation per token. The efficiency bonus (+0.05 * (1 - inv/3))
# means 1 investigation costs only 0.017 off the bonus while giving us
# much better accuracy than going in blind.
INVESTIGATIONS_PER_TOKEN = 1

# Best tool per task type, ranked by how much signal they give.
# We tested all 6 tools across tasks — these consistently help most.
PREFERRED_TOOLS = {
    "contract_analysis": ["contract_functions", "deployer_history", "similar_contracts"],
    "transaction_analysis": ["holder_distribution", "deployer_history", "price_history"],
    "liquidity_analysis": ["price_history", "holder_distribution", "similar_contracts"],
}


# ---------------------------------------------------------------------------
# Structured log helpers (validator parses these from stdout)
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
# LLM calls — triage (quick read) and classify (final verdict)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior blockchain security analyst at a top Web3 auditing firm. "
    "You've reviewed 500+ smart contracts and investigated hundreds of on-chain "
    "scam incidents. You approach each token methodically:\n"
    "1. First identify what type of data you're looking at\n"
    "2. List the specific red flags or green flags you see\n"
    "3. Consider alternative explanations before concluding\n"
    "4. Calibrate confidence based on evidence strength\n\n"
    "You know that most tokens are NOT scams — only flag something when you see "
    "concrete evidence, not just because it has admin functions.\n\n"
    "Always respond with a single valid JSON object and nothing else."
)

# Task-specific red flags — tells the LLM exactly what to hunt for
TASK_RED_FLAGS = {
    "contract_analysis": (
        "How to classify CONTRACTS:\n"
        "- rug_pull: owner-only functions that DIRECTLY move ETH/tokens out "
        "(migrateV2, withdrawAll, emergencyDrain, clearStuckBalance sending to owner). "
        "Must have actual fund-draining capability, not just admin functions.\n"
        "- honeypot: mechanisms that PREVENT selling — high sell tax (>20%), "
        "cooldown timers on transfers, maxSellAmount restrictions, transfer blacklists, "
        "hidden require() in _transfer that blocks certain addresses.\n"
        "- wash_trading: batch transfer infrastructure — rebalanceAllocations(), "
        "marketMaker arrays, fundMarketMakers(), functions that move tokens between "
        "multiple controlled wallets in a single call.\n"
        "- safe: standard ERC20 with OpenZeppelin imports, reasonable tax (<5%), "
        "governance/vesting features, AccessControl or Ownable used for legitimate admin "
        "functions only (pausing, minting with caps, setting tax rates). "
        "Having onlyOwner functions does NOT automatically mean rug pull — what matters "
        "is whether those functions can drain user funds.\n"
    ),
    "transaction_analysis": (
        "How to classify TRANSACTIONS:\n"
        "- rug_pull: deployer/top wallet holding >60% of supply, large sell-offs by "
        "insiders, one-directional outflows, few unique sellers vs many buyers.\n"
        "- honeypot: many buys but almost zero sells, very low unique seller count, "
        "high transaction failure rate on sells, buy/sell ratio >5:1.\n"
        "- wash_trading: trades in exact intervals (3-sec, 5-sec), same wallets as "
        "both buyer and seller, uniform trade sizes, 24h volume that doesn't match "
        "holder count, wallets all created within hours of each other.\n"
        "- safe: organic buy/sell ratio (0.5-2.0), growing unique holders over time, "
        "consistent volume, diverse wallet ages. A moderate Gini coefficient (0.4-0.7) "
        "is normal — only flag extreme concentration (>0.85).\n"
    ),
    "liquidity_analysis": (
        "How to classify LIQUIDITY pools:\n"
        "- rug_pull: LP tokens not locked or lock <30 days, single address holds >80% LP, "
        "deployer removed significant liquidity (>50%), LP value dropped sharply.\n"
        "- honeypot: sells fail at any liquidity depth test, high price impact on sells "
        "but low on buys, TRANSFER_FROM_FAILED errors in sell simulation.\n"
        "- wash_trading: reported TVL >>10x on-chain verified TVL, volume/TVL ratio >10x, "
        "circular net flows between same addresses, volume concentrated in <5 wallets.\n"
        "- safe: locked liquidity (>6 months), multiple LP providers, volume/TVL ratio "
        "between 0.1-3.0, healthy buy/sell volume ratio, stable or growing TVL.\n"
    ),
}


def build_classify_prompt(obs: Dict[str, Any]) -> str:
    """Full classification prompt with all evidence and task-specific guidance."""
    task = obs.get("task_type", "contract_analysis")

    task_desc = {
        "contract_analysis": "Analyse this smart contract for hidden scam mechanisms.",
        "transaction_analysis": "Analyse these on-chain transaction patterns for scam signals.",
        "liquidity_analysis": "Analyse this liquidity pool data for manipulation or exit risk.",
    }.get(task, "Analyse the token data.")

    prompt = (
        f"Task: {task_desc}\n"
        f"Token: {obs.get('token_name', 'Unknown')}\n\n"
        f"=== Base Data ===\n{obs.get('token_data', '')}\n\n"
    )

    # Investigation results — this is the evidence the agent gathered
    inv_results = obs.get("investigation_results", {})
    if inv_results:
        prompt += "=== Investigation Evidence ===\n"
        for tool, result in inv_results.items():
            prompt += f"\n--- {tool} ---\n{result}\n"
        prompt += "\n"

    # Task-specific guidance so the model knows what to look for
    prompt += TASK_RED_FLAGS.get(task, "")
    prompt += "\n"

    prompt += (
        "Based on ALL the evidence above, classify this token.\n\n"
        "Confidence guidelines:\n"
        "- 0.92-1.0: Multiple independent red flags pointing to the same conclusion\n"
        "- 0.80-0.91: Clear signal from either base data or investigation evidence\n"
        "- 0.65-0.79: Some signals but room for alternative interpretation\n"
        "- Below 0.65: Guessing — consider if 'safe' is more appropriate\n\n"
        "Respond ONLY with this JSON:\n"
        "{\n"
        '  "verdict": "<rug_pull|honeypot|wash_trading|safe>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning": "<cite specific evidence from the data and investigations>"\n'
        "}\n"
    )
    return prompt


def call_llm(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Make an LLM call and parse the JSON response. Returns {} on failure."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if the model wraps its response
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.startswith("```")
            ).strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning(f"LLM call failed: {exc}")
        return {}


def get_verdict(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Final classification with all evidence."""
    parsed = call_llm(client, build_classify_prompt(obs))

    # Validate verdict
    valid = {"rug_pull", "honeypot", "wash_trading", "safe"}
    if parsed.get("verdict") not in valid:
        parsed["verdict"] = "safe"

    # Validate confidence
    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.5

    if not isinstance(parsed.get("reasoning"), str):
        parsed["reasoning"] = "insufficient evidence for definitive classification"

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
        # HF Spaces can cold-start in 30-60s. Retry a few times so we don't
        # fail the whole submission because the Space was asleep.
        for attempt in range(5):
            try:
                logger.info(f"Connecting to env at {url} (attempt {attempt + 1}/5)")
                env = RugGuardEnv(base_url=url)
                await env.connect()
                return env
            except Exception as exc:
                if attempt < 4:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"Connection failed: {exc}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise

    logger.info(f"No LOCAL_IMAGE_NAME/RUGGUARD_URL set; using default image: {DEFAULT_LOCAL_IMAGE}")
    return await RugGuardEnv.from_docker_image(DEFAULT_LOCAL_IMAGE)


def obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs) if obs else {}


# ---------------------------------------------------------------------------
# Main loop — single investigation + strong classify prompt.
#
# For each of the 45 tokens:
#   1. Investigate: pick the best tool for this task type (hardcoded ranking)
#   2. Classify: LLM call with base data + evidence + task-specific red flags
#
# We tried adaptive triage (extra LLM call to pick tools) but it was slower
# and scored worse. The hardcoded tool rankings work well because each task
# type has a clear "best" investigation tool.
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

            token_name = obs_dict.get("token_name", "?")
            available = obs_dict.get("available_tools", [])
            remaining = obs_dict.get("investigations_remaining", 0)
            preferred = PREFERRED_TOOLS.get(current_task, [])

            # --- Phase 1: Investigate with the best tool for this task ---
            investigations_done = 0
            while investigations_done < INVESTIGATIONS_PER_TOKEN and remaining > 0 and available:
                tool = None
                for t in preferred:
                    if t in available:
                        tool = t
                        break
                if tool is None:
                    tool = available[0]

                try:
                    step_result = await env.step(RugGuardAction(
                        action_type="investigate", tool=tool,
                    ))
                    obs = step_result.observation
                    obs_dict = obs_to_dict(obs)
                    available = obs_dict.get("available_tools", [])
                    remaining = obs_dict.get("investigations_remaining", 0)
                    investigations_done += 1
                    logger.info(f"  Investigated: {tool} for {token_name}")
                except Exception as exc:
                    logger.warning(f"  Investigation failed: {exc}")
                    break

            # --- Phase 2: Classify with task-specific prompt ---
            verdict_data = get_verdict(openai_client, obs_dict)

            action_str = (
                f"{verdict_data['verdict']}|conf={verdict_data['confidence']:.2f}|"
                f"inv={investigations_done}|"
                f"{verdict_data['reasoning'][:70]}"
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
                f"  {current_task} | {token_name} | "
                f"verdict={verdict_data['verdict']} conf={verdict_data['confidence']:.2f} | "
                f"reward={reward:.4f}"
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

    # Emit structured logs for the validator
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
