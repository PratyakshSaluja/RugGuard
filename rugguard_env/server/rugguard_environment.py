"""
RugGuard Environment Implementation.

An LLM-agent environment for detecting crypto token scams.

Episode structure:
  - 45 token classifications across 3 task types (15 each)
  - Per token: agent can investigate (up to 3 times) then must classify
  - Investigation results are pre-baked in the dataset (no runtime generation)
  - Classification submits the verdict and advances to the next token

Reward function (5 components, max 1.0 per step):
  - +0.50 correct verdict
  - +0.20 correct vulnerability type (scam tokens only)
  - +0.15 confidence calibration
  - +0.10 partial credit for close-but-wrong
  - +0.05 investigation efficiency bonus

Difficulty: samples are tagged easy/medium/hard and ordered progressively.

Design note:
  Episode state is stored as instance variables, NOT in RugGuardState.
  The OpenEnv HTTP server only serialises base State fields between requests.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from ..models import RugGuardAction, RugGuardObservation, RugGuardState
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from models import RugGuardAction, RugGuardObservation, RugGuardState

logger = logging.getLogger(__name__)

STEPS_PER_TASK = 15
TOTAL_STEPS = 45
TASK_ORDER = [
    "contract_analysis",
    "transaction_analysis",
    "liquidity_analysis",
]
DATA_FILES = {
    "contract_analysis": "contracts.json",
    "transaction_analysis": "transactions.json",
    "liquidity_analysis": "liquidity.json",
}

MAX_INVESTIGATIONS_PER_TOKEN = 3
ALL_TOOLS = [
    "holder_distribution",
    "contract_functions",
    "deployer_history",
    "social_signals",
    "similar_contracts",
    "price_history",
]

SUPPORTS_CONCURRENT_SESSIONS = False

# Partial credit: closer wrong answers get some reward
PARTIAL_CREDIT = {
    ("rug_pull", "honeypot"): 0.05,
    ("rug_pull", "wash_trading"): 0.02,
    ("honeypot", "rug_pull"): 0.05,
    ("honeypot", "wash_trading"): 0.02,
    ("wash_trading", "rug_pull"): 0.02,
    ("wash_trading", "honeypot"): 0.02,
}


def _data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "data")


def _compute_reward(
    verdict: str,
    confidence: float,
    ground_truth_label: str,
    ground_truth_vuln: Optional[str],
    num_investigations: int,
) -> float:
    reward = 0.0
    correct = verdict == ground_truth_label

    if correct:
        reward += 0.50
    if ground_truth_vuln is not None and correct:
        reward += 0.20
    if correct:
        reward += 0.15 * confidence
    else:
        reward += 0.15 * (1.0 - confidence)
    if not correct:
        reward += PARTIAL_CREDIT.get((ground_truth_label, verdict), 0.0)
    if correct:
        reward += 0.05 * (1.0 - num_investigations / MAX_INVESTIGATIONS_PER_TOKEN)

    return round(min(1.0, max(0.0, reward)), 4)


class RugGuardEnvironment(Environment):
    """
    Crypto token scam detection environment with multi-step investigation.

    Each token in the episode:
      1. Agent receives base observation with difficulty tier
      2. Agent can investigate (up to 3 times) using pre-baked data
      3. Agent must classify (submit verdict)

    Samples are ordered by difficulty within each task (easy → medium → hard).
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        steps_per_task: int = STEPS_PER_TASK,
        seed: Optional[int] = None,
        task_filter: Optional[str] = None,
    ):
        self._data_dir = data_dir or _data_dir()
        self._steps_per_task = steps_per_task
        self._task_filter = task_filter if task_filter in TASK_ORDER else None
        active_tasks = [self._task_filter] if self._task_filter else TASK_ORDER
        self._active_tasks = active_tasks
        self._total_steps = steps_per_task * len(active_tasks)
        self._seed = seed

        self._datasets: Dict[str, List[dict]] = {}
        for task in TASK_ORDER:
            path = os.path.join(self._data_dir, DATA_FILES[task])
            with open(path, "r", encoding="utf-8") as fh:
                self._datasets[task] = json.load(fh)["samples"]
            logger.info(f"Loaded {len(self._datasets[task])} samples for {task}")

        self._task_queue: List[dict] = []
        self._ep_step: int = 0
        self._ep_done: bool = False
        self._ep_reward: float = 0.0
        self._last_reward: float = 0.0
        self._last_reasoning: str = ""
        self._episode_id: str = str(uuid4())
        self._current_investigations: Dict[str, str] = {}
        self._investigations_used: int = 0

        self._state = RugGuardState(
            episode_id=self._episode_id,
            step_count=0,
        )

    def _build_queue(self, rng: random.Random) -> List[dict]:
        queue = []
        for task in self._active_tasks:
            pool = self._datasets[task]
            chosen = rng.sample(pool, min(self._steps_per_task, len(pool)))
            # Sort by difficulty within task
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            chosen.sort(key=lambda s: diff_order.get(s.get("difficulty", "medium"), 1))
            for sample in chosen:
                queue.append({
                    "task_type": task,
                    "token_name": sample["token_name"],
                    "token_data": sample["token_data"],
                    "label": sample["label"],
                    "vuln": sample.get("vulnerability_type"),
                    "difficulty": sample.get("difficulty", "medium"),
                    "investigations": sample.get("investigations", {}),
                })
        return queue

    def _do_reset(self, seed: Optional[int], episode_id: Optional[str]) -> None:
        effective_seed = seed if seed is not None else self._seed
        rng = random.Random(effective_seed)
        self._task_queue = self._build_queue(rng)
        self._ep_step = 0
        self._ep_done = False
        self._ep_reward = 0.0
        self._last_reward = 0.0
        self._last_reasoning = ""
        self._episode_id = episode_id or str(uuid4())
        self._current_investigations = {}
        self._investigations_used = 0
        self._state = RugGuardState(
            episode_id=self._episode_id,
            step_count=0,
        )
        logger.info(
            f"Episode {self._episode_id} started — {self._total_steps} tokens, "
            f"tasks: {self._active_tasks}"
        )

    def _current_obs(self, reward: float = 0.0, done: bool = False,
                     echoed: str = "") -> RugGuardObservation:
        if done or self._ep_step >= len(self._task_queue):
            return RugGuardObservation(
                task_type=self._task_queue[-1]["task_type"] if self._task_queue else "contract_analysis",
                token_name="",
                token_data="",
                investigation_results={},
                available_tools=[],
                investigations_remaining=0,
                step_number=self._ep_step,
                total_steps=self._total_steps,
                last_reward=reward,
                echoed_message=echoed,
                done=True,
                reward=reward,
            )

        item = self._task_queue[self._ep_step]
        used_tools = list(self._current_investigations.keys())
        available = [t for t in ALL_TOOLS if t not in used_tools]
        remaining = MAX_INVESTIGATIONS_PER_TOKEN - self._investigations_used

        return RugGuardObservation(
            task_type=item["task_type"],
            token_name=item["token_name"],
            token_data=item["token_data"],
            investigation_results=dict(self._current_investigations),
            available_tools=available if remaining > 0 else [],
            investigations_remaining=remaining,
            step_number=self._ep_step + 1,
            total_steps=self._total_steps,
            last_reward=self._last_reward,
            echoed_message=echoed,
            done=False,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RugGuardObservation:
        self._do_reset(seed, episode_id)
        return self._current_obs()

    def step(
        self,
        action: RugGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RugGuardObservation:
        if not self._task_queue:
            self._do_reset(None, None)

        if self._ep_done:
            return self._current_obs(done=True, echoed="Episode done. Call reset().")

        current = self._task_queue[self._ep_step]

        # --- INVESTIGATE action ---
        if action.action_type == "investigate":
            if self._investigations_used >= MAX_INVESTIGATIONS_PER_TOKEN:
                return self._current_obs(
                    echoed="No investigations remaining. You must classify."
                )
            tool = action.tool
            if tool is None or tool in self._current_investigations:
                return self._current_obs(
                    echoed=f"Invalid tool or already used: {tool}"
                )

            # Look up pre-baked investigation result from dataset
            investigations = current.get("investigations", {})
            result = investigations.get(tool, f"No data available for {tool}")
            self._current_investigations[tool] = result
            self._investigations_used += 1
            self._state.step_count += 1

            logger.debug(
                f"Investigation: token={current['token_name']} tool={tool} "
                f"remaining={MAX_INVESTIGATIONS_PER_TOKEN - self._investigations_used}"
            )
            return self._current_obs(echoed=f"Investigation complete: {tool}")

        # --- CLASSIFY action ---
        verdict = action.verdict or "safe"
        confidence = action.confidence if action.confidence is not None else 0.5
        reasoning = action.reasoning or ""

        reward = _compute_reward(
            verdict=verdict,
            confidence=confidence,
            ground_truth_label=current["label"],
            ground_truth_vuln=current["vuln"],
            num_investigations=self._investigations_used,
        )

        self._ep_step += 1
        self._ep_reward += reward
        self._last_reward = reward
        self._last_reasoning = reasoning
        self._state.step_count += 1

        # Reset per-token investigation state
        self._current_investigations = {}
        self._investigations_used = 0

        done = self._ep_step >= self._total_steps
        self._ep_done = done

        logger.debug(
            f"Classify {self._ep_step}/{self._total_steps} | "
            f"task={current['task_type']} verdict={verdict} "
            f"truth={current['label']} reward={reward}"
        )

        if done:
            logger.info(f"Episode {self._episode_id} done — cumulative={self._ep_reward:.4f}")
            obs = self._current_obs(reward=reward, done=True, echoed=reasoning)
            obs.metadata = {"cumulative_reward": self._ep_reward, "episode_id": self._episode_id}
            return obs

        return self._current_obs(reward=reward, echoed=reasoning)

    @property
    def state(self) -> RugGuardState:
        return self._state
