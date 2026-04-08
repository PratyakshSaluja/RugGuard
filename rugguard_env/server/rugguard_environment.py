"""
RugGuard Environment Implementation.

An LLM-agent environment for detecting crypto token scams.
The agent receives token data across three task types and must classify each token
as: rug_pull | honeypot | wash_trading | safe.

Episode structure:
  - 15 steps total: 5 contract_analysis, 5 transaction_analysis, 5 liquidity_analysis
  - Samples drawn from bundled JSON datasets (no external API calls)
  - Rewards in [0, 1]: +0.5 correct verdict, +0.3 correct vuln type, +0.2 calibration

Design note:
  Episode state (task_queue, step_number, ground_truth) is stored as instance
  variables, NOT in RugGuardState. The OpenEnv HTTP server only serialises base
  State fields (episode_id, step_count) between requests; custom Pydantic fields
  are dropped on round-trip. Instance variables on the environment object persist
  because the server reuses a single environment instance per session pool slot.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
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

SUPPORTS_CONCURRENT_SESSIONS = False


def _data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "data")


def _compute_reward(
    verdict: str,
    confidence: float,
    ground_truth_label: str,
    ground_truth_vuln: Optional[str],
) -> float:
    """
    Reward components (clamped to [0, 1]):
      +0.5  correct verdict
      +0.3  correct vulnerability type (scam tokens only)
      +0.2  confidence calibration bonus
    """
    reward = 0.0
    correct = verdict == ground_truth_label

    if correct:
        reward += 0.5
    if ground_truth_vuln is not None and correct:
        reward += 0.3
    if correct:
        reward += 0.2 * confidence
    else:
        reward += 0.2 * (1.0 - confidence)

    return round(min(1.0, max(0.0, reward)), 4)


class RugGuardEnvironment(Environment):
    """
    Crypto token scam detection environment for LLM agents.

    Args:
        data_dir: Override path to the data/ directory (default: auto-detected)
        steps_per_task: Number of samples per task type per episode (default: 5)
        seed: Optional fixed seed for reproducible sample ordering
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
        # Optional single-task mode (e.g. "contract_analysis"). When set, the
        # episode contains steps_per_task samples of that task only — used by
        # the baseline inference script to score each task independently.
        self._task_filter = task_filter if task_filter in TASK_ORDER else None
        active_tasks = [self._task_filter] if self._task_filter else TASK_ORDER
        self._active_tasks = active_tasks
        self._total_steps = steps_per_task * len(active_tasks)
        self._seed = seed

        # Load all datasets at startup
        self._datasets: Dict[str, List[dict]] = {}
        for task in TASK_ORDER:
            path = os.path.join(self._data_dir, DATA_FILES[task])
            with open(path, "r", encoding="utf-8") as fh:
                self._datasets[task] = json.load(fh)["samples"]
            logger.info(f"Loaded {len(self._datasets[task])} samples for {task}")

        # Episode state — kept as instance vars so they survive HTTP round-trips
        self._task_queue: List[dict] = []   # [{task_type, sample, label, vuln}, ...]
        self._ep_step: int = 0              # steps completed this episode
        self._ep_done: bool = False
        self._ep_reward: float = 0.0
        self._last_reward: float = 0.0
        self._last_reasoning: str = ""
        self._episode_id: str = str(uuid4())

        # Minimal Pydantic state (only for OpenEnv base compatibility)
        self._state = RugGuardState(
            episode_id=self._episode_id,
            step_count=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_queue(self, rng: random.Random) -> List[dict]:
        """Sample steps_per_task items from each active task dataset."""
        queue = []
        for task in self._active_tasks:
            pool = self._datasets[task]
            chosen = rng.sample(pool, min(self._steps_per_task, len(pool)))
            for sample in chosen:
                queue.append({
                    "task_type": task,
                    "token_name": sample["token_name"],
                    "token_data": sample["token_data"],
                    "label": sample["label"],
                    "vuln": sample.get("vulnerability_type"),
                })
        return queue

    def _do_reset(self, seed: Optional[int], episode_id: Optional[str]) -> None:
        """Shared reset logic (populates instance vars)."""
        effective_seed = seed if seed is not None else self._seed
        rng = random.Random(effective_seed)
        self._task_queue = self._build_queue(rng)
        self._ep_step = 0
        self._ep_done = False
        self._ep_reward = 0.0
        self._last_reward = 0.0
        self._last_reasoning = ""
        self._episode_id = episode_id or str(uuid4())
        self._state = RugGuardState(
            episode_id=self._episode_id,
            step_count=0,
        )
        logger.info(
            f"Episode {self._episode_id} started — "
            f"{self._total_steps} steps, tasks: "
            + str([q["task_type"] for q in self._task_queue[:3]]) + "..."
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
        item = self._task_queue[0]
        return RugGuardObservation(
            task_type=item["task_type"],
            token_name=item["token_name"],
            token_data=item["token_data"],
            step_number=1,
            total_steps=self._total_steps,
            last_reward=0.0,
            echoed_message="",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: RugGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RugGuardObservation:
        # Auto-init if reset() was never called
        if not self._task_queue:
            self._do_reset(None, None)

        if self._ep_done:
            item = self._task_queue[min(self._ep_step, len(self._task_queue) - 1)]
            return RugGuardObservation(
                task_type=item["task_type"],
                token_name="",
                token_data="",
                step_number=self._ep_step,
                total_steps=self._total_steps,
                last_reward=self._last_reward,
                echoed_message="Episode done. Call reset().",
                done=True,
                reward=0.0,
            )

        # Current item (what the agent just classified)
        current = self._task_queue[self._ep_step]

        reward = _compute_reward(
            verdict=action.verdict,
            confidence=action.confidence,
            ground_truth_label=current["label"],
            ground_truth_vuln=current["vuln"],
        )

        self._ep_step += 1
        self._ep_reward += reward
        self._last_reward = reward
        self._last_reasoning = action.reasoning
        self._state.step_count += 1

        done = self._ep_step >= self._total_steps
        self._ep_done = done

        logger.debug(
            f"Step {self._ep_step}/{self._total_steps} | "
            f"task={current['task_type']} verdict={action.verdict} "
            f"truth={current['label']} reward={reward}"
        )

        if done:
            logger.info(f"Episode {self._episode_id} done — cumulative={self._ep_reward:.4f}")
            return RugGuardObservation(
                task_type=current["task_type"],
                token_name="",
                token_data="",
                step_number=self._ep_step,
                total_steps=self._total_steps,
                last_reward=reward,
                echoed_message=action.reasoning,
                done=True,
                reward=reward,
                metadata={"cumulative_reward": self._ep_reward, "episode_id": self._episode_id},
            )

        # Next item
        nxt = self._task_queue[self._ep_step]
        return RugGuardObservation(
            task_type=nxt["task_type"],
            token_name=nxt["token_name"],
            token_data=nxt["token_data"],
            step_number=self._ep_step + 1,
            total_steps=self._total_steps,
            last_reward=reward,
            echoed_message=action.reasoning,
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> RugGuardState:
        return self._state
