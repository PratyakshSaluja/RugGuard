"""
Data models for the RugGuard Environment.

Defines the typed Action, Observation, and State for crypto token scam detection.
Agents receive token data and must classify it as rug_pull, honeypot, wash_trading, or safe.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, field_validator

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State

VerdictType = Literal["rug_pull", "honeypot", "wash_trading", "safe"]
TaskType = Literal["contract_analysis", "transaction_analysis", "liquidity_analysis"]


class RugGuardAction(Action):
    """
    Agent verdict on a token's safety classification.

    Attributes:
        verdict: Classification — rug_pull | honeypot | wash_trading | safe
        confidence: Agent confidence in the verdict [0.0, 1.0]
        reasoning: Free-text explanation for the classification decision
    """

    verdict: VerdictType = Field(
        ..., description="Classification verdict for the token"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score [0, 1]"
    )
    reasoning: str = Field(
        ..., min_length=1, description="Agent's reasoning for the classification"
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class RugGuardObservation(Observation):
    """
    Observation returned to the agent each step.

    Attributes:
        task_type: Which analysis task is active (contract/transaction/liquidity)
        token_name: Name or symbol of the token under analysis
        token_data: Raw data string for the agent to analyse
        step_number: Current step within the episode (1-indexed)
        total_steps: Total steps in the episode (always 45)
        last_reward: Reward received on the previous step (0.0 on first step)
        echoed_message: Echo of the agent's last reasoning (empty on first step)
    """

    task_type: TaskType = Field(..., description="Type of analysis task")
    token_name: str = Field(..., description="Token name or symbol")
    token_data: str = Field(..., description="Raw token data for analysis")
    step_number: int = Field(..., ge=1, description="Current step (1-indexed)")
    total_steps: int = Field(45, description="Total steps per episode")
    last_reward: float = Field(0.0, description="Reward from the previous step")
    echoed_message: str = Field("", description="Echo of last agent reasoning")


class RugGuardState(State):
    """
    Full server-side state for the RugGuard episode.

    Attributes:
        current_task: Active task type
        step_number: Steps completed so far
        cumulative_reward: Total reward accumulated this episode
        done: Whether the episode has ended
        ground_truth_label: True label for the current sample
        ground_truth_vuln: Vulnerability type (None for safe tokens)
        task_queue: Ordered list of (task_type, sample_index) pairs for episode
        episode_samples: Loaded sample dicts for current episode
    """

    current_task: TaskType = "contract_analysis"
    step_number: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    ground_truth_label: VerdictType = "safe"
    ground_truth_vuln: Optional[str] = None
    # Note: task_queue is kept as an instance variable on the environment,
    # not in State, because OpenEnv only persists base State fields between requests.
