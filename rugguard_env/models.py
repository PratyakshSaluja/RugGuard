"""
Data models for the RugGuard Environment.

Defines the typed Action, Observation, and State for crypto token scam detection.

Action space:
  - "investigate": request additional info about the current token before classifying
  - "classify": submit a verdict (rug_pull | honeypot | wash_trading | safe)

The agent gets a base observation, can investigate up to N times for more data,
then must classify. This creates multi-step reasoning per token.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from openenv.core.env_server.types import Action, Observation, State

VerdictType = Literal["rug_pull", "honeypot", "wash_trading", "safe"]
TaskType = Literal["contract_analysis", "transaction_analysis", "liquidity_analysis"]
ActionType = Literal["investigate", "classify"]

# Investigation tools the agent can request
InvestigationTool = Literal[
    "holder_distribution",
    "contract_functions",
    "deployer_history",
    "social_signals",
    "similar_contracts",
    "price_history",
]


class RugGuardAction(Action):
    """
    Agent action — either investigate further or classify the token.

    For action_type="investigate":
        - tool: which investigation tool to use
        - verdict/confidence/reasoning are ignored

    For action_type="classify":
        - verdict, confidence, reasoning are required
        - tool is ignored
    """

    action_type: ActionType = Field(
        default="classify",
        description="'investigate' to request more info, 'classify' to submit verdict",
    )
    tool: Optional[InvestigationTool] = Field(
        default=None,
        description="Investigation tool to use (only for action_type='investigate')",
    )
    verdict: Optional[VerdictType] = Field(
        default=None,
        description="Classification verdict (only for action_type='classify')",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0, 1] (only for action_type='classify')",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for the classification",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        if v is None:
            return v
        return max(0.0, min(1.0, float(v)))


class RugGuardObservation(Observation):
    """
    Observation returned to the agent each step.

    Attributes:
        task_type: Which analysis task is active
        token_name: Token name or symbol
        token_data: Base data for this token
        investigation_results: Results from previous investigate actions on this token
        available_tools: Investigation tools the agent can still use
        investigations_remaining: How many more investigations allowed for this token
        step_number: Current step within the episode (1-indexed)
        total_steps: Total steps in the episode
        last_reward: Reward from previous classification
        echoed_message: Echo of last agent reasoning
    """

    task_type: TaskType = Field(..., description="Type of analysis task")
    token_name: str = Field(..., description="Token name or symbol")
    token_data: str = Field(..., description="Base token data for analysis")
    investigation_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Results from previous investigations on this token",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Investigation tools still available",
    )
    investigations_remaining: int = Field(
        default=3,
        description="Number of investigations still allowed for this token",
    )
    step_number: int = Field(..., ge=1, description="Current step (1-indexed)")
    total_steps: int = Field(45, description="Total steps per episode")
    last_reward: float = Field(0.0, description="Reward from the previous step")
    echoed_message: str = Field("", description="Echo of last agent reasoning")


class RugGuardState(State):
    """
    Minimal server-side state for OpenEnv compatibility.
    Actual episode state is kept as instance variables on the environment.
    """

    current_task: TaskType = "contract_analysis"
    step_number: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    ground_truth_label: VerdictType = "safe"
    ground_truth_vuln: Optional[str] = None
