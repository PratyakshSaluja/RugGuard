"""
RugGuard Environment Client.

HTTP/WebSocket client for a running RugGuardEnvironment server.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import RugGuardAction, RugGuardObservation, RugGuardState
except ImportError:
    from models import RugGuardAction, RugGuardObservation, RugGuardState


class RugGuardEnv(EnvClient[RugGuardAction, RugGuardObservation, RugGuardState]):
    """
    Client for the RugGuard token scam detection environment.

    Example::

        with RugGuardEnv(base_url="http://localhost:8000") as env:
            obs_result = env.reset()
            obs = obs_result.observation

            # Investigate first
            result = env.step(RugGuardAction(
                action_type="investigate",
                tool="holder_distribution",
            ))

            # Then classify
            result = env.step(RugGuardAction(
                action_type="classify",
                verdict="rug_pull",
                confidence=0.85,
                reasoning="High concentration + deployer history flags",
            ))
    """

    def _step_payload(self, action: RugGuardAction) -> Dict:
        payload = {"action_type": action.action_type}
        if action.action_type == "investigate":
            payload["tool"] = action.tool
        else:
            payload["verdict"] = action.verdict
            payload["confidence"] = action.confidence
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[RugGuardObservation]:
        obs_data = payload.get("observation", {})
        observation = RugGuardObservation(
            task_type=obs_data.get("task_type", "contract_analysis"),
            token_name=obs_data.get("token_name", ""),
            token_data=obs_data.get("token_data", ""),
            investigation_results=obs_data.get("investigation_results", {}),
            available_tools=obs_data.get("available_tools", []),
            investigations_remaining=obs_data.get("investigations_remaining", 0),
            step_number=obs_data.get("step_number", 1),
            total_steps=obs_data.get("total_steps", 45),
            last_reward=obs_data.get("last_reward", 0.0),
            echoed_message=obs_data.get("echoed_message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> RugGuardState:
        return RugGuardState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", "contract_analysis"),
            step_number=payload.get("step_number", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            ground_truth_label=payload.get("ground_truth_label", "safe"),
            ground_truth_vuln=payload.get("ground_truth_vuln"),
        )
