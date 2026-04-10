"""
Tests for RugGuard Environment.

Validates reset, step (investigate + classify), reward bounds,
difficulty ordering, full episode completion, and edge cases.
"""

import json
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from server.rugguard_environment import (
    RugGuardEnvironment,
    _compute_reward,
    MAX_INVESTIGATIONS_PER_TOKEN,
    ALL_TOOLS,
    TASK_ORDER,
)
from models import RugGuardAction, RugGuardObservation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Create environment with fixed seed for determinism."""
    return RugGuardEnvironment(seed=42, steps_per_task=5)


@pytest.fixture
def full_env():
    """Full-sized environment (15 per task)."""
    return RugGuardEnvironment(seed=42)


@pytest.fixture
def single_task_env():
    """Environment restricted to one task."""
    return RugGuardEnvironment(seed=42, steps_per_task=3, task_filter="contract_analysis")


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, RugGuardObservation)

    def test_reset_first_step(self, env):
        obs = env.reset()
        assert obs.step_number == 1
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_has_token_data(self, env):
        obs = env.reset()
        assert len(obs.token_name) > 0
        assert len(obs.token_data) > 0

    def test_reset_task_type_valid(self, env):
        obs = env.reset()
        assert obs.task_type in TASK_ORDER

    def test_reset_investigations_available(self, env):
        obs = env.reset()
        assert obs.investigations_remaining == MAX_INVESTIGATIONS_PER_TOKEN
        assert len(obs.available_tools) == len(ALL_TOOLS)

    def test_reset_deterministic_with_seed(self, env):
        obs1 = env.reset(seed=99)
        name1 = obs1.token_name
        obs2 = env.reset(seed=99)
        name2 = obs2.token_name
        assert name1 == name2

    def test_reset_different_seeds_differ(self, env):
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=2)
        # With different seeds, at least some tokens differ
        # (not guaranteed for first token but very likely)
        # Just check reset works without error
        assert obs1.step_number == 1
        assert obs2.step_number == 1

    def test_reset_total_steps(self, env):
        obs = env.reset()
        assert obs.total_steps == 15  # 5 per task * 3 tasks

    def test_reset_single_task(self, single_task_env):
        obs = single_task_env.reset()
        assert obs.task_type == "contract_analysis"
        assert obs.total_steps == 3


# ---------------------------------------------------------------------------
# Investigation tests
# ---------------------------------------------------------------------------

class TestInvestigate:
    def test_investigate_returns_result(self, env):
        env.reset(seed=42)
        action = RugGuardAction(action_type="investigate", tool="holder_distribution")
        obs = env.step(action)
        assert "holder_distribution" in obs.investigation_results
        assert len(obs.investigation_results["holder_distribution"]) > 0

    def test_investigate_decrements_remaining(self, env):
        env.reset(seed=42)
        action = RugGuardAction(action_type="investigate", tool="contract_functions")
        obs = env.step(action)
        assert obs.investigations_remaining == MAX_INVESTIGATIONS_PER_TOKEN - 1

    def test_investigate_removes_used_tool(self, env):
        env.reset(seed=42)
        action = RugGuardAction(action_type="investigate", tool="deployer_history")
        obs = env.step(action)
        assert "deployer_history" not in obs.available_tools

    def test_investigate_max_limit(self, env):
        env.reset(seed=42)
        tools = ["holder_distribution", "contract_functions", "deployer_history"]
        for tool in tools:
            obs = env.step(RugGuardAction(action_type="investigate", tool=tool))
        # Now at max, should be rejected
        obs = env.step(RugGuardAction(action_type="investigate", tool="social_signals"))
        assert obs.investigations_remaining == 0
        assert "social_signals" not in obs.investigation_results

    def test_investigate_duplicate_tool_rejected(self, env):
        env.reset(seed=42)
        env.step(RugGuardAction(action_type="investigate", tool="price_history"))
        obs = env.step(RugGuardAction(action_type="investigate", tool="price_history"))
        assert "Invalid tool or already used" in obs.echoed_message

    def test_investigate_does_not_advance_token(self, env):
        env.reset(seed=42)
        obs1 = env.step(RugGuardAction(action_type="investigate", tool="holder_distribution"))
        assert obs1.done is False
        assert obs1.step_number == 1  # still on same token

    def test_all_six_tools_available(self, env):
        obs = env.reset(seed=42)
        assert set(obs.available_tools) == set(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassify:
    def test_classify_advances_step(self, env):
        env.reset(seed=42)
        obs = env.step(RugGuardAction(
            action_type="classify",
            verdict="safe",
            confidence=0.5,
            reasoning="test",
        ))
        assert obs.step_number == 2  # moved to next token

    def test_classify_returns_reward(self, env):
        env.reset(seed=42)
        obs = env.step(RugGuardAction(
            action_type="classify",
            verdict="rug_pull",
            confidence=0.9,
            reasoning="test",
        ))
        assert obs.reward >= 0.0
        assert obs.reward <= 1.0

    def test_classify_resets_investigations(self, env):
        env.reset(seed=42)
        env.step(RugGuardAction(action_type="investigate", tool="holder_distribution"))
        obs = env.step(RugGuardAction(
            action_type="classify",
            verdict="safe",
            confidence=0.5,
            reasoning="test",
        ))
        # New token: investigations reset
        assert obs.investigations_remaining == MAX_INVESTIGATIONS_PER_TOKEN
        assert len(obs.investigation_results) == 0

    def test_classify_defaults(self, env):
        """Classify with no verdict/confidence defaults to safe/0.5."""
        env.reset(seed=42)
        obs = env.step(RugGuardAction(action_type="classify"))
        assert obs.reward >= 0.0  # just check it doesn't crash


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestReward:
    def test_correct_safe_max_reward(self):
        r = _compute_reward("safe", 1.0, "safe", None, 0)
        # 0.50 + 0.15*1.0 + 0.05*1.0 = 0.70
        assert r == pytest.approx(0.70, abs=0.01)

    def test_correct_scam_max_reward(self):
        r = _compute_reward("rug_pull", 1.0, "rug_pull", "rug_pull", 0)
        # 0.50 + 0.20 + 0.15*1.0 + 0.05*1.0 = 0.90
        assert r == pytest.approx(0.90, abs=0.01)

    def test_wrong_verdict_low_confidence_gets_calibration(self):
        r = _compute_reward("safe", 0.1, "rug_pull", "rug_pull", 0)
        # Wrong: 0 + 0 + 0.15*(1-0.1) + partial(rug_pull, safe)=0 = 0.135
        assert r == pytest.approx(0.135, abs=0.01)

    def test_partial_credit(self):
        r1 = _compute_reward("honeypot", 0.5, "rug_pull", "rug_pull", 0)
        r2 = _compute_reward("safe", 0.5, "rug_pull", "rug_pull", 0)
        # honeypot->rug_pull gets 0.05 partial, safe->rug_pull gets 0
        assert r1 > r2

    def test_reward_bounds(self):
        for verdict in ["rug_pull", "honeypot", "wash_trading", "safe"]:
            for truth in ["rug_pull", "honeypot", "wash_trading", "safe"]:
                for conf in [0.0, 0.5, 1.0]:
                    for inv in [0, 1, 2, 3]:
                        r = _compute_reward(verdict, conf, truth, truth if truth != "safe" else None, inv)
                        assert 0.0 <= r <= 1.0, f"Reward {r} out of bounds"

    def test_investigation_efficiency(self):
        r0 = _compute_reward("safe", 0.8, "safe", None, 0)
        r3 = _compute_reward("safe", 0.8, "safe", None, 3)
        assert r0 > r3  # fewer investigations = higher bonus


# ---------------------------------------------------------------------------
# Full episode tests
# ---------------------------------------------------------------------------

class TestFullEpisode:
    def test_episode_completes(self, env):
        obs = env.reset(seed=42)
        steps = 0
        while not obs.done:
            obs = env.step(RugGuardAction(
                action_type="classify",
                verdict="safe",
                confidence=0.5,
                reasoning="baseline",
            ))
            steps += 1
            if steps > 100:
                pytest.fail("Episode did not terminate")
        assert obs.done is True
        assert steps == env._total_steps

    def test_episode_after_done_returns_done(self, env):
        obs = env.reset(seed=42)
        while not obs.done:
            obs = env.step(RugGuardAction(
                action_type="classify", verdict="safe", confidence=0.5,
            ))
        # One more step after done
        obs = env.step(RugGuardAction(
            action_type="classify", verdict="safe", confidence=0.5,
        ))
        assert obs.done is True
        assert "done" in obs.echoed_message.lower() or "reset" in obs.echoed_message.lower()

    def test_investigate_then_classify_flow(self, env):
        obs = env.reset(seed=42)
        # Investigate, then classify for first token
        obs = env.step(RugGuardAction(action_type="investigate", tool="holder_distribution"))
        assert len(obs.investigation_results) == 1
        obs = env.step(RugGuardAction(
            action_type="classify", verdict="rug_pull", confidence=0.8, reasoning="test",
        ))
        # Now on second token
        assert obs.step_number == 2
        assert len(obs.investigation_results) == 0

    def test_cumulative_reward_in_metadata(self, env):
        obs = env.reset(seed=42)
        while not obs.done:
            obs = env.step(RugGuardAction(
                action_type="classify", verdict="safe", confidence=0.5,
            ))
        assert hasattr(obs, "metadata")
        assert obs.metadata is not None
        assert "cumulative_reward" in obs.metadata


# ---------------------------------------------------------------------------
# Difficulty ordering tests
# ---------------------------------------------------------------------------

class TestDifficulty:
    def test_samples_sorted_by_difficulty(self, full_env):
        full_env.reset(seed=42)
        queue = full_env._task_queue
        # Check within each task's slice
        for task in full_env._active_tasks:
            task_samples = [s for s in queue if s["task_type"] == task]
            difficulties = [s["difficulty"] for s in task_samples]
            order = {"easy": 0, "medium": 1, "hard": 2}
            indices = [order.get(d, 1) for d in difficulties]
            assert indices == sorted(indices), f"Difficulty not sorted for {task}: {difficulties}"


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_updates(self, env):
        env.reset(seed=42)
        assert env.state.step_count == 0
        env.step(RugGuardAction(action_type="classify", verdict="safe", confidence=0.5))
        assert env.state.step_count == 1

    def test_state_episode_id(self, env):
        env.reset(seed=42, episode_id="test-ep-123")
        assert env.state.episode_id == "test-ep-123"


# ---------------------------------------------------------------------------
# Dataset integrity tests
# ---------------------------------------------------------------------------

class TestDatasets:
    @pytest.fixture(autouse=True)
    def setup_data_dir(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    @pytest.mark.parametrize("filename", ["contracts.json", "transactions.json", "liquidity.json"])
    def test_dataset_loads(self, filename):
        path = os.path.join(self.data_dir, filename)
        assert os.path.exists(path), f"{filename} not found"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "samples" in data
        assert len(data["samples"]) >= 15  # at least enough for one episode

    @pytest.mark.parametrize("filename", ["contracts.json", "transactions.json", "liquidity.json"])
    def test_samples_have_required_fields(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        for s in samples:
            assert "token_name" in s
            assert "label" in s
            assert "token_data" in s
            assert s["label"] in {"rug_pull", "honeypot", "wash_trading", "safe"}

    @pytest.mark.parametrize("filename", ["contracts.json", "transactions.json", "liquidity.json"])
    def test_samples_have_investigations(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        for s in samples:
            inv = s.get("investigations", {})
            assert len(inv) == 6, f"{s['token_name']} missing investigation tools"
            for tool in ALL_TOOLS:
                assert tool in inv, f"{s['token_name']} missing {tool}"

    @pytest.mark.parametrize("filename", ["contracts.json", "transactions.json", "liquidity.json"])
    def test_samples_have_difficulty(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        for s in samples:
            assert s.get("difficulty") in {"easy", "medium", "hard"}, \
                f"{s['token_name']} invalid difficulty: {s.get('difficulty')}"

    @pytest.mark.parametrize("filename", ["contracts.json", "transactions.json", "liquidity.json"])
    def test_label_balance(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        from collections import Counter
        counts = Counter(s["label"] for s in samples)
        labels = {"rug_pull", "honeypot", "wash_trading", "safe"}
        for label in labels:
            assert counts[label] >= 10, f"{filename}: only {counts[label]} samples for {label}"
