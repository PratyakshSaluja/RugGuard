"""
RugGuard Environment Implementation.

An LLM-agent environment for detecting crypto token scams.

Episode structure:
  - 45 token classifications across 3 task types (15 each)
  - Per token: agent can investigate (up to 3 times) then must classify
  - Investigation actions don't consume a "token step" — they add info
  - Classification submits the verdict and advances to the next token

Reward function:
  - +0.5 correct verdict
  - +0.2 correct vulnerability type (scam tokens only)
  - +0.15 confidence calibration
  - +0.1 partial credit for nearby verdicts
  - +0.05 investigation efficiency bonus

Design note:
  Episode state is stored as instance variables, NOT in RugGuardState.
  The OpenEnv HTTP server only serialises base State fields between requests.
"""

import json
import logging
import os
import random
import hashlib
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

# Partial credit matrix — closer wrong answers get some reward
# (true_label, predicted_label) -> partial credit [0, 0.1]
PARTIAL_CREDIT = {
    ("rug_pull", "honeypot"): 0.05,
    ("rug_pull", "wash_trading"): 0.02,
    ("honeypot", "rug_pull"): 0.05,
    ("honeypot", "wash_trading"): 0.02,
    ("wash_trading", "rug_pull"): 0.02,
    ("wash_trading", "honeypot"): 0.02,
    # saying "safe" when it's a scam or vice versa = 0
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
    """
    Reward components (clamped to [0, 1]):
      +0.50  correct verdict
      +0.20  correct vulnerability type (scam tokens only)
      +0.15  confidence calibration
      +0.10  partial credit for close-but-wrong
      +0.05  investigation efficiency
    """
    reward = 0.0
    correct = verdict == ground_truth_label

    # Correct verdict
    if correct:
        reward += 0.50

    # Correct vulnerability type
    if ground_truth_vuln is not None and correct:
        reward += 0.20

    # Confidence calibration
    if correct:
        reward += 0.15 * confidence
    else:
        reward += 0.15 * (1.0 - confidence)

    # Partial credit for nearby wrong answers
    if not correct:
        reward += PARTIAL_CREDIT.get((ground_truth_label, verdict), 0.0)

    # Investigation efficiency: reward for using fewer investigations
    # 0 investigations = full bonus, 3 = no bonus
    investigation_bonus = 0.05 * (1.0 - num_investigations / MAX_INVESTIGATIONS_PER_TOKEN)
    if correct:
        reward += investigation_bonus

    return round(min(1.0, max(0.0, reward)), 4)


def _generate_investigation_result(
    tool: str,
    sample: dict,
    label: str,
    rng: random.Random,
) -> str:
    """Generate investigation results that provide useful but not definitive clues."""
    token_name = sample["token_name"]

    if tool == "holder_distribution":
        if label == "rug_pull":
            return (
                f"Holder Analysis for {token_name}:\n"
                f"- Top wallet holds {rng.randint(60,92)}% of supply\n"
                f"- Top 5 wallets hold {rng.randint(85,98)}% of supply\n"
                f"- {rng.randint(50,500)} total holders\n"
                f"- Gini coefficient: {rng.uniform(0.85,0.98):.2f} (highly concentrated)\n"
                f"- Top wallet funded via bridge/mixer {rng.randint(1,7)} days before token creation"
            )
        elif label == "honeypot":
            return (
                f"Holder Analysis for {token_name}:\n"
                f"- {rng.randint(500,5000)} holders (growing — buy-only pressure)\n"
                f"- Top wallet: {rng.randint(15,40)}% (deployer/whitelisted)\n"
                f"- Median wallet balance: ${rng.randint(50,500)}\n"
                f"- No wallets have successfully reduced their position\n"
                f"- Holder count only increases — zero exits in {rng.randint(5,30)} days"
            )
        elif label == "wash_trading":
            return (
                f"Holder Analysis for {token_name}:\n"
                f"- {rng.randint(10,30)} total holders\n"
                f"- Top {rng.randint(5,15)} wallets created within same {rng.randint(1,24)}h window\n"
                f"- All active trading wallets funded from {rng.randint(1,3)} source address(es)\n"
                f"- Wallet age average: {rng.randint(1,14)} days\n"
                f"- No wallets have any other token holdings"
            )
        else:  # safe
            return (
                f"Holder Analysis for {token_name}:\n"
                f"- {rng.randint(5000,100000)} holders\n"
                f"- Top wallet: {rng.uniform(1.5,4):.1f}% (protocol treasury, multisig)\n"
                f"- Top 10: {rng.uniform(15,30):.0f}% (includes DEX pairs and protocols)\n"
                f"- Gini: {rng.uniform(0.4,0.65):.2f} (well distributed)\n"
                f"- Organic growth: +{rng.uniform(2,8):.0f}% holders/month"
            )

    elif tool == "contract_functions":
        if label == "rug_pull":
            funcs = rng.sample([
                "migrateV2(address)", "emergencyWithdraw()", "recoverTokens(address,uint256)",
                "setRouter(address)", "clearStuckBalance()", "withdrawETH()",
            ], k=rng.randint(2, 4))
            return (
                f"Contract Function Analysis for {token_name}:\n"
                f"- Owner-only functions: {', '.join(funcs)}\n"
                f"- Ownership renounced: No\n"
                f"- Functions that move ETH/tokens to owner: {rng.randint(2,4)}\n"
                f"- No timelock on admin functions\n"
                f"- Contract is {'not ' if rng.random()>0.4 else ''}verified on block explorer"
            )
        elif label == "honeypot":
            return (
                f"Contract Function Analysis for {token_name}:\n"
                f"- updateFees(uint256,uint256) — no upper bound validation\n"
                f"- Current sell tax: {rng.randint(80,99)}% (set via updateFees)\n"
                f"- authorize(address[],bool) — whitelist for fee exemption\n"
                f"- {rng.randint(1,3)} whitelisted addresses found\n"
                f"- Transfer function has conditional logic based on sender/pair address"
            )
        elif label == "wash_trading":
            return (
                f"Contract Function Analysis for {token_name}:\n"
                f"- batchDistribute(address[],uint256[]) — bulk token transfers\n"
                f"- batchCollect(address[],uint256[]) — reclaim tokens from addresses\n"
                f"- {rng.randint(5,20)} addresses registered as 'market makers'\n"
                f"- rebalanceAllocations() called {rng.randint(50,500)} times in 24h\n"
                f"- Circular transfer pattern visible in internal transactions"
            )
        else:
            return (
                f"Contract Function Analysis for {token_name}:\n"
                f"- Standard ERC20 functions + {'governance voting' if rng.random()>0.5 else 'staking'}\n"
                f"- Ownership: {'renounced' if rng.random()>0.4 else 'multisig (3/5)'}\n"
                f"- No functions that transfer tokens/ETH to specific addresses\n"
                f"- All admin functions have appropriate access control\n"
                f"- Audited by {rng.choice(['Certik', 'OpenZeppelin', 'Trail of Bits'])}"
            )

    elif tool == "deployer_history":
        if label in ("rug_pull", "honeypot"):
            return (
                f"Deployer History for {token_name}:\n"
                f"- Deployer address age: {rng.randint(1,30)} days\n"
                f"- Previous contracts deployed: {rng.randint(2,8)}\n"
                f"- {rng.randint(1,5)} previous contracts flagged as scam\n"
                f"- Funded via {rng.choice(['Tornado Cash', 'cross-chain bridge', 'new CEX withdrawal'])}\n"
                f"- No ENS name or social verification"
            )
        elif label == "wash_trading":
            return (
                f"Deployer History for {token_name}:\n"
                f"- Deployer created {rng.randint(3,12)} contracts in past {rng.randint(30,90)} days\n"
                f"- {rng.randint(2,6)} contracts have similar bytecode patterns\n"
                f"- All contracts have batch transfer/collect functions\n"
                f"- Deployer also operates the market maker wallets\n"
                f"- No public identity or team information"
            )
        else:
            return (
                f"Deployer History for {token_name}:\n"
                f"- Deployer address age: {rng.randint(365,1800)} days\n"
                f"- Known entity: {rng.choice(['Yes (verified team)', 'Multisig with known signers', 'DAO governance'])}\n"
                f"- Previous contracts: {rng.randint(3,15)} (all active, no scam flags)\n"
                f"- ENS: {token_name.lower()}.eth\n"
                f"- Funded via {rng.choice(['Coinbase', 'known treasury multisig', 'protocol revenue'])}"
            )

    elif tool == "social_signals":
        if label == "rug_pull":
            return (
                f"Social Analysis for {token_name}:\n"
                f"- Twitter: {rng.choice(['deleted', 'inactive since rug', 'fake followers detected'])} ({rng.randint(1000,50000)} followers)\n"
                f"- Telegram: {rng.choice(['deleted', 'admin left', 'muted all members'])}\n"
                f"- Website: {rng.choice(['domain expired', 'CloudFlare error', 'template site'])}\n"
                f"- Paid promotions detected from {rng.randint(2,10)} influencers\n"
                f"- No doxxed team members"
            )
        elif label == "honeypot":
            return (
                f"Social Analysis for {token_name}:\n"
                f"- Twitter: active, {rng.randint(5000,30000)} followers (some bought)\n"
                f"- Telegram: {rng.randint(500,5000)} members, many complaints about selling issues\n"
                f"- Common user complaint: 'cannot sell', 'transaction reverts', 'stuck'\n"
                f"- Team response: 'working on DEX integration' / 'slippage issue being fixed'\n"
                f"- Website: professional-looking, no team information"
            )
        elif label == "wash_trading":
            return (
                f"Social Analysis for {token_name}:\n"
                f"- Promoted as 'highest volume token on {rng.choice(['BSC', 'Arbitrum'])}'\n"
                f"- Twitter: {rng.randint(500,5000)} followers, mostly bots\n"
                f"- Marketing focused entirely on volume/ranking metrics\n"
                f"- Listed on CMC/CoinGecko with inflated volume stats\n"
                f"- No organic community discussion about utility or development"
            )
        else:
            return (
                f"Social Analysis for {token_name}:\n"
                f"- Twitter: {rng.randint(10000,200000)} followers, organic engagement\n"
                f"- Discord: {rng.randint(5000,50000)} members, active development discussion\n"
                f"- GitHub: {rng.randint(50,500)} stars, {rng.randint(10,100)} contributors, regular commits\n"
                f"- Team: {'fully doxxed' if rng.random()>0.3 else 'pseudonymous but long-standing reputation'}\n"
                f"- Media coverage: {rng.choice(['CoinDesk', 'The Block', 'Bankless'])} features"
            )

    elif tool == "similar_contracts":
        if label in ("rug_pull", "honeypot"):
            return (
                f"Similar Contract Analysis for {token_name}:\n"
                f"- Bytecode similarity: {rng.randint(85,99)}% match with {rng.randint(3,15)} known scam contracts\n"
                f"- Common patterns: {rng.choice(['hidden owner functions', 'dynamic fee modification', 'blacklist mechanism'])}\n"
                f"- Matched scam contracts caused total losses of ${rng.randint(500000,10000000):,}\n"
                f"- Contract template appears to be from known scam toolkit\n"
                f"- TokenSniffer score: {rng.randint(5,30)}/100"
            )
        elif label == "wash_trading":
            return (
                f"Similar Contract Analysis for {token_name}:\n"
                f"- Contract contains batch transfer functions seen in {rng.randint(5,20)} other tokens\n"
                f"- Similar contracts all have 'market maker' registration system\n"
                f"- {rng.randint(3,8)} contracts deployed by same entity with identical structure\n"
                f"- All similar tokens show volume anomalies on DEX aggregators\n"
                f"- TokenSniffer score: {rng.randint(30,50)}/100"
            )
        else:
            return (
                f"Similar Contract Analysis for {token_name}:\n"
                f"- Contract uses standard OpenZeppelin patterns\n"
                f"- No similarity to known scam contract templates\n"
                f"- Similar architecture to {rng.choice(['Aave', 'Compound', 'Uniswap'])} contracts\n"
                f"- TokenSniffer score: {rng.randint(80,100)}/100\n"
                f"- GoPlus Security: no issues detected"
            )

    elif tool == "price_history":
        if label == "rug_pull":
            return (
                f"Price History for {token_name}:\n"
                f"- Launch price: ${rng.uniform(0.001, 0.1):.4f}\n"
                f"- ATH: ${rng.uniform(0.5, 10):.2f} (day {rng.randint(1,7)})\n"
                f"- Crash: -{rng.randint(90,99)}% in {rng.randint(5,60)} minutes\n"
                f"- Current: ${rng.uniform(0.00001, 0.001):.6f}\n"
                f"- Pattern: pump-and-dump with single sharp decline event"
            )
        elif label == "honeypot":
            return (
                f"Price History for {token_name}:\n"
                f"- Launch price: ${rng.uniform(0.01, 0.5):.4f}\n"
                f"- Current: ${rng.uniform(0.5, 20):.2f}\n"
                f"- Trend: monotonically increasing (no corrections)\n"
                f"- Volatility: {rng.uniform(1,5):.1f}% daily (abnormally low)\n"
                f"- Pattern: one-way price movement, no organic sell pressure"
            )
        elif label == "wash_trading":
            return (
                f"Price History for {token_name}:\n"
                f"- Price range: ${rng.uniform(0.1, 1):.2f} - ${rng.uniform(1, 2):.2f}\n"
                f"- Volatility: {rng.uniform(0.1,1):.1f}% daily (suspiciously stable)\n"
                f"- Volume bars: uniform height (no natural variation)\n"
                f"- Price moves in exact {rng.uniform(0.01,0.1):.2f}% increments\n"
                f"- Pattern: algorithmically controlled, no organic market dynamics"
            )
        else:
            return (
                f"Price History for {token_name}:\n"
                f"- Token age: {rng.randint(90,730)} days\n"
                f"- ATH: ${rng.uniform(5, 100):.2f} | ATL: ${rng.uniform(0.1, 2):.2f}\n"
                f"- 30d change: {rng.uniform(-20,30):+.1f}%\n"
                f"- Volatility: {rng.uniform(5,15):.0f}% daily (normal for market cap tier)\n"
                f"- Pattern: organic price discovery with healthy corrections"
            )

    return f"No data available for tool '{tool}'"


class RugGuardEnvironment(Environment):
    """
    Crypto token scam detection environment with multi-step investigation.

    Each token in the episode:
      1. Agent receives base observation
      2. Agent can investigate (up to 3 times) to gather more data
      3. Agent must classify (submit verdict)

    Args:
        data_dir: Override path to data/ directory
        steps_per_task: Samples per task type per episode
        seed: Fixed seed for reproducibility
        task_filter: Restrict to single task type
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

        # Load datasets
        self._datasets: Dict[str, List[dict]] = {}
        for task in TASK_ORDER:
            path = os.path.join(self._data_dir, DATA_FILES[task])
            with open(path, "r", encoding="utf-8") as fh:
                self._datasets[task] = json.load(fh)["samples"]
            logger.info(f"Loaded {len(self._datasets[task])} samples for {task}")

        # Episode state
        self._task_queue: List[dict] = []
        self._ep_step: int = 0
        self._ep_done: bool = False
        self._ep_reward: float = 0.0
        self._last_reward: float = 0.0
        self._last_reasoning: str = ""
        self._episode_id: str = str(uuid4())

        # Per-token investigation state
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
        """Build observation for current token."""
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

            # Generate investigation result based on label
            seed_val = hash(f"{self._episode_id}_{self._ep_step}_{tool}")
            rng = random.Random(seed_val)
            result = _generate_investigation_result(
                tool, current, current["label"], rng
            )
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
