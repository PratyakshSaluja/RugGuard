---
title: rugguard-env
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# RugGuard Environment

A benchmark environment for evaluating how well LLM agents can detect crypto token scams. Built on the OpenEnv framework.

The idea is simple: agents look at smart contracts, transaction histories, and liquidity pool data — the same stuff a real blockchain security analyst would review — and decide whether a token is a scam or not.

## Why this exists

Rug pulls, honeypots, and wash trading schemes drain billions from crypto users every year. Most detection tools are rule-based and easy to bypass. We wanted to see if LLM agents could do better by actually reasoning about the code and data, not just pattern matching.

The dataset is built from patterns found in 2,391 real-world rug pull incidents and actual smart contract vulnerabilities. It's not toy data.

## How it works

Each episode has 45 tokens across 3 task types (15 each). For every token, the agent can:

1. **Investigate** (optional, up to 3 times) — pick from 6 tools to gather more info
2. **Classify** (required) — submit a verdict: `rug_pull`, `honeypot`, `wash_trading`, or `safe`

Tokens are ordered easy-to-hard within each task, so the agent gets warmed up before hitting the tricky ones.

### The 3 tasks

| Task | What the agent sees | Difficulty |
|------|-------------------|------------|
| `contract_analysis` | Solidity source code with subtle backdoors (hidden migration functions, owner-only drains, fee manipulation) | Easier — the scam logic is in the code |
| `transaction_analysis` | On-chain transaction patterns, wallet histories, transfer flows | Medium — need to spot wash trading loops and suspicious wallet clusters |
| `liquidity_analysis` | LP pool metrics, price data, lock status, DEX liquidity | Harder — need to understand DeFi mechanics to spot exit scam setups |

### Investigation tools

Before classifying, agents can use up to 3 of these to gather evidence:

- `holder_distribution` — wallet concentration, top holders, funding sources
- `contract_functions` — owner permissions, dangerous functions, timelocks
- `deployer_history` — deployer wallet age, past contracts, scam associations
- `social_signals` — social media presence, community health, paid promotions
- `similar_contracts` — bytecode similarity to known scams, security scores
- `price_history` — price action, volume patterns, liquidity depth

All investigation results are pre-baked into the dataset (no runtime generation), so there's no label leakage and results are deterministic.

## Reward design

Each classification gets scored on 5 components (max 1.0 per token):

| Component | Points | How it works |
|-----------|--------|-------------|
| Correct verdict | +0.50 | Did you get the right answer? |
| Vulnerability type | +0.20 | For scam tokens: bonus for correct classification |
| Confidence calibration | +0.15 | High confidence when right, low confidence when wrong |
| Partial credit | +0.02-0.05 | Close answers get some credit (e.g. calling a rug pull a honeypot) |
| Investigation efficiency | +0.05 | Bonus for using fewer investigations when you're right |

The reward signal is dense — you get useful feedback every single step, not just at the end. Good for RL training.

## Dataset stats

- 120 samples per task (360 total across all 3 datasets)
- 30 per label (balanced: rug_pull, honeypot, wash_trading, safe)
- Each sample has all 6 investigation tool results pre-computed
- 3 difficulty tiers: 40 easy / 40 medium / 40 hard per dataset
- Patterns sourced from real incidents, not made up

## Action space

### Investigate

```python
RugGuardAction(
    action_type="investigate",
    tool="holder_distribution",
)
```

### Classify

```python
RugGuardAction(
    action_type="classify",
    verdict="rug_pull",
    confidence=0.85,
    reasoning="migrateV2 lets owner drain all ETH, deployer linked to 3 prior scams",
)
```

## Observation space

```python
RugGuardObservation(
    task_type="contract_analysis",
    token_name="ZephyrLend",
    token_data="// SPDX-License-Identifier...",
    investigation_results={"holder_distribution": "Top wallet holds 88%..."},
    available_tools=["contract_functions", "deployer_history", ...],
    investigations_remaining=2,
    step_number=3,
    total_steps=45,
    last_reward=0.85,
    echoed_message="",
    done=False,
    reward=0.0,
)
```

## Quick start

```python
from rugguard_env import RugGuardEnv, RugGuardAction

with RugGuardEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    # gather some evidence first
    result = env.step(RugGuardAction(
        action_type="investigate",
        tool="holder_distribution",
    ))
    obs = result.observation
    print(obs.investigation_results)

    # make your call
    result = env.step(RugGuardAction(
        action_type="classify",
        verdict="rug_pull",
        confidence=0.9,
        reasoning="top wallet holds 88% of supply, funded via Tornado Cash",
    ))
    print(f"reward: {result.reward}, done: {result.done}")
```

## Running it

```bash
# local
cd rugguard_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000

# docker
docker build -t rugguard-env:latest .
docker run -p 8000:8000 rugguard-env:latest
```

## Config

| Env var | Default | What it does |
|---------|---------|-------------|
| `RUGGUARD_STEPS_PER_TASK` | `15` | Samples per task per episode |
| `RUGGUARD_SEED` | random | Fixed seed for reproducible episodes |
| `RUGGUARD_TASK_FILTER` | all | Restrict to one task type |
| `PORT` | `8000` | Server port |

## Tests

```bash
cd rugguard_env
python -m pytest tests/ -v
```

48 tests covering reset, investigate, classify, rewards, full episodes, difficulty ordering, dataset integrity, and edge cases.

## Deploying to HF Spaces

```bash
cd rugguard_env
hf upload <your-org>/rugguard-env . . --repo-type space
```
