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

Crypto token scam detection environment for LLM agents with **multi-step
investigation**. Agents gather evidence using investigation tools, then classify
tokens as `rug_pull`, `honeypot`, `wash_trading`, or `safe`.

## Motivation

Crypto rug pulls, honeypots, and wash-trading scams drain billions of dollars
from retail users every year. RugGuard turns scam triage into a proper agent
benchmark: the agent must reason about Solidity source code, on-chain
transaction histories, and liquidity pool dynamics — three concrete sub-skills
a real security analyst uses every day.

**Data sources:** Patterns derived from 2,391 validated real-world rug pull
incidents and real smart contract vulnerability patterns.

## Tasks

| # | Task | Difficulty | What the agent sees |
|---|------|-----------|---------------------|
| 1 | `contract_analysis` | Easy | Solidity source with subtle backdoors |
| 2 | `transaction_analysis` | Medium | On-chain tx patterns |
| 3 | `liquidity_analysis` | Hard | LP pool metrics and anomalies |

Each task: 15 samples per episode (45 total). 120 samples per dataset (30/label).

## Action Space

Two-phase per token:

### Phase 1: Investigate (optional, up to 3 per token)

```python
RugGuardAction(
    action_type="investigate",
    tool="holder_distribution",  # one of 6 tools
)
```

**Available tools:** `holder_distribution`, `contract_functions`,
`deployer_history`, `social_signals`, `similar_contracts`, `price_history`

### Phase 2: Classify (required)

```python
RugGuardAction(
    action_type="classify",
    verdict="rug_pull",      # rug_pull | honeypot | wash_trading | safe
    confidence=0.85,         # [0.0, 1.0]
    reasoning="Evidence...", # free-text
)
```

## Observation Space

```python
RugGuardObservation(
    task_type: str,                        # current analysis task
    token_name: str,                       # token symbol
    token_data: str,                       # base data for analysis
    investigation_results: Dict[str, str], # results from prior investigations
    available_tools: List[str],            # tools still available
    investigations_remaining: int,         # investigations left (max 3)
    step_number: int,                      # 1-indexed
    total_steps: int,                      # 45
    last_reward: float,
    echoed_message: str,
    done: bool,
    reward: float,
)
```

## Reward Design

| Component | Points | Condition |
|-----------|--------|-----------|
| Correct verdict | +0.50 | `verdict == ground_truth_label` |
| Correct vulnerability type | +0.20 | Correct verdict on scam token |
| Confidence calibration | +0.15 x conf | When correct |
| Confidence calibration | +0.15 x (1-conf) | When wrong |
| Partial credit | +0.02-0.05 | Close-but-wrong answers |
| Investigation efficiency | +0.05 x (1-inv/3) | Fewer investigations when correct |

Max per step: **1.0**. Dense, multi-component signal for RL training.

## Quick Start

```python
from rugguard_env import RugGuardEnv, RugGuardAction

with RugGuardEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    # Investigate
    result = env.step(RugGuardAction(
        action_type="investigate",
        tool="holder_distribution",
    ))
    obs = result.observation
    print(obs.investigation_results)

    # Classify
    result = env.step(RugGuardAction(
        action_type="classify",
        verdict="rug_pull",
        confidence=0.9,
        reasoning="High holder concentration + deployer flagged in prior scams",
    ))
    print(result.reward, result.done)
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RUGGUARD_STEPS_PER_TASK` | `15` | Samples per task per episode |
| `RUGGUARD_SEED` | (random) | Fixed seed for reproducibility |
| `RUGGUARD_TASK_FILTER` | (none) | Restrict to one task type |
| `PORT` | `8000` | HTTP server port |

## Running Locally

```bash
cd rugguard_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or Docker
docker build -t rugguard-env:latest .
docker run -p 8000:8000 rugguard-env:latest
```

## Deploy to HF Spaces

```bash
cd rugguard_env
huggingface-cli upload <your-org>/rugguard-env . . --repo-type space
```
