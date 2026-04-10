---
title: rugguard-env
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# RugGuard: Can Your AI Detect a $2.8B Problem?

In 2023 alone, crypto rug pulls, honeypots, and wash trading scams stole **$2.8 billion** from retail investors. Existing detection tools are mostly rule-based — they check a list of known red flags and move on. Sophisticated scammers have already learned to work around them.

RugGuard takes a different approach. Instead of hard-coding rules, we built an environment where **LLM agents learn to think like security analysts**. They read Solidity code, trace transaction patterns, analyze liquidity pools, and make judgment calls — the same workflow a human auditor follows, but at scale.

## What makes this different

Most token classification benchmarks give the agent all the information upfront and ask for a label. That's a quiz, not an investigation.

**RugGuard is an investigation.** Each token starts with limited information. The agent has to decide what to investigate, interpret the results, and then make a call. Just like a real analyst would.

```
┌─────────────────────────────────────────────────────────┐
│                   Per-Token Agent Flow                   │
│                                                         │
│  Base data (contract/txs/LP)                            │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────┐     ┌──────────────────────────┐       │
│  │ INVESTIGATE  │────▶│ Pick from 6 tools:       │       │
│  │ (up to 3x)  │     │ • holder_distribution    │       │
│  └──────┬──────┘     │ • contract_functions     │       │
│         │            │ • deployer_history       │       │
│         │            │ • social_signals         │       │
│         │            │ • similar_contracts      │       │
│         │            │ • price_history          │       │
│         │            └──────────────────────────┘       │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │  CLASSIFY   │──▶ verdict + confidence + reasoning    │
│  └─────────────┘                                        │
│         │                                               │
│         ▼                                               │
│  Reward: accuracy + vuln type + calibration + efficiency│
└─────────────────────────────────────────────────────────┘
```

The agent that calls everything a scam won't score well (safe tokens exist). The agent that uses all 3 investigations every time loses the efficiency bonus. The agent that guesses high confidence on wrong answers gets punished by the calibration component. **You actually have to be good at this.**

## The three tasks

Each episode runs 45 tokens across 3 task types (15 each), ordered easy → hard within each task:

### 1. Contract Analysis
The agent reads Solidity source code and hunts for backdoors. Not the obvious `function stealAllMoney()` kind — we're talking about `migrateV2()` functions disguised as upgrades, `recoverTokens()` that "recovers" everything to the owner, and fee manipulation that slowly drains value.

### 2. Transaction Analysis
Raw on-chain patterns: who's buying, who's selling, how often, and where the money flows. Wash trading shows up as perfectly timed trades between the same wallets. Rug pulls show up as massive deployer sell-offs. Honeypots show 500 buys and 3 sells.

### 3. Liquidity Analysis
DeFi-native analysis: LP lock status, TVL manipulation, price impact asymmetry. The hardest task because the scam signals are statistical, not structural. A reported TVL that's 200x higher than on-chain reality screams wash trading, but you need to actually do the math.

## Reward function

Five components per classification, max 1.0 per token:

| Component | Points | What it rewards |
|-----------|--------|----------------|
| Correct verdict | +0.50 | Getting the right answer |
| Vulnerability type | +0.20 | Identifying the *specific* scam type (not just "it's bad") |
| Confidence calibration | +0.15 | Being sure when right, uncertain when wrong |
| Partial credit | +0.02-0.05 | Close calls (e.g., calling a rug pull a honeypot — wrong but not clueless) |
| Investigation efficiency | +0.05 | Using fewer tools when you're right (rewarding decisive analysis) |

This isn't pass/fail. The reward is **dense and multi-dimensional**, giving useful gradient signal at every step. An RL agent training on this gets feedback on not just what it got wrong, but *how* it got wrong.

## Dataset

- **360 samples** across 3 datasets (120 each)
- **Balanced**: 30 per label (rug_pull, honeypot, wash_trading, safe)
- **Difficulty tiers**: 40 easy / 40 medium / 40 hard per dataset
- **Pre-baked investigations**: All 6 tool results computed per sample (no runtime generation, no label leakage)
- **Real patterns**: Contract backdoors based on real exploit patterns from 2,391 validated rug pull incidents. Transaction patterns modeled on actual wash trading schemes. LP data based on real DeFi pool manipulation tactics.

## Baseline agent performance

Using Qwen2.5-72B-Instruct with task-specific chain-of-thought prompts and 1 investigation per token:

| Task | Score | Notes |
|------|-------|-------|
| contract_analysis | 0.81 | Strong on scam detection, occasionally over-flags safe tokens |
| transaction_analysis | 0.82 | Good at wash trading and rug pull patterns |
| liquidity_analysis | 0.83 | Best task — LP manipulation signals are clear |

Theoretical maximum is ~0.85 (safe tokens cap at 0.70 due to no vulnerability type bonus). The agent is at **96% of theoretical max**.

## Quick start

```python
from rugguard_env import RugGuardEnv, RugGuardAction

with RugGuardEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    # gather evidence
    result = env.step(RugGuardAction(
        action_type="investigate",
        tool="contract_functions",
    ))

    # make your call
    result = env.step(RugGuardAction(
        action_type="classify",
        verdict="rug_pull",
        confidence=0.95,
        reasoning="migrateV2 gives owner unrestricted ETH withdrawal",
    ))
    print(f"reward: {result.reward}")
```

## Running locally

```bash
cd rugguard_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000

# or docker
docker build -t rugguard-env:latest .
docker run -p 8000:8000 rugguard-env:latest
```

## Tests

```bash
python -m pytest tests/ -v   # 48 tests, all passing
```

Covers reset, investigation, classification, reward math, full episodes, difficulty ordering, dataset integrity, and edge cases.

## Config

| Env var | Default | Description |
|---------|---------|-------------|
| `RUGGUARD_STEPS_PER_TASK` | `15` | Tokens per task per episode |
| `RUGGUARD_SEED` | random | Fixed seed for reproducibility |
| `RUGGUARD_TASK_FILTER` | all | Restrict to single task type |
| `PORT` | `8000` | Server port |

## Architecture

```
rugguard_env/
├── server/
│   ├── app.py                    # FastAPI + OpenEnv create_app
│   └── rugguard_environment.py   # Core environment (reset/step/state)
├── data/
│   ├── contracts.json            # 120 smart contract samples
│   ├── transactions.json         # 120 transaction pattern samples
│   └── liquidity.json            # 120 liquidity pool samples
├── models.py                     # Typed Action/Observation/State
├── client.py                     # HTTP client for remote env
├── tests/
│   └── test_environment.py       # 48 unit tests
├── openenv.yaml                  # OpenEnv spec
└── Dockerfile                    # HF Spaces deployment
```
