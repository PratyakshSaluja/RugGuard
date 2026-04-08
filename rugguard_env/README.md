---
title: rugguard-env
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# RugGuard Environment

Crypto token scam detection environment for LLM agents. Agents analyse token data across
three task types and classify each token as `rug_pull`, `honeypot`, `wash_trading`, or `safe`.

## Motivation

Crypto rug pulls, honeypots, and wash-trading scams drain billions of dollars from
retail users every year. Today, scam triage is done manually by security researchers
or by brittle rule-based scanners that miss novel attack patterns. RugGuard turns this
into a proper RL/agent benchmark: the agent must reason about Solidity source code,
on-chain transaction histories, and liquidity pool dynamics — three concrete sub-skills
a real security analyst uses every day. A model that scores well here is genuinely
useful for protecting users from on-chain fraud.

## Tasks

The environment defines **3 grader-backed tasks**, each with a deterministic
score in `[0, 1]`. Difficulty progresses from surface-level pattern matching to
deeper multi-signal reasoning.

| # | Task | Difficulty | What the agent sees | What it must classify |
|---|------|-----------|---------------------|-----------------------|
| 1 | `contract_analysis` | **Easy** | Solidity source snippet | Hidden mint, owner-only drain, transfer blocks, etc. |
| 2 | `transaction_analysis` | **Medium** | On-chain tx pattern (holder distribution, sell/buy ratios, dev wallet activity) | Wash trading, slow rug, sandwich victims |
| 3 | `liquidity_analysis` | **Hard** | LP pool metrics (lock status, depth, removal events, paired token) | Imminent exit liquidity, fake locks, soft rugs |

Each task runs for 15 samples per episode (45 total). The grader (`_compute_reward`)
is a pure function of `(verdict, confidence, ground_truth_label, vulnerability_type)`
— deterministic and reproducible.

## Quick Start

```python
from envs.rugguard_env import RugGuardEnv, RugGuardAction

with RugGuardEnv(base_url="http://localhost:7860") as env:
    result = env.reset()
    obs = result.observation
    print(obs.task_type, obs.token_name)
    print(obs.token_data)

    result = env.step(RugGuardAction(
        verdict="rug_pull",
        confidence=0.9,
        reasoning="Owner holds 95% of supply with no lock and a rugPull() function.",
    ))
    print(result.reward, result.done)
```

## Episode Structure

| Parameter | Value |
|-----------|-------|
| Steps per episode | 45 |
| Steps per task type | 15 |
| Task types | `contract_analysis`, `transaction_analysis`, `liquidity_analysis` |
| Terminal condition | All 45 samples classified |

15 samples per task gives statistically meaningful per-task scores
(~6.7% per-sample resolution) without exceeding the 20-min runtime budget.

## Action Space

```python
RugGuardAction(
    verdict: Literal["rug_pull", "honeypot", "wash_trading", "safe"],
    confidence: float,   # [0.0, 1.0]
    reasoning: str,      # free-text explanation
)
```

## Observation Space

```python
RugGuardObservation(
    task_type: str,       # "contract_analysis" | "transaction_analysis" | "liquidity_analysis"
    token_name: str,      # Token symbol
    token_data: str,      # Raw data for analysis
    step_number: int,     # Current step (1-indexed)
    total_steps: int,     # Always 15
    last_reward: float,   # Reward from previous step
    echoed_message: str,  # Echo of last reasoning
    done: bool,
    reward: float,
)
```

## Reward Design

| Component | Points | Condition |
|-----------|--------|-----------|
| Correct verdict | +0.5 | `verdict == ground_truth_label` |
| Correct vulnerability type | +0.3 | Correct verdict on scam token |
| Confidence calibration | +0.2×confidence | When correct |
| Confidence calibration | +0.2×(1-confidence) | When wrong |

Maximum reward per step: **1.0**. All rewards clamped to `[0, 1]`.

## Data

All samples are static JSON bundled in `data/` — no external API calls.

| File | Task Type | Samples | Split |
|------|-----------|---------|-------|
| `data/contracts.json` | `contract_analysis` | 52 | ~50/50 scam/safe |
| `data/transactions.json` | `transaction_analysis` | 52 | ~50/50 scam/safe |
| `data/liquidity.json` | `liquidity_analysis` | 52 | ~50/50 scam/safe |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RUGGUARD_STEPS_PER_TASK` | `15` | Samples per task type per episode |
| `RUGGUARD_SEED` | (random) | Fixed seed for reproducible sampling |
| `PORT` | `7860` | HTTP server port |

## Running Locally

```bash
# Install dependencies
cd envs/rugguard_env
pip install -e .

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or via Docker
docker build -t rugguard-env -f server/Dockerfile .
docker run -p 7860:7860 rugguard-env
```

## Validation

```bash
PYTHONPATH=src:envs uv run python -c \
  "from envs.rugguard_env.server.rugguard_environment import RugGuardEnvironment; e = RugGuardEnvironment(); print(e.reset())"
```

## Baseline Inference

Run the baseline LLM agent against a deployed RugGuard server:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your_hf_token>
export RUGGUARD_URL=http://localhost:8000   # or your HF Space URL
python inference.py
```

The script (`inference.py` in repo root) uses the **OpenAI client** as required
by the spec, walks the full 15-step episode, and emits one `[START]/[STEP]*/[END]`
block per task type.

### Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via the HF Inference Router
(15 samples per task, single seed):

| Task | Score (0–1) | Success |
|------|-------------|---------|
| `contract_analysis` | **0.77** | true |
| `transaction_analysis` | **0.78** | true |
| `liquidity_analysis` | **0.73** | true |

Total runtime: **~4.5 minutes** for 45 steps (well under the 20-min budget).

The dataset includes adversarial near-miss samples (fake CertiK audit
comments, ownership-renounced rugs, honeypots that mimic legitimate
transfer-tax tokens, soft rugs with time-delayed drains). Qwen2.5-72B
hovers around 0.73–0.78 across all three tasks — a clear signal that
frontier models still have headroom before saturating this benchmark.

## Deploy to HF Spaces

```bash
cd envs/rugguard_env
openenv build
openenv push --repo-id <your-org>/rugguard-env
```
