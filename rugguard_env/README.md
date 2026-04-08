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

with RugGuardEnv(base_url="http://localhost:8000") as env:
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
    total_steps: int,     # Always 45 (15 per task × 3 tasks)
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

| File | Task Type | Samples | Scam / Safe |
|------|-----------|---------|-------------|
| `data/contracts.json`    | `contract_analysis`    | 55 | 32 / 23 |
| `data/transactions.json` | `transaction_analysis` | 52 | 30 / 22 |
| `data/liquidity.json`    | `liquidity_analysis`   | 52 | 30 / 22 |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RUGGUARD_STEPS_PER_TASK` | `15` | Samples per task type per episode |
| `RUGGUARD_SEED` | (random) | Fixed seed for reproducible sampling |
| `RUGGUARD_TASK_FILTER` | (none) | Restrict episode to one task type |
| `PORT` | `8000` | HTTP server port |

## Running Locally

```bash
# Install dependencies
cd rugguard_env
pip install -e .

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or via Docker
docker build -t rugguard-env:latest .
docker run -p 8000:8000 rugguard-env:latest
```

## Validation

```bash
PYTHONPATH=src:envs uv run python -c \
  "from envs.rugguard_env.server.rugguard_environment import RugGuardEnvironment; e = RugGuardEnvironment(); print(e.reset())"
```

## Baseline Inference

Run the baseline LLM agent against a deployed RugGuard server:

```bash
export API_BASE_URL=https://router.huggingface.co/v1    # or validator proxy
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_KEY=<your_api_key>                           # validator injects this
export LOCAL_IMAGE_NAME=rugguard-env:latest             # script spins up the container
python inference.py
```

The script (`inference.py` in repo root) uses the **OpenAI client** as required
by the spec, calls `RugGuardEnv.from_docker_image(LOCAL_IMAGE_NAME)` to spin
up its own container, walks the full 45-step episode, and emits one
`[START]/[STEP]*/[END]` block per task type.

### Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via the HF Inference Router
(15 samples per task, single seed):

| Task | Score (0–1) | Success |
|------|-------------|---------|
| `contract_analysis`    | **0.75** | true |
| `transaction_analysis` | **0.83** | true |
| `liquidity_analysis`   | **0.53** | true |

Total runtime: **~4.5 minutes** for 45 steps (well under the 20-min budget).

The dataset includes adversarial near-miss samples (fake CertiK audit
comments, ownership-renounced rugs, honeypots that mimic legitimate
transfer-tax tokens, soft rugs with time-delayed drains). Qwen2.5-72B
ranges from ~0.53 on the hardest task (`liquidity_analysis`) up to
~0.83 on `transaction_analysis` — a clear signal that frontier models
still have meaningful headroom before saturating this benchmark.

## Deploy to HF Spaces

```bash
cd rugguard_env
huggingface-cli upload <your-org>/rugguard-env . . --repo-type space
```
