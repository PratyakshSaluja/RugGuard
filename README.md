# RugGuard — OpenEnv Environment

**Meta PyTorch x Scaler OpenEnv Bootcamp Hackathon**

RugGuard is an OpenEnv-compliant environment for training and evaluating LLM
agents on real-world **crypto token scam detection**. Unlike simple classification
benchmarks, RugGuard requires multi-step investigation: agents must decide which
tools to use, gather evidence, and then classify tokens under uncertainty.

- **HF Space:** https://huggingface.co/spaces/Pratyakshh/rugguard-env
- **Env code:** [`rugguard_env/`](rugguard_env/)
- **Baseline runner:** [`inference.py`](inference.py)

## Why this environment

Crypto rug pulls, honeypots, and wash-trading scams drain billions of dollars
from retail users every year. Scam triage today is either done manually by
security researchers or by brittle rule-based scanners that miss novel attack
patterns. RugGuard turns this triage into a proper agent benchmark: the agent
must reason about Solidity source code, on-chain transaction histories, and
liquidity pool dynamics — three concrete sub-skills a real security analyst
uses every day.

**Data sources:** Token patterns derived from real-world validated rug pull
incidents ([dianxiang-sun/rug_pull_dataset](https://github.com/dianxiang-sun/rug_pull_dataset) — 2,391 ETH/BSC incidents),
smart contract vulnerability patterns from [smartbugs-curated](https://github.com/smartbugs/smartbugs-curated),
and realistic DeFi protocol structures.

## Action space: Investigate then classify

Each token presents a two-phase decision:

1. **Investigate** (optional, up to 3 per token): request additional data
   using one of 6 investigation tools before committing to a verdict
2. **Classify** (required): submit verdict, confidence, and reasoning

### Investigation tools

| Tool | What it reveals |
|------|----------------|
| `holder_distribution` | Wallet concentration, Gini coefficient, funding sources |
| `contract_functions` | Owner-only functions, fee mechanisms, access patterns |
| `deployer_history` | Deployer wallet age, prior contracts, scam flags |
| `social_signals` | Social media activity, community sentiment, team info |
| `similar_contracts` | Bytecode similarity to known scams, TokenSniffer score |
| `price_history` | Price trajectory, volatility patterns, anomalies |

Agents must balance investigation depth (more info = better classification)
against efficiency (fewer investigations = bonus reward).

## Tasks

| # | Task | Difficulty | What the agent sees | Must classify |
|---|------|-----------|---------------------|---------------|
| 1 | `contract_analysis`    | Easy   | Solidity source with subtle backdoors                           | Hidden drain, sell restrictions, volume manipulation |
| 2 | `transaction_analysis` | Medium | On-chain tx patterns (holder dist, sell/buy ratios, timing)     | Wash trading, slow rug, trapped funds               |
| 3 | `liquidity_analysis`   | Hard   | LP pool metrics (lock status, depth, removal events)            | Exit liquidity, fake locks, oracle manipulation      |

Each task runs for **15 samples** per episode (**45 classifications total**).
Datasets contain **120 samples per task** (30 per label), with realistic
token names and non-obvious scam patterns derived from real incidents.

## Reward function

| Component                  | Points              | Condition                         |
|----------------------------|---------------------|-----------------------------------|
| Correct verdict            | +0.50               | `verdict == ground_truth_label`   |
| Correct vulnerability type | +0.20               | Correct verdict on scam token     |
| Confidence calibration     | +0.15 x confidence  | When correct                      |
| Confidence calibration     | +0.15 x (1-conf)    | When wrong                        |
| Partial credit             | +0.02-0.05          | Close-but-wrong (e.g. rug_pull predicted as honeypot) |
| Investigation efficiency   | +0.05 x (1-inv/3)   | Bonus for fewer investigations when correct |

Per-step reward is clamped to `[0, 1]`. The reward function provides dense,
multi-component signal suitable for RL training.

## Baseline scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via LiteLLM proxy
(15 samples per task, 1 investigation per token):

| Task                   | Score | Success |
|------------------------|-------|---------|
| `contract_analysis`    | 0.75  | Yes     |
| `transaction_analysis` | 0.83  | Yes     |
| `liquidity_analysis`   | 0.53  | Yes     |

## Running the baseline

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_KEY=<your_api_key>
export LOCAL_IMAGE_NAME=rugguard-env:latest
python inference.py
```

## Running the environment locally

```bash
cd rugguard_env
docker build -t rugguard-env:latest .
docker run -p 8000:8000 rugguard-env:latest

# Sanity check
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Investigate
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "investigate", "tool": "holder_distribution"}'

# Classify
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "verdict": "rug_pull", "confidence": 0.85, "reasoning": "High concentration + deployer flags"}'
```

See [`rugguard_env/README.md`](rugguard_env/README.md) for the full
action/observation space reference and deployment instructions.
