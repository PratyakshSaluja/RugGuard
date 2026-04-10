"""
Enrich RugGuard datasets with:
  1. Pre-baked investigation results per sample (eliminates runtime label leakage)
  2. Difficulty tiers (easy / medium / hard) per sample

Run after generate_datasets.py to augment the JSON files.
"""

import json
import os
import random
import hashlib

random.seed(123)

DATA_DIR = os.path.join(os.path.dirname(__file__), "rugguard_env", "data")

ALL_TOOLS = [
    "holder_distribution",
    "contract_functions",
    "deployer_history",
    "social_signals",
    "similar_contracts",
    "price_history",
]


def _rng_for(token_name, tool):
    """Deterministic RNG per (token, tool) pair."""
    seed = int(hashlib.md5(f"{token_name}_{tool}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


# =========================================================================
# Investigation generators — one function per tool, per label
# These produce realistic but non-trivially-revealing investigation text
# =========================================================================

def inv_holder_distribution(name, label, difficulty):
    r = _rng_for(name, "holder_distribution")

    if label == "rug_pull":
        if difficulty == "easy":
            top = r.randint(80, 95)
            return (f"Holder Analysis for {name}:\n"
                    f"- Top wallet holds {top}% of total supply\n"
                    f"- Top 5 wallets hold {r.randint(top, 99)}% of supply\n"
                    f"- Only {r.randint(50, 300)} total holders\n"
                    f"- Gini coefficient: {r.uniform(0.90, 0.98):.2f} (extremely concentrated)\n"
                    f"- Top wallet funded via Tornado Cash {r.randint(1, 5)} days before deployment\n"
                    f"- No distribution event or airdrop detected")
        elif difficulty == "medium":
            top = r.randint(55, 75)
            return (f"Holder Analysis for {name}:\n"
                    f"- Top wallet holds {top}% of supply (labeled as 'team vesting')\n"
                    f"- {r.randint(200, 800)} holders, growing slowly\n"
                    f"- Gini coefficient: {r.uniform(0.78, 0.88):.2f}\n"
                    f"- Top wallet has no timelock or vesting contract\n"
                    f"- Second largest wallet ({r.randint(5, 15)}%) is DEX pair\n"
                    f"- Wallet age: {r.randint(3, 21)} days")
        else:  # hard
            top = r.randint(25, 45)
            return (f"Holder Analysis for {name}:\n"
                    f"- Top wallet holds {top}% (claims to be DAO treasury)\n"
                    f"- {r.randint(500, 2000)} holders across {r.randint(2, 4)} chains\n"
                    f"- Gini: {r.uniform(0.65, 0.78):.2f} (moderate concentration)\n"
                    f"- Top wallet is an EOA, not a multisig\n"
                    f"- Cluster analysis: top 3 wallets funded from same CEX withdrawal\n"
                    f"- Distribution looks organic at surface level")

    elif label == "honeypot":
        if difficulty == "easy":
            return (f"Holder Analysis for {name}:\n"
                    f"- {r.randint(800, 3000)} holders (only increasing, zero exits)\n"
                    f"- Top wallet: {r.randint(20, 40)}% (deployer)\n"
                    f"- No wallets have successfully reduced position in {r.randint(10, 30)} days\n"
                    f"- Median holding: ${r.randint(50, 300)}\n"
                    f"- Holder count growth: +{r.randint(50, 200)}/day (all buys)")
        elif difficulty == "medium":
            return (f"Holder Analysis for {name}:\n"
                    f"- {r.randint(1500, 5000)} holders\n"
                    f"- Top wallet: {r.randint(10, 25)}% (labeled marketing wallet)\n"
                    f"- {r.randint(2, 5)} wallets have sold — all pre-launch wallets\n"
                    f"- Post-launch sellers: 0 successful\n"
                    f"- Gini: {r.uniform(0.55, 0.70):.2f}\n"
                    f"- Holder velocity: growing {r.randint(3, 8)}% daily")
        else:  # hard
            return (f"Holder Analysis for {name}:\n"
                    f"- {r.randint(3000, 10000)} holders, appears well-distributed\n"
                    f"- Top wallet: {r.randint(5, 12)}% (team, with vesting UI)\n"
                    f"- Gini: {r.uniform(0.45, 0.60):.2f} (looks healthy)\n"
                    f"- {r.randint(5, 15)} wallets have transferred tokens wallet-to-wallet\n"
                    f"- But 0 successful DEX sells from non-whitelisted addresses\n"
                    f"- Subtle: wallet-to-wallet works, DEX sells don't")

    elif label == "wash_trading":
        if difficulty == "easy":
            n = r.randint(5, 12)
            return (f"Holder Analysis for {name}:\n"
                    f"- Only {n} total holders (all active traders)\n"
                    f"- All {n} wallets created within {r.randint(1, 6)} hours of each other\n"
                    f"- All funded from single source address\n"
                    f"- No wallets hold any other tokens\n"
                    f"- Average wallet age: {r.randint(1, 7)} days")
        elif difficulty == "medium":
            n = r.randint(15, 30)
            return (f"Holder Analysis for {name}:\n"
                    f"- {n} holders, {r.randint(10, n)} are active traders\n"
                    f"- Cluster analysis: {r.randint(2, 4)} distinct wallet groups, each from one source\n"
                    f"- Some wallets hold dust amounts of other tokens (cover)\n"
                    f"- Wallet creation spread over {r.randint(2, 14)} days\n"
                    f"- Net token flow between top wallets: approximately zero")
        else:  # hard
            n = r.randint(30, 80)
            return (f"Holder Analysis for {name}:\n"
                    f"- {n} holders across {r.randint(2, 3)} chains\n"
                    f"- Gini: {r.uniform(0.50, 0.65):.2f} (moderate)\n"
                    f"- {r.randint(10, 30)} active traders with legitimate-looking portfolios\n"
                    f"- Subtle: trade timing analysis shows {r.randint(85, 95)}% of trades within same block\n"
                    f"- Gas price fingerprinting: {r.randint(70, 90)}% identical priority fees\n"
                    f"- Appears organic unless timing/gas patterns are checked")

    else:  # safe
        h = r.randint(5000, 150000)
        if difficulty == "easy":
            return (f"Holder Analysis for {name}:\n"
                    f"- {h:,} holders, growing organically +{r.uniform(1, 5):.1f}%/month\n"
                    f"- Top wallet: {r.uniform(1.5, 3.5):.1f}% (protocol treasury, multisig 3/5)\n"
                    f"- Top 10: {r.uniform(12, 22):.0f}% (includes DEX pairs)\n"
                    f"- Gini: {r.uniform(0.40, 0.55):.2f} (well distributed)\n"
                    f"- Ownership renounced, no admin keys")
        elif difficulty == "medium":
            return (f"Holder Analysis for {name}:\n"
                    f"- {h:,} holders\n"
                    f"- Top wallet: {r.uniform(3, 8):.1f}% (team vesting, locked 12mo)\n"
                    f"- {r.randint(5, 15)} whale wallets hold >{r.uniform(0.5, 1.5):.1f}% each\n"
                    f"- Gini: {r.uniform(0.50, 0.65):.2f} (moderate)\n"
                    f"- Smart money: {r.randint(3, 10)} Nansen-labeled fund wallets")
        else:  # hard — safe but looks suspicious
            return (f"Holder Analysis for {name}:\n"
                    f"- {h:,} holders\n"
                    f"- Top wallet: {r.uniform(8, 18):.1f}% (looks concentrated, but it's Binance hot wallet)\n"
                    f"- Gini: {r.uniform(0.60, 0.72):.2f} (elevated but explained by CEX custody)\n"
                    f"- {r.randint(2, 5)} large wallets are CEX deposit addresses\n"
                    f"- Actual user distribution excluding CEXs: Gini {r.uniform(0.38, 0.50):.2f}\n"
                    f"- Ownership: multisig with {r.randint(3, 7)} public signers")


def inv_contract_functions(name, label, difficulty):
    r = _rng_for(name, "contract_functions")

    if label == "rug_pull":
        if difficulty == "easy":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Owner-only: withdrawAll(), emergencyDrain(), clearStuckBalance()\n"
                    f"- Can transfer all ETH and tokens to owner at any time\n"
                    f"- Ownership NOT renounced, no timelock\n"
                    f"- {r.randint(3, 6)} functions move funds to owner address\n"
                    f"- Contract is not verified on block explorer")
        elif difficulty == "medium":
            funcs = r.sample(["migrateV2(address)", "recoverTokens(address,uint256)",
                              "setRouter(address)", "rebalanceTreasury(uint256)"], k=2)
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Owner-only: {', '.join(funcs)}\n"
                    f"- Functions disguised as maintenance/migration utilities\n"
                    f"- {funcs[0]} can send contract's entire ETH balance to arbitrary address\n"
                    f"- No timelock on admin functions\n"
                    f"- Contract verified but comments are misleading")
        else:  # hard
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Standard ERC20 with staking module\n"
                    f"- Owner-only: setRewardRate(), toggleStaking(), withdrawETH()\n"
                    f"- withdrawETH() sends to treasury (which is owner-controlled EOA)\n"
                    f"- setWithdrawalTimelock(0) can disable all withdrawal delays\n"
                    f"- Subtle: rebalanceTreasury() moves staked tokens without user consent\n"
                    f"- Audit: self-published, not from recognized firm")

    elif label == "honeypot":
        if difficulty == "easy":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- updateFees(uint256,uint256) — NO upper bound check\n"
                    f"- Current sell tax: {r.randint(85, 99)}%\n"
                    f"- Buy tax: {r.randint(3, 8)}% (looks normal)\n"
                    f"- authorize(address[],bool) — whitelist exempts deployer from fees\n"
                    f"- {r.randint(1, 3)} addresses currently whitelisted")
        elif difficulty == "medium":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- setMinHoldDuration(uint256) — owner can set to any value\n"
                    f"- Currently set to {r.randint(30, 365)} days (users can't sell before then)\n"
                    f"- authorize() exempts specific addresses from hold requirement\n"
                    f"- Transfer function checks: if (to == pair && !authorized[from]) revert\n"
                    f"- Buy path has no restrictions")
        else:  # hard
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Standard-looking ERC20 with anti-snipe mechanism\n"
                    f"- setCooldown(uint256) — set to {r.randint(1, 5)} seconds currently\n"
                    f"- BUT: cooldown resets on every incoming transfer (including reflections)\n"
                    f"- Net effect: if token has reflections, sell cooldown never expires\n"
                    f"- Mechanism is not obvious from reading setCooldown() alone\n"
                    f"- Requires tracing _update() logic to discover")

    elif label == "wash_trading":
        if difficulty == "easy":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- batchDistribute(address[],uint256[]) — bulk send tokens\n"
                    f"- batchCollect(address[],uint256[]) — reclaim tokens back\n"
                    f"- {r.randint(5, 20)} registered 'market maker' addresses\n"
                    f"- rebalanceAllocations() called {r.randint(200, 800)} times in 24h\n"
                    f"- Circular transfer infrastructure is explicit in code")
        elif difficulty == "medium":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- addMarketMaker(address) — {r.randint(8, 25)} registered\n"
                    f"- fundMarketMakers(uint256[]) — bulk fund from owner\n"
                    f"- rebalanceAllocations() — shuffle tokens between market makers\n"
                    f"- Functions labeled as 'liquidity management'\n"
                    f"- No external DEX interaction, all transfers are internal")
        else:  # hard
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Uses AccessControl with OPERATOR_ROLE\n"
                    f"- batchDistribute/batchCollect behind OPERATOR_ROLE (not owner)\n"
                    f"- Operator is a separate contract (proxy pattern)\n"
                    f"- Surface analysis shows standard governance token\n"
                    f"- Need to trace OPERATOR_ROLE grant to find circular transfer logic\n"
                    f"- rebaseIndex manipulation affects balanceOf() display")

    else:  # safe
        if difficulty == "easy":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Standard ERC20, no owner functions\n"
                    f"- Immutable: no admin, no pause, no fee changes\n"
                    f"- Minted at deployment, no further mint possible\n"
                    f"- Audited by {r.choice(['Certik', 'OpenZeppelin', 'Trail of Bits'])}\n"
                    f"- Ownership renounced at deployment")
        elif difficulty == "medium":
            return (f"Contract Function Analysis for {name}:\n"
                    f"- ERC20Votes + governance\n"
                    f"- Mint capped at totalSupply/100 per tx, maxSupply enforced\n"
                    f"- Minter role held by governance timelock (48h delay)\n"
                    f"- No functions transfer tokens/ETH to specific addresses\n"
                    f"- All admin behind {r.choice(['3/5 multisig', 'governance timelock', 'DAO vote'])}")
        else:  # hard — safe but looks suspicious
            return (f"Contract Function Analysis for {name}:\n"
                    f"- Has withdrawETH() and recoverTokens() — looks risky\n"
                    f"- BUT: both gated behind 72h timelock + multisig approval\n"
                    f"- recoverTokens cannot withdraw the native token (hardcoded check)\n"
                    f"- withdrawETH sends to immutable treasury (not owner EOA)\n"
                    f"- Audit: {r.choice(['Spearbit', 'Trail of Bits'])} — no critical findings\n"
                    f"- Appears dangerous but safeguards are real")


def inv_deployer_history(name, label, difficulty):
    r = _rng_for(name, "deployer_history")

    if label in ("rug_pull", "honeypot"):
        if difficulty == "easy":
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {r.randint(1, 10)} days\n"
                    f"- {r.randint(3, 8)} previous contracts, {r.randint(2, 5)} flagged as scam\n"
                    f"- Funded via {r.choice(['Tornado Cash', 'Railgun', 'cross-chain bridge'])}\n"
                    f"- No ENS, no social verification\n"
                    f"- Pattern: deploy → promote → drain → repeat")
        elif difficulty == "medium":
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {r.randint(15, 60)} days\n"
                    f"- {r.randint(2, 5)} previous contracts, {r.randint(1, 2)} inactive (abandoned)\n"
                    f"- Funded via {r.choice(['new CEX withdrawal', 'cross-chain bridge'])}\n"
                    f"- Twitter linked but account is {r.randint(1, 14)} days old\n"
                    f"- Previous contracts have similar bytecode patterns")
        else:  # hard
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {r.randint(90, 365)} days (looks established)\n"
                    f"- {r.randint(5, 12)} previous contracts, none flagged\n"
                    f"- Funded via legitimate CEX ({r.choice(['Coinbase', 'Kraken'])})\n"
                    f"- BUT: wallet was purchased/acquired (activity gap of {r.randint(30, 90)} days)\n"
                    f"- Pre-gap and post-gap transaction patterns are completely different\n"
                    f"- ENS: {name.lower()}.eth (registered {r.randint(2, 10)} days ago)")

    elif label == "wash_trading":
        if difficulty == "easy":
            return (f"Deployer History for {name}:\n"
                    f"- Created {r.randint(5, 15)} contracts in past {r.randint(14, 30)} days\n"
                    f"- All contracts have batch transfer functions\n"
                    f"- Also operates {r.randint(5, 20)} market maker wallets\n"
                    f"- No public identity\n"
                    f"- All previous tokens show volume anomalies")
        elif difficulty == "medium":
            return (f"Deployer History for {name}:\n"
                    f"- {r.randint(3, 8)} contracts deployed, spread across {r.randint(2, 3)} chains\n"
                    f"- Contracts share common library code (batch operations)\n"
                    f"- Deployer funds market maker wallets through intermediaries\n"
                    f"- Some previous tokens are still active with inflated volume\n"
                    f"- No audit history on any deployment")
        else:
            return (f"Deployer History for {name}:\n"
                    f"- Deployer appears legitimate: {r.randint(180, 500)} day old wallet\n"
                    f"- {r.randint(8, 20)} contracts, mostly standard DeFi tools\n"
                    f"- BUT: {r.randint(3, 5)} contracts share uncommon pattern of internal batch transfers\n"
                    f"- Market maker wallets are 2 hops away (not directly linked)\n"
                    f"- Would need graph analysis to connect deployer to wash wallets")

    else:  # safe
        age = r.randint(365, 2000)
        if difficulty == "easy":
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {age} days\n"
                    f"- Known entity: verified team with public identities\n"
                    f"- {r.randint(5, 20)} previous contracts, all active, zero scam flags\n"
                    f"- Funded via {r.choice(['Coinbase', 'protocol revenue', 'known VC'])}\n"
                    f"- ENS: {name.lower()}.eth (registered {r.randint(180, 700)} days ago)")
        elif difficulty == "medium":
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {age} days\n"
                    f"- Pseudonymous but long reputation in DeFi\n"
                    f"- {r.randint(3, 10)} contracts, clean track record\n"
                    f"- Active in governance of {r.choice(['Aave', 'Compound', 'Uniswap'])}\n"
                    f"- Funded via known DAO treasury")
        else:
            return (f"Deployer History for {name}:\n"
                    f"- Wallet age: {age} days\n"
                    f"- Recent large ETH transfers (looks like extraction but it's VC funding round)\n"
                    f"- {r.randint(5, 15)} contracts including 2 that were sunset (normal lifecycle)\n"
                    f"- Sunset contracts had proper migration, users notified\n"
                    f"- Team doxxed via Crunchbase, {r.choice(['a16z', 'Paradigm', 'Polychain'])} backed")


def inv_social_signals(name, label, difficulty):
    r = _rng_for(name, "social_signals")

    if label == "rug_pull":
        if difficulty == "easy":
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: deleted\n"
                    f"- Telegram: admin left, group muted\n"
                    f"- Website: domain expired\n"
                    f"- {r.randint(3, 10)} paid influencer promotions detected\n"
                    f"- No doxxed team, anonymous founders")
        elif difficulty == "medium":
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: {r.randint(5000, 20000)} followers, {r.randint(30, 60)}% bots\n"
                    f"- Telegram: {r.randint(500, 3000)} members, moderation stopped {r.randint(2, 7)} days ago\n"
                    f"- Website: template site, stock images\n"
                    f"- Whitepaper: AI-generated, no technical depth\n"
                    f"- Team photos: reverse image search matches stock photos")
        else:
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: {r.randint(15000, 50000)} followers, appears active\n"
                    f"- Telegram: {r.randint(2000, 8000)} members, active discussion\n"
                    f"- BUT: engagement pattern is bot-like (same accounts reply within seconds)\n"
                    f"- Website: professional, but WHOIS registered {r.randint(5, 20)} days ago\n"
                    f"- Team: LinkedIn profiles exist but have thin history\n"
                    f"- Looks legitimate until you check engagement authenticity")

    elif label == "honeypot":
        if difficulty == "easy":
            return (f"Social Analysis for {name}:\n"
                    f"- Telegram: many users complaining 'cannot sell', 'transaction reverts'\n"
                    f"- Team response: 'slippage issue being fixed'\n"
                    f"- Twitter: active but ignoring sell complaints\n"
                    f"- Reddit: multiple posts about trapped funds\n"
                    f"- Website: no team information")
        elif difficulty == "medium":
            return (f"Social Analysis for {name}:\n"
                    f"- Telegram: {r.randint(1000, 5000)} members, complaints being deleted\n"
                    f"- Team blames 'DEX integration issues' for sell failures\n"
                    f"- Twitter: {r.randint(5000, 15000)} followers, promotional content only\n"
                    f"- Some users report successful sells (these are whitelisted insiders)\n"
                    f"- Website: professional-looking but no team page")
        else:
            return (f"Social Analysis for {name}:\n"
                    f"- Social presence appears healthy\n"
                    f"- Telegram: {r.randint(3000, 10000)} members, mostly positive\n"
                    f"- Negative comments exist but are rare and quickly addressed\n"
                    f"- Team claims 'anti-bot cooldown' explains sell delays\n"
                    f"- Twitter KOLs promoting it (paid, but not obviously so)\n"
                    f"- Difficult to distinguish from legitimate launch friction")

    elif label == "wash_trading":
        if difficulty == "easy":
            return (f"Social Analysis for {name}:\n"
                    f"- Marketing: entirely focused on 'TOP VOLUME ON BSC' claims\n"
                    f"- Twitter: {r.randint(500, 3000)} followers, mostly bots\n"
                    f"- No discussion of utility, technology, or roadmap\n"
                    f"- Listed on volume-ranking sites as #1\n"
                    f"- Zero organic community")
        elif difficulty == "medium":
            return (f"Social Analysis for {name}:\n"
                    f"- Promoted as 'fastest growing DeFi protocol'\n"
                    f"- Twitter: {r.randint(3000, 10000)} followers with purchased engagement\n"
                    f"- Discord: exists but {r.randint(60, 80)}% of messages are from bots\n"
                    f"- Medium blog: repurposed AI content about 'revolutionizing DeFi'\n"
                    f"- No developer community or GitHub activity")
        else:
            return (f"Social Analysis for {name}:\n"
                    f"- Appears legitimate: active Twitter, Discord, blog\n"
                    f"- {r.randint(8000, 25000)} Twitter followers with decent engagement\n"
                    f"- BUT: engagement spikes correlate exactly with volume spikes\n"
                    f"- Social activity drops to zero between trading bot cycles\n"
                    f"- GitHub has commits but all from same author, no real development\n"
                    f"- Sophisticated social facade hiding volume manipulation")

    else:  # safe
        if difficulty == "easy":
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: {r.randint(20000, 200000)} followers, high organic engagement\n"
                    f"- Discord: {r.randint(10000, 50000)} members, active dev discussion\n"
                    f"- GitHub: {r.randint(100, 500)} stars, {r.randint(15, 80)} contributors\n"
                    f"- Team: fully doxxed, regular AMAs\n"
                    f"- Media: featured in {r.choice(['CoinDesk', 'The Block', 'Bankless'])}")
        elif difficulty == "medium":
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: {r.randint(10000, 50000)} followers, organic growth\n"
                    f"- Discord: {r.randint(3000, 15000)} members\n"
                    f"- GitHub: active development, weekly commits\n"
                    f"- Team: pseudonymous but known in ecosystem for {r.randint(2, 5)} years\n"
                    f"- Partnerships: {r.choice(['Chainlink', 'The Graph', 'LayerZero'])} integration")
        else:
            return (f"Social Analysis for {name}:\n"
                    f"- Twitter: {r.randint(5000, 15000)} followers (modest but real)\n"
                    f"- Some FUD threads exist (competitors spreading doubt)\n"
                    f"- Team has been quiet lately (building, not marketing)\n"
                    f"- GitHub: consistent commits but low star count\n"
                    f"- Looks under-the-radar but fundamentals are strong\n"
                    f"- Community small but highly engaged")


def inv_similar_contracts(name, label, difficulty):
    r = _rng_for(name, "similar_contracts")

    if label in ("rug_pull", "honeypot"):
        if difficulty == "easy":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Bytecode: {r.randint(90, 99)}% match with {r.randint(5, 20)} known scam contracts\n"
                    f"- Known scam toolkit detected\n"
                    f"- Matched contracts caused ${r.randint(1, 15)}M in losses\n"
                    f"- TokenSniffer: {r.randint(5, 20)}/100\n"
                    f"- GoPlus: CRITICAL warnings")
        elif difficulty == "medium":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Bytecode: {r.randint(70, 85)}% match with {r.randint(3, 8)} flagged contracts\n"
                    f"- Modified version of known template (custom variable names)\n"
                    f"- Core drain/restriction logic preserved\n"
                    f"- TokenSniffer: {r.randint(25, 40)}/100\n"
                    f"- GoPlus: {r.randint(2, 4)} medium-severity warnings")
        else:
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Bytecode: {r.randint(40, 60)}% match (heavily modified from template)\n"
                    f"- Original template linked to {r.randint(1, 3)} known scams\n"
                    f"- Modifications add legitimate-looking features on top\n"
                    f"- TokenSniffer: {r.randint(45, 60)}/100 (borderline)\n"
                    f"- GoPlus: 1 low-severity warning\n"
                    f"- Requires manual analysis to confirm malicious intent")

    elif label == "wash_trading":
        if difficulty == "easy":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- {r.randint(5, 15)} contracts by same deployer with identical batch transfer logic\n"
                    f"- All similar tokens show volume anomalies\n"
                    f"- TokenSniffer: {r.randint(20, 40)}/100\n"
                    f"- Pattern: deploy → inflate volume → get listed → abandon")
        elif difficulty == "medium":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Contract shares market maker infrastructure with {r.randint(3, 8)} tokens\n"
                    f"- Batch operation pattern seen in known wash trading setups\n"
                    f"- TokenSniffer: {r.randint(35, 50)}/100\n"
                    f"- Similar contracts spread across {r.randint(2, 3)} chains")
        else:
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Uses standard OpenZeppelin AccessControl (common pattern)\n"
                    f"- Batch distribution functions exist in legitimate airdrop contracts too\n"
                    f"- TokenSniffer: {r.randint(50, 65)}/100\n"
                    f"- Distinguishing wash trading infra from legitimate market making requires context\n"
                    f"- {r.randint(2, 4)} structurally similar contracts, unclear if same entity")

    else:  # safe
        if difficulty == "easy":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Standard OpenZeppelin patterns, no custom backdoors\n"
                    f"- No similarity to known scam templates\n"
                    f"- TokenSniffer: {r.randint(85, 100)}/100\n"
                    f"- GoPlus: no issues detected\n"
                    f"- Architecture similar to {r.choice(['Aave', 'Compound', 'Uniswap'])} contracts")
        elif difficulty == "medium":
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Clean architecture, audited patterns\n"
                    f"- TokenSniffer: {r.randint(80, 95)}/100\n"
                    f"- GoPlus: 1 informational finding (non-critical)\n"
                    f"- Similar to {r.randint(10, 30)} legitimate DeFi protocols\n"
                    f"- No known scam contract matches")
        else:
            return (f"Similar Contract Analysis for {name}:\n"
                    f"- Has some patterns that overlap with flagged contracts ({r.randint(20, 35)}% match)\n"
                    f"- BUT: overlap is in common ERC20 boilerplate, not scam logic\n"
                    f"- TokenSniffer: {r.randint(70, 85)}/100 (docked for admin functions that have timelocks)\n"
                    f"- GoPlus: 1 medium warning (owner can pause — but behind governance)\n"
                    f"- False positive risk: high for automated scanners")


def inv_price_history(name, label, difficulty):
    r = _rng_for(name, "price_history")

    if label == "rug_pull":
        if difficulty == "easy":
            return (f"Price History for {name}:\n"
                    f"- Launch: ${r.uniform(0.001, 0.05):.4f}\n"
                    f"- ATH: ${r.uniform(1, 20):.2f} (day {r.randint(1, 5)})\n"
                    f"- Crash: -{r.randint(95, 99)}% in {r.randint(5, 30)} minutes\n"
                    f"- Current: ${r.uniform(0.00001, 0.0005):.6f}\n"
                    f"- Classic pump-and-dump chart pattern")
        elif difficulty == "medium":
            return (f"Price History for {name}:\n"
                    f"- Gradual rise over {r.randint(7, 21)} days: ${r.uniform(0.01, 0.1):.3f} → ${r.uniform(1, 5):.2f}\n"
                    f"- Multiple smaller dips ({r.randint(10, 25)}%) with recovery (manufactured)\n"
                    f"- Final drop: -{r.randint(85, 98)}% in {r.randint(1, 4)} hours\n"
                    f"- Pattern: slow build with fake recoveries → sudden rug\n"
                    f"- Post-rug: flatline at near-zero")
        else:
            return (f"Price History for {name}:\n"
                    f"- {r.randint(30, 90)} day history, organic-looking growth\n"
                    f"- Multiple natural-looking corrections ({r.randint(15, 30)}% pullbacks)\n"
                    f"- Steady decline in last {r.randint(5, 14)} days (-{r.randint(40, 70)}%)\n"
                    f"- NOT a sudden crash — slow bleed as deployer sells in small batches\n"
                    f"- Looks like market downturn, but coincides with team wallet movements\n"
                    f"- Hard to distinguish from legitimate bear market without on-chain data")

    elif label == "honeypot":
        if difficulty == "easy":
            return (f"Price History for {name}:\n"
                    f"- Only goes up: ${r.uniform(0.01, 0.1):.3f} → ${r.uniform(1, 15):.2f}\n"
                    f"- Zero corrections (no one can sell)\n"
                    f"- Volatility: {r.uniform(0.5, 2):.1f}% daily (abnormally low)\n"
                    f"- Monotonically increasing — clear honeypot signature\n"
                    f"- Volume is all buys, no sells registered")
        elif difficulty == "medium":
            return (f"Price History for {name}:\n"
                    f"- Mostly upward: +{r.randint(200, 800)}% since launch\n"
                    f"- Small dips exist ({r.randint(2, 5)}%) but only from whitelisted wallet sells\n"
                    f"- Volatility: {r.uniform(2, 5):.1f}% daily (low for micro-cap)\n"
                    f"- Organic tokens at this market cap show {r.randint(10, 25)}% daily swings\n"
                    f"- Price stability is suspiciously artificial")
        else:
            return (f"Price History for {name}:\n"
                    f"- Price chart looks somewhat normal with {r.randint(3, 8)} visible corrections\n"
                    f"- BUT: all 'corrections' are from same {r.randint(2, 4)} wallets\n"
                    f"- Actual organic sell volume: 0\n"
                    f"- Corrections are manufactured to make chart look healthy\n"
                    f"- Volatility: {r.uniform(4, 8):.1f}% (engineered to appear normal)\n"
                    f"- Requires sell-source analysis to detect deception")

    elif label == "wash_trading":
        if difficulty == "easy":
            return (f"Price History for {name}:\n"
                    f"- Tight range: ${r.uniform(0.5, 1):.2f} - ${r.uniform(1, 1.5):.2f}\n"
                    f"- Volatility: {r.uniform(0.05, 0.5):.2f}% daily (impossibly stable)\n"
                    f"- Price moves in exact ${r.uniform(0.001, 0.01):.3f} increments\n"
                    f"- Volume bars: uniform height (algorithmic)\n"
                    f"- No organic market dynamics visible")
        elif difficulty == "medium":
            return (f"Price History for {name}:\n"
                    f"- Appears to have normal volatility ({r.uniform(3, 8):.0f}% daily)\n"
                    f"- BUT: price always returns to same level within {r.randint(4, 24)} hours\n"
                    f"- Mean-reversion too perfect — natural markets have drift\n"
                    f"- Trade sizes suspiciously uniform\n"
                    f"- Volume spikes at exactly same time each day")
        else:
            return (f"Price History for {name}:\n"
                    f"- Chart looks organic at daily resolution\n"
                    f"- Intraday: {r.randint(80, 95)}% of price movement in {r.randint(2, 5)} minute windows\n"
                    f"- Rest of day: near-zero volatility (bots inactive)\n"
                    f"- Volume autocorrelation: {r.uniform(0.85, 0.98):.2f} (>0.7 indicates algorithmic)\n"
                    f"- Need tick-level data to distinguish from organic trading\n"
                    f"- Daily candles look normal, 1-minute candles reveal bot pattern")

    else:  # safe
        if difficulty == "easy":
            return (f"Price History for {name}:\n"
                    f"- Token age: {r.randint(180, 730)} days\n"
                    f"- ATH: ${r.uniform(10, 100):.2f} | ATL: ${r.uniform(0.5, 5):.2f}\n"
                    f"- 30d: {r.uniform(-15, 20):+.1f}% (tracks market conditions)\n"
                    f"- Volatility: {r.uniform(5, 12):.0f}% daily (normal for tier)\n"
                    f"- Healthy corrections and recoveries throughout history")
        elif difficulty == "medium":
            return (f"Price History for {name}:\n"
                    f"- {r.randint(90, 400)} day history\n"
                    f"- Correlated with ETH price (beta: {r.uniform(0.8, 1.5):.1f})\n"
                    f"- {r.randint(3, 8)} significant drawdowns ({r.randint(20, 40)}%), all recovered\n"
                    f"- Volume follows news cycle (organic)\n"
                    f"- No single-block large trades dominating price action")
        else:
            return (f"Price History for {name}:\n"
                    f"- Recent sharp drop: -{r.randint(25, 45)}% in {r.randint(3, 7)} days\n"
                    f"- Looks like rug pull chart at first glance\n"
                    f"- BUT: drop coincides with broader market selloff\n"
                    f"- Volume spiked from organic panic selling, not insider exits\n"
                    f"- Currently recovering: +{r.randint(5, 15)}% from local low\n"
                    f"- No unusual wallet movements during drawdown")


# Dispatcher
INVESTIGATION_GENERATORS = {
    "holder_distribution": inv_holder_distribution,
    "contract_functions": inv_contract_functions,
    "deployer_history": inv_deployer_history,
    "social_signals": inv_social_signals,
    "similar_contracts": inv_similar_contracts,
    "price_history": inv_price_history,
}


def assign_difficulty(samples):
    """Assign difficulty tiers: 10 easy, 10 medium, 10 hard per label."""
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    for label, items in by_label.items():
        random.shuffle(items)
        for i, s in enumerate(items):
            if i < 10:
                s["difficulty"] = "easy"
            elif i < 20:
                s["difficulty"] = "medium"
            else:
                s["difficulty"] = "hard"


def add_investigations(samples):
    """Pre-generate all 6 investigation results for every sample."""
    for s in samples:
        name = s["token_name"]
        label = s["label"]
        diff = s["difficulty"]
        invs = {}
        for tool, gen_fn in INVESTIGATION_GENERATORS.items():
            invs[tool] = gen_fn(name, label, diff)
        s["investigations"] = invs


def sort_by_difficulty(samples):
    """Sort samples within each task group: easy first, then medium, then hard."""
    order = {"easy": 0, "medium": 1, "hard": 2}
    samples.sort(key=lambda s: order.get(s.get("difficulty", "medium"), 1))


def main():
    for fname in ["contracts.json", "transactions.json", "liquidity.json"]:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        samples = data["samples"]
        assign_difficulty(samples)
        add_investigations(samples)
        sort_by_difficulty(samples)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Stats
        diff_counts = {}
        for s in samples:
            d = s["difficulty"]
            diff_counts[d] = diff_counts.get(d, 0) + 1
        print(f"{fname}: {len(samples)} samples | difficulty: {diff_counts} | "
              f"investigations: {len(samples[0]['investigations'])} tools per sample")


if __name__ == "__main__":
    main()
