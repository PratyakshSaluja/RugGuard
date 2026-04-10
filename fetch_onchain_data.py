"""
fetch_onchain_data.py — Fetch real on-chain token data from public APIs.

Pulls live blockchain data for known scam and safe tokens, then outputs
enriched JSON that can be merged into RugGuard datasets.

APIs used (all free, no key required):
  - GoPlus Security API: token security analysis, holder info, honeypot detection
  - Etherscan API: contract source code, deployer info (free key required)
  - CoinGecko API: price history, market data (free demo plan)

Usage:
    # Set your free Etherscan API key (get one at etherscan.io/myapikey)
    export ETHERSCAN_API_KEY=YourFreeKey

    python fetch_onchain_data.py
    python fetch_onchain_data.py --chain 1 --output fetched_tokens.json
    python fetch_onchain_data.py --addresses 0xabc,0xdef --chain 56
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# Chain IDs: 1=Ethereum, 56=BSC, 137=Polygon, 42161=Arbitrum
CHAIN_CONFIGS = {
    1: {
        "name": "Ethereum",
        "etherscan_base": "https://api.etherscan.io/api",
        "coingecko_platform": "ethereum",
    },
    56: {
        "name": "BSC",
        "etherscan_base": "https://api.bscscan.com/api",
        "coingecko_platform": "binance-smart-chain",
    },
    137: {
        "name": "Polygon",
        "etherscan_base": "https://api.polygonscan.com/api",
        "coingecko_platform": "polygon-pos",
    },
}

# Known tokens to fetch — mix of scams and legitimate tokens for balanced data
# These are real addresses on Ethereum mainnet
KNOWN_TOKENS = {
    1: [
        # === SAFE / LEGITIMATE ===
        {"address": "0xdac17f958d2ee523a2206206994597c13d831ec7", "label": "safe", "name": "USDT"},
        {"address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "label": "safe", "name": "USDC"},
        {"address": "0x6b175474e89094c44da98b954eedeac495271d0f", "label": "safe", "name": "DAI"},
        {"address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "label": "safe", "name": "UNI"},
        {"address": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", "label": "safe", "name": "AAVE"},
        {"address": "0x514910771af9ca656af840dff83e8264ecf986ca", "label": "safe", "name": "LINK"},
        {"address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "label": "safe", "name": "WBTC"},
        {"address": "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2", "label": "safe", "name": "MKR"},

        # === KNOWN SCAMS (historically flagged) ===
        {"address": "0x9469d013805bffb7d3debe5e7839237e535ec483", "label": "rug_pull", "name": "RING Financial"},
        {"address": "0xa2b4c0af19cc16a6cfacce81f192b024d625817d", "label": "honeypot", "name": "KISHU"},
        {"address": "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", "label": "safe", "name": "SHIB"},
        {"address": "0xc5fb36dd2fb59d3b98deff88425a3f425ee469ed", "label": "honeypot", "name": "Dejitaru Tsuka"},
        {"address": "0x15874d65e649880c2614e7a480cb7c9a55787ff6", "label": "rug_pull", "name": "EthereumMax"},
    ],
}

# Rate limiting
GOPLUS_DELAY = 0.5      # seconds between GoPlus calls
ETHERSCAN_DELAY = 0.25  # seconds between Etherscan calls (5/sec free)
COINGECKO_DELAY = 2.0   # seconds between CoinGecko calls (30/min free)


# ---------------------------------------------------------------------------
# GoPlus Security API
# ---------------------------------------------------------------------------

def fetch_goplus_security(chain_id: int, address: str) -> Dict[str, Any]:
    """Fetch token security data from GoPlus (free, no key)."""
    url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"
    params = {"contract_addresses": address.lower()}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 1 and data.get("result"):
            return data["result"].get(address.lower(), {})
    except Exception as e:
        print(f"  [GoPlus] Error for {address}: {e}", file=sys.stderr)

    return {}


def parse_goplus_to_investigations(gp: Dict[str, Any], token_name: str) -> Dict[str, str]:
    """Convert raw GoPlus response into investigation-style text."""
    results = {}

    # --- holder_distribution ---
    holders = gp.get("holder_count", "?")
    creator_pct = gp.get("creator_percent", "?")
    owner_pct = gp.get("owner_percent", "?")
    lp_holders = gp.get("lp_holder_count", "?")
    top_holders = gp.get("holders", [])

    holder_text = f"Holder Analysis for {token_name} (on-chain):\n"
    holder_text += f"- Total holders: {holders}\n"
    holder_text += f"- Creator balance %: {creator_pct}\n"
    holder_text += f"- Owner balance %: {owner_pct}\n"
    holder_text += f"- LP holders: {lp_holders}\n"
    if top_holders:
        holder_text += "- Top holders:\n"
        for h in top_holders[:5]:
            addr = h.get("address", "?")[:10] + "..."
            pct = h.get("percent", "?")
            locked = " (locked)" if h.get("is_locked") == 1 else ""
            holder_text += f"  {addr} — {float(pct)*100:.1f}%{locked}\n"
    results["holder_distribution"] = holder_text.strip()

    # --- contract_functions ---
    contract_text = f"Contract Function Analysis for {token_name} (on-chain):\n"
    flags = {
        "is_proxy": "Is proxy contract",
        "is_mintable": "Mintable",
        "can_take_back_ownership": "Can reclaim ownership",
        "owner_change_balance": "Owner can change balances",
        "hidden_owner": "Hidden owner",
        "selfdestruct": "Has selfdestruct",
        "external_call": "Makes external calls",
        "is_open_source": "Open source",
    }
    for key, desc in flags.items():
        val = gp.get(key, "?")
        status = "YES" if val == "1" else "NO" if val == "0" else str(val)
        contract_text += f"- {desc}: {status}\n"
    results["contract_functions"] = contract_text.strip()

    # --- deployer_history ---
    deployer_text = f"Deployer History for {token_name} (on-chain):\n"
    deployer_text += f"- Creator address: {gp.get('creator_address', '?')}\n"
    deployer_text += f"- Owner address: {gp.get('owner_address', '?')}\n"
    deployer_text += f"- Creator balance: {gp.get('creator_balance', '?')}\n"
    deployer_text += f"- Owner balance: {gp.get('owner_balance', '?')}\n"
    trust = gp.get("trust_list", "?")
    deployer_text += f"- On trust list: {'YES' if trust == '1' else 'NO' if trust == '0' else trust}\n"
    results["deployer_history"] = deployer_text.strip()

    # --- social_signals ---
    social_text = f"Social Analysis for {token_name} (on-chain):\n"
    social_text += f"- Token name: {gp.get('token_name', '?')}\n"
    social_text += f"- Token symbol: {gp.get('token_symbol', '?')}\n"
    # GoPlus doesn't have social data, so mark as limited
    is_true = gp.get("is_true_token", "?")
    social_text += f"- Verified/true token: {'YES' if is_true == '1' else 'NO' if is_true == '0' else 'unknown'}\n"
    fake = gp.get("fake_token", None)
    if fake:
        social_text += f"- Fake token warning: {fake}\n"
    social_text += f"- Note: Limited social data from on-chain analysis\n"
    results["social_signals"] = social_text.strip()

    # --- similar_contracts ---
    similar_text = f"Similar Contract Analysis for {token_name} (on-chain):\n"
    similar_text += f"- Open source: {'YES' if gp.get('is_open_source') == '1' else 'NO'}\n"
    similar_text += f"- Is proxy: {'YES' if gp.get('is_proxy') == '1' else 'NO'}\n"
    note = gp.get("note", "")
    if note:
        similar_text += f"- Security note: {note}\n"
    # Aggregate risk score from flags
    risk_flags = sum(1 for k in [
        "is_honeypot", "honeypot_with_same_creator", "is_blacklisted",
        "is_whitelisted", "transfer_pausable", "cannot_sell_all",
        "selfdestruct", "owner_change_balance",
    ] if gp.get(k) == "1")
    similar_text += f"- Risk flags triggered: {risk_flags}/8\n"
    similar_text += f"- Anti-whale: {'YES' if gp.get('is_anti_whale') == '1' else 'NO'}\n"
    results["similar_contracts"] = similar_text.strip()

    # --- price_history ---
    price_text = f"Price/Liquidity for {token_name} (on-chain):\n"
    buy_tax = gp.get("buy_tax", "?")
    sell_tax = gp.get("sell_tax", "?")
    price_text += f"- Buy tax: {buy_tax}\n"
    price_text += f"- Sell tax: {sell_tax}\n"
    price_text += f"- Cannot buy: {'YES' if gp.get('cannot_buy') == '1' else 'NO'}\n"
    price_text += f"- Cannot sell all: {'YES' if gp.get('cannot_sell_all') == '1' else 'NO'}\n"
    price_text += f"- Is honeypot: {'YES' if gp.get('is_honeypot') == '1' else 'NO'}\n"

    dex_list = gp.get("dex", [])
    if dex_list:
        price_text += "- DEX liquidity:\n"
        for d in dex_list[:5]:
            liq = d.get("liquidity", "?")
            name = d.get("name", "?")
            price_text += f"  {name}: ${float(liq):,.0f}\n" if liq != "?" else f"  {name}: ?\n"
    results["price_history"] = price_text.strip()

    return results


# ---------------------------------------------------------------------------
# Etherscan API (free key required)
# ---------------------------------------------------------------------------

def fetch_etherscan_source(chain_id: int, address: str) -> Optional[str]:
    """Fetch verified contract source code from Etherscan."""
    if not ETHERSCAN_API_KEY:
        return None

    config = CHAIN_CONFIGS.get(chain_id)
    if not config:
        return None

    url = config["etherscan_base"]
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": ETHERSCAN_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1" and data.get("result"):
            source = data["result"][0].get("SourceCode", "")
            if source:
                # Truncate very long contracts
                return source[:8000] if len(source) > 8000 else source
    except Exception as e:
        print(f"  [Etherscan] Error for {address}: {e}", file=sys.stderr)

    return None


def fetch_etherscan_txlist(chain_id: int, address: str, page: int = 1,
                           count: int = 20) -> List[Dict]:
    """Fetch recent transactions for a contract."""
    if not ETHERSCAN_API_KEY:
        return []

    config = CHAIN_CONFIGS.get(chain_id)
    if not config:
        return []

    url = config["etherscan_base"]
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": address,
        "page": page,
        "offset": count,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1":
            return data.get("result", [])
    except Exception as e:
        print(f"  [Etherscan TX] Error for {address}: {e}", file=sys.stderr)

    return []


# ---------------------------------------------------------------------------
# CoinGecko API (free demo plan, no key)
# ---------------------------------------------------------------------------

def fetch_coingecko_data(chain_id: int, address: str) -> Dict[str, Any]:
    """Fetch token market data from CoinGecko."""
    config = CHAIN_CONFIGS.get(chain_id)
    if not config:
        return {}

    platform = config["coingecko_platform"]
    url = f"https://api.coingecko.com/api/v3/coins/{platform}/contract/{address.lower()}"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            print("  [CoinGecko] Rate limited, skipping", file=sys.stderr)
            return {}
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [CoinGecko] Error for {address}: {e}", file=sys.stderr)

    return {}


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_token(chain_id: int, address: str, name: str, label: str) -> Dict[str, Any]:
    """Fetch all available data for a single token."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Fetching: {name} ({address[:10]}...) — expected: {label}", file=sys.stderr)

    result = {
        "token_name": name,
        "address": address,
        "chain_id": chain_id,
        "label": label,
        "vulnerability_type": label if label != "safe" else None,
        "sources": [],
    }

    # 1. GoPlus Security
    print("  [1/3] GoPlus Security...", file=sys.stderr)
    gp = fetch_goplus_security(chain_id, address)
    if gp:
        result["goplus_raw"] = gp
        result["investigations"] = parse_goplus_to_investigations(gp, name)
        result["sources"].append("goplus")

        # Use GoPlus honeypot flag to validate/suggest label
        is_honeypot = gp.get("is_honeypot", "0") == "1"
        cannot_sell = gp.get("cannot_sell_all", "0") == "1"
        if is_honeypot or cannot_sell:
            result["goplus_flags"] = "HONEYPOT_DETECTED"
        risk_count = sum(1 for k in [
            "is_honeypot", "selfdestruct", "owner_change_balance",
            "hidden_owner", "cannot_sell_all",
        ] if gp.get(k) == "1")
        result["goplus_risk_score"] = risk_count
    time.sleep(GOPLUS_DELAY)

    # 2. Etherscan source code + recent txs
    if ETHERSCAN_API_KEY:
        print("  [2/3] Etherscan source...", file=sys.stderr)
        source = fetch_etherscan_source(chain_id, address)
        if source:
            result["token_data"] = source
            result["sources"].append("etherscan_source")
        time.sleep(ETHERSCAN_DELAY)

        print("  [2b/3] Etherscan transactions...", file=sys.stderr)
        txs = fetch_etherscan_txlist(chain_id, address, count=20)
        if txs:
            # Summarize txs for investigation data
            tx_summary = f"Recent Transactions for {name} (on-chain, last {len(txs)}):\n"
            unique_froms = set()
            unique_tos = set()
            total_value = 0
            for tx in txs:
                unique_froms.add(tx.get("from", ""))
                unique_tos.add(tx.get("to", ""))
                try:
                    val = int(tx.get("value", 0)) / (10 ** int(tx.get("tokenDecimal", 18)))
                    total_value += val
                except (ValueError, ZeroDivisionError):
                    pass
            tx_summary += f"- Unique senders: {len(unique_froms)}\n"
            tx_summary += f"- Unique receivers: {len(unique_tos)}\n"
            tx_summary += f"- Total volume (sample): {total_value:,.2f} tokens\n"
            # Check for circular patterns
            circular = unique_froms & unique_tos
            if circular:
                tx_summary += f"- Addresses appearing as both sender & receiver: {len(circular)}\n"
            result.setdefault("investigations", {})
            # Enhance holder_distribution with tx data
            if "holder_distribution" in result.get("investigations", {}):
                result["investigations"]["holder_distribution"] += f"\n\nTransaction Pattern:\n{tx_summary}"
            result["sources"].append("etherscan_tx")
        time.sleep(ETHERSCAN_DELAY)
    else:
        print("  [2/3] Skipping Etherscan (no ETHERSCAN_API_KEY set)", file=sys.stderr)

    # 3. CoinGecko market data
    print("  [3/3] CoinGecko market data...", file=sys.stderr)
    cg = fetch_coingecko_data(chain_id, address)
    if cg:
        market = cg.get("market_data", {})
        price = market.get("current_price", {}).get("usd", "?")
        mcap = market.get("market_cap", {}).get("usd", "?")
        vol24 = market.get("total_volume", {}).get("usd", "?")
        ath = market.get("ath", {}).get("usd", "?")
        ath_change = market.get("ath_change_percentage", {}).get("usd", "?")

        price_text = result.get("investigations", {}).get("price_history", "")
        cg_text = f"\n\nMarket Data (CoinGecko):\n"
        cg_text += f"- Current price: ${price}\n"
        cg_text += f"- Market cap: ${mcap:,.0f}\n" if isinstance(mcap, (int, float)) else f"- Market cap: {mcap}\n"
        cg_text += f"- 24h volume: ${vol24:,.0f}\n" if isinstance(vol24, (int, float)) else f"- 24h volume: {vol24}\n"
        cg_text += f"- ATH: ${ath}\n"
        cg_text += f"- From ATH: {ath_change}%\n" if ath_change != "?" else ""

        result.setdefault("investigations", {})
        result["investigations"]["price_history"] = price_text + cg_text
        result["sources"].append("coingecko")
    time.sleep(COINGECKO_DELAY)

    # If no source code from Etherscan, use GoPlus summary as token_data
    if "token_data" not in result:
        result["token_data"] = json.dumps({
            k: gp.get(k) for k in [
                "token_name", "token_symbol", "holder_count", "total_supply",
                "is_open_source", "is_proxy", "is_mintable", "is_honeypot",
                "buy_tax", "sell_tax", "cannot_buy", "cannot_sell_all",
                "owner_address", "creator_address", "is_blacklisted",
            ] if gp.get(k) is not None
        }, indent=2) if gp else f"No data available for {name}"

    # Set difficulty based on risk clarity
    risk_score = result.get("goplus_risk_score", 0)
    if label == "safe":
        result["difficulty"] = "easy" if risk_score == 0 else "medium" if risk_score <= 1 else "hard"
    else:
        result["difficulty"] = "easy" if risk_score >= 3 else "medium" if risk_score >= 1 else "hard"

    print(f"  Done: {len(result.get('sources', []))} sources, "
          f"risk_score={risk_score}, difficulty={result['difficulty']}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch on-chain token data for RugGuard")
    parser.add_argument("--chain", type=int, default=1, help="Chain ID (1=ETH, 56=BSC)")
    parser.add_argument("--addresses", type=str, default="",
                        help="Comma-separated contract addresses (overrides built-in list)")
    parser.add_argument("--output", type=str, default="fetched_tokens.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    chain_id = args.chain

    if args.addresses:
        tokens = [
            {"address": a.strip(), "label": "unknown", "name": f"Token_{i}"}
            for i, a in enumerate(args.addresses.split(","))
        ]
    else:
        tokens = KNOWN_TOKENS.get(chain_id, [])

    if not tokens:
        print(f"No tokens configured for chain {chain_id}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching {len(tokens)} tokens on chain {chain_id} "
          f"({CHAIN_CONFIGS.get(chain_id, {}).get('name', '?')})", file=sys.stderr)

    results = []
    for token in tokens:
        data = fetch_token(
            chain_id=chain_id,
            address=token["address"],
            name=token["name"],
            label=token["label"],
        )
        # Remove raw GoPlus data from output (too verbose for dataset)
        data.pop("goplus_raw", None)
        results.append(data)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"chain_id": chain_id, "fetched": len(results), "tokens": results},
                  f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Saved {len(results)} tokens to {output_path}", file=sys.stderr)
    print(f"Sources used: GoPlus{' + Etherscan' if ETHERSCAN_API_KEY else ''} + CoinGecko", file=sys.stderr)

    # Print summary table
    print(f"\n{'Token':<20} {'Label':<14} {'Risk':<5} {'Diff':<8} {'Sources'}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    for t in results:
        print(f"{t['token_name']:<20} {t['label']:<14} {t.get('goplus_risk_score','?'):<5} "
              f"{t.get('difficulty','?'):<8} {','.join(t.get('sources',[]))}", file=sys.stderr)


if __name__ == "__main__":
    main()
