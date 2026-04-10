"""
Generate realistic RugGuard datasets using real rug pull addresses and patterns.

Sources:
- dianxiang-sun/rug_pull_dataset (2,391 validated rug pulls from ETH/BSC)
- smartbugs-curated (Solidity vulnerability patterns)
- Real-world DeFi token patterns

Outputs:
  rugguard_env/data/contracts.json    — Solidity source code analysis
  rugguard_env/data/transactions.json — On-chain transaction patterns
  rugguard_env/data/liquidity.json    — Liquidity pool metrics
"""

import csv
import json
import random
import hashlib
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Load real rug pull addresses/types
# ---------------------------------------------------------------------------
REAL_RUGS = []
rug_csv = os.path.join(
    os.environ.get("TEMP", "/tmp"),
    "rug_pull_dataset", "rugpull_full_dataset_new.csv"
)
if os.path.exists(rug_csv):
    with open(rug_csv) as f:
        for row in csv.DictReader(f):
            REAL_RUGS.append({
                "address": row["address"].strip(),
                "chain": row["Chain"].strip(),
                "losses": row["Losses"].strip(),
                "type": row["Type"].strip(),
                "root_cause": row["Root Causes"].strip(),
            })

# ---------------------------------------------------------------------------
# Realistic token name pools (NOT obvious scam names)
# ---------------------------------------------------------------------------
LEGIT_SOUNDING_NAMES = [
    "AuraFinance", "VelodromeV3", "SynapseProtocol", "LayerBridge", "NexusYield",
    "EtherVault", "PulseDAO", "NovaSwap", "ZenithToken", "ArcticDEX",
    "CosmicLend", "PrismStake", "QuantumSwap", "StellarBridge", "OmegaFi",
    "AtlasProtocol", "MeridianDAO", "OrbitLend", "SpectrumDEX", "VortexYield",
    "CatalystFi", "EclipseSwap", "HorizonDAO", "InfinityVault", "KineticDEX",
    "LunarStake", "MantleFi", "NebulaBridge", "OptimaSwap", "PegasusYield",
    "RadiantLend", "SolaceDAO", "TerraNova", "UltraSwap", "VertexFi",
    "WaveDAO", "XenithProtocol", "ZephyrLend", "AetherBridge", "BlazeDEX",
    "CelestisFi", "DawnProtocol", "EmberSwap", "FluxDAO", "GravityLend",
    "HaloFi", "IonSwap", "JadeProtocol", "KryptonDAO", "LithiumFi",
    "MagnetSwap", "NimbusFi", "OasisLend", "PhoenixDAO", "QuasarDEX",
    "ResonanceFi", "SummitSwap", "TitanBridge", "UnityDAO", "ValorFi",
    "WindsorProtocol", "YieldNest", "ZionBridge", "AlphaVault", "BetaSwap",
    "CoreDAO", "DeltaLend", "EpochFi", "FrostSwap", "GammaProtocol",
    "HelixDAO", "IndexVault", "JunctionFi", "KaizenSwap", "LatticeDAO",
    "MatrixLend", "NodeFi", "OracleSwap", "PivotDAO", "QuestFi",
    "RelayBridge", "ShardProtocol", "TridentSwap", "UmbraDAO", "VaultFi",
    "WarpBridge", "XenonSwap", "YugenDAO", "ZealotFi", "AnchorLend",
    "BreezeDEX", "CircuitFi", "DriftSwap", "EquinoxDAO", "ForgeProtocol",
    "GlacierFi", "HarborSwap", "IgniteLend", "JetDAO",
]

# Realistic DeFi project naming patterns
SAFE_PREFIXES = [
    "Aave", "Compound", "Maker", "Uniswap", "Curve", "Lido", "Rocket",
    "Convex", "Balancer", "Yearn", "Sushi", "Pancake", "dYdX", "GMX",
]

def rand_addr():
    return "0x" + hashlib.sha256(str(random.random()).encode()).hexdigest()[:40]

def rand_pair():
    return random.choice(["WETH", "WBNB", "USDC", "USDT", "BUSD", "DAI"])

def rand_dex():
    return random.choice([
        "Uniswap V2", "Uniswap V3", "PancakeSwap V2", "SushiSwap",
        "Camelot", "Trader Joe", "Raydium", "Orca",
    ])

def rand_chain():
    return random.choice(["ETH Mainnet", "BSC", "Arbitrum", "Polygon", "Base"])

def rand_loss():
    return random.randint(5000, 15000000)

# ---------------------------------------------------------------------------
# CONTRACT dataset generators — realistic Solidity with subtle patterns
# ---------------------------------------------------------------------------

def gen_contract_rug_pull(name, difficulty="medium"):
    """Generate contracts with hidden drain mechanisms."""
    templates = [
        # 1: Hidden in legitimate-looking migration function
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract {name} is ERC20, Ownable, ReentrancyGuard {{
    uint256 public constant MAX_SUPPLY = 100_000_000e18;
    uint256 public taxRate = 300; // 3%
    address public taxWallet;
    mapping(address => bool) public isExcludedFromTax;
    bool public tradingEnabled;

    constructor() ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        taxWallet = msg.sender;
        _mint(msg.sender, MAX_SUPPLY);
        isExcludedFromTax[msg.sender] = true;
    }}

    function enableTrading() external onlyOwner {{
        tradingEnabled = true;
    }}

    function setTaxRate(uint256 _rate) external onlyOwner {{
        require(_rate <= 500, "Tax too high");
        taxRate = _rate;
    }}

    function _update(address from, address to, uint256 amount) internal override {{
        if (!isExcludedFromTax[from] && !isExcludedFromTax[to]) {{
            require(tradingEnabled, "Trading not enabled");
            uint256 tax = (amount * taxRate) / 10000;
            super._update(from, taxWallet, tax);
            amount -= tax;
        }}
        super._update(from, to, amount);
    }}

    /// @notice Migrate liquidity to V2 contract
    function migrateV2(address newContract) external onlyOwner {{
        uint256 balance = address(this).balance;
        (bool sent, ) = newContract.call{{value: balance}}("");
        require(sent, "Migration failed");
    }}

    /// @notice Emergency token recovery for stuck tokens
    function recoverTokens(address token, uint256 amount) external onlyOwner {{
        IERC20(token).transfer(owner(), amount);
    }}

    receive() external payable {{}}
}}""",
        # 2: Backdoor via selfdestruct in helper contract
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IRouter {{
    function swapExactTokensForETH(uint256, uint256, address[] calldata, address, uint256) external;
    function WETH() external view returns (address);
}}

contract {name} is ERC20, Ownable {{
    IRouter public router;
    address public pair;
    uint256 public maxWalletSize = 2_000_000e18;
    mapping(address => bool) private _isExcluded;
    bool private _swapping;

    constructor(address _router) ERC20("{name}", "{name[:3].upper()}T") Ownable(msg.sender) {{
        router = IRouter(_router);
        _mint(msg.sender, 100_000_000e18);
        _isExcluded[msg.sender] = true;
        _isExcluded[address(this)] = true;
    }}

    function setPair(address _pair) external onlyOwner {{
        pair = _pair;
    }}

    function setMaxWallet(uint256 _max) external onlyOwner {{
        require(_max >= 1_000_000e18, "Too low");
        maxWalletSize = _max;
    }}

    function _update(address from, address to, uint256 amount) internal override {{
        if (!_isExcluded[to] && to != pair) {{
            require(balanceOf(to) + amount <= maxWalletSize, "Max wallet");
        }}
        super._update(from, to, amount);
    }}

    function clearStuckBalance() external onlyOwner {{
        payable(owner()).transfer(address(this).balance);
    }}

    function clearStuckTokens(address token) external onlyOwner {{
        uint256 bal = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner(), bal);
    }}

    receive() external payable {{}}
}}""",
        # 3: Timelock that owner can bypass
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract {name} is ERC20, Pausable, Ownable {{
    uint256 public constant INITIAL_SUPPLY = 50_000_000e18;
    address public treasuryWallet;
    uint256 public withdrawalTimelock = 48 hours;
    mapping(address => uint256) public lastWithdrawal;
    uint256 public totalStaked;
    mapping(address => uint256) public stakedBalance;

    constructor(address _treasury) ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        treasuryWallet = _treasury;
        _mint(address(this), INITIAL_SUPPLY);
    }}

    function stake(uint256 amount) external whenNotPaused {{
        _transfer(msg.sender, address(this), amount);
        stakedBalance[msg.sender] += amount;
        totalStaked += amount;
    }}

    function unstake(uint256 amount) external whenNotPaused {{
        require(stakedBalance[msg.sender] >= amount, "Insufficient stake");
        require(block.timestamp >= lastWithdrawal[msg.sender] + withdrawalTimelock, "Timelock");
        stakedBalance[msg.sender] -= amount;
        totalStaked -= amount;
        lastWithdrawal[msg.sender] = block.timestamp;
        _transfer(address(this), msg.sender, amount);
    }}

    function setWithdrawalTimelock(uint256 _timelock) external onlyOwner {{
        withdrawalTimelock = _timelock;
    }}

    /// @dev Admin function to rebalance treasury allocation
    function rebalanceTreasury(uint256 amount) external onlyOwner {{
        _transfer(address(this), treasuryWallet, amount);
    }}

    function pause() external onlyOwner {{ _pause(); }}
    function unpause() external onlyOwner {{ _unpause(); }}

    receive() external payable {{}}

    function withdrawETH() external onlyOwner {{
        payable(treasuryWallet).transfer(address(this).balance);
    }}
}}""",
    ]
    return random.choice(templates)


def gen_contract_honeypot(name, difficulty="medium"):
    """Generate contracts with hidden sell restrictions."""
    templates = [
        # 1: Hidden fee manipulation — sell tax can be set to 100%
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract {name} is ERC20, Ownable {{
    uint256 public buyTax = 500;  // 5%
    uint256 public sellTax = 500; // 5%
    address public marketingWallet;
    address public pair;
    mapping(address => bool) public isExcluded;
    bool public limitsInEffect = true;
    uint256 public maxTxAmount;

    constructor() ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        marketingWallet = msg.sender;
        _mint(msg.sender, 1_000_000_000e18);
        maxTxAmount = totalSupply() / 50; // 2%
        isExcluded[msg.sender] = true;
        isExcluded[address(this)] = true;
    }}

    function setPair(address _pair) external onlyOwner {{
        pair = _pair;
    }}

    function removeLimits() external onlyOwner {{
        limitsInEffect = false;
    }}

    function updateFees(uint256 _buyTax, uint256 _sellTax) external onlyOwner {{
        buyTax = _buyTax;
        sellTax = _sellTax;
    }}

    function _update(address from, address to, uint256 amount) internal override {{
        if (limitsInEffect && !isExcluded[from] && !isExcluded[to]) {{
            require(amount <= maxTxAmount, "Max tx");
        }}

        uint256 fee = 0;
        if (to == pair && !isExcluded[from]) {{
            fee = (amount * sellTax) / 10000;
        }} else if (from == pair && !isExcluded[to]) {{
            fee = (amount * buyTax) / 10000;
        }}

        if (fee > 0) {{
            super._update(from, address(this), fee);
            amount -= fee;
        }}
        super._update(from, to, amount);
    }}

    function swapAndSend() external onlyOwner {{
        uint256 bal = balanceOf(address(this));
        _approve(address(this), address(pair), bal);
        payable(marketingWallet).transfer(address(this).balance);
    }}

    receive() external payable {{}}
}}""",
        # 2: Hidden blacklist via approval mechanism
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract {name} is ERC20, Ownable {{
    mapping(address => bool) private _authorized;
    mapping(address => uint256) private _holdTimestamp;
    address public pair;
    uint256 public minHoldDuration = 0;
    bool public antiSnipeEnabled = true;
    uint256 public launchBlock;

    constructor() ERC20("{name}", "{name[:3].upper()}") Ownable(msg.sender) {{
        _mint(msg.sender, 500_000_000e18);
        _authorized[msg.sender] = true;
    }}

    function openTrading(address _pair) external onlyOwner {{
        pair = _pair;
        launchBlock = block.number;
    }}

    function setAntiSnipe(bool _enabled) external onlyOwner {{
        antiSnipeEnabled = _enabled;
    }}

    function setMinHoldDuration(uint256 _duration) external onlyOwner {{
        minHoldDuration = _duration;
    }}

    function authorize(address[] calldata addrs, bool status) external onlyOwner {{
        for (uint i = 0; i < addrs.length; i++) {{
            _authorized[addrs[i]] = status;
        }}
    }}

    function _update(address from, address to, uint256 amount) internal override {{
        if (from == pair) {{
            _holdTimestamp[to] = block.timestamp;
        }}
        if (to == pair && !_authorized[from]) {{
            if (antiSnipeEnabled) {{
                require(
                    block.timestamp >= _holdTimestamp[from] + minHoldDuration,
                    "Hold period not met"
                );
            }}
        }}
        super._update(from, to, amount);
    }}
}}""",
        # 3: Transfer cooldown that gets extended indefinitely
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract {name} is ERC20, Ownable, ReentrancyGuard {{
    uint256 public constant MAX_SUPPLY = 200_000_000e18;
    address public pair;
    mapping(address => uint256) public lastTransfer;
    uint256 public cooldownPeriod = 30; // seconds
    mapping(address => bool) public isWhitelisted;
    uint256 public maxWalletPercent = 200; // 2%

    constructor() ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        _mint(msg.sender, MAX_SUPPLY);
        isWhitelisted[msg.sender] = true;
    }}

    function setPair(address _pair) external onlyOwner {{
        pair = _pair;
        isWhitelisted[_pair] = true;
    }}

    function setCooldown(uint256 _seconds) external onlyOwner {{
        cooldownPeriod = _seconds;
    }}

    function setMaxWalletPercent(uint256 _percent) external onlyOwner {{
        require(_percent >= 100, "Min 1%");
        maxWalletPercent = _percent;
    }}

    function whitelist(address[] calldata addrs, bool status) external onlyOwner {{
        for (uint i = 0; i < addrs.length; i++) {{
            isWhitelisted[addrs[i]] = status;
        }}
    }}

    function _update(address from, address to, uint256 amount) internal override {{
        if (!isWhitelisted[from] && !isWhitelisted[to]) {{
            require(
                block.timestamp >= lastTransfer[from] + cooldownPeriod,
                "Cooldown active"
            );
            if (to != pair) {{
                require(
                    balanceOf(to) + amount <= (totalSupply() * maxWalletPercent) / 10000,
                    "Exceeds max wallet"
                );
            }}
        }}
        lastTransfer[from] = block.timestamp;
        super._update(from, to, amount);
    }}
}}""",
    ]
    return random.choice(templates)


def gen_contract_wash_trading(name, difficulty="medium"):
    """Generate contracts with wash trading infrastructure."""
    templates = [
        # 1: Hidden batch transfer for volume inflation
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract {name} is ERC20, AccessControl {{
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000e18;
    address public treasury;
    uint256 public rebaseIndex = 1e18;

    constructor(address _treasury) ERC20("{name}", "{name[:4].upper()}") {{
        treasury = _treasury;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        _mint(_treasury, INITIAL_SUPPLY);
    }}

    /// @notice Distribute rewards to multiple holders
    function batchDistribute(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyRole(OPERATOR_ROLE) {{
        require(recipients.length == amounts.length, "Length mismatch");
        for (uint i = 0; i < recipients.length; i++) {{
            _transfer(treasury, recipients[i], amounts[i]);
        }}
    }}

    /// @notice Collect tokens back to treasury for next distribution cycle
    function batchCollect(
        address[] calldata holders,
        uint256[] calldata amounts
    ) external onlyRole(OPERATOR_ROLE) {{
        for (uint i = 0; i < holders.length; i++) {{
            _transfer(holders[i], treasury, amounts[i]);
        }}
    }}

    function setRebaseIndex(uint256 _index) external onlyRole(OPERATOR_ROLE) {{
        rebaseIndex = _index;
    }}

    function balanceOf(address account) public view override returns (uint256) {{
        return (super.balanceOf(account) * rebaseIndex) / 1e18;
    }}
}}""",
        # 2: Looks like a normal DEX aggregator but routes through self
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface ISwapRouter {{
    function swapExactTokensForTokens(
        uint256 amountIn, uint256 amountOutMin,
        address[] calldata path, address to, uint256 deadline
    ) external returns (uint256[] memory);
}}

contract {name} is ERC20, Ownable {{
    uint256 public constant SUPPLY = 500_000_000e18;
    address[] public marketMakers;
    mapping(address => bool) public isMarketMaker;
    uint256 public dailyVolumeTarget;
    address public router;

    constructor(address _router) ERC20("{name}", "{name[:3].upper()}") Ownable(msg.sender) {{
        router = _router;
        _mint(msg.sender, SUPPLY);
    }}

    function addMarketMaker(address mm) external onlyOwner {{
        marketMakers.push(mm);
        isMarketMaker[mm] = true;
        _approve(mm, router, type(uint256).max);
    }}

    function removeMarketMaker(uint256 idx) external onlyOwner {{
        isMarketMaker[marketMakers[idx]] = false;
        marketMakers[idx] = marketMakers[marketMakers.length - 1];
        marketMakers.pop();
    }}

    function setDailyTarget(uint256 _target) external onlyOwner {{
        dailyVolumeTarget = _target;
    }}

    /// @notice Rebalance market maker allocations for optimal spread
    function rebalanceAllocations(
        uint256[] calldata fromIdx,
        uint256[] calldata toIdx,
        uint256[] calldata amounts
    ) external onlyOwner {{
        for (uint i = 0; i < fromIdx.length; i++) {{
            _transfer(marketMakers[fromIdx[i]], marketMakers[toIdx[i]], amounts[i]);
        }}
    }}

    function fundMarketMakers(uint256[] calldata amounts) external onlyOwner {{
        for (uint i = 0; i < marketMakers.length && i < amounts.length; i++) {{
            _transfer(msg.sender, marketMakers[i], amounts[i]);
        }}
    }}
}}""",
    ]
    return random.choice(templates)


def gen_contract_safe(name, difficulty="medium"):
    """Generate legitimate-looking contracts that are actually safe."""
    templates = [
        # 1: Standard ERC20 with vesting — looks like it could be a scam but isn't
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract {name} is ERC20, Ownable {{
    uint256 public constant MAX_SUPPLY = 100_000_000e18;
    uint256 public immutable vestingStart;
    uint256 public constant VESTING_DURATION = 365 days;
    uint256 public teamAllocation;
    uint256 public teamClaimed;

    constructor() ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        vestingStart = block.timestamp;
        teamAllocation = MAX_SUPPLY * 15 / 100; // 15% team
        _mint(address(this), teamAllocation);
        _mint(msg.sender, MAX_SUPPLY - teamAllocation); // 85% to LP/community
    }}

    function claimVested() external onlyOwner {{
        uint256 elapsed = block.timestamp - vestingStart;
        if (elapsed > VESTING_DURATION) elapsed = VESTING_DURATION;
        uint256 totalVested = (teamAllocation * elapsed) / VESTING_DURATION;
        uint256 claimable = totalVested - teamClaimed;
        require(claimable > 0, "Nothing to claim");
        teamClaimed += claimable;
        _transfer(address(this), owner(), claimable);
    }}

    function renounceOwnership() public override onlyOwner {{
        super.renounceOwnership();
    }}
}}""",
        # 2: Governance token with proper access control
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract {name} is ERC20Votes, AccessControl {{
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    uint256 public immutable maxSupply;
    uint256 public immutable mintCap; // max per tx

    constructor(uint256 _maxSupply)
        ERC20("{name}", "{name[:4].upper()}")
        ERC20Permit("{name}")
    {{
        maxSupply = _maxSupply;
        mintCap = _maxSupply / 100;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _mint(msg.sender, _maxSupply / 5); // 20% initial
    }}

    function mint(address to, uint256 amount) external onlyRole(MINTER_ROLE) {{
        require(amount <= mintCap, "Exceeds mint cap");
        require(totalSupply() + amount <= maxSupply, "Exceeds max supply");
        _mint(to, amount);
    }}
}}""",
        # 3: Staking contract with proper mechanics
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract {name} is ERC20, ERC20Burnable, ReentrancyGuard, Ownable {{
    uint256 public constant MAX_SUPPLY = 50_000_000e18;
    uint256 public rewardRate = 100; // 1% per epoch
    mapping(address => uint256) public staked;
    mapping(address => uint256) public stakeTimestamp;
    uint256 public totalStaked;
    bool public stakingEnabled = true;

    constructor() ERC20("{name}", "{name[:4].upper()}") Ownable(msg.sender) {{
        _mint(msg.sender, MAX_SUPPLY);
    }}

    function stake(uint256 amount) external nonReentrant {{
        require(stakingEnabled, "Staking paused");
        _claimReward(msg.sender);
        _transfer(msg.sender, address(this), amount);
        staked[msg.sender] += amount;
        totalStaked += amount;
        stakeTimestamp[msg.sender] = block.timestamp;
    }}

    function unstake(uint256 amount) external nonReentrant {{
        require(staked[msg.sender] >= amount, "Insufficient");
        _claimReward(msg.sender);
        staked[msg.sender] -= amount;
        totalStaked -= amount;
        _transfer(address(this), msg.sender, amount);
    }}

    function _claimReward(address user) internal {{
        if (staked[user] > 0 && stakeTimestamp[user] > 0) {{
            uint256 elapsed = block.timestamp - stakeTimestamp[user];
            uint256 reward = (staked[user] * rewardRate * elapsed) / (10000 * 365 days);
            if (reward > 0 && totalSupply() + reward <= MAX_SUPPLY) {{
                _mint(user, reward);
            }}
        }}
        stakeTimestamp[user] = block.timestamp;
    }}

    function setRewardRate(uint256 _rate) external onlyOwner {{
        require(_rate <= 1000, "Max 10%");
        rewardRate = _rate;
    }}

    function toggleStaking() external onlyOwner {{
        stakingEnabled = !stakingEnabled;
    }}
}}""",
        # 4: Simple immutable token — no admin functions at all
        f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract {name} is ERC20 {{
    constructor(address treasury, address community, address lp) ERC20("{name}", "{name[:4].upper()}") {{
        _mint(treasury, 30_000_000e18);   // 30% treasury (multisig)
        _mint(community, 50_000_000e18);  // 50% community airdrop
        _mint(lp, 20_000_000e18);         // 20% initial liquidity
    }}
}}""",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# TRANSACTION dataset generators
# ---------------------------------------------------------------------------

def gen_tx_rug_pull(name, addr):
    chain = rand_chain()
    deployer = rand_addr()
    loss = rand_loss()
    templates = [
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Transaction Pattern Analysis (Last 30 Days):
- Deployer ({deployer[:10]}...{deployer[-4:]}) holds {random.randint(60,95)}% of supply in single wallet
- Initial liquidity added: ${loss:,} via {rand_dex()}
- {random.randint(800,5000)} buy transactions from {random.randint(200,2000)} unique wallets in first {random.randint(24,96)}h
- Deployer executed {random.randint(2,8)} large sell transactions in {random.randint(5,30)} minute window
- Post-sell token price decline: {random.randint(85,99)}%
- Deployer wallet funded via Tornado Cash / cross-chain bridge {random.randint(2,14)} days before deployment
- Team wallet ({rand_addr()[:10]}...) transferred {random.randint(70,100)}% of holdings to CEX hot wallet
- No token lock contract deployed despite claims in documentation
- Social media accounts (Twitter, Telegram) deleted within {random.randint(1,12)} hours of sell event
- Current 24h volume: ${random.randint(0,500)} (effectively zero)""",
        f"""Token: {name} | Network: {chain} | Contract: {addr}
On-Chain Activity Summary:
- Token age: {random.randint(3,45)} days | Holders: {random.randint(100,3000)} (peak: {random.randint(2000,15000)})
- Buy/sell ratio: {random.uniform(15,50):.1f}:1 (extreme buy pressure → sudden reversal)
- Developer wallet activity: {random.randint(3,12)} transfers to {random.randint(2,5)} intermediary wallets
- Intermediary wallets converge to single CEX deposit address
- Liquidity removal: {random.randint(80,100)}% of LP tokens removed in block #{random.randint(18000000,20000000)}
- Time between last marketing push and LP removal: {random.randint(2,48)} hours
- Smart contract verified on Etherscan: {"Yes" if random.random() > 0.3 else "No"}
- Ownership renounced: {"No" if random.random() > 0.2 else "Yes (but via proxy with hidden admin)"}
- Connected wallets (graph analysis): deployer linked to {random.randint(2,8)} previously flagged contracts""",
    ]
    return random.choice(templates)


def gen_tx_honeypot(name, addr):
    chain = rand_chain()
    templates = [
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Transaction Pattern Analysis (Last 14 Days):
- Total buy transactions: {random.randint(1000,8000)}
- Total sell transactions: {random.randint(3,20)} (all from {random.randint(1,3)} whitelisted addresses)
- Failed sell attempts: {random.randint(500,5000)} ({random.uniform(95,100):.1f}% failure rate for non-whitelisted)
- Average buy size: ${random.randint(50,500)}
- Whitelisted wallet sells: ${random.randint(100000,2000000):,} total across {random.randint(3,15)} transactions
- Error messages in failed sells: "Transfer failed", "Insufficient output", "TRANSFER_FROM_FAILED"
- Gas wasted on failed sell attempts: ~${random.randint(5000,80000):,}
- Token price appears stable/rising (no organic sell pressure)
- Buy tax observed: {random.randint(3,8)}% | Sell tax effective: {random.randint(90,100)}% (hidden)
- Contract has modifiable fee function with no upper bound check
- Deployer last activity: {random.randint(1,6)} hours ago (still active)""",
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Behavioral Analysis:
- Unique buyers: {random.randint(500,5000)} | Unique sellers: {random.randint(1,5)}
- Median hold time: {random.randint(3,30)} days (forced — cannot sell)
- Transfer success rate (non-deployer → DEX): {random.uniform(0,5):.1f}%
- Transfer success rate (non-deployer → wallet): {random.uniform(60,100):.0f}% (can transfer, can't swap)
- Router approval events: {random.randint(800,5000)} (users trying to sell)
- Actual swap executions: {random.randint(2,10)} (only whitelisted)
- Price movement: +{random.randint(50,500)}% since launch (buy-only pressure)
- Liquidity pool depth: ${random.randint(100000,3000000):,} (looks healthy)
- Contract uses dynamic fee structure: _sellTax variable changed {random.randint(3,15)} times
- Current _sellTax value: {random.randint(9000,9999)} basis points ({random.randint(90,99)}%)""",
    ]
    return random.choice(templates)


def gen_tx_wash_trading(name, addr):
    chain = rand_chain()
    unique_wallets = random.randint(5,25)
    real_volume = random.randint(500, 5000)
    fake_volume = real_volume * random.randint(500, 20000)
    templates = [
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Volume Analysis (24h):
- Reported volume: ${fake_volume:,}
- Unique trading wallets: {unique_wallets}
- Average trades per wallet: {random.randint(50,500)}
- Wallet cluster analysis: {unique_wallets - random.randint(0,3)} of {unique_wallets} wallets funded from same source
- Circular flow detected: A→B→C→...→A pattern, {random.randint(100,2000)} cycles in 24h
- Average trade size: ${fake_volume // (unique_wallets * random.randint(50,200)):,}
- Price impact per trade: {random.uniform(0.0001,0.01):.4f}% (impossibly low for market cap)
- Market cap: ${random.randint(200000,5000000):,}
- Volume/Market cap ratio: {fake_volume / random.randint(500000,2000000):.1f}x (healthy range: 0.01-0.5x)
- Organic volume estimate: <${real_volume:,}
- All {unique_wallets} wallets created within {random.randint(1,48)} hour window
- Wallet funding pattern: identical amounts from Binance/OKX hot wallet""",
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Trading Pattern Anomalies:
- 24h transactions: {random.randint(2000,50000)} | Unique participants: {unique_wallets}
- Trade timing: {random.uniform(85,99):.0f}% of trades occur in {random.randint(2,5)}-second intervals (bot signature)
- Bid-ask spread: {random.uniform(0.001,0.01):.3f}% (artificially tight)
- Order book depth vs actual fills: {random.randint(50,200)}x discrepancy
- Top {unique_wallets} wallets account for {random.uniform(95,99.9):.1f}% of all volume
- Net token flow among top wallets: ~0 (circular, zero-sum)
- CMC/CoinGecko reported volume: ${fake_volume:,}
- On-chain verifiable volume: ${real_volume:,}
- Discrepancy factor: {fake_volume // max(real_volume, 1):,}x
- None of the top trading wallets have any other token holdings
- Gas spending pattern: identical gas price for all trades from cluster""",
    ]
    return random.choice(templates)


def gen_tx_safe(name, addr):
    chain = rand_chain()
    holders = random.randint(5000, 200000)
    templates = [
        f"""Token: {name} | Network: {chain} | Contract: {addr}
Transaction Summary (30 Days):
- Total holders: {holders:,}
- Buy/sell ratio: {random.uniform(0.8,1.4):.2f} (balanced two-way market)
- Daily active traders: {random.randint(200, 5000)}
- Largest wallet: {random.uniform(1.5,4.0):.1f}% of supply (team vesting contract, time-locked)
- Top 10 wallets: {random.uniform(15,35):.1f}% of supply (includes DEX pairs and known protocols)
- Average daily volume: ${random.randint(100000,5000000):,}
- Volume consistent over {random.randint(30,180)} days (no anomalous spikes)
- Deployer wallet: {random.uniform(0,2):.1f}% of current supply
- Ownership: {"Renounced" if random.random() > 0.3 else "Multisig (3/5 signers)"}
- Contract verified on Etherscan: Yes
- Audit: {random.choice(["Certik", "Trail of Bits", "OpenZeppelin", "Hacken", "PeckShield"])} (score: {random.randint(85,98)}/100)
- No wallet clustering anomalies detected
- Organic social activity: {random.randint(5000,50000)} Twitter followers, active Discord""",
        f"""Token: {name} | Network: {chain} | Contract: {addr}
On-Chain Health Metrics:
- Token age: {random.randint(90,730)} days
- Holder growth: +{random.uniform(1,8):.1f}% per month (organic)
- Gini coefficient of holdings: {random.uniform(0.4,0.65):.2f} (moderately distributed)
- Whale concentration (>1% holders): {random.randint(3,8)} addresses
- Smart money flow (Nansen labels): {random.randint(5,30)} labeled fund wallets hold positions
- DEX pairs: {random.randint(2,6)} active ({rand_dex()}, {rand_dex()})
- CEX listings: {random.randint(1,5)} ({"Binance, " if random.random()>0.5 else ""}{"Coinbase, " if random.random()>0.5 else ""}KuCoin)
- Unique daily transactions: {random.randint(500,10000)}
- Failed transaction rate: {random.uniform(0.5,3):.1f}% (normal range)
- No blacklist function in contract
- Token utility: {random.choice(["Governance + fee sharing", "Gas token for L2", "Staking rewards", "Protocol revenue sharing"])}""",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# LIQUIDITY dataset generators
# ---------------------------------------------------------------------------

def gen_liq_rug_pull(name, addr):
    dex = rand_dex()
    pair_token = rand_pair()
    init_liq = random.randint(100000, 10000000)
    templates = [
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Liquidity Pool Analysis:
- Initial LP: ${init_liq:,} (100% provided by deployer)
- LP token distribution: 1 address holds {random.uniform(95,100):.1f}% of LP tokens
- LP lock status: {"Not locked" if random.random() > 0.3 else f"Locked via {random.choice(['Unicrypt', 'Team Finance', 'PinkSale'])} — but lock contract has admin override"}
- Current pool TVL: ${random.randint(100, 5000)} (was ${init_liq:,})
- LP removal events: {random.randint(1,5)} transactions removing {random.uniform(90,100):.1f}% of liquidity
- Time from listing to LP removal: {random.randint(24,720)} hours
- Price impact of LP removal: -{random.uniform(95,99.9):.1f}%
- Remaining pool depth: ${random.randint(50,2000)} (dust)
- Impermanent loss for non-deployer LPs: {random.uniform(90,100):.0f}%
- Deployer received {random.randint(50,500)} {pair_token} from LP removal
- Post-removal: zero buy/sell activity""",
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
LP Risk Assessment:
- Pool created: {random.randint(3,60)} days ago
- Initial liquidity: ${init_liq:,}
- LP providers: {random.randint(1,3)} unique addresses (all linked to deployer)
- LP token lock: {random.choice(["None", "Self-custodied (not in lock contract)", "Locked but deployer is lock contract owner"])}
- Liquidity concentration: {random.uniform(95,100):.1f}% in single address
- Unusual pattern: deployer added and removed liquidity {random.randint(3,10)} times in {random.randint(7,30)} days
- Each removal slightly larger than previous addition (incremental drain)
- Cumulative extraction: ${int(init_liq * random.uniform(0.3,0.8)):,}
- Current TVL trend: declining {random.uniform(5,15):.0f}% per day
- No other DEX pairs exist for this token
- Token contract has approve(pair, MAX_UINT) called by deployer""",
    ]
    return random.choice(templates)


def gen_liq_honeypot(name, addr):
    dex = rand_dex()
    pair_token = rand_pair()
    tvl = random.randint(500000, 8000000)
    templates = [
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Liquidity Pool Analysis:
- Pool TVL: ${tvl:,} — appears healthy on surface
- LP locked: Yes, via {random.choice(["Unicrypt", "Team Finance", "PinkSale"])} for {random.randint(6,24)} months
- Buy slippage required: {random.randint(5,12)}% (slightly elevated but functional)
- Sell slippage required: >{random.randint(50,99)}% (transactions fail at normal slippage)
- Successful swap attempts (buy): {random.randint(2000,10000)}
- Successful swap attempts (sell): {random.randint(2,15)} (deployer wallets only)
- Failed swap attempts (sell): {random.randint(1000,8000)}
- Pool price: artificially stable — no sell pressure creates illusion of support
- Price movement: +{random.randint(20,300)}% since launch (only buys execute)
- LP lock is genuine but irrelevant — trapped users cannot exit regardless
- Router interaction analysis: sell transactions revert at token contract level, not router
- Fee structure: buy {random.randint(3,8)}% | sell {random.randint(80,99)}% (hidden, not shown in contract read functions)""",
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Pool Health Indicators:
- TVL: ${tvl:,} | Daily volume: ${random.randint(50000,500000):,}
- Volume composition: {random.uniform(95,100):.0f}% buys, {random.uniform(0,5):.0f}% sells
- Liquidity depth for $10K buy: {random.uniform(0.5,2):.1f}% price impact (good)
- Liquidity depth for $10K sell: FAILED — reverts with "TransferHelper: TRANSFER_FROM_FAILED"
- LP providers: {random.randint(1,5)} addresses
- Pool age: {random.randint(5,45)} days
- Price chart: monotonically increasing (suspicious — no natural corrections)
- Comparison: similar market cap tokens show ±{random.randint(5,20)}% daily swings
- Contract _maxSellAmount or equivalent: set to {random.randint(0,100)} tokens (effectively zero)
- Token approval does not guarantee swap execution — contract-level block""",
    ]
    return random.choice(templates)


def gen_liq_wash_trading(name, addr):
    dex = rand_dex()
    pair_token = rand_pair()
    real_tvl = random.randint(20000, 200000)
    fake_tvl = real_tvl * random.randint(50, 300)
    templates = [
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Liquidity & Volume Analysis:
- Reported TVL: ${fake_tvl:,}
- On-chain verified TVL (reserve0 * price + reserve1): ${real_tvl:,}
- TVL discrepancy: {fake_tvl // max(real_tvl, 1)}x — price oracle manipulation
- Pool reserves: {random.randint(10000,500000):,} {name[:4].upper()} + {random.uniform(10,500):.1f} {pair_token}
- Price oracle: uses spot price (single-block, easily manipulated)
- Reported 24h volume: ${random.randint(5000000,100000000):,}
- Volume/TVL ratio: {random.uniform(50,500):.0f}x (healthy: 0.1-2x)
- LP providers: {random.randint(2,5)} addresses (all same entity)
- Swap events: {random.randint(5000,50000)} in 24h from {random.randint(5,20)} wallets
- Net flow analysis: circular — tokens return to origin within {random.randint(2,10)} hops
- MEV bot activity: {random.randint(0,5)} bots, unusual absence of arbitrage
- Real organic users estimate: <{random.randint(10,50)}""",
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Pool Anomaly Report:
- Listed on: {random.choice(["CoinMarketCap", "CoinGecko", "DexScreener"])} (top {random.randint(100,500)} by volume)
- Reported metrics: Volume ${random.randint(10000000,500000000):,} | TVL ${fake_tvl:,}
- Actual on-chain reserves: ${real_tvl:,}
- How TVL is inflated: token price calculated from manipulated internal oracle
- Swap analysis: same {random.randint(5,15)} wallets executing {random.randint(100,1000)} trades/day
- Trade size distribution: uniform ${random.randint(1000,50000)}-${random.randint(50000,200000)} (no natural distribution)
- Real-world comparison: organic tokens show log-normal trade size distribution
- Slippage per trade: {random.uniform(0.001,0.01):.4f}% (impossibly low given real TVL)
- Gas analysis: all trades from cluster use priority fee {random.randint(1,3)} gwei (identical)
- No trades from MetaMask/Rabby signatures — all direct contract calls
- CoinGecko trust score: {random.randint(1,3)}/10""",
    ]
    return random.choice(templates)


def gen_liq_safe(name, addr):
    dex = rand_dex()
    pair_token = rand_pair()
    tvl = random.randint(1000000, 50000000)
    templates = [
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Liquidity Pool Analysis:
- Pool TVL: ${tvl:,}
- LP providers: {random.randint(50,500)} unique addresses
- Largest LP provider: {random.uniform(3,12):.1f}% of pool (institutional market maker)
- LP lock: {random.choice(["Community-distributed — no single entity can drain", f"Team LP ({random.randint(10,30)}%) locked via Team Finance for {random.randint(12,36)} months", "Protocol-owned liquidity via Olympus-style bonding"])}
- Pool age: {random.randint(90,730)} days
- Historical TVL: stable/growing over {random.randint(3,18)} months
- Price impact for $100K trade: {random.uniform(0.3,2):.1f}%
- Daily volume: ${random.randint(500000,10000000):,}
- Volume/TVL ratio: {random.uniform(0.05,0.5):.2f}x (healthy)
- Fee tier: {random.choice(["0.3%", "0.05%", "1%"])} | Accumulated fees: ${random.randint(100000,5000000):,}
- LP APY: {random.uniform(5,25):.1f}% (organic fee generation)
- Multiple DEX pairs exist ({random.randint(2,5)} pools across {random.randint(1,3)} chains)
- No admin functions that can affect pool operation""",
        f"""Token: {name}/{pair_token} Pool ({dex}) | Contract: {addr}
Pool Health Assessment:
- TVL: ${tvl:,} | Rank: #{random.randint(50,500)} on {dex}
- Liquidity distribution: {"Concentrated" if random.random()>0.5 else "Full-range"} ({random.randint(50,200)} active positions)
- Largest position: {random.uniform(2,8):.1f}% of TVL
- Pool utilization rate: {random.uniform(30,80):.0f}% (active trading range)
- Impermanent loss (30d): {random.uniform(-5,-0.5):.1f}% (normal range)
- Fee revenue (30d): ${random.randint(10000,500000):,}
- Buy/sell volume ratio: {random.uniform(0.8,1.3):.2f} (balanced)
- Unique traders (30d): {random.randint(500,10000)}
- MEV activity: {random.uniform(2,8):.0f}% of volume (normal for this TVL tier)
- Oracle: Chainlink price feed ({random.choice(["8", "18"])} decimals, {random.randint(10,60)}s heartbeat)
- Pool parameters: immutable (no admin functions)
- Audit: pool factory audited by {random.choice(["Trail of Bits", "OpenZeppelin", "Spearbit"])}""",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------------

def build_dataset(gen_funcs, count_per_label, use_real_addrs=True):
    """Build a dataset with balanced labels.

    gen_funcs: dict mapping label -> (generator_fn, vuln_type_or_None)
    """
    samples = []
    name_pool = list(LEGIT_SOUNDING_NAMES)
    random.shuffle(name_pool)
    name_idx = 0

    real_rug_addrs = [r["address"] for r in REAL_RUGS]
    random.shuffle(real_rug_addrs)
    addr_idx = 0

    for label, (gen_fn, vuln_type) in gen_funcs.items():
        for i in range(count_per_label):
            name = name_pool[name_idx % len(name_pool)]
            name_idx += 1

            if use_real_addrs and label != "safe" and addr_idx < len(real_rug_addrs):
                addr = real_rug_addrs[addr_idx]
                addr_idx += 1
            else:
                addr = rand_addr()

            if label == "safe":
                token_data = gen_fn(name, addr)
            else:
                token_data = gen_fn(name, addr)

            sample = {
                "token_name": name,
                "label": label,
                "vulnerability_type": vuln_type,
                "token_data": token_data,
            }
            samples.append(sample)

    random.shuffle(samples)
    return {"samples": samples}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "rugguard_env", "data")

    # --- Contracts dataset (30 per label = 120 total) ---
    contracts = build_dataset(
        {
            "rug_pull": (gen_contract_rug_pull, "rug_pull"),
            "honeypot": (gen_contract_honeypot, "honeypot"),
            "wash_trading": (gen_contract_wash_trading, "wash_trading"),
            "safe": (gen_contract_safe, None),
        },
        count_per_label=30,
        use_real_addrs=False,  # contracts don't use addresses in data
    )
    with open(os.path.join(out_dir, "contracts.json"), "w") as f:
        json.dump(contracts, f, indent=2)
    print(f"contracts.json: {len(contracts['samples'])} samples")

    # --- Transactions dataset (30 per label = 120 total) ---
    transactions = build_dataset(
        {
            "rug_pull": (gen_tx_rug_pull, "rug_pull"),
            "honeypot": (gen_tx_honeypot, "honeypot"),
            "wash_trading": (gen_tx_wash_trading, "wash_trading"),
            "safe": (gen_tx_safe, None),
        },
        count_per_label=30,
    )
    with open(os.path.join(out_dir, "transactions.json"), "w") as f:
        json.dump(transactions, f, indent=2)
    print(f"transactions.json: {len(transactions['samples'])} samples")

    # --- Liquidity dataset (30 per label = 120 total) ---
    liquidity = build_dataset(
        {
            "rug_pull": (gen_liq_rug_pull, "rug_pull"),
            "honeypot": (gen_liq_honeypot, "honeypot"),
            "wash_trading": (gen_liq_wash_trading, "wash_trading"),
            "safe": (gen_liq_safe, None),
        },
        count_per_label=30,
    )
    with open(os.path.join(out_dir, "liquidity.json"), "w") as f:
        json.dump(liquidity, f, indent=2)
    print(f"liquidity.json: {len(liquidity['samples'])} samples")

    # Print label distribution
    for fname, ds in [("contracts", contracts), ("transactions", transactions), ("liquidity", liquidity)]:
        labels = {}
        for s in ds["samples"]:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
        print(f"  {fname}: {labels}")


if __name__ == "__main__":
    main()
