# verifiablex402

> Powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas)

Transaction integrity analyzer for x402 payments on Base. It looks at a wallet's
USDC transaction history, runs it through a small neural network, and tells you
whether the activity looks like real commerce, wash trading, circular payments,
or something else. The classification comes with a zero-knowledge proof (via
JOLT Atlas) so anyone can verify the result without seeing the raw data.

## How it works — plain English

Imagine you run an online store that accepts crypto payments. A customer shows up
wanting to pay with USDC on Base. Before you accept the payment, you want to
know: is this wallet legit, or is it a bot farming fake volume?

That is the problem verifiablex402 solves. Here is exactly what happens when you
point it at a wallet address:

### Step 1: Pull the transaction history

The indexer connects to a Base mainnet RPC node and fetches every USDC Transfer
event log involving the wallet over a configurable window (default ~7 days /
302,400 blocks). It deduplicates the logs, then makes two additional passes over
the RPC:

- **Timestamp backfill** — for each unique block number in the results, it calls
  `eth_getBlockByNumber` to get the block's Unix timestamp. A HashMap cache
  avoids duplicate calls when multiple transactions share a block.
- **Gas backfill** — for each transaction, it calls `eth_getTransactionReceipt`
  to get `gasUsed` and `effectiveGasPrice`.

RPC calls use **exponential backoff retry** (3 attempts with 200ms/400ms/800ms
delays). HTTP 429 (rate limit) and 5xx server errors are treated as retryable.

These two backfill steps are critical. Without real timestamps, seven of the 24
features (average time between transactions, time regularity, burst score, night
ratio, weekend ratio, transactions per day, activity span) would all be zero.
Without real gas data, the gas efficiency feature would be zero. The model was
trained on non-zero values for all of these, so missing data degrades accuracy.

### Step 2: Extract 24 features

The feature extractor (`TransactionFeatures::extract`) computes 24 numbers from
the raw transaction list. These capture the shape of the wallet's behavior:

- **Volume metrics** — how many transactions, how many unique counterparties,
  how spread out the counterparty distribution is (Shannon entropy).
- **Value metrics** — average, standard deviation, min, max, range ratio. Are
  the amounts diverse or suspiciously identical?
- **Pattern metrics** — self-transfer ratio (sending to yourself), circular path
  score (A sends to B, B sends to C, C sends back to A), identical amount ratio.
- **Timing metrics** — average interval between transactions, regularity
  (coefficient of variation), burst score (peak rate vs. average rate), night
  ratio, weekend ratio, transactions per day, activity span in days.
- **Efficiency metrics** — gas usage patterns, inflow/outflow balance, average
  block gap, unique values ratio, small transaction ratio, round amount ratio.

Each feature is clipped to a known range and linearly scaled to the integer
interval [0, 128] (fixed-point representation with scale factor 2^7). This
produces a 24-element `i32` vector ready for the neural network.

### Step 3: Run the neural network

The model is a 3-layer multi-layer perceptron (MLP):

```
Input [1x24] → Dense [24x36] + ReLU → Dense [36x36] + ReLU → Dense [36x5]
```

Total: 2,417 parameters, all baked into the Rust binary as const arrays (no
external weight files). The forward pass runs in integer arithmetic on the
fixed-point inputs.

The 5 output values are raw logits — one per class. The code takes the argmax to
get the predicted class, computes confidence as the margin between the top score
and the runner-up (divided by 128, capped at 1.0), and runs softmax to get
normalized probabilities for the receipt.

### Step 4: Make a decision

The enforcement module maps classifications to decisions:

| Classification | Decision | Meaning |
|---|---|---|
| GENUINE_COMMERCE | Allow | Normal buying/selling patterns |
| LOW_ACTIVITY | Allow | Not enough data to judge |
| SCRIPTED_BENIGN | Allow | Automated but legitimate (payroll, subscriptions) |
| CIRCULAR_PAYMENTS | Flag (or Deny above 85% confidence) | Money cycling through intermediaries |
| WASH_TRADING | Deny | Fake volume, bot-like behavior |

### Step 5: Issue a Guardrail Receipt

The result is packaged into a Guardrail Receipt — a JSON document containing:

- **Receipt ID** — unique identifier (SHA-256 of timestamp + CSPRNG random bytes).
- **Subject commitment** — SHA-256 hash of the 24-element input feature vector.
  This binds the receipt to the exact inputs without revealing them.
- **Evaluation** — classification, decision, confidence, per-class softmax scores,
  human-readable reasoning.
- **Guardrail metadata** — domain, policy ID, model hash (SHA-256 of the
  serialized model bytecode, versioned with a "v1" prefix).
- **Nonce** — 32 bytes of cryptographically secure randomness (from
  `rand::thread_rng()`, a CSPRNG) to prevent receipt replay.
- **Payment info** — optional x402 payment details (network, asset, amount,
  payer, payee, tx hash) if the evaluation was paid for.

### Step 6: Generate a ZK proof (mandatory)

Every evaluation generates a zero-knowledge proof using JOLT Atlas with Dory
polynomial commitments. This proves that the neural network was executed
correctly on the committed inputs. Proofs are mandatory — there is no way to
skip proof generation.

The proof system works like this:

1. **Preprocessing** — the model structure is compiled into a circuit. This is
   expensive (~1-2 minutes) but cached via `OnceLock`, so it only happens once
   per process lifetime.
2. **Proving** — JOLT produces a SNARK proving the forward pass was computed
   correctly. The proof includes the program I/O (inputs and outputs).
3. **Local verification** — the proof is verified immediately before saving
   (fail-closed design). If local verification fails, the entire operation
   errors out.
4. **Serialization** — the proof is serialized, base64-encoded, and stored in
   a JSON proof file and embedded in the receipt.

Anyone can later verify the proof without re-running the model or seeing the raw
wallet data. They just need the proof bytes, the program I/O, and the model hash.

### The whole pipeline

```
Wallet address
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│   Indexer    │────▶│  Feature Extract  │────▶│    MLP     │
│  (Base RPC)  │     │   (24 features)   │     │ [24→36→36→5]│
└──────────────┘     └──────────────────┘     └─────┬──────┘
                                                     │
                                              ┌──────▼──────┐
                                              │ Enforcement │
                                              │ Allow/Flag/ │
                                              │    Deny     │
                                              └──────┬──────┘
                                                     │
                              ┌───────────────┬──────▼──────┐
                              │  ZK Proof     │  Guardrail  │
                              │ (JOLT Atlas)  │   Receipt   │
                              │  (optional)   │   (JSON)    │
                              └───────────────┴─────────────┘
```

## Project structure

```
verifiablex402/
├── src/
│   ├── main.rs              # CLI (analyze, scan, verify, serve, models)
│   ├── lib.rs               # Library root, run_guardrail()
│   ├── config.rs            # TOML config file loader
│   ├── transaction.rs       # Data types + 24-feature extraction
│   ├── encoding.rs          # Feature normalization to [0, 128]
│   ├── enforcement.rs       # Allow / Flag / Deny decision logic
│   ├── receipt.rs           # Guardrail Receipt construction (CSPRNG nonces)
│   ├── proving.rs           # JOLT Atlas ZK proof generation/verification
│   ├── indexer.rs           # Base mainnet JSON-RPC indexer (retry + backfill)
│   ├── server.rs            # Axum HTTP server (CORS, auth, metrics, cache)
│   ├── payment.rs           # On-chain USDC payment verification
│   ├── auth.rs              # API key middleware (X-API-Key header)
│   ├── cache.rs             # LRU wallet result cache (moka)
│   ├── metrics.rs           # Prometheus metrics recording
│   └── models/
│       └── tx_integrity.rs  # MLP [24 → 36 → 36 → 5]
├── tests/
│   └── integration.rs       # Integration tests
├── contracts/
│   ├── src/
│   │   └── IntegrityAttestation.sol  # On-chain attestation (OpenZeppelin AccessControl)
│   ├── test/
│   │   └── IntegrityAttestation.t.sol  # Foundry tests
│   └── foundry.toml
├── training-data/
│   ├── generate_synthetic.py  # Generate labeled training data (10,000 samples)
│   ├── train_classifier.py    # Train PyTorch model + export metrics
│   ├── export_to_rust.py      # Quantize weights → Rust const arrays
│   └── validate_model.py      # Verify quantization accuracy (< 1% delta)
├── docs/
│   └── adr/
│       ├── 001-fixed-point-arithmetic.md
│       ├── 002-jolt-atlas-proof-system.md
│       └── 003-synthetic-training-data.md
├── .github/
│   └── workflows/
│       └── ci.yml             # Rust + Solidity CI pipeline
├── openapi.yaml               # OpenAPI 3.1 specification
├── Dockerfile                 # Workspace-context Docker build
├── render.yaml                # Render deployment config
└── Cargo.toml
```

## Getting started

### 1. Prerequisites

You need the [jolt-atlas](https://github.com/ICME-Lab/jolt-atlas) workspace
checked out, since this crate depends on `onnx-tracer`, `zkml-jolt-core`, and
`jolt-core` from that workspace.

```
# Expected directory layout:
jolt-atlas/
├── onnx-tracer/
├── zkml-jolt-core/
├── verifiablex402/       # this repo
└── Cargo.toml            # workspace root (lists "verifiablex402" in members)
```

You also need:
- **Rust 1.88+** (the `rust-toolchain.toml` pins this)
- **RISC-V target**: `rustup target add riscv32im-unknown-none-elf`
- **Python 3.8+** with PyTorch (only if retraining the model)
- The `dory_srs_22_variables.srs` file in the project root (symlinked from
  jolt-atlas; needed for ZK proof generation)

### 2. Build

```bash
cd /path/to/jolt-atlas
cargo build -p verifiablex402 --release
```

The binary lands at `target/release/verifiablex402`.

### 3. Train the model (optional)

The crate ships with trained weights. To retrain on synthetic data:

```bash
cd training-data/

# Generate 10,000 labeled samples (2,000 per class, 10% edge cases)
python generate_synthetic.py

# Train the MLP (~100 epochs), exports training_metrics.json
python train_classifier.py

# Validate quantization accuracy (must be < 1% delta)
python validate_model.py

# Export quantized weights as Rust const arrays
python export_to_rust.py
```

Copy the output from `export_to_rust.py` into `src/models/tx_integrity.rs`,
replacing the existing `W1`, `B1`, `W2`, `B2`, `W3`, `B3` arrays. Then rebuild.

### 4. Analyze a wallet

```bash
# Analyze a wallet address (fetches from Base mainnet, always generates ZK proof)
verifiablex402 analyze --wallet 0xYourWalletAddress

# Output as JSON
verifiablex402 analyze --wallet 0xYourWalletAddress --format json

# Save receipt to file
verifiablex402 analyze --wallet 0xYourWalletAddress --output receipt.json

# Analyze from a local JSON file (offline)
verifiablex402 analyze --wallet 0xAny --input wallet_activity.json
```

### 5. Verify a receipt

```bash
# Verify receipt structure and consistency
verifiablex402 verify-receipt --input receipt.json

# Also verify the embedded ZK proof
verifiablex402 verify-receipt --input receipt.json --verify-proof
```

### 6. Run the HTTP server

```bash
# Basic
verifiablex402 serve

# With options
verifiablex402 serve --bind 0.0.0.0:8080 --rate-limit 120 --require-payment

# With structured logging at debug level
RUST_LOG=verifiablex402=debug verifiablex402 serve
```

Endpoints:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | Open | Health check, model hash, uptime, RPC status |
| `GET` | `/metrics` | Open | Prometheus metrics (scrape format) |
| `POST` | `/guardrail/integrity` | API key | Evaluate wallet integrity (from features or activity data) |
| `POST` | `/api/v1/scan` | API key | Scan a wallet by address (fetches from Base) |

The `/health` and `/metrics` endpoints are always open (needed for Render health
checks and Prometheus scraping). The `/guardrail/integrity` and `/api/v1/scan`
endpoints require an API key when keys are configured.

HTTP status codes:

| Code | Meaning |
|------|---------|
| `200` | Evaluation succeeded |
| `400` | Bad request (e.g. wrong number of features) |
| `401` | Missing or invalid API key |
| `402` | Payment required (`--require-payment` enabled, no valid payment) |
| `429` | Rate limit exceeded |
| `500` | Internal error during evaluation |
| `502` | Failed to reach Base RPC or payment verification error |

See [`openapi.yaml`](openapi.yaml) for the full API specification.

Example request to `/guardrail/integrity`:

```json
{
  "wallet_activity": {
    "wallet_address": "0xabc...",
    "chain_id": 8453,
    "from_block": 1000,
    "to_block": 2000,
    "transactions": [
      {
        "tx_hash": "0x...",
        "from": "0xabc...",
        "to": "0xdef...",
        "value": 5000000,
        "timestamp": 1700000000,
        "block_number": 1000,
        "gas_used": 21000,
        "gas_price": 1000000000
      }
    ]
  },
}
```

Note: ZK proofs are always generated for every evaluation. The `generate_proof`
field is accepted but ignored for backward compatibility.

Example request to `/api/v1/scan`:

```json
{
  "wallet_address": "0xabc...",
  "lookback_blocks": 302400,
  "payment": {
    "network": "eip155:8453",
    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "amount": "5000",
    "payer": "0xabc...",
    "payee": "0xdef...",
    "tx_hash": "0x...",
    "scheme": "exact"
  }
}
```

Server flags:

```
--bind ADDR              Address to listen on (default: 127.0.0.1:8080)
--max-proofs N           Max concurrent ZK proof generations (default: 4)
--rate-limit N           Requests per minute per IP, 0 = unlimited (default: 60)
--rpc-url URL            Base RPC endpoint (default: https://mainnet.base.org)
--require-payment        Enable x402 payment gating
```

ZK proofs are mandatory for all evaluations — every response includes a JOLT
Atlas proof. The `--max-proofs` flag controls concurrency to manage resource
usage.

All server flags can also be set via a config file (see below).

### 7. Configuration file

verifiablex402 loads an optional TOML configuration file from
`~/.config/verifiablex402/config.toml`. CLI arguments take priority over config
file values, which take priority over defaults.

Example `~/.config/verifiablex402/config.toml`:

```toml
rpc_url = "https://base-mainnet.g.alchemy.com/v2/YOUR_KEY"
bind = "0.0.0.0:8080"
rate_limit_rpm = 120
max_concurrent_proofs = 2
require_payment = true

# USDC contract address (defaults to canonical Base USDC)
usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Payment payee address (for on-chain verification)
payment_payee = "0xYourPayeeAddress"

# Allowed CORS origins (empty = allow any origin)
allowed_origins = ["https://yourdomain.com"]

# API keys for authentication (empty = no auth required)
api_keys = ["your-secret-key-1", "your-secret-key-2"]

# Wallet result cache settings
cache_ttl_seconds = 300     # 5 minutes
cache_max_entries = 1000
```

### 8. Batch scan

```bash
# wallets.txt: one address per line
verifiablex402 scan --input wallets.txt --output-dir results/
```

Batch scans run with parallelism (4 concurrent wallet scans by default) using
`futures::stream::buffer_unordered` for throughput.

### 9. Deploy the on-chain attestation contract (optional)

The `contracts/` directory contains a Foundry project with
`IntegrityAttestation.sol`. This contract uses OpenZeppelin AccessControl for
role-based authorization and stores proof hashes and wallet classifications
on-chain.

```bash
cd contracts/

# Install dependencies
forge install OpenZeppelin/openzeppelin-contracts --no-commit
forge install foundry-rs/forge-std --no-commit

# Build and test
forge build
forge test -vv

# Deploy to Base
forge script --broadcast ...
```

### 10. Show model info

```bash
verifiablex402 models
```

Prints the model architecture, parameter count, class labels, and hash.

## Classification and decision mapping

| Classification | Default decision | When |
|---|---|---|
| GENUINE_COMMERCE | Allow | Normal commerce patterns |
| LOW_ACTIVITY | Allow | Insufficient data to judge |
| SCRIPTED_BENIGN | Allow | Automated but legitimate |
| CIRCULAR_PAYMENTS | Flag (Deny above 85% confidence) | Suspicious cycling |
| WASH_TRADING | Deny | Fake volume detected |

## The 24 input features

| # | Feature | What it measures | Normalization range |
|---|---------|-----------------|-------------------|
| 0 | tx_count | Total transactions in the window | [0, 500] |
| 1 | unique_counterparties | Distinct addresses interacted with | [0, 200] |
| 2 | counterparty_entropy | Shannon entropy of counterparty distribution | [0, 5.3] |
| 3 | avg_value | Mean transaction value (USDC) | [0, 100 USDC] |
| 4 | std_value | Standard deviation of values | [0, 100 USDC] |
| 5 | max_value | Largest single transaction | [0, 1000 USDC] |
| 6 | min_value | Smallest single transaction | [0, 100 USDC] |
| 7 | value_range_ratio | (max - min) / max | [0, 1] |
| 8 | identical_amount_ratio | Fraction of txs with repeated exact amounts | [0, 1] |
| 9 | self_transfer_ratio | Fraction sent back to self | [0, 1] |
| 10 | circular_path_score | A->B->C->A loop detection score | [0, 1] |
| 11 | avg_time_between_tx | Mean seconds between consecutive txs | [0, 86400] |
| 12 | time_regularity | Coefficient of variation of inter-tx intervals | [0, 3] |
| 13 | burst_score | Peak tx/minute / average tx/minute | [0, 100] |
| 14 | night_ratio | Fraction of txs between midnight and 6 AM UTC | [0, 1] |
| 15 | weekend_ratio | Fraction of txs on Saturday/Sunday | [0, 1] |
| 16 | tx_per_day | Average transactions per day | [0, 100] |
| 17 | gas_efficiency | Average gas_used / gas_price ratio | [0, 1] |
| 18 | inflow_outflow_ratio | Received value / total value | [0, 1] |
| 19 | avg_block_gap | Mean block number gap between txs | [0, 10000] |
| 20 | unique_values_ratio | Distinct values / total tx count | [0, 1] |
| 21 | small_tx_ratio | Fraction of txs below 1 USDC | [0, 1] |
| 22 | round_amount_ratio | Fraction of txs with round numbers | [0, 1] |
| 23 | activity_span_days | Days between first and last tx | [0, 365] |

All features are clipped to their normalization range and linearly scaled to the
integer interval [0, 128] (fixed-point scale = 2^7).

## Security

- **API key authentication** — protected endpoints (`/guardrail/integrity`,
  `/api/v1/scan`) require an `X-API-Key` header when API keys are configured.
  When no keys are set, all requests pass through (backward compatible).
- **On-chain payment verification** — when `--require-payment` is enabled and a
  `payment_payee` is configured, the server verifies payments on-chain by
  fetching the transaction receipt and checking for a USDC Transfer event
  matching the expected payee and amount.
- **Configurable CORS** — set `allowed_origins` in the config file to restrict
  which origins can call the API. Empty or unset means allow any origin
  (development mode).
- **CSPRNG nonces** — receipt IDs and nonces use `rand::thread_rng()`, a
  cryptographically secure random number generator.
- **Fail-closed verification** — ZK proofs are verified locally before saving.
  If verification fails, the operation errors. Model hash mismatches cause
  immediate rejection.
- **Rate limiting** — per-IP rate limiting via the `governor` crate. Configurable
  requests per minute, defaults to 60.
- **Structured logging** — all server activity is logged via `tracing` with
  structured fields (wallet address, classification, processing time). Control
  verbosity with `RUST_LOG` (e.g. `RUST_LOG=verifiablex402=debug`).

## Observability

### Health check

`GET /health` returns server status including RPC connectivity:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_hash": "sha256:...",
  "model_name": "tx-integrity",
  "model_params": 2417,
  "uptime_seconds": 3600,
  "rpc_connected": true
}
```

The `status` field is `"ok"` when the RPC node is reachable (tested with a
3-second timeout `eth_blockNumber` call) and `"degraded"` otherwise. The endpoint
always returns HTTP 200 to avoid restart flapping on transient RPC issues.

### Prometheus metrics

`GET /metrics` returns metrics in Prometheus text exposition format:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `evaluations_total` | Counter | `classification`, `decision` | Total evaluations performed |
| `evaluation_duration_ms` | Histogram | — | Evaluation latency |
| `rpc_calls_total` | Counter | `method`, `success` | Total RPC calls made |
| `rpc_call_duration_ms` | Histogram | — | RPC call latency |
| `rate_limit_hits_total` | Counter | — | Rate limit rejections |
| `payment_checks_total` | Counter | `result` | Payment verification outcomes |
| `cache_hits_total` | Counter | — | Wallet cache hits |
| `cache_misses_total` | Counter | — | Wallet cache misses |

### Wallet result cache

The `/api/v1/scan` endpoint caches results keyed by `wallet_address:lookback_blocks`.
Configurable via `cache_ttl_seconds` (default 300s) and `cache_max_entries`
(default 1000). Uses the `moka` crate (concurrent LRU with TTL eviction).

## Docker

The Dockerfile expects the **jolt-atlas workspace root** as the build context:

```bash
# From the jolt-atlas workspace root:
docker build -t verifiablex402 -f verifiablex402/Dockerfile .
docker run -p 10000:10000 verifiablex402
```

The image verifies SRS file checksums at startup before launching the server.

## CI

The GitHub Actions CI pipeline (`.github/workflows/ci.yml`) runs on push to
`main` and pull requests:

- **Rust job**: clones jolt-atlas workspace, runs `cargo fmt --check`,
  `cargo clippy -D warnings`, and `cargo test --release` for the
  `verifiablex402` package.
- **Solidity job**: installs Foundry, runs `forge build` and `forge test -vv`
  in the `contracts/` directory.

## Smart contract

`IntegrityAttestation.sol` uses OpenZeppelin `AccessControl` for role management:

- **`DEFAULT_ADMIN_ROLE`** — manages all roles (granted to deployer)
- **`ATTESTER_ROLE`** — can submit proof attestations and record wallet
  classifications (granted to deployer, additional attesters added via
  `grantRole`)

Key functions:

| Function | Access | Description |
|----------|--------|-------------|
| `attestProof(bytes32)` | `ATTESTER_ROLE` | Register a proof hash on-chain |
| `recordClassification(address, Classification, uint8, bytes32)` | `ATTESTER_ROLE` | Record wallet classification with proof |
| `isWalletSuspicious(address)` | Public view | Returns true for CircularPayments or WashTrading |
| `isProofHashValid(bytes32)` | Public view | Check if a proof hash was attested |
| `getWalletClassification(address)` | Public view | Get latest classification for a wallet |
| `grantRole(bytes32, address)` | `DEFAULT_ADMIN_ROLE` | Add new attesters or admins |
| `revokeRole(bytes32, address)` | `DEFAULT_ADMIN_ROLE` | Remove attesters or admins |

## Tests

```bash
# Run all Rust tests
cargo test -p verifiablex402 --release

# Run just integration tests
cargo test -p verifiablex402 --test integration --release

# Run the prove-verify integration test
cargo test -p verifiablex402 --test integration --release test_tx_integrity_prove_verify

# Run Solidity tests
cd contracts && forge test -vv
```

## Architecture decision records

- [ADR 001: Fixed-Point Arithmetic](docs/adr/001-fixed-point-arithmetic.md) — why i32 at scale 128, not f32
- [ADR 002: JOLT Atlas Proof System](docs/adr/002-jolt-atlas-proof-system.md) — why JOLT Atlas over other ZK systems
- [ADR 003: Synthetic Training Data](docs/adr/003-synthetic-training-data.md) — current approach + known limitations

## License

MIT
