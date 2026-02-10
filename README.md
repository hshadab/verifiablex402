# verifiablex402

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

### Step 6: Generate a ZK proof (optional)

If `--prove` or `generate_proof: true` is specified, the system generates a
zero-knowledge proof using JOLT Atlas with Dory polynomial commitments. This
proves that the neural network was executed correctly on the committed inputs.

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
│   ├── proving.rs           # JOLT Atlas ZK proof generation/verification (cached preprocessing)
│   ├── indexer.rs           # Base mainnet JSON-RPC indexer (with timestamp + gas backfill)
│   ├── server.rs            # Axum HTTP server (CORS, tracing, payment enforcement)
│   └── models/
│       └── tx_integrity.rs  # MLP [24 → 36 → 36 → 5]
├── training-data/
│   ├── generate_synthetic.py  # Generate labeled training data
│   ├── train_classifier.py    # Train the PyTorch model
│   └── export_to_rust.py      # Quantize weights → Rust const arrays
├── contracts/
│   └── src/
│       └── IntegrityAttestation.sol  # On-chain proof attestation
├── tests/
│   └── integration.rs        # 24 integration tests
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

# Generate 1,000 labeled samples (200 per class)
python generate_synthetic.py

# Train the MLP (~100 epochs)
python train_classifier.py

# Export quantized weights as Rust const arrays
python export_to_rust.py
```

Copy the output from `export_to_rust.py` into `src/models/tx_integrity.rs`,
replacing the existing `W1`, `B1`, `W2`, `B2`, `W3`, `B3` arrays. Then rebuild.

### 4. Analyze a wallet

```bash
# Analyze a wallet address (fetches from Base mainnet via RPC)
verifiablex402 analyze --wallet 0xYourWalletAddress

# With ZK proof generation
verifiablex402 analyze --wallet 0xYourWalletAddress --prove

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

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check, model hash, uptime |
| `POST` | `/guardrail/integrity` | Evaluate wallet integrity (from features or activity data) |
| `POST` | `/api/v1/scan` | Scan a wallet by address (fetches from Base) |

HTTP status codes:

| Code | Meaning |
|------|---------|
| `200` | Evaluation succeeded |
| `400` | Bad request (e.g. wrong number of features) |
| `402` | Payment required (`--require-payment` enabled, no payment in request) |
| `429` | Rate limit exceeded |
| `500` | Internal error during evaluation |
| `502` | Failed to reach Base RPC |

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
  "generate_proof": false
}
```

Example request to `/api/v1/scan`:

```json
{
  "wallet_address": "0xabc...",
  "lookback_blocks": 302400,
  "generate_proof": false,
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
--require-proof          Require proof for every request
--rate-limit N           Requests per minute per IP, 0 = unlimited (default: 60)
--rpc-url URL            Base RPC endpoint (default: https://mainnet.base.org)
--require-payment        Enable x402 payment gating
```

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
```

### 8. Batch scan

```bash
# wallets.txt: one address per line
verifiablex402 scan --input wallets.txt --output-dir results/
```

### 9. Deploy the on-chain attestation contract (optional)

The `contracts/` directory contains a Foundry project with
`IntegrityAttestation.sol`. This contract stores proof hashes and wallet
classifications on-chain.

```bash
cd contracts/
forge build
forge test
forge script --broadcast ...  # deploy to Base
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

- **CSPRNG nonces** — receipt IDs and nonces use `rand::thread_rng()`, a
  cryptographically secure random number generator. Earlier versions used
  `RandomState`-based hashing which is not cryptographically secure.
- **Fail-closed verification** — ZK proofs are verified locally before saving.
  If verification fails, the operation errors. Model hash mismatches cause
  immediate rejection.
- **x402 payment enforcement** — when `--require-payment` is enabled, the server
  returns HTTP 402 unless the request includes a valid payment with a non-empty
  `tx_hash`.
- **Rate limiting** — per-IP rate limiting via the `governor` crate. Configurable
  requests per minute, defaults to 60.
- **CORS** — the server includes a CORS layer (via `tower-http`) allowing any
  origin with GET and POST methods. Tighten `allow_origin` for production.
- **Structured logging** — all server activity is logged via `tracing` with
  structured fields (wallet address, classification, processing time). Control
  verbosity with `RUST_LOG` (e.g. `RUST_LOG=verifiablex402=debug`).

## Tests

```bash
# Run all tests (49 total: 25 unit + 24 integration)
cargo test -p verifiablex402

# Run just integration tests
cargo test -p verifiablex402 --test integration

# Run the prove-verify integration test (requires --release for speed)
cargo test -p verifiablex402 --test integration --release test_tx_integrity_prove_verify
```

## License

MIT
