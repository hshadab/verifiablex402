# ADR 003: Synthetic Training Data

## Status

Accepted (with known limitations)

## Context

Training an on-chain transaction classifier requires labeled wallet data across 5 behavioral categories. Real labeled data is difficult to obtain:

- No public labeled datasets of wallet transaction patterns exist
- Manual labeling requires domain expertise and is subjective
- Privacy concerns with using real wallet data for training
- Need balanced classes (real data would be heavily skewed toward genuine commerce)

## Decision

Use **synthetically generated training data** with hand-crafted feature distributions per class.

### Generation approach (`generate_synthetic.py`):

Each class has a generator function that samples 24 features from class-specific distributions:

| Class | Key distinguishing features |
|-------|---------------------------|
| GENUINE_COMMERCE | High counterparty diversity/entropy, varied amounts, moderate tx rate |
| LOW_ACTIVITY | Very few transactions (0-15), low tx/day, sparse |
| SCRIPTED_BENIGN | High tx count, very regular timing (low CV), diverse counterparties |
| CIRCULAR_PAYMENTS | High circular_path_score, high self_transfer_ratio, few counterparties |
| WASH_TRADING | High identical_amount_ratio, very low diversity, high burst_score, bot-like timing |

### Dataset size:

- **2,000 samples per class** (10,000 total)
- **10% edge cases** per class (0-1 tx wallets, max-value outliers, mixed patterns)
- Features normalized to `[0, 128]` to match the Rust fixed-point representation

## Consequences

### Known limitations:

1. **Distribution gap**: Synthetic distributions may not match real-world wallet behavior. The boundaries between classes (especially circular_payments vs. wash_trading) are idealized.
2. **Feature independence**: Each feature is sampled independently — real features have correlations (e.g., high tx_count implies lower avg_time_between_tx) that are not fully captured.
3. **No adversarial samples**: The training data does not include wallets deliberately designed to evade detection.
4. **Label noise**: Edge cases intentionally blur class boundaries, but real-world ambiguity may be different.

### Mitigations:

- `validate_model.py` checks that quantization loss is < 1%
- Per-class accuracy is tracked in `training_metrics.json`
- The model outputs confidence scores and a decision escalation threshold (85% for circular→deny)
- Retraining with real labeled data is the recommended path once available

### Retraining workflow:

1. Run `generate_synthetic.py` (or supply real data as `training_features.csv`)
2. Run `train_classifier.py` → produces `tx_integrity_model.pt` + `training_metrics.json`
3. Run `validate_model.py` → verifies quantization accuracy
4. Run `export_to_rust.py` → paste updated weights into `src/models/tx_integrity.rs`
5. Rebuild: `cargo build -p verifiablex402 --release`
