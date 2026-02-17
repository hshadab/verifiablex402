#!/usr/bin/env python3
"""Generate synthetic transaction data for the x402 integrity classifier.

Produces 5 classes of labeled data:
0 - GENUINE_COMMERCE: moderate volume, diverse counterparties, varying amounts
1 - LOW_ACTIVITY: <3 tx/day, sparse
2 - SCRIPTED_BENIGN: high volume, regular timing, diverse counterparties
3 - CIRCULAR_PAYMENTS: high self-transfer, A->B->C->A loops
4 - WASH_TRADING: high identical amounts, minimal diversity, bot-like timing

Output: training_features.csv (24 features + label column)
"""

import csv
import random
import math
import os

random.seed(42)

NUM_SAMPLES_PER_CLASS = 2000
NUM_FEATURES = 24
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "training_features.csv")

FEATURE_NAMES = [
    "tx_count", "unique_counterparties", "counterparty_entropy",
    "avg_value", "std_value", "max_value", "min_value",
    "value_range_ratio", "identical_amount_ratio", "self_transfer_ratio",
    "circular_path_score", "avg_time_between_tx", "time_regularity",
    "burst_score", "night_ratio", "weekend_ratio", "tx_per_day",
    "gas_efficiency", "inflow_outflow_ratio", "avg_block_gap",
    "unique_values_ratio", "small_tx_ratio", "round_amount_ratio",
    "activity_span_days",
]

# Normalization ranges (same as Rust clip_scale)
RANGES = [
    (0, 500), (0, 200), (0, 5.3), (0, 1e8), (0, 1e8),
    (0, 1e9), (0, 1e8), (0, 1), (0, 1), (0, 1),
    (0, 1), (0, 86400), (0, 3), (0, 100), (0, 1),
    (0, 1), (0, 100), (0, 1), (0, 1), (0, 10000),
    (0, 1), (0, 1), (0, 1), (0, 365),
]


def clip_normalize(value, lo, hi):
    """Clip to [lo, hi] and normalize to [0, 128]."""
    clamped = max(lo, min(hi, value))
    return int((clamped - lo) / (hi - lo) * 128)


def gen_genuine_commerce(edge_case=False):
    """Generate features for genuine commerce wallet."""
    if edge_case:
        # Edge: very high tx count outlier
        tx_count = random.uniform(400, 500)
        unique_cp = random.uniform(150, 200)
        entropy = random.uniform(4.5, 5.3)
    else:
        tx_count = random.uniform(20, 300)
        unique_cp = random.uniform(10, 150)
        entropy = random.uniform(2.0, 5.0)
    avg_val = random.uniform(1e6, 5e7)
    std_val = random.uniform(5e5, 3e7)
    max_val = avg_val + random.uniform(1e6, 5e7)
    min_val = max(0, avg_val - random.uniform(5e5, 2e7))
    range_ratio = random.uniform(0.5, 0.95)
    identical_ratio = random.uniform(0.05, 0.3)
    self_ratio = random.uniform(0, 0.05)
    circular = random.uniform(0, 0.1)
    avg_time = random.uniform(300, 43200)
    regularity = random.uniform(0.3, 2.0)
    burst = random.uniform(1, 15)
    night = random.uniform(0.05, 0.25)
    weekend = random.uniform(0.15, 0.35)
    tx_day = random.uniform(1, 30)
    gas_eff = random.uniform(0.1, 0.8)
    io_ratio = random.uniform(0.3, 0.7)
    block_gap = random.uniform(50, 5000)
    unique_vals = random.uniform(0.5, 0.95)
    small_ratio = random.uniform(0.05, 0.3)
    round_ratio = random.uniform(0.1, 0.5)
    span = random.uniform(7, 180)

    raw = [tx_count, unique_cp, entropy, avg_val, std_val, max_val, min_val,
           range_ratio, identical_ratio, self_ratio, circular, avg_time,
           regularity, burst, night, weekend, tx_day, gas_eff, io_ratio,
           block_gap, unique_vals, small_ratio, round_ratio, span]
    return [clip_normalize(v, lo, hi) for v, (lo, hi) in zip(raw, RANGES)]


def gen_low_activity(edge_case=False):
    """Generate features for low activity wallet."""
    if edge_case:
        # Edge: 0-1 transaction wallet
        tx_count = random.uniform(0, 1)
        unique_cp = random.uniform(0, 1)
        entropy = 0
        tx_day = random.uniform(0, 0.01)
    else:
        tx_count = random.uniform(1, 15)
        unique_cp = random.uniform(1, 8)
        entropy = random.uniform(0, 2.0)
        tx_day = random.uniform(0.01, 2)
    avg_val = random.uniform(1e5, 2e7)
    std_val = random.uniform(0, 1e7)
    max_val = avg_val + random.uniform(0, 1e7)
    min_val = max(0, avg_val - random.uniform(0, 5e6))
    range_ratio = random.uniform(0, 0.8)
    identical_ratio = random.uniform(0.1, 0.8)
    self_ratio = random.uniform(0, 0.2)
    circular = random.uniform(0, 0.1)
    avg_time = random.uniform(10000, 86400)
    regularity = random.uniform(0, 2.5)
    burst = random.uniform(1, 5)
    night = random.uniform(0, 0.5)
    weekend = random.uniform(0, 0.5)
    gas_eff = random.uniform(0.1, 0.9)
    io_ratio = random.uniform(0.2, 0.8)
    block_gap = random.uniform(500, 10000)
    unique_vals = random.uniform(0.3, 1.0)
    small_ratio = random.uniform(0, 0.5)
    round_ratio = random.uniform(0, 0.6)
    span = random.uniform(1, 60)

    raw = [tx_count, unique_cp, entropy, avg_val, std_val, max_val, min_val,
           range_ratio, identical_ratio, self_ratio, circular, avg_time,
           regularity, burst, night, weekend, tx_day, gas_eff, io_ratio,
           block_gap, unique_vals, small_ratio, round_ratio, span]
    return [clip_normalize(v, lo, hi) for v, (lo, hi) in zip(raw, RANGES)]


def gen_scripted_benign(edge_case=False):
    """Generate features for scripted benign wallet (payroll, subscriptions)."""
    if edge_case:
        # Edge: max-value outlier (large payroll)
        tx_count = random.uniform(400, 500)
        avg_val = random.uniform(8e7, 1e8)
    else:
        tx_count = random.uniform(100, 500)
        avg_val = random.uniform(5e6, 8e7)
    unique_cp = random.uniform(20, 200)
    entropy = random.uniform(2.5, 5.3)
    std_val = random.uniform(1e5, 5e6)
    max_val = avg_val + random.uniform(1e5, 1e7)
    min_val = max(0, avg_val - random.uniform(1e5, 5e6))
    range_ratio = random.uniform(0.1, 0.5)
    identical_ratio = random.uniform(0.2, 0.6)
    self_ratio = random.uniform(0, 0.05)
    circular = random.uniform(0, 0.05)
    avg_time = random.uniform(60, 3600)
    regularity = random.uniform(0.05, 0.5)
    burst = random.uniform(1, 10)
    night = random.uniform(0.05, 0.2)
    weekend = random.uniform(0.05, 0.2)
    tx_day = random.uniform(10, 100)
    gas_eff = random.uniform(0.2, 0.7)
    io_ratio = random.uniform(0.3, 0.7)
    block_gap = random.uniform(10, 500)
    unique_vals = random.uniform(0.3, 0.8)
    small_ratio = random.uniform(0.05, 0.3)
    round_ratio = random.uniform(0.3, 0.7)
    span = random.uniform(14, 365)

    raw = [tx_count, unique_cp, entropy, avg_val, std_val, max_val, min_val,
           range_ratio, identical_ratio, self_ratio, circular, avg_time,
           regularity, burst, night, weekend, tx_day, gas_eff, io_ratio,
           block_gap, unique_vals, small_ratio, round_ratio, span]
    return [clip_normalize(v, lo, hi) for v, (lo, hi) in zip(raw, RANGES)]


def gen_circular_payments(edge_case=False):
    """Generate features for circular payments wallet."""
    if edge_case:
        # Edge: mixed pattern — moderate circular with some genuine features
        circular = random.uniform(0.2, 0.4)
        unique_cp = random.uniform(10, 30)
        entropy = random.uniform(1.5, 3.0)
    else:
        circular = random.uniform(0.3, 1.0)
        unique_cp = random.uniform(3, 20)
        entropy = random.uniform(0.5, 2.5)
    tx_count = random.uniform(20, 200)
    avg_val = random.uniform(5e6, 5e7)
    std_val = random.uniform(1e5, 1e7)
    max_val = avg_val + random.uniform(1e6, 2e7)
    min_val = max(0, avg_val - random.uniform(5e5, 1e7))
    range_ratio = random.uniform(0.2, 0.7)
    identical_ratio = random.uniform(0.3, 0.7)
    self_ratio = random.uniform(0.1, 0.5)
    avg_time = random.uniform(60, 7200)
    regularity = random.uniform(0.2, 1.5)
    burst = random.uniform(3, 50)
    night = random.uniform(0.1, 0.6)
    weekend = random.uniform(0.1, 0.5)
    tx_day = random.uniform(3, 50)
    gas_eff = random.uniform(0.1, 0.5)
    io_ratio = random.uniform(0.35, 0.65)
    block_gap = random.uniform(10, 2000)
    unique_vals = random.uniform(0.1, 0.5)
    small_ratio = random.uniform(0.05, 0.3)
    round_ratio = random.uniform(0.2, 0.6)
    span = random.uniform(3, 90)

    raw = [tx_count, unique_cp, entropy, avg_val, std_val, max_val, min_val,
           range_ratio, identical_ratio, self_ratio, circular, avg_time,
           regularity, burst, night, weekend, tx_day, gas_eff, io_ratio,
           block_gap, unique_vals, small_ratio, round_ratio, span]
    return [clip_normalize(v, lo, hi) for v, (lo, hi) in zip(raw, RANGES)]


def gen_wash_trading(edge_case=False):
    """Generate features for wash trading wallet."""
    if edge_case:
        # Edge: mixed pattern — some value variation to test boundaries
        identical_ratio = random.uniform(0.4, 0.6)
        unique_vals = random.uniform(0.15, 0.3)
    else:
        identical_ratio = random.uniform(0.6, 1.0)
        unique_vals = random.uniform(0.01, 0.2)
    tx_count = random.uniform(50, 500)
    unique_cp = random.uniform(1, 10)
    entropy = random.uniform(0, 1.5)
    avg_val = random.uniform(1e6, 5e7)
    std_val = random.uniform(0, 5e5)
    max_val = avg_val + random.uniform(0, 2e6)
    min_val = max(0, avg_val - random.uniform(0, 1e6))
    range_ratio = random.uniform(0, 0.2)
    self_ratio = random.uniform(0.2, 0.8)
    circular = random.uniform(0.2, 0.9)
    avg_time = random.uniform(10, 600)
    regularity = random.uniform(0, 0.3)
    burst = random.uniform(10, 100)
    night = random.uniform(0.2, 0.8)
    weekend = random.uniform(0.2, 0.6)
    tx_day = random.uniform(10, 100)
    gas_eff = random.uniform(0.1, 0.4)
    io_ratio = random.uniform(0.4, 0.6)
    block_gap = random.uniform(1, 100)
    small_ratio = random.uniform(0, 0.3)
    round_ratio = random.uniform(0.5, 1.0)
    span = random.uniform(1, 30)

    raw = [tx_count, unique_cp, entropy, avg_val, std_val, max_val, min_val,
           range_ratio, identical_ratio, self_ratio, circular, avg_time,
           regularity, burst, night, weekend, tx_day, gas_eff, io_ratio,
           block_gap, unique_vals, small_ratio, round_ratio, span]
    return [clip_normalize(v, lo, hi) for v, (lo, hi) in zip(raw, RANGES)]


GENERATORS = [
    gen_genuine_commerce,
    gen_low_activity,
    gen_scripted_benign,
    gen_circular_payments,
    gen_wash_trading,
]

CLASS_NAMES = [
    "GENUINE_COMMERCE",
    "LOW_ACTIVITY",
    "SCRIPTED_BENIGN",
    "CIRCULAR_PAYMENTS",
    "WASH_TRADING",
]

# 10% of samples per class are edge cases
EDGE_CASE_RATIO = 0.10


def main():
    rows = []
    n_edge = int(NUM_SAMPLES_PER_CLASS * EDGE_CASE_RATIO)
    n_normal = NUM_SAMPLES_PER_CLASS - n_edge

    for label, gen_fn in enumerate(GENERATORS):
        for _ in range(n_normal):
            features = gen_fn(edge_case=False)
            rows.append(features + [label])
        for _ in range(n_edge):
            features = gen_fn(edge_case=True)
            rows.append(features + [label])

    random.shuffle(rows)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples ({NUM_SAMPLES_PER_CLASS} per class, {n_edge} edge cases each)")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
