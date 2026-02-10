//! Core data types for x402 transaction integrity analysis.
//!
//! Defines wallet activity, transaction records, classification enums,
//! and the 24-feature extraction pipeline for the MLP classifier.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Classification enums
// ---------------------------------------------------------------------------

/// Classification of wallet transaction patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrityClassification {
    /// Normal merchant/service payments with diverse counterparties
    GenuineCommerce,
    /// Wallet with very few transactions (insufficient data)
    LowActivity,
    /// High-volume automated but legitimate usage (payroll, subscriptions)
    ScriptedBenign,
    /// Funds cycling through intermediaries back to origin
    CircularPayments,
    /// Fake volume with identical amounts, minimal diversity, bot-like timing
    WashTrading,
}

impl IntegrityClassification {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::GenuineCommerce,
            1 => Self::LowActivity,
            2 => Self::ScriptedBenign,
            3 => Self::CircularPayments,
            4 => Self::WashTrading,
            _ => Self::LowActivity, // default fallback
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GenuineCommerce => "GENUINE_COMMERCE",
            Self::LowActivity => "LOW_ACTIVITY",
            Self::ScriptedBenign => "SCRIPTED_BENIGN",
            Self::CircularPayments => "CIRCULAR_PAYMENTS",
            Self::WashTrading => "WASH_TRADING",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::GenuineCommerce => 0,
            Self::LowActivity => 1,
            Self::ScriptedBenign => 2,
            Self::CircularPayments => 3,
            Self::WashTrading => 4,
        }
    }

    /// Whether this classification should result in a deny decision.
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::WashTrading)
    }

    /// Whether this classification should result in a flag decision.
    pub fn is_flag(&self) -> bool {
        matches!(self, Self::CircularPayments)
    }
}

/// Decision: Allow, Flag, or Deny.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrityDecision {
    Allow,
    Flag,
    Deny,
}

impl IntegrityDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Flag => "flag",
            Self::Deny => "deny",
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction data types
// ---------------------------------------------------------------------------

/// A single x402 transaction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct X402Transaction {
    pub tx_hash: String,
    pub from: String,
    pub to: String,
    /// Value in smallest unit (e.g., USDC with 6 decimals)
    pub value: u64,
    pub timestamp: u64,
    pub block_number: u64,
    pub gas_used: u64,
    pub gas_price: u64,
}

/// Wallet activity over a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletActivity {
    pub wallet_address: String,
    pub chain_id: u64,
    pub from_block: u64,
    pub to_block: u64,
    pub transactions: Vec<X402Transaction>,
}

// ---------------------------------------------------------------------------
// 24-feature extraction
// ---------------------------------------------------------------------------

/// 24 features extracted from wallet transaction history.
///
/// Feature indices:
/// 0: tx_count                   — total transaction count
/// 1: unique_counterparties      — number of distinct to/from addresses
/// 2: counterparty_entropy       — Shannon entropy of counterparty distribution
/// 3: avg_value                  — average transaction value (normalized)
/// 4: std_value                  — standard deviation of values
/// 5: max_value                  — maximum single transaction value
/// 6: min_value                  — minimum single transaction value
/// 7: value_range_ratio          — (max - min) / max
/// 8: identical_amount_ratio     — fraction of txs with most common amount
/// 9: self_transfer_ratio        — fraction of txs where from == to or back-to-self
/// 10: circular_path_score       — fraction of counterparties forming A→B→C→A loops
/// 11: avg_time_between_tx       — mean inter-transaction interval (seconds)
/// 12: time_regularity           — coefficient of variation of inter-tx intervals
/// 13: burst_score               — max tx/minute ÷ avg tx/minute
/// 14: night_ratio               — fraction of txs between 00:00–06:00 UTC
/// 15: weekend_ratio             — fraction of txs on Saturday/Sunday
/// 16: tx_per_day                — average transactions per day
/// 17: gas_efficiency            — average gas_used / gas_price ratio (normalized)
/// 18: inflow_outflow_ratio      — inflow / (inflow + outflow) for the wallet
/// 19: avg_block_gap             — average blocks between consecutive txs
/// 20: unique_values_ratio       — distinct values / total txs
/// 21: small_tx_ratio            — fraction of txs below 1 USDC (< 1_000_000 in 6-dec)
/// 22: round_amount_ratio        — fraction of txs with round amounts (multiples of 1 USDC)
/// 23: activity_span_days        — time span from first to last tx in days
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionFeatures {
    pub tx_count: f64,
    pub unique_counterparties: f64,
    pub counterparty_entropy: f64,
    pub avg_value: f64,
    pub std_value: f64,
    pub max_value: f64,
    pub min_value: f64,
    pub value_range_ratio: f64,
    pub identical_amount_ratio: f64,
    pub self_transfer_ratio: f64,
    pub circular_path_score: f64,
    pub avg_time_between_tx: f64,
    pub time_regularity: f64,
    pub burst_score: f64,
    pub night_ratio: f64,
    pub weekend_ratio: f64,
    pub tx_per_day: f64,
    pub gas_efficiency: f64,
    pub inflow_outflow_ratio: f64,
    pub avg_block_gap: f64,
    pub unique_values_ratio: f64,
    pub small_tx_ratio: f64,
    pub round_amount_ratio: f64,
    pub activity_span_days: f64,
}

impl TransactionFeatures {
    /// Extract all 24 features from a wallet's transaction history.
    pub fn extract(activity: &WalletActivity) -> Self {
        let txs = &activity.transactions;
        let wallet = activity.wallet_address.to_lowercase();
        let n = txs.len() as f64;

        if txs.is_empty() {
            return Self::zeros();
        }

        // Basic counts
        let tx_count = n;

        // Counterparty analysis
        let mut counterparty_counts: HashMap<String, usize> = HashMap::new();
        for tx in txs {
            let cp = if tx.from.to_lowercase() == wallet {
                tx.to.to_lowercase()
            } else {
                tx.from.to_lowercase()
            };
            *counterparty_counts.entry(cp).or_insert(0) += 1;
        }
        let unique_counterparties = counterparty_counts.len() as f64;

        // Shannon entropy of counterparty distribution
        let counterparty_entropy = {
            let total = counterparty_counts.values().sum::<usize>() as f64;
            if total == 0.0 {
                0.0
            } else {
                counterparty_counts
                    .values()
                    .map(|&c| {
                        let p = c as f64 / total;
                        if p > 0.0 { -p * p.ln() } else { 0.0 }
                    })
                    .sum::<f64>()
            }
        };

        // Value statistics
        let values: Vec<f64> = txs.iter().map(|tx| tx.value as f64).collect();
        let avg_value = values.iter().sum::<f64>() / n;
        let std_value = {
            let variance = values.iter().map(|v| (v - avg_value).powi(2)).sum::<f64>() / n;
            variance.sqrt()
        };
        let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let value_range_ratio = if max_value > 0.0 {
            (max_value - min_value) / max_value
        } else {
            0.0
        };

        // Identical amount ratio
        let mut amount_counts: HashMap<u64, usize> = HashMap::new();
        for tx in txs {
            *amount_counts.entry(tx.value).or_insert(0) += 1;
        }
        let max_same_amount = *amount_counts.values().max().unwrap_or(&0) as f64;
        let identical_amount_ratio = max_same_amount / n;

        // Self-transfer ratio
        let self_transfers = txs
            .iter()
            .filter(|tx| tx.from.to_lowercase() == tx.to.to_lowercase())
            .count() as f64;
        let self_transfer_ratio = self_transfers / n;

        // Circular path detection (A→B→C→A)
        let circular_path_score = compute_circular_score(txs, &wallet);

        // Timing analysis
        let mut timestamps: Vec<u64> = txs.iter().map(|tx| tx.timestamp).collect();
        timestamps.sort();

        let intervals: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] as f64) - (w[0] as f64))
            .collect();

        let avg_time_between_tx = if intervals.is_empty() {
            0.0
        } else {
            intervals.iter().sum::<f64>() / intervals.len() as f64
        };

        // Time regularity: coefficient of variation of intervals
        let time_regularity = if intervals.is_empty() || avg_time_between_tx == 0.0 {
            0.0
        } else {
            let var = intervals
                .iter()
                .map(|i| (i - avg_time_between_tx).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;
            var.sqrt() / avg_time_between_tx
        };

        // Burst score: max tx/minute ÷ avg tx/minute
        let burst_score = compute_burst_score(&timestamps);

        // Night ratio (00:00-06:00 UTC)
        let night_count = txs
            .iter()
            .filter(|tx| {
                let hour = (tx.timestamp % 86400) / 3600;
                hour < 6
            })
            .count() as f64;
        let night_ratio = night_count / n;

        // Weekend ratio
        let weekend_count = txs
            .iter()
            .filter(|tx| {
                // Unix epoch (Jan 1 1970) was a Thursday (day 4)
                let day_of_week = ((tx.timestamp / 86400) + 4) % 7;
                day_of_week == 0 || day_of_week == 6 // Sunday or Saturday
            })
            .count() as f64;
        let weekend_ratio = weekend_count / n;

        // Transactions per day
        let span_seconds = if timestamps.len() >= 2 {
            (timestamps.last().unwrap() - timestamps.first().unwrap()) as f64
        } else {
            86400.0 // default 1 day
        };
        let activity_span_days = (span_seconds / 86400.0).max(1.0);
        let tx_per_day = n / activity_span_days;

        // Gas efficiency
        let gas_efficiency = {
            let gas_ratios: Vec<f64> = txs
                .iter()
                .filter(|tx| tx.gas_price > 0)
                .map(|tx| tx.gas_used as f64 / tx.gas_price as f64)
                .collect();
            if gas_ratios.is_empty() {
                0.0
            } else {
                gas_ratios.iter().sum::<f64>() / gas_ratios.len() as f64
            }
        };

        // Inflow/outflow ratio
        let (inflow, outflow) = txs.iter().fold((0u64, 0u64), |(inf, outf), tx| {
            if tx.to.to_lowercase() == wallet {
                (inf + tx.value, outf)
            } else {
                (inf, outf + tx.value)
            }
        });
        let inflow_outflow_ratio = if inflow + outflow > 0 {
            inflow as f64 / (inflow + outflow) as f64
        } else {
            0.5
        };

        // Average block gap
        let mut blocks: Vec<u64> = txs.iter().map(|tx| tx.block_number).collect();
        blocks.sort();
        let block_gaps: Vec<f64> = blocks
            .windows(2)
            .map(|w| (w[1] as f64) - (w[0] as f64))
            .collect();
        let avg_block_gap = if block_gaps.is_empty() {
            0.0
        } else {
            block_gaps.iter().sum::<f64>() / block_gaps.len() as f64
        };

        // Unique values ratio
        let unique_values: HashSet<u64> = txs.iter().map(|tx| tx.value).collect();
        let unique_values_ratio = unique_values.len() as f64 / n;

        // Small transaction ratio (< 1 USDC = 1_000_000 in 6-decimal)
        let small_count = txs.iter().filter(|tx| tx.value < 1_000_000).count() as f64;
        let small_tx_ratio = small_count / n;

        // Round amount ratio (multiple of 1 USDC = 1_000_000)
        let round_count = txs
            .iter()
            .filter(|tx| tx.value > 0 && tx.value % 1_000_000 == 0)
            .count() as f64;
        let round_amount_ratio = round_count / n;

        Self {
            tx_count,
            unique_counterparties,
            counterparty_entropy,
            avg_value,
            std_value,
            max_value,
            min_value,
            value_range_ratio,
            identical_amount_ratio,
            self_transfer_ratio,
            circular_path_score,
            avg_time_between_tx,
            time_regularity,
            burst_score,
            night_ratio,
            weekend_ratio,
            tx_per_day,
            gas_efficiency,
            inflow_outflow_ratio,
            avg_block_gap,
            unique_values_ratio,
            small_tx_ratio,
            round_amount_ratio,
            activity_span_days,
        }
    }

    /// Convert features to normalized i32 vector in [0, 128] (fixed-point scale=7).
    pub fn to_normalized_vec(&self) -> Vec<i32> {
        let scale = 128.0;
        vec![
            clip_scale(self.tx_count, 0.0, 500.0, scale),
            clip_scale(self.unique_counterparties, 0.0, 200.0, scale),
            clip_scale(self.counterparty_entropy, 0.0, 5.3, scale),  // ln(200) ≈ 5.3
            clip_scale(self.avg_value, 0.0, 100_000_000.0, scale),   // 100 USDC
            clip_scale(self.std_value, 0.0, 100_000_000.0, scale),
            clip_scale(self.max_value, 0.0, 1_000_000_000.0, scale), // 1000 USDC
            clip_scale(self.min_value, 0.0, 100_000_000.0, scale),
            clip_scale(self.value_range_ratio, 0.0, 1.0, scale),
            clip_scale(self.identical_amount_ratio, 0.0, 1.0, scale),
            clip_scale(self.self_transfer_ratio, 0.0, 1.0, scale),
            clip_scale(self.circular_path_score, 0.0, 1.0, scale),
            clip_scale(self.avg_time_between_tx, 0.0, 86400.0, scale), // 1 day max
            clip_scale(self.time_regularity, 0.0, 3.0, scale),
            clip_scale(self.burst_score, 0.0, 100.0, scale),
            clip_scale(self.night_ratio, 0.0, 1.0, scale),
            clip_scale(self.weekend_ratio, 0.0, 1.0, scale),
            clip_scale(self.tx_per_day, 0.0, 100.0, scale),
            clip_scale(self.gas_efficiency, 0.0, 1.0, scale),
            clip_scale(self.inflow_outflow_ratio, 0.0, 1.0, scale),
            clip_scale(self.avg_block_gap, 0.0, 10000.0, scale),
            clip_scale(self.unique_values_ratio, 0.0, 1.0, scale),
            clip_scale(self.small_tx_ratio, 0.0, 1.0, scale),
            clip_scale(self.round_amount_ratio, 0.0, 1.0, scale),
            clip_scale(self.activity_span_days, 0.0, 365.0, scale),
        ]
    }

    /// Return a zero-valued feature set (for empty wallets).
    pub fn zeros() -> Self {
        Self {
            tx_count: 0.0,
            unique_counterparties: 0.0,
            counterparty_entropy: 0.0,
            avg_value: 0.0,
            std_value: 0.0,
            max_value: 0.0,
            min_value: 0.0,
            value_range_ratio: 0.0,
            identical_amount_ratio: 0.0,
            self_transfer_ratio: 0.0,
            circular_path_score: 0.0,
            avg_time_between_tx: 0.0,
            time_regularity: 0.0,
            burst_score: 0.0,
            night_ratio: 0.0,
            weekend_ratio: 0.0,
            tx_per_day: 0.0,
            gas_efficiency: 0.0,
            inflow_outflow_ratio: 0.5,
            avg_block_gap: 0.0,
            unique_values_ratio: 0.0,
            small_tx_ratio: 0.0,
            round_amount_ratio: 0.0,
            activity_span_days: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Clip value to [lo, hi] and scale to [0, scale].
fn clip_scale(value: f64, lo: f64, hi: f64, scale: f64) -> i32 {
    let clamped = value.max(lo).min(hi);
    let normalized = (clamped - lo) / (hi - lo);
    (normalized * scale) as i32
}

/// Compute circular path score: fraction of counterparties involved in A→B→C→A loops.
fn compute_circular_score(txs: &[X402Transaction], wallet: &str) -> f64 {
    // Build adjacency: who did the wallet send to?
    let mut outgoing: HashMap<String, HashSet<String>> = HashMap::new();

    for tx in txs {
        let from = tx.from.to_lowercase();
        let to = tx.to.to_lowercase();
        outgoing.entry(from).or_default().insert(to);
    }

    // Check for cycles: wallet → B → C → wallet
    let wallet_targets = match outgoing.get(wallet) {
        Some(targets) => targets.clone(),
        None => return 0.0,
    };

    let mut circular_counterparties = HashSet::new();
    for b in &wallet_targets {
        if let Some(b_targets) = outgoing.get(b.as_str()) {
            for c in b_targets {
                if let Some(c_targets) = outgoing.get(c.as_str()) {
                    if c_targets.contains(wallet) {
                        circular_counterparties.insert(b.clone());
                        circular_counterparties.insert(c.clone());
                    }
                }
            }
        }
    }

    let total_counterparties: HashSet<String> = txs
        .iter()
        .flat_map(|tx| {
            vec![tx.from.to_lowercase(), tx.to.to_lowercase()]
        })
        .filter(|addr| addr != wallet)
        .collect();

    if total_counterparties.is_empty() {
        0.0
    } else {
        circular_counterparties.len() as f64 / total_counterparties.len() as f64
    }
}

/// Compute burst score: max tx/minute ÷ avg tx/minute.
fn compute_burst_score(sorted_timestamps: &[u64]) -> f64 {
    if sorted_timestamps.len() < 2 {
        return 1.0;
    }

    let first = sorted_timestamps[0];
    let last = *sorted_timestamps.last().unwrap();
    let total_minutes = ((last - first) as f64 / 60.0).max(1.0);
    let avg_per_minute = sorted_timestamps.len() as f64 / total_minutes;

    // Count max txs in any single minute window
    let mut max_in_minute = 1u64;
    let mut window_start = 0;
    for i in 0..sorted_timestamps.len() {
        while sorted_timestamps[i] - sorted_timestamps[window_start] > 60 {
            window_start += 1;
        }
        let count = (i - window_start + 1) as u64;
        if count > max_in_minute {
            max_in_minute = count;
        }
    }

    if avg_per_minute > 0.0 {
        max_in_minute as f64 / avg_per_minute
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_round_trip() {
        for i in 0..5 {
            let class = IntegrityClassification::from_index(i);
            assert_eq!(class.index(), i);
        }
    }

    #[test]
    fn test_feature_extraction_empty() {
        let activity = WalletActivity {
            wallet_address: "0xabc".to_string(),
            chain_id: 8453,
            from_block: 0,
            to_block: 100,
            transactions: vec![],
        };
        let features = TransactionFeatures::extract(&activity);
        assert_eq!(features.tx_count, 0.0);
    }

    #[test]
    fn test_feature_normalization_bounds() {
        let features = TransactionFeatures {
            tx_count: 1000.0, // exceeds max of 500
            unique_counterparties: 50.0,
            counterparty_entropy: 3.0,
            avg_value: 50_000_000.0,
            std_value: 10_000_000.0,
            max_value: 100_000_000.0,
            min_value: 1_000_000.0,
            value_range_ratio: 0.9,
            identical_amount_ratio: 0.3,
            self_transfer_ratio: 0.1,
            circular_path_score: 0.0,
            avg_time_between_tx: 3600.0,
            time_regularity: 0.5,
            burst_score: 5.0,
            night_ratio: 0.25,
            weekend_ratio: 0.28,
            tx_per_day: 10.0,
            gas_efficiency: 0.5,
            inflow_outflow_ratio: 0.6,
            avg_block_gap: 100.0,
            unique_values_ratio: 0.8,
            small_tx_ratio: 0.1,
            round_amount_ratio: 0.5,
            activity_span_days: 30.0,
        };

        let vec = features.to_normalized_vec();
        assert_eq!(vec.len(), 24);
        for (i, &v) in vec.iter().enumerate() {
            assert!(v >= 0, "Feature {} value {} < 0", i, v);
            assert!(v <= 128, "Feature {} value {} > 128", i, v);
        }
    }

    #[test]
    fn test_clip_scale() {
        assert_eq!(clip_scale(0.0, 0.0, 1.0, 128.0), 0);
        assert_eq!(clip_scale(1.0, 0.0, 1.0, 128.0), 128);
        assert_eq!(clip_scale(0.5, 0.0, 1.0, 128.0), 64);
        assert_eq!(clip_scale(2.0, 0.0, 1.0, 128.0), 128); // clamped
        assert_eq!(clip_scale(-1.0, 0.0, 1.0, 128.0), 0);  // clamped
    }
}
