//! Guardrail Receipt schema for verifiable transaction integrity evaluations.
//!
//! Receipts provide cryptographic proof that a wallet was evaluated by a specific
//! model, with the evaluation result binding to the input features.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::transaction::{IntegrityClassification, IntegrityDecision, TransactionFeatures};

/// Version of the receipt schema
pub const RECEIPT_VERSION: &str = "1.0.0";

/// A complete guardrail receipt for transaction integrity evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailReceipt {
    /// Schema version
    pub version: String,

    /// Unique receipt identifier
    pub receipt_id: String,

    /// Timestamp of evaluation
    pub timestamp: DateTime<Utc>,

    /// Guardrail metadata
    pub guardrail: GuardrailInfo,

    /// Subject being evaluated
    pub subject: Subject,

    /// Evaluation results
    pub evaluation: Evaluation,

    /// Cryptographic proof
    pub proof: ProofInfo,

    /// Optional payment information (for x402 integration)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment: Option<PaymentInfo>,

    /// Nonce for uniqueness
    pub nonce: String,

    /// Additional metadata
    #[serde(default)]
    pub metadata: ReceiptMetadata,
}

/// Information about the guardrail that produced this receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailInfo {
    /// Domain of the guardrail
    pub domain: String,

    /// Type of action being gated
    pub action_type: String,

    /// Policy identifier
    pub policy_id: String,

    /// Hash of the model used
    pub model_hash: String,
}

/// Subject being evaluated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    /// SHA-256 commitment to the input features
    pub commitment: String,

    /// Human-readable description
    pub description: String,

    /// URI identifying the subject
    pub uri: String,
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// Decision: allow, deny, or flag
    pub decision: String,

    /// Classification label
    pub classification: String,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Scores for each class
    pub scores: IntegrityClassScores,

    /// Human-readable reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// Scores for each integrity class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityClassScores {
    #[serde(rename = "GENUINE_COMMERCE")]
    pub genuine_commerce: f64,
    #[serde(rename = "LOW_ACTIVITY")]
    pub low_activity: f64,
    #[serde(rename = "SCRIPTED_BENIGN")]
    pub scripted_benign: f64,
    #[serde(rename = "CIRCULAR_PAYMENTS")]
    pub circular_payments: f64,
    #[serde(rename = "WASH_TRADING")]
    pub wash_trading: f64,
}

impl IntegrityClassScores {
    pub fn from_raw_scores(raw: &[i32; 5]) -> Self {
        // Convert raw i32 scores to normalized probabilities using softmax
        let scaled: Vec<f64> = raw.iter().map(|&x| (x as f64) / 128.0).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let total: f64 = exp_vals.iter().sum();

        if total == 0.0 || !total.is_finite() {
            return Self {
                genuine_commerce: 0.2,
                low_activity: 0.2,
                scripted_benign: 0.2,
                circular_payments: 0.2,
                wash_trading: 0.2,
            };
        }

        Self {
            genuine_commerce: exp_vals[0] / total,
            low_activity: exp_vals[1] / total,
            scripted_benign: exp_vals[2] / total,
            circular_payments: exp_vals[3] / total,
            wash_trading: exp_vals[4] / total,
        }
    }

    pub fn to_array(&self) -> [f64; 5] {
        [
            self.genuine_commerce,
            self.low_activity,
            self.scripted_benign,
            self.circular_payments,
            self.wash_trading,
        ]
    }
}

/// Proof information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofInfo {
    /// Proof system used
    pub system: String,

    /// Base64-encoded proof bytes
    pub proof_bytes: String,

    /// Hash of the verification key
    pub verification_key_hash: String,

    /// Time to generate proof in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prove_time_ms: Option<u64>,

    /// Serialized program IO for proof verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_io: Option<String>,
}

/// Payment information (for x402 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentInfo {
    /// Network identifier (e.g., "eip155:8453" for Base)
    pub network: String,

    /// Asset contract address (e.g., USDC on Base)
    pub asset: String,

    /// Amount in smallest unit (e.g., "5000" for 0.005 USDC)
    pub amount: String,

    /// Payer address
    pub payer: String,

    /// Payee address
    pub payee: String,

    /// Transaction hash
    pub tx_hash: String,

    /// Payment scheme
    pub scheme: String,
}

/// Additional receipt metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReceiptMetadata {
    /// Version of the prover service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prover_version: Option<String>,

    /// Version of JOLT Atlas
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jolt_atlas_version: Option<String>,

    /// Number of input features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_features_count: Option<usize>,

    /// Number of model parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_params_count: Option<usize>,
}

impl GuardrailReceipt {
    /// Create a new receipt for a transaction integrity evaluation
    pub fn new_integrity_receipt(
        wallet_address: &str,
        chain_id: u64,
        features: &TransactionFeatures,
        classification: IntegrityClassification,
        decision: IntegrityDecision,
        reasoning: &str,
        scores: IntegrityClassScores,
        confidence: f64,
        model_hash: String,
        proof_bytes: String,
        vk_hash: String,
        prove_time_ms: Option<u64>,
        program_io: Option<String>,
        nonce: [u8; 32],
    ) -> Self {
        let receipt_id = generate_receipt_id();
        let feature_vec = features.to_normalized_vec();
        let commitment = compute_commitment(&feature_vec);

        Self {
            version: RECEIPT_VERSION.to_string(),
            receipt_id,
            timestamp: Utc::now(),
            guardrail: GuardrailInfo {
                domain: "integrity".to_string(),
                action_type: "evaluate_wallet".to_string(),
                policy_id: "icme:tx-integrity-v1".to_string(),
                model_hash,
            },
            subject: Subject {
                commitment,
                description: format!(
                    "Wallet transaction integrity: {} on chain {}",
                    wallet_address, chain_id
                ),
                uri: format!("eip155:{}/address/{}", chain_id, wallet_address),
            },
            evaluation: Evaluation {
                decision: decision.as_str().to_string(),
                classification: classification.as_str().to_string(),
                confidence,
                scores,
                reasoning: Some(reasoning.to_string()),
            },
            proof: ProofInfo {
                system: "jolt-atlas".to_string(),
                proof_bytes,
                verification_key_hash: vk_hash,
                prove_time_ms,
                program_io,
            },
            payment: None,
            nonce: hex::encode(nonce),
            metadata: ReceiptMetadata {
                prover_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                jolt_atlas_version: Some("0.5.0".to_string()),
                input_features_count: Some(24),
                model_params_count: Some(2417),
            },
        }
    }

    /// Add payment information to the receipt
    pub fn with_payment(mut self, payment: PaymentInfo) -> Self {
        self.payment = Some(payment);
        self
    }

    /// Verify the receipt's input commitment matches the provided features
    pub fn verify_commitment(&self, features: &TransactionFeatures) -> bool {
        let feature_vec = features.to_normalized_vec();
        let expected = compute_commitment(&feature_vec);
        self.subject.commitment == expected
    }

    /// Check if this receipt indicates the wallet should be blocked
    pub fn is_blocked(&self) -> bool {
        self.evaluation.decision == "deny"
    }

    /// Check if this receipt indicates the wallet needs review
    pub fn is_flagged(&self) -> bool {
        self.evaluation.decision == "flag"
    }
}

/// Generate a unique receipt ID
fn generate_receipt_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut hasher = Sha256::new();
    hasher.update(timestamp.to_le_bytes());
    hasher.update(&rand_bytes());
    let hash = hasher.finalize();

    format!("gr_integrity_{}", &hex::encode(&hash[..8]))
}

/// Compute SHA-256 commitment to feature vector
fn compute_commitment(features: &[i32]) -> String {
    let canonical = serde_json::to_string(features).unwrap_or_default();

    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let hash = hasher.finalize();

    format!("sha256:{}", hex::encode(hash))
}

/// Generate cryptographically secure random bytes.
fn rand_bytes() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
    bytes
}

/// Generate a cryptographically secure random 32-byte nonce.
pub fn generate_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce);
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_serialization() {
        let scores = IntegrityClassScores {
            genuine_commerce: 0.1,
            low_activity: 0.05,
            scripted_benign: 0.05,
            circular_payments: 0.1,
            wash_trading: 0.7,
        };

        let features = TransactionFeatures::zeros();

        let receipt = GuardrailReceipt::new_integrity_receipt(
            "0x1234567890abcdef",
            8453,
            &features,
            IntegrityClassification::WashTrading,
            IntegrityDecision::Deny,
            "Wash trading detected",
            scores,
            0.7,
            "sha256:abc123".to_string(),
            "base64proof".to_string(),
            "sha256:vk123".to_string(),
            Some(1500),
            Some("{\"inputs\":[],\"outputs\":[]}".to_string()),
            [0u8; 32],
        );

        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("gr_integrity_"));
        assert!(json.contains("WASH_TRADING"));
        assert!(json.contains("deny"));
        assert!(json.contains("eip155:8453/address/"));

        // Verify it can be deserialized back
        let parsed: GuardrailReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.evaluation.classification, "WASH_TRADING");
        assert_eq!(parsed.guardrail.domain, "integrity");
        assert_eq!(parsed.guardrail.action_type, "evaluate_wallet");
    }

    #[test]
    fn test_commitment_verification() {
        let features = TransactionFeatures::zeros();
        let scores = IntegrityClassScores {
            genuine_commerce: 0.6,
            low_activity: 0.2,
            scripted_benign: 0.1,
            circular_payments: 0.05,
            wash_trading: 0.05,
        };

        let receipt = GuardrailReceipt::new_integrity_receipt(
            "0xdeadbeef",
            8453,
            &features,
            IntegrityClassification::GenuineCommerce,
            IntegrityDecision::Allow,
            "No issues detected",
            scores,
            0.6,
            "sha256:def456".to_string(),
            "proof".to_string(),
            "sha256:vk456".to_string(),
            None,
            None,
            generate_nonce(),
        );

        // Verify with same features should pass
        assert!(receipt.verify_commitment(&features));
    }

    #[test]
    fn test_scores_from_raw() {
        let raw = [100i32, -50, 20, -30, 80];
        let scores = IntegrityClassScores::from_raw_scores(&raw);
        let arr = scores.to_array();
        let total: f64 = arr.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "Scores should sum to ~1.0, got {}", total);
    }
}
