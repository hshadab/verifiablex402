//! verifiablex402 — x402 Transaction Integrity Analyzer
//!
//! Classifies wallet transaction patterns on Base mainnet into 5 categories
//! via a small MLP, proves the classification with JOLT Atlas zkML,
//! and outputs Guardrail Receipts.
//!
//! # Classification Categories
//!
//! - **GENUINE_COMMERCE**: Normal merchant/service payments
//! - **LOW_ACTIVITY**: Wallet with very few transactions
//! - **SCRIPTED_BENIGN**: Automated but legitimate usage
//! - **CIRCULAR_PAYMENTS**: Funds cycling through intermediaries
//! - **WASH_TRADING**: Fake volume with bot-like patterns

pub mod config;
pub mod encoding;
pub mod enforcement;
pub mod indexer;
pub mod models;
pub mod proving;
pub mod receipt;
pub mod server;
pub mod transaction;

use eyre::{bail, Result};
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::enforcement::derive_decision;
use crate::models::tx_integrity::tx_integrity_model;
use crate::receipt::{generate_nonce, GuardrailReceipt, IntegrityClassScores};
use crate::transaction::{IntegrityClassification, TransactionFeatures};

// ---------------------------------------------------------------------------
// GuardModel enum
// ---------------------------------------------------------------------------

pub enum GuardModel {
    TxIntegrity,
}

impl GuardModel {
    pub fn model_fn(&self) -> fn() -> Model {
        match self {
            Self::TxIntegrity => tx_integrity_model,
        }
    }

    pub fn labels(&self) -> Vec<String> {
        match self {
            Self::TxIntegrity => vec![
                "GENUINE_COMMERCE".into(),
                "LOW_ACTIVITY".into(),
                "SCRIPTED_BENIGN".into(),
                "CIRCULAR_PAYMENTS".into(),
                "WASH_TRADING".into(),
            ],
        }
    }

    pub fn model_hash(&self) -> String {
        hash_model_fn(self.model_fn())
    }

    pub fn input_width(&self) -> usize {
        match self {
            Self::TxIntegrity => 24,
        }
    }

    pub fn max_trace_length(&self) -> usize {
        match self {
            Self::TxIntegrity => 1 << 16,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::TxIntegrity => "tx-integrity",
        }
    }
}

// ---------------------------------------------------------------------------
// Model hash
// ---------------------------------------------------------------------------

/// Version prefix for built-in model hashes.
const MODEL_HASH_VERSION: &str = "v1";

pub fn hash_model_fn(model_fn: fn() -> Model) -> String {
    let model = model_fn();
    let bytecode = onnx_tracer::decode_model(model);
    let serialized =
        serde_json::to_vec(&bytecode).unwrap_or_else(|_| format!("{:?}", bytecode).into_bytes());
    let mut hasher = Sha256::new();
    hasher.update(MODEL_HASH_VERSION.as_bytes());
    hasher.update(&serialized);
    let hash = hasher.finalize();
    format!("sha256:{}", hex::encode(hash))
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".verifiablex402")
}

pub fn proof_dir() -> PathBuf {
    config_dir().join("proofs")
}

// ---------------------------------------------------------------------------
// Core guardrail runner
// ---------------------------------------------------------------------------

/// Run the transaction integrity guardrail on a feature vector.
///
/// Returns (classification, confidence, model_hash, receipt, optional proof_path).
pub fn run_guardrail(
    features: &TransactionFeatures,
    wallet_address: &str,
    chain_id: u64,
    generate_proof: bool,
) -> Result<(GuardrailReceipt, Option<PathBuf>)> {
    let guard = GuardModel::TxIntegrity;
    let model_fn = guard.model_fn();
    let model_hash = guard.model_hash();

    // Encode features
    let feature_vec = encoding::encode_transaction_features(features);
    let input = Tensor::new(Some(&feature_vec), &[1, 24])
        .map_err(|e| eyre::eyre!("tensor error: {:?}", e))?;

    // Forward pass
    let model = model_fn();
    let result = model
        .forward(std::slice::from_ref(&input))
        .map_err(|e| eyre::eyre!("forward error: {}", e))?;
    let data = &result.outputs[0].inner;

    if data.len() < 5 {
        bail!("Expected 5 output classes, got {}", data.len());
    }

    let raw_scores: [i32; 5] = [data[0], data[1], data[2], data[3], data[4]];

    // Argmax
    let (best_idx, &best_val) = data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .unwrap();

    // Confidence: margin over runner-up
    let runner_up = data
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_idx)
        .map(|(_, v)| *v)
        .max()
        .unwrap_or(0);
    let margin = (best_val - runner_up).abs();
    let confidence = (margin as f64 / 128.0).min(1.0);

    let classification = IntegrityClassification::from_index(best_idx);
    let scores = IntegrityClassScores::from_raw_scores(&raw_scores);
    let scores_array = scores.to_array();
    let (decision, reasoning) = derive_decision(classification, &scores_array, confidence);

    // Generate proof if requested
    let (proof_bytes_str, vk_hash, prove_time, program_io_str, proof_path) = if generate_proof {
        let proof_directory = proof_dir();
        let max_trace_length = guard.max_trace_length();

        let input_for_proof = Tensor::new(Some(&feature_vec), &[1, 24])
            .map_err(|e| eyre::eyre!("tensor error: {:?}", e))?;

        let (path, _program_io) = proving::prove_and_save(
            model_fn,
            &input_for_proof,
            &proof_directory,
            &model_hash,
            max_trace_length,
            guard.name(),
        )?;

        // Read proof file to extract proof bytes and program_io
        let proof_content = std::fs::read_to_string(&path)?;
        let proof_json: serde_json::Value = serde_json::from_str(&proof_content)?;

        let pb = proof_json
            .get("proof")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let pio = proof_json
            .get("program_io")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        (pb, model_hash.clone(), None, pio, Some(path))
    } else {
        ("".to_string(), "".to_string(), None, None, None)
    };

    let nonce = generate_nonce();

    let receipt = GuardrailReceipt::new_integrity_receipt(
        wallet_address,
        chain_id,
        features,
        classification,
        decision,
        &reasoning,
        scores,
        confidence,
        model_hash,
        proof_bytes_str,
        vk_hash,
        prove_time,
        program_io_str,
        nonce,
    );

    Ok((receipt, proof_path))
}

/// Check if a decision label string represents a deny decision.
pub fn is_deny_decision(label: &str) -> bool {
    enforcement::is_deny_decision(label)
}
