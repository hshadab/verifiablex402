//! Integration tests for the verifiablex402 transaction integrity analyzer.
//!
//! Run with: cargo test -p verifiablex402 --test integration --release

use ark_bn254::Fr;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::KeccakTranscript;
use onnx_tracer::tensor::Tensor;
use zkml_jolt_core::jolt::JoltSNARK;

use verifiablex402::models::tx_integrity::tx_integrity_model;
use verifiablex402::receipt::{IntegrityClassScores, RECEIPT_VERSION};
use verifiablex402::transaction::{
    IntegrityClassification, IntegrityDecision, TransactionFeatures, WalletActivity, X402Transaction,
};

// ---------------------------------------------------------------------------
// Model output shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_model_output_shape() {
    let model = tx_integrity_model();
    let input = Tensor::new(Some(&[0i32; 24]), &[1, 24]).unwrap();
    let result = model.forward(&[input]).unwrap();

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(
        result.outputs[0].inner.len(),
        5,
        "Expected 5 output classes"
    );
}

#[test]
fn test_model_deterministic() {
    let model = tx_integrity_model();
    let input_data = vec![64i32; 24];

    let input1 = Tensor::new(Some(&input_data), &[1, 24]).unwrap();
    let result1 = model.forward(&[input1]).unwrap();
    let out1 = result1.outputs[0].inner.clone();

    let model2 = tx_integrity_model();
    let input2 = Tensor::new(Some(&input_data), &[1, 24]).unwrap();
    let result2 = model2.forward(&[input2]).unwrap();
    let out2 = result2.outputs[0].inner.clone();

    assert_eq!(out1, out2, "Model should be deterministic");
}

// ---------------------------------------------------------------------------
// Classification tests
// ---------------------------------------------------------------------------

#[test]
fn test_genuine_commerce_classification() {
    let model = tx_integrity_model();
    // Genuine: high tx count, high diversity, high entropy, varied values
    let mut input_vec = vec![0i32; 24];
    input_vec[0] = 80;   // tx_count (moderate-high)
    input_vec[1] = 100;  // unique_counterparties (high)
    input_vec[2] = 90;   // counterparty_entropy (high)
    input_vec[7] = 100;  // value_range_ratio (high variety)
    input_vec[20] = 90;  // unique_values_ratio (high)
    input_vec[23] = 64;  // activity_span_days (moderate)

    let input = Tensor::new(Some(&input_vec), &[1, 24]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let data = &result.outputs[0].inner;

    // Verify output shape
    assert_eq!(data.len(), 5, "Expected 5 output classes, got {:?}", data);
    let max_idx = data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .unwrap()
        .0;
    assert_eq!(max_idx, 0, "Genuine commerce features should classify as GENUINE_COMMERCE (0), got {}", max_idx);
}

#[test]
fn test_wash_trading_classification() {
    let model = tx_integrity_model();
    // Wash trading: identical amounts, low diversity, burst patterns
    let mut input_vec = vec![0i32; 24];
    input_vec[0] = 80;    // tx_count
    input_vec[1] = 10;    // unique_counterparties (low)
    input_vec[2] = 10;    // counterparty_entropy (low)
    input_vec[8] = 120;   // identical_amount_ratio (very high)
    input_vec[9] = 80;    // self_transfer_ratio (high)
    input_vec[10] = 100;  // circular_path_score (high)
    input_vec[13] = 100;  // burst_score (high)
    input_vec[14] = 90;   // night_ratio (high)
    input_vec[20] = 10;   // unique_values_ratio (low)
    input_vec[22] = 100;  // round_amount_ratio (high)

    let input = Tensor::new(Some(&input_vec), &[1, 24]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let data = &result.outputs[0].inner;

    assert_eq!(data.len(), 5, "Expected 5 output classes");
    let max_idx = data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .unwrap()
        .0;
    assert_eq!(max_idx, 4, "Wash trading features should classify as WASH_TRADING (4), got {}", max_idx);
}

#[test]
fn test_low_activity_classification() {
    let model = tx_integrity_model();
    // Low activity: few transactions
    let mut input_vec = vec![0i32; 24];
    input_vec[0] = 5;     // tx_count (very low)
    input_vec[1] = 3;     // unique_counterparties (very low)
    input_vec[16] = 2;    // tx_per_day (very low)
    input_vec[23] = 5;    // activity_span_days (low)

    let input = Tensor::new(Some(&input_vec), &[1, 24]).unwrap();
    let result = model.forward(&[input]).unwrap();
    let data = &result.outputs[0].inner;

    assert_eq!(data.len(), 5, "Expected 5 output classes");
}

// ---------------------------------------------------------------------------
// Feature extraction tests
// ---------------------------------------------------------------------------

#[test]
fn test_feature_extraction_with_transactions() {
    let activity = WalletActivity {
        wallet_address: "0xabc".to_string(),
        chain_id: 8453,
        from_block: 1000,
        to_block: 2000,
        transactions: vec![
            X402Transaction {
                tx_hash: "0x1".to_string(),
                from: "0xabc".to_string(),
                to: "0xdef".to_string(),
                value: 5_000_000, // 5 USDC
                timestamp: 1700000000,
                block_number: 1000,
                gas_used: 21000,
                gas_price: 1000000000,
            },
            X402Transaction {
                tx_hash: "0x2".to_string(),
                from: "0xdef".to_string(),
                to: "0xabc".to_string(),
                value: 10_000_000, // 10 USDC
                timestamp: 1700003600, // 1 hour later
                block_number: 1100,
                gas_used: 21000,
                gas_price: 1000000000,
            },
            X402Transaction {
                tx_hash: "0x3".to_string(),
                from: "0xabc".to_string(),
                to: "0x123".to_string(),
                value: 3_000_000, // 3 USDC
                timestamp: 1700007200, // 2 hours later
                block_number: 1200,
                gas_used: 21000,
                gas_price: 1000000000,
            },
        ],
    };

    let features = TransactionFeatures::extract(&activity);
    assert_eq!(features.tx_count, 3.0);
    assert_eq!(features.unique_counterparties, 2.0); // def, 123
    assert!(features.counterparty_entropy > 0.0);

    let vec = features.to_normalized_vec();
    assert_eq!(vec.len(), 24);
    for (i, &v) in vec.iter().enumerate() {
        assert!(v >= 0, "Feature {} value {} < 0", i, v);
        assert!(v <= 128, "Feature {} value {} > 128", i, v);
    }
}

#[test]
fn test_feature_normalization() {
    let features = TransactionFeatures::zeros();
    let vec = features.to_normalized_vec();

    assert_eq!(vec.len(), 24);
    // Zero features should produce mostly zeros (except inflow_outflow_ratio = 0.5)
    for (i, &v) in vec.iter().enumerate() {
        assert!(v >= 0, "Feature {} should be >= 0, got {}", i, v);
        assert!(v <= 128, "Feature {} should be <= 128, got {}", i, v);
    }
}

// ---------------------------------------------------------------------------
// Receipt construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_receipt_construction() {
    let features = TransactionFeatures::zeros();
    let scores = IntegrityClassScores {
        genuine_commerce: 0.6,
        low_activity: 0.2,
        scripted_benign: 0.1,
        circular_payments: 0.05,
        wash_trading: 0.05,
    };

    let receipt = verifiablex402::receipt::GuardrailReceipt::new_integrity_receipt(
        "0xdeadbeef",
        8453,
        &features,
        IntegrityClassification::GenuineCommerce,
        IntegrityDecision::Allow,
        "No issues detected",
        scores,
        0.6,
        "sha256:test".to_string(),
        "".to_string(),
        "".to_string(),
        None,
        None,
        [0u8; 32],
    );

    // Verify receipt schema
    assert_eq!(receipt.version, RECEIPT_VERSION);
    assert!(receipt.receipt_id.starts_with("gr_integrity_"));
    assert_eq!(receipt.guardrail.domain, "integrity");
    assert_eq!(receipt.guardrail.action_type, "evaluate_wallet");
    assert_eq!(receipt.evaluation.classification, "GENUINE_COMMERCE");
    assert_eq!(receipt.evaluation.decision, "allow");
    assert!(receipt.subject.uri.contains("eip155:8453"));
    assert!(receipt.subject.commitment.starts_with("sha256:"));
    assert!(!receipt.is_blocked());
    assert!(!receipt.is_flagged());

    // Verify JSON serialization round-trip
    let json = serde_json::to_string_pretty(&receipt).unwrap();
    let parsed: verifiablex402::receipt::GuardrailReceipt =
        serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.receipt_id, receipt.receipt_id);
    assert_eq!(parsed.evaluation.classification, "GENUINE_COMMERCE");
}

#[test]
fn test_receipt_deny() {
    let features = TransactionFeatures::zeros();
    let scores = IntegrityClassScores {
        genuine_commerce: 0.05,
        low_activity: 0.05,
        scripted_benign: 0.05,
        circular_payments: 0.1,
        wash_trading: 0.75,
    };

    let receipt = verifiablex402::receipt::GuardrailReceipt::new_integrity_receipt(
        "0xbadwallet",
        8453,
        &features,
        IntegrityClassification::WashTrading,
        IntegrityDecision::Deny,
        "Wash trading detected",
        scores,
        0.75,
        "sha256:test".to_string(),
        "".to_string(),
        "".to_string(),
        None,
        None,
        [0u8; 32],
    );

    assert!(receipt.is_blocked());
    assert!(!receipt.is_flagged());
    assert_eq!(receipt.evaluation.decision, "deny");
}

#[test]
fn test_receipt_flag() {
    let features = TransactionFeatures::zeros();
    let scores = IntegrityClassScores {
        genuine_commerce: 0.1,
        low_activity: 0.05,
        scripted_benign: 0.05,
        circular_payments: 0.6,
        wash_trading: 0.2,
    };

    let receipt = verifiablex402::receipt::GuardrailReceipt::new_integrity_receipt(
        "0xsuspect",
        8453,
        &features,
        IntegrityClassification::CircularPayments,
        IntegrityDecision::Flag,
        "Circular payments detected",
        scores,
        0.5,
        "sha256:test".to_string(),
        "".to_string(),
        "".to_string(),
        None,
        None,
        [0u8; 32],
    );

    assert!(!receipt.is_blocked());
    assert!(receipt.is_flagged());
    assert_eq!(receipt.evaluation.decision, "flag");
}

// ---------------------------------------------------------------------------
// Scores tests
// ---------------------------------------------------------------------------

#[test]
fn test_scores_from_raw() {
    let raw = [100i32, -50, 20, -30, 80];
    let scores = IntegrityClassScores::from_raw_scores(&raw);
    let arr = scores.to_array();

    // Scores should sum to approximately 1.0
    let total: f64 = arr.iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "Scores should sum to ~1.0, got {}",
        total
    );

    // All scores should be positive
    for (i, &s) in arr.iter().enumerate() {
        assert!(s >= 0.0, "Score {} should be >= 0, got {}", i, s);
        assert!(s <= 1.0, "Score {} should be <= 1, got {}", i, s);
    }
}

#[test]
fn test_scores_from_uniform() {
    // All same scores should give uniform distribution
    let raw = [10i32, 10, 10, 10, 10];
    let scores = IntegrityClassScores::from_raw_scores(&raw);
    let arr = scores.to_array();

    for &s in &arr {
        assert!(
            (s - 0.2).abs() < 0.01,
            "Uniform scores should give ~0.2 each, got {}",
            s
        );
    }
}

// ---------------------------------------------------------------------------
// Encoding tests
// ---------------------------------------------------------------------------

#[test]
fn test_encoding_validate_correct_length() {
    let features = vec![64i32; 24];
    let result = verifiablex402::encoding::validate_features(&features).unwrap();
    assert_eq!(result.len(), 24);
}

#[test]
fn test_encoding_validate_wrong_length() {
    let features = vec![64i32; 22];
    assert!(verifiablex402::encoding::validate_features(&features).is_err());
}

#[test]
fn test_encoding_clamps_values() {
    let mut features = vec![64i32; 24];
    features[0] = 200;
    features[1] = -10;

    let result = verifiablex402::encoding::validate_features(&features).unwrap();
    assert_eq!(result[0], 128, "Value over 128 should be clamped to 128");
    assert_eq!(result[1], 0, "Value under 0 should be clamped to 0");
}

// ---------------------------------------------------------------------------
// Enforcement tests
// ---------------------------------------------------------------------------

#[test]
fn test_enforcement_wash_trading_deny() {
    let scores = [0.1, 0.05, 0.05, 0.1, 0.7];
    let (decision, _reason) = verifiablex402::enforcement::derive_decision(
        IntegrityClassification::WashTrading,
        &scores,
        0.7,
    );
    assert_eq!(decision, IntegrityDecision::Deny);
}

#[test]
fn test_enforcement_genuine_allow() {
    let scores = [0.7, 0.1, 0.1, 0.05, 0.05];
    let (decision, _reason) = verifiablex402::enforcement::derive_decision(
        IntegrityClassification::GenuineCommerce,
        &scores,
        0.7,
    );
    assert_eq!(decision, IntegrityDecision::Allow);
}

#[test]
fn test_enforcement_circular_flag() {
    let scores = [0.1, 0.05, 0.05, 0.6, 0.2];
    let (decision, _reason) = verifiablex402::enforcement::derive_decision(
        IntegrityClassification::CircularPayments,
        &scores,
        0.5,
    );
    assert_eq!(decision, IntegrityDecision::Flag);
}

#[test]
fn test_enforcement_circular_escalate_to_deny() {
    let scores = [0.05, 0.02, 0.03, 0.85, 0.05];
    let (decision, _reason) = verifiablex402::enforcement::derive_decision(
        IntegrityClassification::CircularPayments,
        &scores,
        0.90,
    );
    assert_eq!(decision, IntegrityDecision::Deny);
}

// ---------------------------------------------------------------------------
// Classification enum tests
// ---------------------------------------------------------------------------

#[test]
fn test_classification_round_trip() {
    for i in 0..5 {
        let class = IntegrityClassification::from_index(i);
        assert_eq!(class.index(), i);
        assert!(!class.as_str().is_empty());
    }
}

#[test]
fn test_classification_decisions() {
    assert!(IntegrityClassification::WashTrading.is_deny());
    assert!(!IntegrityClassification::WashTrading.is_flag());
    assert!(IntegrityClassification::CircularPayments.is_flag());
    assert!(!IntegrityClassification::CircularPayments.is_deny());
    assert!(!IntegrityClassification::GenuineCommerce.is_deny());
    assert!(!IntegrityClassification::GenuineCommerce.is_flag());
}

// ---------------------------------------------------------------------------
// Model hash test
// ---------------------------------------------------------------------------

#[test]
fn test_model_hash_deterministic() {
    let hash1 = verifiablex402::hash_model_fn(tx_integrity_model);
    let hash2 = verifiablex402::hash_model_fn(tx_integrity_model);
    assert_eq!(hash1, hash2);
    assert!(hash1.starts_with("sha256:"));
}

// ---------------------------------------------------------------------------
// Prove-verify integration test (JOLT Atlas round trip)
// ---------------------------------------------------------------------------

#[test]
fn test_tx_integrity_prove_verify() {
    type PCS = DoryCommitmentScheme;
    type Snark = JoltSNARK<Fr, PCS, KeccakTranscript>;

    let max_trace_length = 1 << 16;
    let preprocessing = Snark::prover_preprocess(tx_integrity_model, max_trace_length);

    let input = Tensor::new(Some(&[64i32; 24]), &[1, 24]).unwrap();
    let (snark, program_io, _debug_info) =
        Snark::prove(&preprocessing, tx_integrity_model, &input);

    let verifier_preprocessing = (&preprocessing).into();
    snark
        .verify(&verifier_preprocessing, program_io, None)
        .expect("proof should verify");
}

// ---------------------------------------------------------------------------
// GuardModel tests
// ---------------------------------------------------------------------------

#[test]
fn test_guard_model_properties() {
    let guard = verifiablex402::GuardModel::TxIntegrity;
    assert_eq!(guard.input_width(), 24);
    assert_eq!(guard.max_trace_length(), 1 << 16);
    assert_eq!(guard.name(), "tx-integrity");
    assert_eq!(guard.labels().len(), 5);
}

// ---------------------------------------------------------------------------
// Payment verification tests (struct validation level)
// ---------------------------------------------------------------------------

#[test]
fn test_payment_info_struct_validation() {
    use verifiablex402::receipt::PaymentInfo;

    let payment = PaymentInfo {
        network: "eip155:8453".to_string(),
        asset: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913".to_string(),
        amount: "5000".to_string(),
        payer: "0xabc".to_string(),
        payee: "0xdef".to_string(),
        tx_hash: "0x123".to_string(),
        scheme: "exact".to_string(),
    };

    assert_eq!(payment.network, "eip155:8453");
    assert!(!payment.tx_hash.is_empty());
}

#[test]
fn test_payment_info_empty_tx_hash() {
    use verifiablex402::receipt::PaymentInfo;

    let payment = PaymentInfo {
        network: "eip155:8453".to_string(),
        asset: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913".to_string(),
        amount: "5000".to_string(),
        payer: "0xabc".to_string(),
        payee: "0xdef".to_string(),
        tx_hash: "".to_string(),
        scheme: "exact".to_string(),
    };

    assert!(payment.tx_hash.is_empty());
}

// ---------------------------------------------------------------------------
// API key middleware tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_api_key_middleware_rejects_without_key() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use std::collections::HashSet;
    use std::sync::Arc;
    use tower::ServiceExt;

    let keys: verifiablex402::auth::ApiKeySet = Arc::new(
        vec!["test-key-123".to_string()]
            .into_iter()
            .collect::<HashSet<_>>(),
    );

    let app = Router::new()
        .route("/protected", get(|| async { "ok" }))
        .route_layer(middleware::from_fn_with_state(
            keys.clone(),
            verifiablex402::auth::require_api_key,
        ))
        .with_state(keys);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_api_key_middleware_accepts_valid_key() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use std::collections::HashSet;
    use std::sync::Arc;
    use tower::ServiceExt;

    let keys: verifiablex402::auth::ApiKeySet = Arc::new(
        vec!["test-key-123".to_string()]
            .into_iter()
            .collect::<HashSet<_>>(),
    );

    let app = Router::new()
        .route("/protected", get(|| async { "ok" }))
        .route_layer(middleware::from_fn_with_state(
            keys.clone(),
            verifiablex402::auth::require_api_key,
        ))
        .with_state(keys);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("x-api-key", "test-key-123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_api_key_middleware_passthrough_when_no_keys() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use std::collections::HashSet;
    use std::sync::Arc;
    use tower::ServiceExt;

    let keys: verifiablex402::auth::ApiKeySet =
        Arc::new(HashSet::<String>::new());

    let app = Router::new()
        .route("/protected", get(|| async { "ok" }))
        .route_layer(middleware::from_fn_with_state(
            keys.clone(),
            verifiablex402::auth::require_api_key,
        ))
        .with_state(keys);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Health endpoint test
// ---------------------------------------------------------------------------

#[test]
fn test_health_response_shape() {
    use verifiablex402::server::HealthResponse;

    let response = HealthResponse {
        status: "ok".to_string(),
        version: "0.1.0".to_string(),
        model_hash: "sha256:abc".to_string(),
        model_name: "tx-integrity".to_string(),
        model_params: 2417,
        uptime_seconds: 42,
        rpc_connected: true,
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["model_params"], 2417);
    assert_eq!(json["rpc_connected"], true);
    assert!(json["model_hash"].as_str().unwrap().starts_with("sha256:"));
}

// ---------------------------------------------------------------------------
// Cache tests
// ---------------------------------------------------------------------------

#[test]
fn test_wallet_cache_key_format() {
    let key = verifiablex402::cache::WalletCache::key("0xABC", 302400);
    assert_eq!(key, "0xabc:302400");
}

#[test]
fn test_wallet_cache_insert_and_get() {
    let cache = verifiablex402::cache::WalletCache::new(300, 100);
    let features = TransactionFeatures::zeros();
    let scores = IntegrityClassScores {
        genuine_commerce: 0.8,
        low_activity: 0.05,
        scripted_benign: 0.05,
        circular_payments: 0.05,
        wash_trading: 0.05,
    };

    let receipt = verifiablex402::receipt::GuardrailReceipt::new_integrity_receipt(
        "0xtest",
        8453,
        &features,
        IntegrityClassification::GenuineCommerce,
        IntegrityDecision::Allow,
        "test",
        scores,
        0.8,
        "sha256:test".to_string(),
        "".to_string(),
        "".to_string(),
        None,
        None,
        [0u8; 32],
    );

    cache.insert("test:302400".to_string(), receipt.clone());
    let cached = cache.get("test:302400");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().evaluation.classification, "GENUINE_COMMERCE");

    // Miss for different key
    assert!(cache.get("other:302400").is_none());
}
