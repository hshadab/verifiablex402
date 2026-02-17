//! HTTP server for the x402 Transaction Integrity Analyzer.
//!
//! Provides REST API endpoints for wallet transaction integrity evaluation
//! with optional ZK proof generation and x402 payment gating.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use eyre::Result;
use governor::{Quota, RateLimiter};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};
use tower_http::cors::{Any, CorsLayer};

use crate::indexer::BaseIndexer;
use crate::models::tx_integrity::tx_integrity_model;
use crate::payment::PaymentVerifier;
use crate::receipt::{GuardrailReceipt, PaymentInfo};
use crate::transaction::{TransactionFeatures, WalletActivity};
use crate::hash_model_fn;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to
    pub bind_addr: SocketAddr,
    /// Maximum concurrent proof generations
    pub max_concurrent_proofs: usize,
    /// Whether to require proof generation
    pub require_proof: bool,
    /// Rate limit in requests per minute per IP (0 = no limit)
    pub rate_limit_rpm: u32,
    /// Base RPC URL for wallet scanning
    pub rpc_url: String,
    /// Whether to require x402 payment
    pub require_payment: bool,
    /// Payment amount in USDC smallest unit (default: 5000 = $0.005)
    pub payment_amount: String,
    /// Payment payee address (for on-chain verification)
    pub payment_payee: Option<String>,
    /// USDC contract address
    pub usdc_contract: Option<String>,
    /// Allowed CORS origins (None/empty = allow any)
    pub allowed_origins: Option<Vec<String>>,
    /// API keys for authentication (None/empty = no auth)
    pub api_keys: Option<Vec<String>>,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache entries
    pub cache_max_entries: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().unwrap(),
            max_concurrent_proofs: 4,
            require_proof: false,
            rate_limit_rpm: 60,
            rpc_url: "https://mainnet.base.org".to_string(),
            require_payment: false,
            payment_amount: "5000".to_string(),
            payment_payee: None,
            usdc_contract: None,
            allowed_origins: None,
            api_keys: None,
            cache_ttl_seconds: 300,
            cache_max_entries: 1000,
        }
    }
}

/// Request for integrity evaluation
#[derive(Debug, Deserialize)]
pub struct IntegrityRequest {
    /// The input for evaluation
    #[serde(flatten)]
    pub input: IntegrityInput,

    /// Whether to generate a ZK proof
    #[serde(default)]
    pub generate_proof: bool,

    /// Optional payment information
    #[serde(default)]
    pub payment: Option<PaymentInfo>,
}

/// Input for integrity evaluation — flexible like ClawGuard
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum IntegrityInput {
    /// Full wallet activity data
    WalletActivity {
        wallet_activity: WalletActivity,
    },
    /// Pre-computed 24-dim feature vector
    Features {
        features: Vec<i32>,
        #[serde(default = "default_wallet")]
        wallet_address: String,
        #[serde(default = "default_chain_id")]
        chain_id: u64,
    },
    /// Structured transaction features
    TransactionFeatures {
        transaction_features: TransactionFeatures,
        wallet_address: String,
        #[serde(default = "default_chain_id")]
        chain_id: u64,
    },
}

fn default_wallet() -> String {
    "unknown".to_string()
}
fn default_chain_id() -> u64 {
    8453
}

/// Request to scan a wallet by address
#[derive(Debug, Deserialize)]
pub struct ScanWalletRequest {
    /// Wallet address to scan
    pub wallet_address: String,
    /// Number of blocks to look back (default: ~7 days at 2s/block)
    #[serde(default = "default_lookback")]
    pub lookback_blocks: u64,
    /// Whether to generate a ZK proof
    #[serde(default)]
    pub generate_proof: bool,
    /// Optional payment information
    #[serde(default)]
    pub payment: Option<PaymentInfo>,
}

fn default_lookback() -> u64 {
    302400 // ~7 days at 2s/block
}

/// Response from integrity evaluation
#[derive(Debug, Serialize)]
pub struct IntegrityResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<GuardrailReceipt>,
    pub processing_time_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model_hash: String,
    pub model_name: String,
    pub model_params: usize,
    pub uptime_seconds: u64,
    pub rpc_connected: bool,
}

/// Type alias for per-IP rate limiters
type IpRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

/// Server state
pub struct ServerState {
    pub config: ServerConfig,
    pub model_hash: String,
    pub start_time: Instant,
    pub proof_semaphore: Semaphore,
    pub rate_limiters: Mutex<HashMap<std::net::IpAddr, Arc<IpRateLimiter>>>,
    pub indexer: BaseIndexer,
    pub payment_verifier: Option<PaymentVerifier>,
    pub cache: crate::cache::WalletCache,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        let model_hash = hash_model_fn(tx_integrity_model);
        let max_proofs = config.max_concurrent_proofs;
        let mut indexer = BaseIndexer::new(&config.rpc_url);
        if let Some(ref usdc) = config.usdc_contract {
            indexer = indexer.with_usdc_contract(usdc);
        }

        let payment_verifier = config.payment_payee.as_ref().map(|payee| {
            PaymentVerifier::new(
                &config.rpc_url,
                payee,
                config.usdc_contract.as_deref(),
            )
        });

        let cache = crate::cache::WalletCache::new(
            config.cache_ttl_seconds,
            config.cache_max_entries,
        );

        Self {
            config,
            model_hash,
            start_time: Instant::now(),
            proof_semaphore: Semaphore::new(max_proofs),
            rate_limiters: Mutex::new(HashMap::new()),
            indexer,
            payment_verifier,
            cache,
        }
    }

    pub async fn get_rate_limiter(&self, ip: std::net::IpAddr) -> Option<Arc<IpRateLimiter>> {
        if self.config.rate_limit_rpm == 0 {
            return None;
        }

        let mut limiters = self.rate_limiters.lock().await;

        if let Some(limiter) = limiters.get(&ip) {
            return Some(Arc::clone(limiter));
        }

        let quota = Quota::per_minute(NonZeroU32::new(self.config.rate_limit_rpm).unwrap());
        let limiter = Arc::new(RateLimiter::direct(quota));
        limiters.insert(ip, Arc::clone(&limiter));

        if limiters.len() > 10000 {
            tracing::warn!("rate limiter map exceeded 10000 entries, clearing");
            limiters.clear();
            limiters.insert(ip, Arc::clone(&limiter));
        }

        Some(limiter)
    }
}

/// Run the HTTP server
pub async fn run_server(config: ServerConfig) -> Result<()> {
    use axum::{
        middleware,
        routing::{get, post},
        Router,
    };
    use std::collections::HashSet;

    let rate_limit_rpm = config.rate_limit_rpm;
    let prometheus_handle = crate::metrics::install_prometheus_recorder();

    // Build API key set
    let api_keys: crate::auth::ApiKeySet = Arc::new(
        config
            .api_keys
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>(),
    );
    let has_api_keys = !api_keys.is_empty();

    let state = Arc::new(ServerState::new(config.clone()));

    // Build CORS layer
    let cors = match &config.allowed_origins {
        Some(origins) if !origins.is_empty() => {
            let allowed: Vec<axum::http::HeaderValue> = origins
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(allowed)
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                .allow_headers([
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderName::from_static("x-api-key"),
                ])
        }
        _ => CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers(Any),
    };

    // Protected routes (require API key when configured)
    let protected = Router::new()
        .route("/guardrail/integrity", post(integrity_handler))
        .route("/api/v1/scan", post(scan_handler))
        .route_layer(middleware::from_fn_with_state(
            api_keys.clone(),
            crate::auth::require_api_key,
        ))
        .with_state(state.clone());

    // Open routes (health, metrics)
    let open = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(move || {
            let handle = prometheus_handle.clone();
            async move { handle.render() }
        }))
        .with_state(state.clone());

    let app = Router::new()
        .merge(open)
        .merge(protected)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    tracing::info!("verifiablex402 server listening on {}", config.bind_addr);
    tracing::info!("Endpoints: GET /health, GET /metrics, POST /guardrail/integrity, POST /api/v1/scan");
    if rate_limit_rpm > 0 {
        tracing::info!(rate_limit_rpm, "rate limiting enabled");
    }
    if config.require_payment {
        tracing::info!("x402 payment enforcement enabled");
    }
    if has_api_keys {
        tracing::info!("API key authentication enabled");
    }

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Health check handler
async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    // Check RPC connectivity with 3s timeout
    let rpc_connected = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        state.indexer.current_block(),
    )
    .await
    .map(|r| r.is_ok())
    .unwrap_or(false);

    let status = if rpc_connected {
        "ok".to_string()
    } else {
        "degraded".to_string()
    };

    let response = HealthResponse {
        status,
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_hash: state.model_hash.clone(),
        model_name: "tx-integrity".to_string(),
        model_params: 2417,
        uptime_seconds: state.start_time.elapsed().as_secs(),
        rpc_connected,
    };
    axum::Json(response)
}

/// Integrity evaluation handler
async fn integrity_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    axum::Json(request): axum::Json<IntegrityRequest>,
) -> (StatusCode, axum::Json<IntegrityResponse>) {
    let start = Instant::now();
    let client_ip = addr.ip();

    // Check rate limit
    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            tracing::warn!(%client_ip, "rate limit exceeded");
            crate::metrics::record_rate_limit_hit();
            return (
                StatusCode::TOO_MANY_REQUESTS,
                axum::Json(IntegrityResponse {
                    success: false,
                    error: Some(format!(
                        "Rate limit exceeded. Maximum {} requests per minute.",
                        state.config.rate_limit_rpm
                    )),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }),
            );
        }
    }

    // Check payment requirement
    if state.config.require_payment {
        match &request.payment {
            None => {
                crate::metrics::record_payment_check("missing");
                return (
                    StatusCode::PAYMENT_REQUIRED,
                    axum::Json(IntegrityResponse {
                        success: false,
                        error: Some("x402 payment required".into()),
                        receipt: None,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }),
                );
            }
            Some(payment) if payment.tx_hash.is_empty() => {
                crate::metrics::record_payment_check("empty_hash");
                return (
                    StatusCode::PAYMENT_REQUIRED,
                    axum::Json(IntegrityResponse {
                        success: false,
                        error: Some("x402 payment required (empty tx_hash)".into()),
                        receipt: None,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }),
                );
            }
            Some(payment) => {
                if let Some(ref verifier) = state.payment_verifier {
                    match verifier.verify_payment(payment, &state.config.payment_amount).await {
                        Ok(true) => {
                            crate::metrics::record_payment_check("verified");
                        }
                        Ok(false) => {
                            crate::metrics::record_payment_check("failed");
                            return (
                                StatusCode::PAYMENT_REQUIRED,
                                axum::Json(IntegrityResponse {
                                    success: false,
                                    error: Some("payment verification failed: transfer not found or insufficient amount".into()),
                                    receipt: None,
                                    processing_time_ms: start.elapsed().as_millis() as u64,
                                }),
                            );
                        }
                        Err(e) => {
                            crate::metrics::record_payment_check("error");
                            tracing::warn!(error = %e, "payment verification error");
                            return (
                                StatusCode::BAD_GATEWAY,
                                axum::Json(IntegrityResponse {
                                    success: false,
                                    error: Some(format!("payment verification error: {}", e)),
                                    receipt: None,
                                    processing_time_ms: start.elapsed().as_millis() as u64,
                                }),
                            );
                        }
                    }
                }
            }
        }
    }

    // Extract features based on input type
    let (features, wallet_address, chain_id) = match &request.input {
        IntegrityInput::WalletActivity { wallet_activity } => {
            let features = TransactionFeatures::extract(wallet_activity);
            (
                features,
                wallet_activity.wallet_address.clone(),
                wallet_activity.chain_id,
            )
        }
        IntegrityInput::Features {
            features,
            wallet_address,
            chain_id,
        } => {
            if features.len() != 24 {
                return (
                    StatusCode::BAD_REQUEST,
                    axum::Json(IntegrityResponse {
                        success: false,
                        error: Some(format!("Expected 24 features, got {}", features.len())),
                        receipt: None,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }),
                );
            }
            let tf = features_from_vec(features);
            (tf, wallet_address.clone(), *chain_id)
        }
        IntegrityInput::TransactionFeatures {
            transaction_features,
            wallet_address,
            chain_id,
        } => (
            transaction_features.clone(),
            wallet_address.clone(),
            *chain_id,
        ),
    };

    tracing::info!(%wallet_address, %chain_id, "evaluating wallet integrity");

    // Run the guardrail
    match crate::run_guardrail(&features, &wallet_address, chain_id, request.generate_proof) {
        Ok((mut receipt, _proof_path)) => {
            let classification = receipt.evaluation.classification.clone();
            let decision = receipt.evaluation.decision.clone();
            let processing_time_ms = start.elapsed().as_millis() as u64;
            crate::metrics::record_evaluation(&classification, &decision, processing_time_ms);
            tracing::info!(
                %wallet_address,
                %classification,
                processing_time_ms,
                "evaluation complete"
            );
            if let Some(payment) = request.payment {
                receipt = receipt.with_payment(payment);
            }
            (
                StatusCode::OK,
                axum::Json(IntegrityResponse {
                    success: true,
                    error: None,
                    receipt: Some(receipt),
                    processing_time_ms,
                }),
            )
        }
        Err(e) => {
            tracing::warn!(%wallet_address, error = %e, "evaluation failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(IntegrityResponse {
                    success: false,
                    error: Some(format!("Evaluation failed: {}", e)),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }),
            )
        }
    }
}

/// Scan wallet handler — fetches transactions from Base and analyzes
async fn scan_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    axum::Json(request): axum::Json<ScanWalletRequest>,
) -> (StatusCode, axum::Json<IntegrityResponse>) {
    let start = Instant::now();
    let client_ip = addr.ip();

    // Check rate limit
    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            tracing::warn!(%client_ip, "rate limit exceeded on scan");
            return (
                StatusCode::TOO_MANY_REQUESTS,
                axum::Json(IntegrityResponse {
                    success: false,
                    error: Some("Rate limit exceeded".to_string()),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }),
            );
        }
    }

    // Check payment requirement
    if state.config.require_payment {
        match &request.payment {
            None => {
                return (
                    StatusCode::PAYMENT_REQUIRED,
                    axum::Json(IntegrityResponse {
                        success: false,
                        error: Some("x402 payment required".into()),
                        receipt: None,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }),
                );
            }
            Some(payment) if payment.tx_hash.is_empty() => {
                return (
                    StatusCode::PAYMENT_REQUIRED,
                    axum::Json(IntegrityResponse {
                        success: false,
                        error: Some("x402 payment required (empty tx_hash)".into()),
                        receipt: None,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }),
                );
            }
            Some(payment) => {
                if let Some(ref verifier) = state.payment_verifier {
                    match verifier.verify_payment(payment, &state.config.payment_amount).await {
                        Ok(true) => {}
                        Ok(false) => {
                            return (
                                StatusCode::PAYMENT_REQUIRED,
                                axum::Json(IntegrityResponse {
                                    success: false,
                                    error: Some("payment verification failed".into()),
                                    receipt: None,
                                    processing_time_ms: start.elapsed().as_millis() as u64,
                                }),
                            );
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "payment verification error");
                            return (
                                StatusCode::BAD_GATEWAY,
                                axum::Json(IntegrityResponse {
                                    success: false,
                                    error: Some(format!("payment verification error: {}", e)),
                                    receipt: None,
                                    processing_time_ms: start.elapsed().as_millis() as u64,
                                }),
                            );
                        }
                    }
                }
            }
        }
    }

    let wallet_address = &request.wallet_address;
    tracing::info!(%wallet_address, lookback_blocks = request.lookback_blocks, "scanning wallet");

    // Check cache first
    let cache_key = crate::cache::WalletCache::key(wallet_address, request.lookback_blocks);
    if let Some(cached_receipt) = state.cache.get(&cache_key) {
        crate::metrics::record_cache_hit();
        let processing_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(%wallet_address, "returning cached result");
        return (
            StatusCode::OK,
            axum::Json(IntegrityResponse {
                success: true,
                error: None,
                receipt: Some(cached_receipt),
                processing_time_ms,
            }),
        );
    }
    crate::metrics::record_cache_miss();

    // Fetch wallet activity from Base
    let activity = match state
        .indexer
        .scan_wallet(wallet_address, request.lookback_blocks)
        .await
    {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(%wallet_address, error = %e, "failed to fetch wallet data");
            return (
                StatusCode::BAD_GATEWAY,
                axum::Json(IntegrityResponse {
                    success: false,
                    error: Some(format!("Failed to fetch wallet data: {}", e)),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }),
            );
        }
    };

    let tx_count = activity.transactions.len();
    let features = TransactionFeatures::extract(&activity);

    match crate::run_guardrail(
        &features,
        wallet_address,
        8453,
        request.generate_proof,
    ) {
        Ok((mut receipt, _proof_path)) => {
            let classification = receipt.evaluation.classification.clone();
            let decision = receipt.evaluation.decision.clone();
            let processing_time_ms = start.elapsed().as_millis() as u64;
            crate::metrics::record_evaluation(&classification, &decision, processing_time_ms);
            tracing::info!(
                %wallet_address,
                tx_count,
                %classification,
                processing_time_ms,
                "scan complete"
            );

            // Cache the result
            state.cache.insert(cache_key, receipt.clone());

            if let Some(payment) = request.payment {
                receipt = receipt.with_payment(payment);
            }
            (
                StatusCode::OK,
                axum::Json(IntegrityResponse {
                    success: true,
                    error: None,
                    receipt: Some(receipt),
                    processing_time_ms,
                }),
            )
        }
        Err(e) => {
            tracing::warn!(%wallet_address, error = %e, "scan evaluation failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(IntegrityResponse {
                    success: false,
                    error: Some(format!("Evaluation failed: {}", e)),
                    receipt: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }),
            )
        }
    }
}

/// Convert a raw 24-dim feature vector back to TransactionFeatures
fn features_from_vec(vec: &[i32]) -> TransactionFeatures {
    // Since we can't perfectly invert the normalization, create a features struct
    // with normalized values that will map back to the same vector
    TransactionFeatures {
        tx_count: (vec.get(0).copied().unwrap_or(0) as f64 / 128.0) * 500.0,
        unique_counterparties: (vec.get(1).copied().unwrap_or(0) as f64 / 128.0) * 200.0,
        counterparty_entropy: (vec.get(2).copied().unwrap_or(0) as f64 / 128.0) * 5.3,
        avg_value: (vec.get(3).copied().unwrap_or(0) as f64 / 128.0) * 100_000_000.0,
        std_value: (vec.get(4).copied().unwrap_or(0) as f64 / 128.0) * 100_000_000.0,
        max_value: (vec.get(5).copied().unwrap_or(0) as f64 / 128.0) * 1_000_000_000.0,
        min_value: (vec.get(6).copied().unwrap_or(0) as f64 / 128.0) * 100_000_000.0,
        value_range_ratio: vec.get(7).copied().unwrap_or(0) as f64 / 128.0,
        identical_amount_ratio: vec.get(8).copied().unwrap_or(0) as f64 / 128.0,
        self_transfer_ratio: vec.get(9).copied().unwrap_or(0) as f64 / 128.0,
        circular_path_score: vec.get(10).copied().unwrap_or(0) as f64 / 128.0,
        avg_time_between_tx: (vec.get(11).copied().unwrap_or(0) as f64 / 128.0) * 86400.0,
        time_regularity: (vec.get(12).copied().unwrap_or(0) as f64 / 128.0) * 3.0,
        burst_score: (vec.get(13).copied().unwrap_or(0) as f64 / 128.0) * 100.0,
        night_ratio: vec.get(14).copied().unwrap_or(0) as f64 / 128.0,
        weekend_ratio: vec.get(15).copied().unwrap_or(0) as f64 / 128.0,
        tx_per_day: (vec.get(16).copied().unwrap_or(0) as f64 / 128.0) * 100.0,
        gas_efficiency: vec.get(17).copied().unwrap_or(0) as f64 / 128.0,
        inflow_outflow_ratio: vec.get(18).copied().unwrap_or(0) as f64 / 128.0,
        avg_block_gap: (vec.get(19).copied().unwrap_or(0) as f64 / 128.0) * 10000.0,
        unique_values_ratio: vec.get(20).copied().unwrap_or(0) as f64 / 128.0,
        small_tx_ratio: vec.get(21).copied().unwrap_or(0) as f64 / 128.0,
        round_amount_ratio: vec.get(22).copied().unwrap_or(0) as f64 / 128.0,
        activity_span_days: (vec.get(23).copied().unwrap_or(0) as f64 / 128.0) * 365.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            model_hash: "sha256:abc".to_string(),
            model_name: "tx-integrity".to_string(),
            model_params: 2417,
            uptime_seconds: 100,
            rpc_connected: true,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"model_params\":2417"));
        assert!(json.contains("\"rpc_connected\":true"));
    }
}
