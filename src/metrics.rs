//! Prometheus metrics for the verifiablex402 server.
//!
//! Provides counters and histograms for monitoring evaluations, RPC calls,
//! rate limiting, payment checks, and cache performance.

use metrics::{counter, histogram};

/// Record an evaluation result.
pub fn record_evaluation(classification: &str, decision: &str, duration_ms: u64) {
    counter!("evaluations_total", "classification" => classification.to_string(), "decision" => decision.to_string())
        .increment(1);
    histogram!("evaluation_duration_ms").record(duration_ms as f64);
}

/// Record an RPC call.
pub fn record_rpc_call(method: &str, success: bool, duration_ms: u64) {
    counter!("rpc_calls_total", "method" => method.to_string(), "success" => success.to_string())
        .increment(1);
    histogram!("rpc_call_duration_ms").record(duration_ms as f64);
}

/// Record a rate limit hit.
pub fn record_rate_limit_hit() {
    counter!("rate_limit_hits_total").increment(1);
}

/// Record a payment check result.
pub fn record_payment_check(result: &str) {
    counter!("payment_checks_total", "result" => result.to_string()).increment(1);
}

/// Record a cache hit.
pub fn record_cache_hit() {
    counter!("cache_hits_total").increment(1);
}

/// Record a cache miss.
pub fn record_cache_miss() {
    counter!("cache_misses_total").increment(1);
}

/// Install the Prometheus metrics exporter and return the recorder handle.
pub fn install_prometheus_recorder() -> metrics_exporter_prometheus::PrometheusHandle {
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    builder
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}
