//! API key authentication middleware for the verifiablex402 HTTP server.
//!
//! When API keys are configured, requests to protected endpoints must include
//! a valid key in the `X-API-Key` header. When no keys are configured,
//! all requests pass through (backward compatible).

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

/// Shared set of valid API keys.
pub type ApiKeySet = Arc<HashSet<String>>;

/// Axum middleware that checks for a valid API key in the `X-API-Key` header.
///
/// If the key set is empty, all requests are allowed (no auth mode).
/// Otherwise, requests without a valid key receive 401 Unauthorized.
pub async fn require_api_key(
    axum::extract::State(keys): axum::extract::State<ApiKeySet>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // No keys configured = pass-through
    if keys.is_empty() {
        return Ok(next.run(request).await);
    }

    let api_key = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if keys.contains(key) => Ok(next.run(request).await),
        Some(_) => {
            tracing::warn!("invalid API key provided");
            Err(StatusCode::UNAUTHORIZED)
        }
        None => {
            tracing::warn!("missing X-API-Key header");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}
