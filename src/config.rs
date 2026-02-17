//! Configuration file support for verifiablex402.
//!
//! Loads optional TOML config from `~/.config/verifiablex402/config.toml`.

use serde::Deserialize;

/// Application configuration loaded from TOML file.
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    /// Base RPC URL (e.g., "https://mainnet.base.org")
    pub rpc_url: Option<String>,
    /// Server bind address (e.g., "127.0.0.1:8080")
    pub bind: Option<String>,
    /// Rate limit in requests per minute per IP
    pub rate_limit_rpm: Option<u32>,
    /// Maximum concurrent proof generations
    pub max_concurrent_proofs: Option<usize>,
    /// Whether to require x402 payment
    pub require_payment: Option<bool>,
    /// USDC contract address on Base mainnet (defaults to canonical USDC)
    pub usdc_contract: Option<String>,
    /// Payment payee address (for payment verification)
    pub payment_payee: Option<String>,
    /// Allowed CORS origins (None/empty = allow any)
    pub allowed_origins: Option<Vec<String>>,
    /// API keys for authentication (None/empty = no auth)
    pub api_keys: Option<Vec<String>>,
    /// Cache TTL in seconds (default: 300)
    pub cache_ttl_seconds: Option<u64>,
    /// Maximum cache entries (default: 1000)
    pub cache_max_entries: Option<u64>,
}

impl Config {
    /// Load config from the default path, falling back to defaults on any error.
    pub fn load() -> Self {
        let path = dirs::config_dir()
            .unwrap_or_default()
            .join("verifiablex402")
            .join("config.toml");
        match std::fs::read_to_string(&path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => {
                    tracing::info!(path = %path.display(), "loaded config");
                    config
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to parse config, using defaults");
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!(path = %path.display(), "config file not found, using defaults");
                Self::default()
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read config, using defaults");
                Self::default()
            }
        }
    }
}
