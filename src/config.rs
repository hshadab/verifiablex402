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
}

impl Config {
    /// Load config from the default path, falling back to defaults on any error.
    pub fn load() -> Self {
        let path = dirs::config_dir()
            .unwrap_or_default()
            .join("verifiablex402")
            .join("config.toml");
        match std::fs::read_to_string(&path) {
            Ok(content) => toml::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }
}
