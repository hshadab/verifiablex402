//! LRU wallet result cache for the verifiablex402 server.
//!
//! Caches `GuardrailReceipt` results keyed by `wallet_address:lookback_blocks`
//! to avoid redundant RPC fetches and evaluations for recently scanned wallets.

use std::sync::Arc;
use std::time::Duration;

use moka::sync::Cache;

use crate::receipt::GuardrailReceipt;

/// Cache for wallet scan results.
#[derive(Clone)]
pub struct WalletCache {
    inner: Arc<Cache<String, GuardrailReceipt>>,
}

impl WalletCache {
    /// Create a new wallet cache with the given TTL and max entries.
    pub fn new(ttl_seconds: u64, max_entries: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_entries)
            .time_to_live(Duration::from_secs(ttl_seconds))
            .build();

        Self {
            inner: Arc::new(cache),
        }
    }

    /// Build a cache key from wallet address and lookback blocks.
    pub fn key(wallet_address: &str, lookback_blocks: u64) -> String {
        format!("{}:{}", wallet_address.to_lowercase(), lookback_blocks)
    }

    /// Get a cached receipt if available.
    pub fn get(&self, key: &str) -> Option<GuardrailReceipt> {
        self.inner.get(key)
    }

    /// Insert a receipt into the cache.
    pub fn insert(&self, key: String, receipt: GuardrailReceipt) {
        self.inner.insert(key, receipt);
    }
}
