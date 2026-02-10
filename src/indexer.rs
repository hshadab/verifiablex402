//! Base Mainnet indexer for x402 wallet transactions.
//!
//! Uses raw JSON-RPC calls to fetch ERC-3009 TransferWithAuthorization events
//! from the USDC contract on Base (chain ID 8453).

use eyre::{Result, WrapErr};

use crate::transaction::{WalletActivity, X402Transaction};

/// USDC contract address on Base mainnet
const USDC_BASE: &str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";

/// ERC-3009 TransferWithAuthorization event topic
/// keccak256("TransferWithAuthorization(address,address,uint256,uint256,uint256,bytes32,bytes)")
const _TRANSFER_WITH_AUTH_TOPIC: &str =
    "0xe3eccac9a1e29c3b8e04e2e8d5c2b06f9d609fb2e548d99642ae9d3d5d8c3c9e";

/// Standard ERC-20 Transfer event topic
/// keccak256("Transfer(address,address,uint256)")
const TRANSFER_TOPIC: &str =
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";

/// Default Base mainnet RPC
const DEFAULT_RPC: &str = "https://mainnet.base.org";

/// Base Mainnet indexer for fetching wallet transaction history.
pub struct BaseIndexer {
    rpc_url: String,
    client: reqwest::Client,
}

impl BaseIndexer {
    /// Create a new indexer with the given RPC URL.
    pub fn new(rpc_url: &str) -> Self {
        Self {
            rpc_url: rpc_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new indexer with the default Base mainnet RPC.
    pub fn default_rpc() -> Self {
        Self::new(DEFAULT_RPC)
    }

    /// Get the current block number.
    pub async fn current_block(&self) -> Result<u64> {
        let resp = self
            .rpc_call("eth_blockNumber", serde_json::json!([]))
            .await?;
        let hex_str = resp
            .as_str()
            .ok_or_else(|| eyre::eyre!("invalid block number response"))?;
        parse_hex_u64(hex_str)
    }

    /// Fetch USDC Transfer events involving the given wallet address.
    ///
    /// Looks for ERC-20 Transfer events on the USDC contract where the wallet
    /// is either the sender or receiver.
    pub async fn fetch_wallet_transactions(
        &self,
        wallet: &str,
        from_block: u64,
        to_block: u64,
    ) -> Result<Vec<X402Transaction>> {
        let wallet_padded = pad_address(wallet);

        // Fetch transfers FROM the wallet
        let from_logs = self
            .get_logs(
                USDC_BASE,
                TRANSFER_TOPIC,
                Some(&wallet_padded),
                None,
                from_block,
                to_block,
            )
            .await
            .unwrap_or_default();

        // Fetch transfers TO the wallet
        let to_logs = self
            .get_logs(
                USDC_BASE,
                TRANSFER_TOPIC,
                None,
                Some(&wallet_padded),
                from_block,
                to_block,
            )
            .await
            .unwrap_or_default();

        let mut txs = Vec::new();
        let mut seen_hashes = std::collections::HashSet::new();

        for log in from_logs.iter().chain(to_logs.iter()) {
            if let Some(tx) = parse_transfer_log(log) {
                if seen_hashes.insert(tx.tx_hash.clone()) {
                    txs.push(tx);
                }
            }
        }

        txs.sort_by_key(|tx| tx.block_number);

        // Backfill missing data from RPC
        self.backfill_timestamps(&mut txs).await?;
        self.backfill_gas(&mut txs).await?;

        Ok(txs)
    }

    /// Get the timestamp for a given block number.
    pub async fn get_block_timestamp(&self, block_number: u64) -> Result<u64> {
        let hex_block = format!("0x{:x}", block_number);
        let resp = self
            .rpc_call("eth_getBlockByNumber", serde_json::json!([hex_block, false]))
            .await?;
        let timestamp_hex = resp
            .get("timestamp")
            .and_then(|v| v.as_str())
            .ok_or_else(|| eyre::eyre!("missing timestamp in block response"))?;
        parse_hex_u64(timestamp_hex)
    }

    /// Get the gas receipt for a transaction (gasUsed, effectiveGasPrice).
    pub async fn get_tx_receipt(&self, tx_hash: &str) -> Result<(u64, u64)> {
        let resp = self
            .rpc_call(
                "eth_getTransactionReceipt",
                serde_json::json!([tx_hash]),
            )
            .await?;
        let gas_used = resp
            .get("gasUsed")
            .and_then(|v| v.as_str())
            .map(|s| parse_hex_u64(s).unwrap_or(0))
            .unwrap_or(0);
        let gas_price = resp
            .get("effectiveGasPrice")
            .and_then(|v| v.as_str())
            .map(|s| parse_hex_u64(s).unwrap_or(0))
            .unwrap_or(0);
        Ok((gas_used, gas_price))
    }

    /// Backfill block timestamps for transactions that have timestamp == 0.
    pub async fn backfill_timestamps(&self, txs: &mut [X402Transaction]) -> Result<()> {
        use std::collections::HashMap;
        let mut cache: HashMap<u64, u64> = HashMap::new();

        // Collect unique block numbers
        for tx in txs.iter() {
            if tx.timestamp == 0 {
                cache.entry(tx.block_number).or_insert(0);
            }
        }

        // Fetch timestamps for each unique block
        for (&block, ts) in cache.iter_mut() {
            match self.get_block_timestamp(block).await {
                Ok(t) => *ts = t,
                Err(e) => {
                    eprintln!("WARNING: failed to get timestamp for block {}: {}", block, e);
                }
            }
        }

        // Apply cached timestamps to transactions
        for tx in txs.iter_mut() {
            if tx.timestamp == 0 {
                if let Some(&ts) = cache.get(&tx.block_number) {
                    tx.timestamp = ts;
                }
            }
        }

        Ok(())
    }

    /// Backfill gas data (gas_used, gas_price) for transactions that have gas_used == 0.
    pub async fn backfill_gas(&self, txs: &mut [X402Transaction]) -> Result<()> {
        for tx in txs.iter_mut() {
            if tx.gas_used == 0 {
                match self.get_tx_receipt(&tx.tx_hash).await {
                    Ok((gas_used, gas_price)) => {
                        tx.gas_used = gas_used;
                        tx.gas_price = gas_price;
                    }
                    Err(e) => {
                        eprintln!(
                            "WARNING: failed to get receipt for tx {}: {}",
                            tx.tx_hash, e
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Convenience method: scan a wallet's recent activity.
    pub async fn scan_wallet(
        &self,
        wallet: &str,
        lookback_blocks: u64,
    ) -> Result<WalletActivity> {
        let current = self.current_block().await?;
        let from_block = current.saturating_sub(lookback_blocks);

        let mut transactions = self
            .fetch_wallet_transactions(wallet, from_block, current)
            .await?;

        // Backfill missing data from RPC
        self.backfill_timestamps(&mut transactions).await?;
        self.backfill_gas(&mut transactions).await?;

        Ok(WalletActivity {
            wallet_address: wallet.to_string(),
            chain_id: 8453,
            from_block,
            to_block: current,
            transactions,
        })
    }

    /// Make a JSON-RPC call to the Base node.
    async fn rpc_call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        });

        let resp = self
            .client
            .post(&self.rpc_url)
            .json(&body)
            .send()
            .await
            .wrap_err("RPC request failed")?;

        let json: serde_json::Value = resp.json().await.wrap_err("Failed to parse RPC response")?;

        if let Some(error) = json.get("error") {
            eyre::bail!("RPC error: {}", error);
        }

        json.get("result")
            .cloned()
            .ok_or_else(|| eyre::eyre!("missing result in RPC response"))
    }

    /// Fetch logs matching the given filter.
    async fn get_logs(
        &self,
        contract: &str,
        event_topic: &str,
        topic1: Option<&str>,
        topic2: Option<&str>,
        from_block: u64,
        to_block: u64,
    ) -> Result<Vec<serde_json::Value>> {
        let mut topics = vec![serde_json::Value::String(event_topic.to_string())];

        match topic1 {
            Some(t) => topics.push(serde_json::Value::String(t.to_string())),
            None => topics.push(serde_json::Value::Null),
        }

        if let Some(t) = topic2 {
            topics.push(serde_json::Value::String(t.to_string()));
        }

        let filter = serde_json::json!({
            "address": contract,
            "fromBlock": format!("0x{:x}", from_block),
            "toBlock": format!("0x{:x}", to_block),
            "topics": topics,
        });

        let result = self.rpc_call("eth_getLogs", serde_json::json!([filter])).await?;

        result
            .as_array()
            .cloned()
            .ok_or_else(|| eyre::eyre!("expected array of logs"))
    }
}

/// Parse a Transfer event log into an X402Transaction.
fn parse_transfer_log(log: &serde_json::Value) -> Option<X402Transaction> {
    let tx_hash = log.get("transactionHash")?.as_str()?.to_string();
    let block_hex = log.get("blockNumber")?.as_str()?;
    let block_number = parse_hex_u64(block_hex).ok()?;

    let topics = log.get("topics")?.as_array()?;
    if topics.len() < 3 {
        return None;
    }

    // topics[1] = from address (padded to 32 bytes)
    // topics[2] = to address (padded to 32 bytes)
    let from = unpad_address(topics[1].as_str()?);
    let to = unpad_address(topics[2].as_str()?);

    // data = value (uint256)
    let data = log.get("data")?.as_str()?;
    let value = parse_hex_u64(data).unwrap_or(0);

    Some(X402Transaction {
        tx_hash,
        from,
        to,
        value,
        timestamp: 0, // Will be filled by block timestamp lookup if needed
        block_number,
        gas_used: 0,
        gas_price: 0,
    })
}

/// Pad an address to a 32-byte hex topic (left-pad with zeros).
fn pad_address(addr: &str) -> String {
    let clean = addr.trim_start_matches("0x").to_lowercase();
    format!("0x{:0>64}", clean)
}

/// Extract an address from a 32-byte topic (remove left-padding).
fn unpad_address(topic: &str) -> String {
    let clean = topic.trim_start_matches("0x");
    if clean.len() >= 40 {
        format!("0x{}", &clean[clean.len() - 40..])
    } else {
        format!("0x{}", clean)
    }
}

/// Parse a hex string (0x-prefixed) as u64.
fn parse_hex_u64(hex: &str) -> Result<u64> {
    let clean = hex.trim_start_matches("0x");
    // For very large hex values (uint256), take only the last 16 hex chars (8 bytes)
    let trimmed = if clean.len() > 16 {
        &clean[clean.len() - 16..]
    } else {
        clean
    };
    u64::from_str_radix(trimmed, 16).wrap_err_with(|| format!("invalid hex: {}", hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_address() {
        let addr = "0x1234567890abcdef1234567890abcdef12345678";
        let padded = pad_address(addr);
        assert_eq!(padded.len(), 66); // 0x + 64 chars
        assert!(padded.ends_with("1234567890abcdef1234567890abcdef12345678"));
    }

    #[test]
    fn test_unpad_address() {
        let topic = "0x0000000000000000000000001234567890abcdef1234567890abcdef12345678";
        let addr = unpad_address(topic);
        assert_eq!(addr, "0x1234567890abcdef1234567890abcdef12345678");
    }

    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_hex_u64("0x1").unwrap(), 1);
        assert_eq!(parse_hex_u64("0xff").unwrap(), 255);
        assert_eq!(parse_hex_u64("0x100").unwrap(), 256);
    }
}
