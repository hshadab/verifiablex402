//! On-chain payment verification for x402 payment gating.
//!
//! Verifies that a claimed USDC payment actually occurred on-chain by checking
//! the transaction receipt for a Transfer event matching the expected payee and amount.

use eyre::{Result, WrapErr};

use crate::indexer::DEFAULT_USDC_BASE;
use crate::receipt::PaymentInfo;

/// Standard ERC-20 Transfer event topic: keccak256("Transfer(address,address,uint256)")
const TRANSFER_TOPIC: &str =
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";

/// Verifies on-chain payment transactions.
pub struct PaymentVerifier {
    rpc_url: String,
    client: reqwest::Client,
    expected_payee: String,
    usdc_contract: String,
}

impl PaymentVerifier {
    /// Create a new payment verifier.
    pub fn new(rpc_url: &str, expected_payee: &str, usdc_contract: Option<&str>) -> Self {
        Self {
            rpc_url: rpc_url.to_string(),
            client: reqwest::Client::new(),
            expected_payee: expected_payee.to_lowercase(),
            usdc_contract: usdc_contract
                .unwrap_or(DEFAULT_USDC_BASE)
                .to_lowercase(),
        }
    }

    /// Verify that a payment transaction is valid.
    ///
    /// Checks:
    /// 1. Transaction receipt exists and status == 1 (success)
    /// 2. A USDC Transfer event log exists in the receipt
    /// 3. Transfer `to` matches the expected payee
    /// 4. Transfer amount >= expected amount
    /// 5. Transfer is on the configured USDC contract
    pub async fn verify_payment(
        &self,
        payment: &PaymentInfo,
        expected_amount: &str,
    ) -> Result<bool> {
        if payment.tx_hash.is_empty() {
            return Ok(false);
        }

        let receipt = self.get_tx_receipt(&payment.tx_hash).await?;

        // Check receipt status == 1 (success)
        let status = receipt
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("0x0");
        if status != "0x1" {
            tracing::warn!(tx_hash = %payment.tx_hash, "payment tx failed (status != 0x1)");
            return Ok(false);
        }

        // Parse logs for USDC Transfer event
        let logs = match receipt.get("logs").and_then(|v| v.as_array()) {
            Some(logs) => logs,
            None => {
                tracing::warn!(tx_hash = %payment.tx_hash, "no logs in payment receipt");
                return Ok(false);
            }
        };

        let expected_amount_u64: u64 = expected_amount.parse().unwrap_or(0);

        for log in logs {
            // Check contract address matches USDC
            let log_address = log
                .get("address")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();
            if log_address != self.usdc_contract {
                continue;
            }

            // Check Transfer event topic
            let topics = match log.get("topics").and_then(|v| v.as_array()) {
                Some(t) if t.len() >= 3 => t,
                _ => continue,
            };

            let event_topic = topics[0].as_str().unwrap_or("");
            if event_topic != TRANSFER_TOPIC {
                continue;
            }

            // Extract `to` address from topic[2] (32-byte padded)
            let to_topic = topics[2].as_str().unwrap_or("");
            let to_address = unpad_address(to_topic);

            if to_address != self.expected_payee {
                continue;
            }

            // Extract amount from data
            let data = log.get("data").and_then(|v| v.as_str()).unwrap_or("0x0");
            let amount = parse_hex_u64(data).unwrap_or(0);

            if amount >= expected_amount_u64 {
                tracing::info!(
                    tx_hash = %payment.tx_hash,
                    amount,
                    expected = expected_amount_u64,
                    "payment verified"
                );
                return Ok(true);
            }
        }

        tracing::warn!(
            tx_hash = %payment.tx_hash,
            "no matching USDC Transfer found in receipt"
        );
        Ok(false)
    }

    /// Fetch transaction receipt via RPC.
    async fn get_tx_receipt(&self, tx_hash: &str) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash],
            "id": 1,
        });

        let resp = self
            .client
            .post(&self.rpc_url)
            .json(&body)
            .send()
            .await
            .wrap_err("payment RPC request failed")?;

        let json: serde_json::Value = resp
            .json()
            .await
            .wrap_err("failed to parse payment RPC response")?;

        if let Some(error) = json.get("error") {
            eyre::bail!("payment RPC error: {}", error);
        }

        json.get("result")
            .cloned()
            .ok_or_else(|| eyre::eyre!("missing result in payment RPC response"))
    }
}

/// Extract an address from a 32-byte topic (remove left-padding).
fn unpad_address(topic: &str) -> String {
    let clean = topic.trim_start_matches("0x");
    if clean.len() >= 40 {
        format!("0x{}", &clean[clean.len() - 40..]).to_lowercase()
    } else {
        format!("0x{}", clean).to_lowercase()
    }
}

/// Parse a hex string (0x-prefixed) as u64.
fn parse_hex_u64(hex: &str) -> Result<u64> {
    let clean = hex.trim_start_matches("0x");
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
    fn test_unpad_address() {
        let topic = "0x0000000000000000000000001234567890abcdef1234567890abcdef12345678";
        let addr = unpad_address(topic);
        assert_eq!(addr, "0x1234567890abcdef1234567890abcdef12345678");
    }

    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_hex_u64("0x1388").unwrap(), 5000);
        assert_eq!(parse_hex_u64("0xff").unwrap(), 255);
    }
}
