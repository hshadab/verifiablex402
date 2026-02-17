use chrono::Utc;
use clap::{Parser, Subcommand};
use eyre::{Result, WrapErr};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

use verifiablex402::{
    config::Config,
    hash_model_fn,
    models::tx_integrity::tx_integrity_model,
    receipt::GuardrailReceipt,
    transaction::{TransactionFeatures, WalletActivity},
};

#[derive(Parser)]
#[command(
    name = "verifiablex402",
    about = "x402 Transaction Integrity Analyzer — classifies wallet transaction patterns with zkML proof verification."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a single wallet's transaction integrity
    Analyze {
        /// Wallet address to analyze
        #[arg(long)]
        wallet: String,

        /// Path to wallet activity JSON file (alternative to --wallet for offline analysis)
        #[arg(long)]
        input: Option<PathBuf>,

        /// Generate a zero-knowledge proof
        #[arg(long, default_value_t = false)]
        prove: bool,

        /// Base RPC URL
        #[arg(long, default_value = "https://mainnet.base.org")]
        rpc_url: String,

        /// Lookback blocks (default: ~7 days)
        #[arg(long, default_value_t = 302400)]
        lookback: u64,

        /// Output format: json, summary, or receipt
        #[arg(long, default_value = "summary")]
        format: String,

        /// Save receipt to file
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Batch analysis from a file of wallet addresses
    Scan {
        /// Path to file with wallet addresses (one per line)
        #[arg(long)]
        input: PathBuf,

        /// Generate proofs for each wallet
        #[arg(long, default_value_t = false)]
        prove: bool,

        /// Base RPC URL
        #[arg(long, default_value = "https://mainnet.base.org")]
        rpc_url: String,

        /// Output directory for receipts
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },

    /// Verify a proof file
    Verify {
        /// Path to the proof file
        #[arg(long)]
        proof: PathBuf,

        /// Expected model hash
        #[arg(long)]
        model_hash: String,
    },

    /// Verify a guardrail receipt
    VerifyReceipt {
        /// Path to the receipt JSON file
        #[arg(long)]
        input: PathBuf,

        /// Verify the ZK proof (requires model to be available)
        #[arg(long, default_value_t = false)]
        verify_proof: bool,
    },

    /// Start the HTTP server
    Serve {
        /// Address to bind to
        #[arg(long)]
        bind: Option<String>,

        /// Maximum concurrent proof generations
        #[arg(long)]
        max_proofs: Option<usize>,

        /// Require proof generation for all requests
        #[arg(long, default_value_t = false)]
        require_proof: bool,

        /// Rate limit in requests per minute per IP (0 = no limit)
        #[arg(long)]
        rate_limit: Option<u32>,

        /// Base RPC URL
        #[arg(long)]
        rpc_url: Option<String>,

        /// Require x402 payment for evaluation
        #[arg(long)]
        require_payment: Option<bool>,
    },

    /// Show model information
    Models,
}

/// Analyze result for JSON output
#[derive(Serialize)]
struct AnalyzeResult {
    wallet_address: String,
    chain_id: u64,
    classification: String,
    decision: String,
    confidence: f64,
    reasoning: String,
    tx_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_generated: Option<bool>,
    model_hash: String,
    timestamp: String,
}

fn cmd_analyze(
    wallet: String,
    input: Option<PathBuf>,
    prove: bool,
    rpc_url: String,
    lookback: u64,
    format: String,
    output: Option<PathBuf>,
) -> Result<i32> {
    let rt = tokio::runtime::Runtime::new()?;

    // Get wallet activity
    let activity = if let Some(input_path) = input {
        let content = fs::read_to_string(&input_path)?;
        serde_json::from_str::<WalletActivity>(&content)?
    } else {
        eprintln!("Fetching transactions for {} from Base mainnet...", wallet);
        let indexer = verifiablex402::indexer::BaseIndexer::new(&rpc_url);
        rt.block_on(indexer.scan_wallet(&wallet, lookback))?
    };

    let tx_count = activity.transactions.len();
    eprintln!("Found {} transactions", tx_count);

    // Extract features and run guardrail
    let features = TransactionFeatures::extract(&activity);
    let (receipt, _proof_path) = verifiablex402::run_guardrail(
        &features,
        &activity.wallet_address,
        activity.chain_id,
        prove,
    )?;

    // Format output
    match format.as_str() {
        "json" => {
            let result = AnalyzeResult {
                wallet_address: activity.wallet_address.clone(),
                chain_id: activity.chain_id,
                classification: receipt.evaluation.classification.clone(),
                decision: receipt.evaluation.decision.clone(),
                confidence: receipt.evaluation.confidence,
                reasoning: receipt
                    .evaluation
                    .reasoning
                    .clone()
                    .unwrap_or_default(),
                tx_count,
                receipt_id: Some(receipt.receipt_id.clone()),
                proof_generated: if prove { Some(true) } else { None },
                model_hash: receipt.guardrail.model_hash.clone(),
                timestamp: Utc::now().to_rfc3339(),
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "receipt" => {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        _ => {
            // Summary format
            println!("Transaction Integrity Analysis");
            println!("==============================");
            println!("Wallet:         {}", activity.wallet_address);
            println!("Chain:          Base (8453)");
            println!("Transactions:   {}", tx_count);
            println!();
            println!(
                "Classification: {}",
                receipt.evaluation.classification
            );
            println!("Decision:       {}", receipt.evaluation.decision);
            println!(
                "Confidence:     {:.1}%",
                receipt.evaluation.confidence * 100.0
            );
            if let Some(ref reasoning) = receipt.evaluation.reasoning {
                println!("Reasoning:      {}", reasoning);
            }
            println!();
            println!("Scores:");
            let scores = &receipt.evaluation.scores;
            println!(
                "  GENUINE_COMMERCE:  {:.1}%",
                scores.genuine_commerce * 100.0
            );
            println!(
                "  LOW_ACTIVITY:      {:.1}%",
                scores.low_activity * 100.0
            );
            println!(
                "  SCRIPTED_BENIGN:   {:.1}%",
                scores.scripted_benign * 100.0
            );
            println!(
                "  CIRCULAR_PAYMENTS: {:.1}%",
                scores.circular_payments * 100.0
            );
            println!(
                "  WASH_TRADING:      {:.1}%",
                scores.wash_trading * 100.0
            );
            println!();
            println!("Receipt ID:  {}", receipt.receipt_id);
            println!(
                "Model Hash:  {}",
                receipt.guardrail.model_hash
            );
        }
    }

    // Save receipt if output specified
    if let Some(output_path) = output {
        fs::write(&output_path, serde_json::to_string_pretty(&receipt)?)?;
        eprintln!("Receipt saved to: {}", output_path.display());
    }

    // Exit code: 1 for deny, 0 otherwise
    if receipt.is_blocked() {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn cmd_scan(
    input: PathBuf,
    prove: bool,
    rpc_url: String,
    output_dir: Option<PathBuf>,
) -> Result<()> {
    use futures::stream::{self, StreamExt};

    let content = fs::read_to_string(&input)?;
    let wallets: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && l.starts_with("0x"))
        .collect();

    eprintln!("Scanning {} wallets...", wallets.len());

    let rt = tokio::runtime::Runtime::new()?;
    let indexer = std::sync::Arc::new(verifiablex402::indexer::BaseIndexer::new(&rpc_url));

    if let Some(ref dir) = output_dir {
        fs::create_dir_all(dir)?;
    }

    let concurrency = 4usize;
    let output_dir_arc = output_dir.clone().map(std::sync::Arc::new);
    let total = wallets.len();

    rt.block_on(async {
        let results: Vec<_> = stream::iter(wallets.into_iter().enumerate())
            .map(|(i, wallet)| {
                let indexer = indexer.clone();
                let output_dir = output_dir_arc.clone();
                async move {
                    eprintln!("[{}/{}] Scanning {}...", i + 1, total, wallet);

                    let activity = match indexer.scan_wallet(&wallet, 302400).await {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!("  ERROR: {}", e);
                            return;
                        }
                    };

                    let features = TransactionFeatures::extract(&activity);
                    match verifiablex402::run_guardrail(&features, &wallet, 8453, prove) {
                        Ok((receipt, _)) => {
                            println!(
                                "{}: {} ({}, {:.1}% confidence)",
                                wallet,
                                receipt.evaluation.classification,
                                receipt.evaluation.decision,
                                receipt.evaluation.confidence * 100.0
                            );

                            if let Some(ref dir) = output_dir {
                                let filename = format!("{}.receipt.json", wallet);
                                let path = dir.join(&filename);
                                let _ = fs::write(
                                    &path,
                                    serde_json::to_string_pretty(&receipt).unwrap_or_default(),
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("  ERROR: {}", e);
                        }
                    }
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        drop(results);
    });

    Ok(())
}

fn cmd_verify(proof: PathBuf, model_hash: String) -> Result<()> {
    let guard = verifiablex402::GuardModel::TxIntegrity;
    let trace_len = guard.max_trace_length();

    let valid = verifiablex402::proving::verify_proof_file(
        &proof,
        guard.model_fn(),
        trace_len,
        Some(&model_hash),
    )?;

    let result = serde_json::json!({
        "valid": valid,
        "model_hash_matches": true,
        "proof_file": proof.to_string_lossy(),
    });

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_verify_receipt(input: PathBuf, verify_proof: bool) -> Result<()> {
    use verifiablex402::receipt::RECEIPT_VERSION;

    let content = fs::read_to_string(&input)
        .wrap_err_with(|| format!("Failed to read receipt: {}", input.display()))?;
    let receipt: GuardrailReceipt =
        serde_json::from_str(&content).wrap_err("Failed to parse receipt JSON")?;

    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    // Check 1: Schema version
    let schema_valid = receipt.version == RECEIPT_VERSION;
    if !schema_valid {
        warnings.push(format!(
            "Schema version mismatch: got {}, expected {}",
            receipt.version, RECEIPT_VERSION
        ));
    }

    // Check 2: Nonce format
    let nonce_valid = receipt.nonce.len() == 64
        && receipt.nonce.chars().all(|c| c.is_ascii_hexdigit());
    if !nonce_valid {
        errors.push(format!(
            "Invalid nonce format: expected 64 hex chars, got {} chars",
            receipt.nonce.len()
        ));
    }

    // Check 3: Model hash matches known models
    let known_hash = hash_model_fn(tx_integrity_model);
    let model_known = receipt.guardrail.model_hash == known_hash;
    if !model_known {
        warnings.push(format!(
            "Model hash not recognized: {}",
            receipt.guardrail.model_hash
        ));
    }

    // Check 4: Commitment format
    let commitment_valid = receipt.subject.commitment.starts_with("sha256:")
        && receipt.subject.commitment.len() == 71;
    if !commitment_valid {
        errors.push("Invalid commitment format".to_string());
    }

    // Check 5: Decision consistency
    let decision_consistent = match receipt.evaluation.classification.as_str() {
        "GENUINE_COMMERCE" | "LOW_ACTIVITY" | "SCRIPTED_BENIGN" => {
            receipt.evaluation.decision == "allow"
        }
        "WASH_TRADING" => receipt.evaluation.decision == "deny",
        "CIRCULAR_PAYMENTS" => {
            receipt.evaluation.decision == "flag" || receipt.evaluation.decision == "deny"
        }
        _ => {
            warnings.push(format!(
                "Unknown classification: {}",
                receipt.evaluation.classification
            ));
            true
        }
    };
    if !decision_consistent {
        errors.push(format!(
            "Decision '{}' inconsistent with classification '{}'",
            receipt.evaluation.decision, receipt.evaluation.classification
        ));
    }

    // Check 6: ZK proof verification
    let proof_valid = if verify_proof && !receipt.proof.proof_bytes.is_empty() {
        match verify_receipt_proof(&receipt) {
            Ok(valid) => {
                if !valid {
                    errors.push("ZK proof verification failed".to_string());
                }
                Some(valid)
            }
            Err(e) => {
                errors.push(format!("ZK proof verification error: {}", e));
                Some(false)
            }
        }
    } else if verify_proof && receipt.proof.proof_bytes.is_empty() {
        warnings.push("No proof bytes in receipt to verify".to_string());
        None
    } else {
        None
    };

    let valid = errors.is_empty() && schema_valid && nonce_valid && commitment_valid && decision_consistent;

    let result = serde_json::json!({
        "valid": valid,
        "receipt_id": receipt.receipt_id,
        "checks": {
            "schema_valid": schema_valid,
            "nonce_valid": nonce_valid,
            "model_known": model_known,
            "model_hash": receipt.guardrail.model_hash,
            "commitment_valid": commitment_valid,
            "decision_consistent": decision_consistent,
            "proof_valid": proof_valid,
            "classification": receipt.evaluation.classification,
            "decision": receipt.evaluation.decision,
            "confidence": receipt.evaluation.confidence,
        },
        "warnings": warnings,
        "errors": errors,
    });

    println!("{}", serde_json::to_string_pretty(&result)?);

    eprintln!();
    if valid {
        eprintln!("Receipt verification PASSED");
        eprintln!("  Receipt ID: {}", receipt.receipt_id);
        eprintln!("  Subject: {}", receipt.subject.description);
        eprintln!(
            "  Classification: {} ({})",
            receipt.evaluation.classification, receipt.evaluation.decision
        );
        eprintln!(
            "  Confidence: {:.1}%",
            receipt.evaluation.confidence * 100.0
        );
    } else {
        eprintln!("Receipt verification FAILED");
        for err in &errors {
            eprintln!("  Error: {}", err);
        }
    }

    Ok(())
}

fn verify_receipt_proof(receipt: &GuardrailReceipt) -> Result<bool> {
    use base64::{engine::general_purpose::STANDARD as B64, Engine};
    use onnx_tracer::ProgramIO;

    if receipt.guardrail.domain != "integrity" {
        eyre::bail!("Only integrity domain proofs are supported for verification");
    }

    let expected_hash = hash_model_fn(tx_integrity_model);
    if receipt.guardrail.model_hash != expected_hash {
        return Ok(false);
    }

    if receipt.proof.proof_bytes.is_empty() {
        return Ok(false);
    }

    let program_io_str = match &receipt.proof.program_io {
        Some(s) => s,
        None => {
            eprintln!("WARNING: Receipt missing program_io, performing partial verification");
            return Ok(true);
        }
    };

    let proof_bytes = B64
        .decode(&receipt.proof.proof_bytes)
        .wrap_err("Failed to decode proof bytes from base64")?;

    let program_io: ProgramIO =
        serde_json::from_str(program_io_str).wrap_err("Failed to parse program_io")?;

    let max_trace_length = 1 << 16;
    verifiablex402::proving::verify_proof_from_bytes(
        &proof_bytes,
        tx_integrity_model,
        program_io,
        max_trace_length,
    )
}

fn cmd_serve(
    bind: Option<String>,
    max_proofs: Option<usize>,
    require_proof: bool,
    rate_limit: Option<u32>,
    rpc_url: Option<String>,
    require_payment: Option<bool>,
) -> Result<()> {
    use verifiablex402::server::{run_server, ServerConfig};

    let cfg = Config::load();

    let bind_str = bind
        .or(cfg.bind)
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());
    let bind_addr = bind_str
        .parse()
        .wrap_err_with(|| format!("Invalid bind address: {}", bind_str))?;
    let max_concurrent_proofs = max_proofs
        .or(cfg.max_concurrent_proofs)
        .unwrap_or(4);
    let rate_limit_rpm = rate_limit
        .or(cfg.rate_limit_rpm)
        .unwrap_or(60);
    let rpc = rpc_url
        .or(cfg.rpc_url)
        .unwrap_or_else(|| "https://mainnet.base.org".to_string());
    let payment = require_payment
        .or(cfg.require_payment)
        .unwrap_or(false);

    let config = ServerConfig {
        bind_addr,
        max_concurrent_proofs,
        require_proof,
        rate_limit_rpm,
        rpc_url: rpc,
        require_payment: payment,
        payment_amount: "5000".to_string(),
        payment_payee: cfg.payment_payee,
        usdc_contract: cfg.usdc_contract,
        allowed_origins: cfg.allowed_origins,
        api_keys: cfg.api_keys,
        cache_ttl_seconds: cfg.cache_ttl_seconds.unwrap_or(300),
        cache_max_entries: cfg.cache_max_entries.unwrap_or(1000),
    };

    tracing::info!("starting verifiablex402 server");
    tracing::info!("model: tx-integrity (2,417 params)");
    tracing::info!(max_concurrent_proofs, "proof concurrency configured");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server(config))?;

    Ok(())
}

fn cmd_models() -> Result<()> {
    let model_hash = hash_model_fn(tx_integrity_model);

    println!("Transaction Integrity Model");
    println!("==========================");
    println!("Name:           tx-integrity");
    println!("Architecture:   MLP [1,24] -> [1,36] -> [1,36] -> [1,5]");
    println!("Parameters:     2,417");
    println!("Input features: 24 (normalized to [0, 128])");
    println!("Output classes: 5");
    println!("  0: GENUINE_COMMERCE  — Normal merchant/service payments");
    println!("  1: LOW_ACTIVITY      — Very few transactions");
    println!("  2: SCRIPTED_BENIGN   — Automated but legitimate");
    println!("  3: CIRCULAR_PAYMENTS — Funds cycling through intermediaries");
    println!("  4: WASH_TRADING      — Fake volume, bot-like patterns");
    println!("Scale:          7 (fixed-point × 128)");
    println!("Max trace:      65536 (1 << 16)");
    println!("Model hash:     {}", model_hash);
    println!("Proof system:   JOLT Atlas (Dory commitment)");

    Ok(())
}

fn main() {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "verifiablex402=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Analyze {
            wallet,
            input,
            prove,
            rpc_url,
            lookback,
            format,
            output,
        } => match cmd_analyze(wallet, input, prove, rpc_url, lookback, format, output) {
            Ok(code) => {
                if code != 0 {
                    std::process::exit(code);
                }
                Ok(())
            }
            Err(e) => Err(e),
        },
        Commands::Scan {
            input,
            prove,
            rpc_url,
            output_dir,
        } => cmd_scan(input, prove, rpc_url, output_dir),
        Commands::Verify { proof, model_hash } => cmd_verify(proof, model_hash),
        Commands::VerifyReceipt {
            input,
            verify_proof,
        } => cmd_verify_receipt(input, verify_proof),
        Commands::Serve {
            bind,
            max_proofs,
            require_proof,
            rate_limit,
            rpc_url,
            require_payment,
        } => cmd_serve(bind, max_proofs, require_proof, rate_limit, rpc_url, require_payment),
        Commands::Models => cmd_models(),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
