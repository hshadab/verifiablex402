//! Enforcement types and decision mapping for the transaction integrity analyzer.
//!
//! Three enforcement tiers: Log, Soft, Hard
//! Decision mapping:
//! - WashTrading → Deny
//! - CircularPayments → Flag
//! - GenuineCommerce / LowActivity / ScriptedBenign → Allow

use crate::transaction::{IntegrityClassification, IntegrityDecision};

// ---------------------------------------------------------------------------
// Enforcement level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementLevel {
    /// Log only — no blocking
    Log,
    /// Soft — deny is overridable
    Soft,
    /// Hard — deny is final
    Hard,
}

impl std::str::FromStr for EnforcementLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hard" => Ok(Self::Hard),
            "soft" => Ok(Self::Soft),
            "log" => Ok(Self::Log),
            other => Err(format!(
                "unknown enforcement level '{other}', expected log/soft/hard"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Decision derivation
// ---------------------------------------------------------------------------

/// Confidence threshold above which a Flag can escalate to Deny.
const FLAG_TO_DENY_THRESHOLD: f64 = 0.85;

/// Derive a decision and human-readable reasoning from the classification.
pub fn derive_decision(
    classification: IntegrityClassification,
    _scores: &[f64; 5],
    confidence: f64,
) -> (IntegrityDecision, String) {
    match classification {
        IntegrityClassification::WashTrading => (
            IntegrityDecision::Deny,
            format!(
                "Wash trading pattern detected with {:.1}% confidence. \
                 High identical-amount ratio with minimal counterparty diversity.",
                confidence * 100.0
            ),
        ),
        IntegrityClassification::CircularPayments => {
            if confidence > FLAG_TO_DENY_THRESHOLD {
                (
                    IntegrityDecision::Deny,
                    format!(
                        "Circular payment pattern detected with high confidence ({:.1}%), \
                         escalated from Flag to Deny. Funds cycling through intermediaries.",
                        confidence * 100.0
                    ),
                )
            } else {
                (
                    IntegrityDecision::Flag,
                    format!(
                        "Circular payment pattern detected with {:.1}% confidence. \
                         Potential funds cycling — manual review recommended.",
                        confidence * 100.0
                    ),
                )
            }
        }
        IntegrityClassification::GenuineCommerce => (
            IntegrityDecision::Allow,
            format!(
                "Genuine commerce pattern with {:.1}% confidence. \
                 Diverse counterparties and varied transaction amounts.",
                confidence * 100.0
            ),
        ),
        IntegrityClassification::LowActivity => (
            IntegrityDecision::Allow,
            format!(
                "Low activity wallet with {:.1}% confidence. \
                 Insufficient transaction history to determine patterns.",
                confidence * 100.0
            ),
        ),
        IntegrityClassification::ScriptedBenign => (
            IntegrityDecision::Allow,
            format!(
                "Scripted benign pattern with {:.1}% confidence. \
                 Automated but legitimate usage (e.g., payroll, subscriptions).",
                confidence * 100.0
            ),
        ),
    }
}

/// Check if a decision label string represents a deny decision.
pub fn is_deny_decision(label: &str) -> bool {
    matches!(label, "WASH_TRADING" | "CIRCULAR_PAYMENTS")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wash_trading_deny() {
        let scores = [0.1, 0.05, 0.05, 0.1, 0.7];
        let (decision, _reason) =
            derive_decision(IntegrityClassification::WashTrading, &scores, 0.7);
        assert_eq!(decision, IntegrityDecision::Deny);
    }

    #[test]
    fn test_circular_flag() {
        let scores = [0.1, 0.05, 0.05, 0.6, 0.2];
        let (decision, _reason) =
            derive_decision(IntegrityClassification::CircularPayments, &scores, 0.5);
        assert_eq!(decision, IntegrityDecision::Flag);
    }

    #[test]
    fn test_circular_escalate_to_deny() {
        let scores = [0.05, 0.02, 0.03, 0.85, 0.05];
        let (decision, _reason) =
            derive_decision(IntegrityClassification::CircularPayments, &scores, 0.90);
        assert_eq!(decision, IntegrityDecision::Deny);
    }

    #[test]
    fn test_genuine_allow() {
        let scores = [0.7, 0.1, 0.1, 0.05, 0.05];
        let (decision, _reason) =
            derive_decision(IntegrityClassification::GenuineCommerce, &scores, 0.7);
        assert_eq!(decision, IntegrityDecision::Allow);
    }

    #[test]
    fn test_enforcement_parse() {
        assert_eq!("hard".parse::<EnforcementLevel>().unwrap(), EnforcementLevel::Hard);
        assert_eq!("soft".parse::<EnforcementLevel>().unwrap(), EnforcementLevel::Soft);
        assert_eq!("log".parse::<EnforcementLevel>().unwrap(), EnforcementLevel::Log);
        assert!("invalid".parse::<EnforcementLevel>().is_err());
    }
}
