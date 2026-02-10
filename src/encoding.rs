//! Feature encoding for the transaction integrity model.
//!
//! Validates 24-element feature vectors and ensures all values are in [0, 128].
//! Supports both structured `TransactionFeatures` input and raw `Vec<i32>`.

use crate::transaction::TransactionFeatures;
use eyre::{bail, Result};

const SCALE: i32 = 128;
const FEATURE_DIM: usize = 24;

/// Validate and clamp a raw feature vector to [0, 128].
///
/// Returns an error if the vector is not exactly 24 elements.
pub fn validate_features(features: &[i32]) -> Result<Vec<i32>> {
    if features.len() != FEATURE_DIM {
        bail!(
            "Expected {} features, got {}",
            FEATURE_DIM,
            features.len()
        );
    }

    let clamped: Vec<i32> = features.iter().map(|&v| v.max(0).min(SCALE)).collect();
    Ok(clamped)
}

/// Encode structured TransactionFeatures into a validated i32 vector.
pub fn encode_transaction_features(features: &TransactionFeatures) -> Vec<i32> {
    let raw = features.to_normalized_vec();
    // Clamp to [0, 128] (should already be, but be safe)
    raw.iter().map(|&v| v.max(0).min(SCALE)).collect()
}

/// Encode a raw feature vector, clamping values to [0, 128].
/// Returns zeros if the vector length is wrong.
pub fn encode_raw_or_default(features: &[i32]) -> Vec<i32> {
    match validate_features(features) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARNING: encoding: {}", e);
            vec![0i32; FEATURE_DIM]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_correct_length() {
        let features = vec![64i32; 24];
        let result = validate_features(&features).unwrap();
        assert_eq!(result.len(), 24);
        assert!(result.iter().all(|&v| v == 64));
    }

    #[test]
    fn test_validate_wrong_length() {
        let features = vec![64i32; 22];
        assert!(validate_features(&features).is_err());
    }

    #[test]
    fn test_validate_clamps_values() {
        let mut features = vec![64i32; 24];
        features[0] = 200;  // over max
        features[1] = -10;  // under min

        let result = validate_features(&features).unwrap();
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 0);
    }

    #[test]
    fn test_encode_raw_default_on_wrong_length() {
        let features = vec![64i32; 10]; // wrong length
        let result = encode_raw_or_default(&features);
        assert_eq!(result.len(), 24);
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_encode_transaction_features() {
        let features = TransactionFeatures::zeros();
        let encoded = encode_transaction_features(&features);
        assert_eq!(encoded.len(), 24);
    }
}
