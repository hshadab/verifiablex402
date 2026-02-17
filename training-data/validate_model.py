#!/usr/bin/env python3
"""Validate quantized model accuracy against float model.

Cross-checks i32 quantized forward pass (simulating Rust behavior) against
the PyTorch float forward pass to verify that quantization loss is acceptable.

The threshold is < 1% accuracy delta between float and quantized predictions.

Usage:
    python validate_model.py

Reads: tx_integrity_model.pt, training_features.csv
"""

import csv
import os
import sys
import torch
import torch.nn as nn

DATA_FILE = os.path.join(os.path.dirname(__file__), "training_features.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "tx_integrity_model.pt")

INPUT_DIM = 24
HIDDEN_DIM = 36
OUTPUT_DIM = 5
SCALE = 128  # fixed-point scale (2^7)

# Maximum acceptable accuracy delta between float and quantized
MAX_ACCURACY_DELTA = 0.01  # 1%


class TxIntegrityMLP(nn.Module):
    """3-layer MLP for transaction integrity classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantize_weights(state_dict):
    """Quantize model weights to i32 at scale 128, matching Rust export."""
    quantized = {}
    for key, tensor in state_dict.items():
        quantized[key] = (tensor * SCALE).round().to(torch.int32)
    return quantized


def quantized_forward(quantized_weights, x_i32):
    """Simulate the Rust fixed-point forward pass using i32 arithmetic.

    Input x_i32: [batch, 24] tensor of i32 values in [0, 128]
    """
    # Layer 1: y = ReLU(x @ W1^T + b1)
    # W1 is [36, 24] quantized, b1 is [36] quantized
    # x is [batch, 24], values in [0, 128]
    # Matmul output is at scale 128*128 = 16384, bias at scale 128
    # After matmul + bias*128, divide by 128 to get back to scale 128
    w1 = quantized_weights["fc1.weight"].to(torch.int64)  # [36, 24]
    b1 = quantized_weights["fc1.bias"].to(torch.int64)    # [36]
    x = x_i32.to(torch.int64)

    h1 = torch.matmul(x, w1.T) + b1 * SCALE  # scale = 128*128
    h1 = h1 // SCALE  # back to scale 128
    h1 = torch.clamp(h1, min=0)  # ReLU

    # Layer 2
    w2 = quantized_weights["fc2.weight"].to(torch.int64)
    b2 = quantized_weights["fc2.bias"].to(torch.int64)
    h2 = torch.matmul(h1, w2.T) + b2 * SCALE
    h2 = h2 // SCALE
    h2 = torch.clamp(h2, min=0)  # ReLU

    # Layer 3 (output, no ReLU)
    w3 = quantized_weights["fc3.weight"].to(torch.int64)
    b3 = quantized_weights["fc3.bias"].to(torch.int64)
    out = torch.matmul(h2, w3.T) + b3 * SCALE
    out = out // SCALE

    return out.to(torch.int32)


def load_data():
    """Load test data from CSV."""
    features = []
    labels = []

    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            feat = [int(float(x)) for x in row[:-1]]  # already in [0, 128]
            label = int(row[-1])
            features.append(feat)
            labels.append(label)

    return features, labels


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found: {MODEL_FILE}")
        print("Run train_classifier.py first.")
        sys.exit(1)

    print("Loading model...")
    model = TxIntegrityMLP()
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()

    print("Loading data...")
    features_raw, labels = load_data()
    n = len(labels)
    print(f"Loaded {n} samples")

    # Float forward pass
    X_float = torch.tensor(features_raw, dtype=torch.float32) / SCALE
    with torch.no_grad():
        float_outputs = model(X_float)
    float_preds = float_outputs.argmax(dim=1).tolist()

    float_correct = sum(1 for p, l in zip(float_preds, labels) if p == l)
    float_acc = float_correct / n

    # Quantized forward pass
    quantized = quantize_weights(model.state_dict())
    X_i32 = torch.tensor(features_raw, dtype=torch.int32)
    quant_outputs = quantized_forward(quantized, X_i32)
    quant_preds = quant_outputs.argmax(dim=1).tolist()

    quant_correct = sum(1 for p, l in zip(quant_preds, labels) if p == l)
    quant_acc = quant_correct / n

    # Agreement between float and quantized
    agreement = sum(1 for f, q in zip(float_preds, quant_preds) if f == q)
    agreement_rate = agreement / n

    delta = abs(float_acc - quant_acc)

    print(f"\nResults:")
    print(f"  Float accuracy:     {float_acc:.4f} ({float_correct}/{n})")
    print(f"  Quantized accuracy: {quant_acc:.4f} ({quant_correct}/{n})")
    print(f"  Accuracy delta:     {delta:.4f}")
    print(f"  Agreement rate:     {agreement_rate:.4f} ({agreement}/{n})")
    print(f"  Max allowed delta:  {MAX_ACCURACY_DELTA:.4f}")

    if delta <= MAX_ACCURACY_DELTA:
        print(f"\n  PASS: Quantization loss is acceptable ({delta:.4f} <= {MAX_ACCURACY_DELTA:.4f})")
        return 0
    else:
        print(f"\n  FAIL: Quantization loss too high ({delta:.4f} > {MAX_ACCURACY_DELTA:.4f})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
