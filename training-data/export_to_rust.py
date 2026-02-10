#!/usr/bin/env python3
"""Export trained PyTorch model weights to Rust const arrays.

Reads: tx_integrity_model.pt
Outputs: Rust const arrays for W1/B1/W2/B2/W3/B3

The weights are quantized to i32 at scale=7 (multiplied by 128).
Paste the output into src/models/tx_integrity.rs.
"""

import os
import torch
import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(__file__), "tx_integrity_model.pt")
SCALE = 128  # 2^7


def quantize_weights(tensor, scale=SCALE):
    """Quantize a float tensor to i32 at the given scale."""
    return (tensor.detach().numpy() * scale).astype(np.int32)


def format_rust_array(name, values, cols_per_row=None, comment=None):
    """Format an array as a Rust const declaration."""
    lines = []
    if comment:
        lines.append(f"/// {comment}")
    lines.append(f"const {name}: &[i32] = &[")

    flat = values.flatten().tolist()
    if cols_per_row:
        for i in range(0, len(flat), cols_per_row):
            chunk = flat[i:i + cols_per_row]
            row = ", ".join(str(int(v)) for v in chunk)
            neuron_idx = i // cols_per_row
            lines.append(f"    // Neuron {neuron_idx}")
            lines.append(f"    {row},")
    else:
        row = ", ".join(str(int(v)) for v in flat)
        lines.append(f"    {row},")

    lines.append("];")
    return "\n".join(lines)


def main():
    print(f"Loading model from {MODEL_FILE}")

    # Load the trained model
    from train_classifier import TxIntegrityMLP
    model = TxIntegrityMLP()
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()

    # Extract and quantize weights
    w1 = quantize_weights(model.fc1.weight)  # [36, 24]
    b1 = quantize_weights(model.fc1.bias)    # [36]
    w2 = quantize_weights(model.fc2.weight)  # [36, 36]
    b2 = quantize_weights(model.fc2.bias)    # [36]
    w3 = quantize_weights(model.fc3.weight)  # [5, 36]
    b3 = quantize_weights(model.fc3.bias)    # [5]

    print(f"W1: {w1.shape} ({w1.size} params)")
    print(f"B1: {b1.shape} ({b1.size} params)")
    print(f"W2: {w2.shape} ({w2.size} params)")
    print(f"B2: {b2.shape} ({b2.size} params)")
    print(f"W3: {w3.shape} ({w3.size} params)")
    print(f"B3: {b3.shape} ({b3.size} params)")
    print(f"Total: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size} params")

    # Format as Rust code
    print("\n// ===== Paste the following into src/models/tx_integrity.rs =====\n")
    print(format_rust_array("W1", w1, cols_per_row=24,
                            comment="Layer 1 weights: [36 hidden neurons, 24 input features]"))
    print()
    print(format_rust_array("B1", b1, comment="Layer 1 biases: [36]"))
    print()
    print(format_rust_array("W2", w2, cols_per_row=36,
                            comment="Layer 2 weights: [36 hidden neurons, 36 neurons from layer 1]"))
    print()
    print(format_rust_array("B2", b2, comment="Layer 2 biases: [36]"))
    print()
    print(format_rust_array("W3", w3, cols_per_row=36,
                            comment="Layer 3 (output) weights: [5 output neurons, 36 hidden neurons]"))
    print()
    print(format_rust_array("B3", b3, comment="Layer 3 biases: [5]"))

    # Verify dimensions
    assert w1.shape == (36, 24), f"W1 shape mismatch: {w1.shape}"
    assert b1.shape == (36,), f"B1 shape mismatch: {b1.shape}"
    assert w2.shape == (36, 36), f"W2 shape mismatch: {w2.shape}"
    assert b2.shape == (36,), f"B2 shape mismatch: {b2.shape}"
    assert w3.shape == (5, 36), f"W3 shape mismatch: {w3.shape}"
    assert b3.shape == (5,), f"B3 shape mismatch: {b3.shape}"

    print("\n// ===== End of generated code =====")


if __name__ == "__main__":
    main()
