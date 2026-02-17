# ADR 002: JOLT Atlas Proof System

## Status

Accepted

## Context

The verifiablex402 analyzer needs to produce cryptographic proofs that the classification result was computed honestly — i.e., that the declared model was actually run on the declared inputs. Several ZK proof systems were considered:

| System | Pros | Cons |
|--------|------|------|
| Circom/Groth16 | Small proofs, fast verification | Circuit-based, hard to express ML models, trusted setup |
| Halo2 | No trusted setup, recursive | Complex API, circuit-based |
| RISC Zero | General RISC-V VM | Larger proofs, slower proving |
| JOLT Atlas | Fast proving, general RISC-V, Lasso lookup args | Newer, less battle-tested |

## Decision

Use **JOLT Atlas** with the **Dory polynomial commitment scheme** for proof generation.

### Key reasons:

1. **RISC-V native**: The MLP forward pass compiles to standard Rust → RISC-V, no circuit translation needed
2. **Fast proving**: JOLT's Lasso-based lookup arguments enable faster proving than traditional R1CS systems
3. **Dory commitments**: Transparent setup (no trusted ceremony), efficient verifier
4. **Workspace integration**: JOLT Atlas provides `onnx-tracer` for model tracing and `zkml-jolt-core` for proof generation, both as workspace crates
5. **i32 arithmetic**: Natural fit with the fixed-point scale-128 representation (ADR 001)

### Proof pipeline:

```
TransactionFeatures → [i32; 24] → onnx_tracer::Model::forward() → [i32; 5]
                                    ↓ (inside JOLT VM)
                              JoltSNARK::prove() → (proof_bytes, program_io)
```

## Consequences

- Build requires `riscv32im-unknown-none-elf` target
- Preprocessing step (~1-2 min) is cached in a `OnceLock` for reuse
- Proofs are ~500KB base64-encoded (significant receipt size when proof is included)
- Verification requires the same model function pointer and trace length
- Tied to the JOLT Atlas fork at `github.com/ICME-Lab/zkml-jolt`
