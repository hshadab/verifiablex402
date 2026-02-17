# ADR 001: Fixed-Point Arithmetic (i32 at Scale 128)

## Status

Accepted

## Context

The transaction integrity classifier must run inside a JOLT Atlas zkVM to generate verifiable proofs. The JOLT proving system operates on a RISC-V ISA (riscv32im) which has native 32-bit integer operations but no floating-point unit. Using f32/f64 would require software float emulation, dramatically increasing the trace length and proof generation time.

## Decision

All model weights and feature values use **i32 fixed-point arithmetic at scale 2^7 (128)**:

- **Feature values**: Normalized to `[0, 128]` range (24 integer features)
- **Model weights**: Quantized from PyTorch float32 by multiplying by 128 and rounding to i32
- **Intermediate computations**: Matmul results are at scale 128*128 = 16384, divided by 128 after each layer to return to scale 128
- **Output scores**: Raw i32 values; argmax determines classification

### Why Scale 128 (2^7)?

- **Power of 2**: Division by 128 is a right-shift by 7 bits, which is a single RISC-V instruction (`srai`)
- **Sufficient precision**: 7 bits of fractional precision (~0.8% resolution) is adequate for the MLP's 5-class classification task
- **No overflow**: With 24 input features at max 128 and 36 hidden neurons, the worst-case matmul accumulation fits in i32 (24 * 128 * 128 = 393,216, well under i32 max)

## Consequences

- Quantization introduces small accuracy loss (~0.1-0.5% observed on synthetic data)
- The `validate_model.py` script verifies that float-vs-quantized accuracy delta is < 1%
- All feature normalization ranges are fixed constants — adding new features requires retraining
- Softmax is computed in f64 *outside* the proof (in the receipt construction), so it does not affect proof correctness
