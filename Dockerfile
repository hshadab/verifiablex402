# Build context: jolt-atlas workspace root
#   docker build -t verifiablex402 -f verifiablex402/Dockerfile .
FROM rust:1.88-bookworm AS builder

RUN rustup target add riscv32im-unknown-none-elf
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace root files
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Copy workspace members needed by verifiablex402
COPY onnx-tracer/ onnx-tracer/
COPY zkml-jolt-core/ zkml-jolt-core/
COPY verifiablex402/ verifiablex402/

# Build
RUN cargo build --release --package verifiablex402

# --- Runtime ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/verifiablex402 /app/verifiablex402
COPY --from=builder /build/dory_srs_22_variables.srs /app/
COPY --from=builder /build/dory_srs_24_variables.srs /app/

# SRS checksum verification
RUN sha256sum dory_srs_*.srs > srs_checksums.txt

EXPOSE 10000
ENV RUST_LOG=verifiablex402=info

CMD ["sh", "-c", "sha256sum -c srs_checksums.txt && exec /app/verifiablex402 serve --bind 0.0.0.0:10000"]
