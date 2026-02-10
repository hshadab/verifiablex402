FROM rust:1.88-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone the jolt-atlas workspace (contains onnx-tracer, zkml-jolt-core)
RUN git clone --depth 1 https://github.com/ICME-Lab/jolt-atlas.git .

# Remove the workspace's version of verifiablex402 (if any) and replace with ours
RUN rm -rf verifiablex402

# Copy our code into the workspace
COPY . verifiablex402/

# Build in release mode
RUN cargo build -p verifiablex402 --release

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/verifiablex402 /usr/local/bin/verifiablex402

# Copy SRS file if present in workspace
COPY --from=builder /build/dory_srs_22_variables.srs /app/dory_srs_22_variables.srs
COPY --from=builder /build/dory_srs_24_variables.srs /app/dory_srs_24_variables.srs

WORKDIR /app

ENV RUST_LOG=verifiablex402=info

EXPOSE 8080

CMD ["verifiablex402", "serve", "--bind", "0.0.0.0:8080"]
