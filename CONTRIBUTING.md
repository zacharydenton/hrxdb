# Contributing to hrxdb

hrxdb is an embedded GPU-powered vector database for unified-memory systems,
built on HRX and Loom.
The current execution target is AMD Strix Halo (`gfx1151`) on Linux x86_64.
Start with the [README](README.md) and [API and execution guide](docs/guide.md).

## Build and check

Use Rust 1.88 or newer. Compilation, CPU tests, and documentation do not require
GPU hardware or download a native runtime. CI checks the minimum supported Rust
version and stable:

```sh
cargo fmt --all -- --check
HRX_OFFLINE=1 cargo clippy --locked --all-targets -- -D warnings
HRX_OFFLINE=1 cargo test --locked --all-targets
HRX_OFFLINE=1 cargo test --locked --doc
HRX_OFFLINE=1 RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
cargo package --locked
```

Use `cargo +1.88.0` to check the minimum supported version explicitly. Cargo
packaging requires a clean working tree unless you add `--allow-dirty` while
iterating. The runtime/compiler dependency is pinned to the qualified bundle.

## Hardware validation

Follow the [runtime setup](https://github.com/zacharydenton/hrx-rs/blob/v0.5.0/docs/GPU-NPU.md#native-setup)
on a supported host. First use provisions the pinned native bundle; after
provisioning, run with `HRX_OFFLINE=1` to prevent downloads:

```sh
HRX_OFFLINE=1 cargo test --locked --release -- --ignored --test-threads=1 --skip batch::bench
HRX_OFFLINE=1 cargo run --locked --release --example search
HRX_OFFLINE=1 cargo run --locked --release --example album_scores
HRX_OFFLINE=1 cargo run --locked --release --example device_search
HRX_OFFLINE=1 cargo run --locked --release --example subset_scores
```

The largest correctness test allocates about 9.4 GB. Run hardware checks
sequentially, separately from benchmarks and other GPU workloads. Ordinary CI
cannot validate GPU execution. If you cannot run hardware checks, say so in
the pull request so a maintainer can qualify the change.

Kernel changes need relevant hardware coverage, including applicable tails,
padding, invalid inputs, ties, shard boundaries, and stream ordering. Compare
against the existing CPU reference or a known result. Preserve documented
numerical and ownership contracts, or explain the intended API change.

## Performance changes

Use generated data and record the command, Rust and runtime versions, GPU,
shape, schedule, warmup, and latency distribution. Keep ingestion and compilation
separate from warmed search latency. Compare implementations under the same
conditions; interleave runs when possible. Small shapes and batch widths matter
as well as the large-corpus target. See [recorded methodology](https://github.com/zacharydenton/hrxdb/blob/main/results/README.md)
and `cargo run --release --bin hrxdb-bench -- --help`.

## Issues and pull requests

Open an [issue](https://github.com/zacharydenton/hrxdb/issues) with a small
reproducer, expected and actual behavior, crate or commit version, Rust version,
OS/glibc and AMD driver versions, GPU target, and corpus/query dimensions.
For device API problems, show buffer layouts and stream/event ordering.
Use synthetic vectors instead of personal collections or proprietary embeddings.

Describe the concrete problem, resulting behavior, and checks run in a pull
request. Update API documentation and runnable examples when contracts change.
Keep benchmark artifacts focused on evidence needed to assess the change.
