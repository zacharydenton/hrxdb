# hrxdb

An embedded GPU vector index for AMD Strix Halo, written in Rust.
[Loom](https://github.com/ROCm/hrx-system) kernels perform cosine search and
top-k; [HRX](https://github.com/zacharydenton/hrx-rs) owns the resident buffers
and executes the work. Custom kernels can use the same corpus without reading
intermediate vectors back to the CPU.

| Recorded workload | Median completion time |
| --- | ---: |
| One top-10 query over 10M × 384 vectors | 33.7 ms |
| Sixty top-5 queries over 6,909,092 × 384 vectors | 80.7 ms |
| Append 50 rows to 7M × 768 vectors | 0.54 ms |
| Replace 100 scattered rows in an unshared 7M × 768 corpus | 0.88 ms |

Local `gfx1151` measurements from September 2026 with generated data; ingestion
and compilation are excluded. The rows are separate workloads, not a batching
speedup comparison. [Results and methodology](results/README.md).

## Features

- Exhaustive cosine search over FP16 vectors, with FP32 scores and deterministic ties.
- Up to 256 queries per batch and top-k up to 1,024.
- Append and update snapshots while existing readers keep their original rows.
- Shared corpus handles, independent search workspaces and native snapshot files.
- Device queries, row gathering and custom scoring followed by GPU top-k.

Search is exact over the stored representation; quantization can change rankings
relative to the original vectors. Embeddings, metadata and server APIs belong to
the application.

## Quick start

Requires Rust 1.91+, Linux x86-64 and Strix Halo (`gfx1151`).

```toml
[dependencies]
hrxdb = "0.3.1"
```

```rust
use hrxdb::{Corpus, Device};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
    ])?;
    let mut searcher = corpus.searcher()?;
    let neighbors = searcher.search(&[1.0, 0.0, 0.0], 2)?;
    assert_eq!(neighbors[0].id, 0);
    Ok(())
}
```

IDs are insertion positions. `Corpus::build` normalizes and quantizes FP32 input;
`build_fp16` preserves supplied FP16 values. From a checkout, run
`cargo run --locked --release --example search`.

First GPU use downloads HRX's native bundle. Execution requires amdgpu/KFD access,
glibc 2.43+ and compatible system libraries.
[Native setup](https://github.com/zacharydenton/hrx-rs/blob/main/docs/GPU-NPU.md#native-setup).

## Use an existing HRX context

`Corpus::build_in` and `build_fp16_in` allocate in an encoder's `ModelContext`.
Keep queries, gathered rows and scores resident through a custom pipeline,
then download the selected IDs and scores.

[Storage and snapshots](docs/storage.md) · [Custom scoring](docs/scoring.md) ·
[Layouts and synchronization](docs/guide.md) · [API reference](https://docs.rs/hrxdb)

## Development

```sh
cargo test --locked --all-targets
cargo test --locked --doc
cargo clippy --locked --all-targets -- -D warnings
cargo run --locked --release --bin hrxdb-bench -- --rows 100000 --samples 10
```

CPU checks need no GPU. Hardware tests run separately:

```sh
HRX_OFFLINE=1 cargo test --locked --release -- --ignored --test-threads=1 --skip batch::bench
```

The largest test allocates about 9.4 GB.
[Profiling](docs/profiling.md) · [Contributing](CONTRIBUTING.md) ·
[Changelog](CHANGELOG.md) · [Releases](RELEASE.md)

## License

[MIT](LICENSE). The HRX runtime and compiler have separate
[third-party notices](https://github.com/zacharydenton/hrx-rs/blob/main/THIRD-PARTY.md).
