# hrxdb

**A GPU-powered vector database for unified-memory systems, built on HRX and Loom.**

hrxdb is an embedded vector database designed for unified-memory systems like
AMD Strix Halo. [HRX](https://github.com/ROCm/hrx-system) manages GPU buffers,
streams, and execution; [Loom](https://github.com/ROCm/hrx-system/tree/main/loom)
kernels implement exact cosine search and top-k.
Your own Loom kernels can work directly with the same resident corpus.
Gather named rows, compute application-specific scores, and select top-k
without copying intermediate vectors or score matrices back to the CPU.

The library exposes a Rust API through
[`hrx-rs`](https://github.com/zacharydenton/hrx-rs). Execution currently targets
**AMD Strix Halo (`gfx1151`) on Linux x86_64**. Rust 1.88+ is required; other GPUs
and operating systems are not supported for execution in this release.

Unified memory makes large resident collections possible without a separate
discrete GPU memory pool. hrxdb builds around that model: ingest once, reuse
the corpus across searches and custom scoring, and read back only what the
application needs. Host ingestion still copies into owned runtime buffers;
unified memory does not make every operation zero-copy or remove bandwidth limits.

- **Exact cosine search:** exhaustive FP16 corpus scans, FP32 scores, deterministic
  ties, exclusions, and top-k up to 1,024.
- **Shared-read batches:** up to 64 queries per batch with bounded score tiles
  and running GPU top-k.
- **Composable storage:** cheap-clone `Corpus` handles, independent `Searcher`
  workspaces, read-only shard bindings, and device-resident row gathering.
- **Custom GPU pipelines:** standalone `TopK`, device query/result buffers,
  completion events, and persistent exclusion bitmaps.
- **Large resident corpora:** automatic sharding, direct FP16 ingestion,
  validated zero-copy adoption of device buffers, and memory accounting.

hrxdb is an embedded vector index and GPU computation library. Persistence,
embedding generation, application metadata, updates, approximate search, and
server APIs belong to the application. Search is exact over the **stored
quantized representation**; FP16 rounding and FP32 arithmetic can change
rankings relative to the original vectors.

## Quick start

Add the dependency:

```toml
[dependencies]
hrxdb = "0.1"
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

IDs are zero-based insertion positions. Keep external IDs and metadata in
parallel application-owned storage. `Corpus::build` accepts a sized iterator
of FP32 rows, normalizes and quantizes them, and uploads in bounded chunks.
`Corpus::build_fp16` preserves supplied little-endian FP16 values and computes
their inverse norms. Finite, nonzero rows are required.

From a checkout, run the [search example](examples/search.rs):

```sh
cargo run --locked --release --example search
```

Building does not initialize GPU hardware or download native code. First GPU
or compiler use provisions the native bundle pinned by `hrx-rs =0.4.0`.
Execution needs the AMD kernel driver, `/dev/kfd` and render-device permissions,
compatible C/C++ runtime libraries and `libatomic`, and **glibc 2.43+**
(the bundle's baseline is Ubuntu 26.04). See the
[HRX setup guide](https://github.com/zacharydenton/hrx-rs/blob/v0.4.0/docs/GPU-NPU.md#native-setup).
After provisioning, `HRX_OFFLINE=1` prevents runtime downloads.

## Choose your operation

| Task | API |
|---|---|
| Search one query or a batch | `Searcher::search`, `search_batch` |
| Reuse host result capacity | `search_into`, `search_batch_into`, `scores_into` |
| Share one corpus across workers | `Corpus::clone`, `Corpus::searcher` |
| Gather selected rows for a custom kernel | `Corpus::gather_into` |
| Bind resident vectors without a copy | `Corpus::shards` |
| Adopt owned FP16 device allocations | `Corpus::from_device` |
| Search with GPU-produced queries | `Searcher::search_device`, `DeviceQueries` |
| Rank application-defined GPU scores | `TopK`, `ScoreBatch` |
| Keep results and exclusions on the GPU | `DeviceNeighbors`, `DeviceExclusions` |
| Account for shared storage and workspace | `Corpus::memory_usage`, `Searcher::memory_usage` |

`Corpus` is `Clone + Send + Sync`; clones share immutable allocations. Each
`Searcher` owns its stream and workspace. Independent workers allow independent
submission, but do not guarantee overlapping execution, higher throughput,
priority, or latency isolation on a saturated GPU. Use batching to share corpus
reads across queries. Use events to order work between streams.

Public `MAX_BATCH` and `MAX_K` expose the supported query and result ceilings.
Logical dimensions are 1–16,384; corpora support up to 2^30 rows, subject to
available memory. Internal shards contain at most 2^32 FP16 elements and include
readable row slack for custom tiled kernels. Only logical rows participate in
search; custom kernels must mask slack.

See the [API and execution guide](docs/guide.md) for buffer layouts,
normalization, exclusions, memory limits, and synchronization contracts.
See the [API reference](https://docs.rs/hrxdb/0.1.0/hrxdb/), or generate it locally
with `cargo doc --locked --no-deps --open`.

## Build your own scoring pipeline

```text
resident corpus → gather named subsets → custom scoring kernel → GPU top-k
                                                                    ↓
                                                      selected IDs and scores
```

`gather_into` writes compact normalized FP32 rows, preserving requested order
and duplicates while resolving shards internally. `TopK` consumes your own
row-major score matrix and returns score-column IDs. Album grouping, custom
reductions, and metadata stay in application code.

Runnable examples:

- [Subset comparisons](examples/subset_scores.rs): gather two blocks, score
  them on the GPU, and read back only the winners.
- [Album ranking](examples/album_scores.rs): bind corpus vectors and an album
  side array, reduce scores by album, then select top-k.
- [Device search](examples/device_search.rs): queue queries and update visited
  IDs on the GPU, with event-ordered result consumption.

The custom scoring kernels illustrate interoperability; they are not tuned GEMMs.
Applications using HRX types directly should also depend on
`hrx = { package = "hrx-rs", version = "=0.4.0" }`.

## Measured performance

Recorded on local gfx1151 with generated data, 2026-09-11:

| Workload | Median host-completion latency |
|---|---:|
| One top-10 query over 10M × 384 vectors | 33.7 ms |
| Sixty top-5 queries over 6,909,092 × 384 vectors | 80.7 ms |

These are separate runs, not a batching speedup comparison. The sixty-query
run was **1.42× faster than the preceding batch implementation** in an
interleaved comparison with identical returned IDs and score bits. Smaller
shapes can be slower. Timings exclude ingestion and compilation; clocks and
other system activity were not controlled. They are not embedding-quality
measurements or guarantees for a particular collection.

The [results and raw samples](https://github.com/zacharydenton/hrxdb/blob/main/results/README.md) include methodology,
memory use, compiler reports, and reproduction commands. To benchmark locally:

```sh
cargo run --locked --release --bin hrxdb-bench -- --rows 100000 --samples 10
cargo run --locked --release --bin hrxdb-bench -- --rows 6909092 --batch 60 --k 5 --samples 10
```

## Development and release

CPU tests and documentation work without GPU hardware:

```sh
cargo test --locked --all-targets
cargo test --locked --doc
cargo clippy --locked --all-targets -- -D warnings
```

On a prepared gfx1151 host, run hardware correctness tests separately from
performance benchmarks:

```sh
HRX_OFFLINE=1 cargo test --locked --release -- --ignored --test-threads=1 --skip batch::bench
```

The largest test allocates about 9.4 GB. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, validation, and reporting issues; [CHANGELOG.md](CHANGELOG.md) for
the release contents; and [RELEASE.md](RELEASE.md) for publication steps.

MIT licensed. The separately distributed HRX runtime and compiler have their
own [third-party notices](https://github.com/zacharydenton/hrx-rs/blob/v0.4.0/THIRD-PARTY.md).
