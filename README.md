# hrxdb

An immutable, GPU-resident flat vector index on [hrx-rs](https://github.com/zacharydenton/hrx-rs). It exhaustively
searches FP16 vectors using FP32 cosine scores and returns the best 1–32 matches.
The first target is **one query over 10 million 384-dimensional vectors** on Linux
with a gfx1151 GPU (Strix Halo). This is a library and benchmark, with no server,
persistence, updates, filtering, or ANN index yet.

## Quick start

Add the dependency to your application:

```toml
[dependencies]
hrxdb = "0.1"
```

```rust
use hrxdb::{Device, FlatIndex};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let mut db = FlatIndex::build(&device, 3, [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
    ])?;
    let neighbors = db.search(&[1.0, 0.0, 0.0], 2)?;
    assert_eq!(neighbors[0].id, 0);
    Ok(())
}
```

`build` accepts an `ExactSizeIterator` of anything implementing `AsRef<[f32]>`,
including borrowed slices or generated rows. Ingestion converts and uploads in
approximately 4 MiB chunks; it does not retain a second host corpus. IDs are
zero-based insertion positions, not external IDs. Keep application IDs in a
parallel array and resolve only the returned matches:

```rust
let external_ids = ["document-a", "document-b", "document-c"];
for neighbor in neighbors {
    println!("{}: {}", external_ids[neighbor.id as usize], neighbor.similarity);
}
```

`FlatIndex` is `Send`: build it on one thread and move it to a worker thread.
It is not `Sync`; searches require `&mut self`. Use `Arc<Mutex<FlatIndex>>` when
multiple callers should share one index with serialized queries. The index owns
its device resources, so the original `Device` handle need not outlive it.
See the runnable [search example](examples/search.rs).

The constructor normalizes each row, rounds it to FP16, and stores an FP32
inverse norm for the rounded row. Queries are normalized to FP32. Scores are
FP32 dot products corrected by that stored inverse norm. Search is exhaustive
over this **quantized representation**; FP16 rounding and FP32 arithmetic can
change rankings relative to the original vectors. Scores are not clamped.
Equal computed scores prefer the lower ID. Small indexes return `min(k, len)`
neighbors, and an empty index returns an empty vector for a valid query.

Zero-norm and nonfinite vectors, dimension mismatches, and k outside 1–32 are
errors. Logical dimensions are 1–16,384 and padded to a multiple of 128; the
primary 384-dimensional case needs no padding. Row count is limited to 2^30.
Available GPU memory and compiler resources may impose tighter limits for large
shapes. The tested dimension set is 1, 3, 127, 384, and 769.

## Kernel families

[`scan_family.loom`](kernels/scan_family.loom) uses Loom configuration, dependent
vector/view types, template providers, and explicit loop unrolling. Rust embeds
the authored source and supplies specialization values; it generates no kernel
source strings. Cosine and checksum providers share the same memory schedule.

The tuned default uses 128 threads, wave32, two rows per wave, and four adjacent
FP16 components per lane per load. At dimension 384 each lane reads 12 components
of each row, reuses query values, accumulates in FP32, and reduces within its
wave. Corpus values go directly into registers, with no LDS staging. Loom's
multidimensional views lower the corpus address into 64-bit pointer arithmetic;
tests place unique matches on both sides of 4 GiB and at the end of 7.68 GB.

[`select_family.loom`](kernels/select_family.loom) partitions the score array
into groups of 1,024. Each 256-thread group holds four candidates per thread,
repeatedly reduces to the maximum score and minimum matching ID, and invalidates
the winner. Loom workgroup reductions handle subgroup exchange, LDS publication,
and barriers. The same family selects from intermediate candidate pairs until
only the final top-k remains. Keeping each partition's top-k preserves the
global top-k. Invalid lanes use negative infinity and a sentinel ID.

At 10M × 384, storage is 7.68 GB of vectors, 40 MB of inverse norms, 40 MB of
scores, and about 5 MB of selection scratch. Scoring writes and selection reads
40 MB each, approximately 1% additional traffic relative to vectors. Corpus,
query, result, and scratch allocations are reused. Query/result host memory is
page-aligned and imported once into HRX, with explicit completion before host
access. Searches compile no kernels and allocate no new GPU buffers. Returned
`Vec<Neighbor>` values allocate on the host.

## Benchmark

Rust 1.88+, Linux x86_64, an AMD kernel driver, KFD/render permissions, and
a gfx1151 GPU are required for execution. The pinned native HRX bundle needs
glibc 2.43 or newer (Ubuntu 26.04 is its distribution baseline); see the
[HRX installation guide](https://github.com/zacharydenton/hrx-rs/blob/main/docs/GPU-NPU.md#native-setup).
Building and generating documentation do not initialize the GPU. The crate
uses the published `hrx-rs 0.4.0` dependency and needs no sibling checkout. HRX provisions its pinned runtime automatically; use
`HRX_OFFLINE=1` after provisioning to prohibit downloads.

```sh
cargo run --release --bin hrxdb-bench -- --output default.json
cargo run --release --bin hrxdb-bench -- --sweep --output sweep.json
# A smaller run:
cargo run --release --bin hrxdb-bench -- --rows 100000 --samples 10
```

Defaults: 10M rows, 384 dimensions, k=10, three warmups and 30 measured queries.
`--sweep` compares all 16 combinations of 128/256 threads, 1/2/4/8 rows per wave,
and 2/4 components per load on the same corpus. It selects the minimum median
full-search latency, preferring fewer VGPRs for results within 1%. Optional
`llvm-readobj` diagnostics supply register/LDS/scratch usage; set
`HRXDB_LLVM_READOBJ` to override its executable. Without it, selection uses
latency alone and resource fields are null. Saved benchmark reports use cache-relative
artifact identifiers so they contain no user-specific filesystem paths.

Each sample measures separately completed read-control, scoring, and full-search
invocations. Full search includes query normalization/publication, scan, GPU
selection, and result readback. Compilation and ingestion are excluded. HRX has
no GPU timestamps, so all timings use host wall time with explicit completion.
Queries vary between samples; no query batching is used. Useful GB/s divides
**unpadded FP16 vector bytes** by elapsed time, using decimal GB. It excludes
padding, norms, query, scores, and candidate traffic from the numerator. The
read control consumes each FP16 value into a checksum; it is a matching kernel
baseline, not a hardware memory-controller counter.

Before timing each configuration, the benchmark compares GPU selection with a
CPU top-k over the entire score array. It also compares sampled dots, including
the tail and 4 GiB boundary, and all returned similarities with CPU calculations
over identically quantized rows. It separately reports original-vs-quantized
top-k overlap on a generated subset of up to 8,192 rows; that statistic is not a
quality estimate for real embedding datasets.

See [measured results](https://github.com/zacharydenton/hrxdb/blob/main/results/README.md). The 256 GB/s value is a stretch target;
the practical acceptance criterion is at least 90% of the matching read control.

## Validation

```sh
cargo test --release
cargo test --release -- --ignored --test-threads=1
cargo clippy --all-targets -- -D warnings
```

Hardware tests cover all 16 schedules, CPU score/ranking comparisons, padded
dimensions, tails, empty/small indexes, invalid vectors, repeated queries, ties,
negative scores, all k values, and multi-level selection with exact synthetic
scores. The 10M-row test allocates approximately 7.8 GB and checks unique matches
around the 4 GiB boundary and at the allocation tail. Run hardware tests
separately from performance measurements to avoid GPU contention.


The repository's [release guide](RELEASE.md) covers packaging and publication.
See [CHANGELOG.md](CHANGELOG.md) for version history. Contributions should include
appropriate CPU or opt-in GPU coverage; report bugs with the GPU target, driver,
crate version, and a minimal reproducer. Hardware tests do not run in ordinary CI.
