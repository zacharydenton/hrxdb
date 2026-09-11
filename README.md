# hrxdb

An immutable, GPU-resident flat vector index on [hrx-rs](https://github.com/zacharydenton/hrx-rs). It exhaustively
searches FP16 vectors using FP32 cosine scores and returns the best 1–1,024 matches,
with optional excluded IDs. Large corpora are sharded internally.
The first target is **one query over 10 million 384-dimensional vectors** on Linux
with a gfx1151 GPU (Strix Halo). This is a library and benchmark, with no server,
persistence, updates, metadata filters, or ANN index yet.

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

For a corpus already stored as FP16, use `build_fp16` (or
`build_fp16_with_config`). Each row is an `AsRef<[u8]>` containing exactly
`dimensions * 2` little-endian IEEE 754 binary16 bytes, without padding:

```rust
let db = FlatIndex::build_fp16(&device, dimensions, corpus_bytes.chunks_exact(dimensions * 2))?;
```

The bytes are preserved, zero padding is added, and inverse norms are computed
in one pass. Rows need not be normalized. This avoids allocating FP32 rows and
rounding them back to FP16. Reject incomplete trailing rows when reading a file:
`chunks_exact` omits any remainder. Both constructors require an accurately
sized iterator and upload in bounded chunks.

`FlatIndex` is `Send`: build it on one thread and move it to a worker thread.
It is not `Sync`; searches require `&mut self`. Use `Arc<Mutex<FlatIndex>>` when
multiple callers should share one index with serialized queries. The index owns
its device resources, so the original `Device` handle need not outlive it.
See the runnable [search example](examples/search.rs).

The FP32 constructor normalizes each row, rounds it to FP16, and stores an FP32
inverse norm for the rounded row. Queries are normalized to FP32. Scores are
FP32 dot products corrected by that stored inverse norm. Search is exhaustive
over this **quantized representation**; FP16 rounding and FP32 arithmetic can
change rankings relative to the original vectors. Scores are not clamped.
For `build_fp16`, the quantized representation is the supplied values; preserving
them may produce different scores from normalizing FP32 rows before rounding.
Equal computed scores prefer the lower ID. Small indexes return `min(k, len)`
neighbors, and an empty index returns an empty vector for a valid query.

Zero-norm and nonfinite vectors, dimension mismatches, and k outside 1–1,024 are
errors. Logical dimensions are 1–16,384 and padded to a multiple of 128; the
primary 384-dimensional case needs no padding. Row count is limited to 2^30,
and each internal allocation contains at most **2^32 FP16 elements** (8 GiB
of vector storage), respecting the compiler's 32-bit element-index limit.
This is 8,388,608 rows per shard at 512 dimensions or 11,184,810 at 384.
Both constructors automatically split larger corpora. For example, 8,942,135 ×
512 faces use two shards while retaining one index and global insertion IDs.
Scans write each shard's scores into its range of a shared score array; GPU
selection finds the best k across the entire corpus. `shard_count()` reports
the number of corpus allocations.
Available GPU memory and compiler resources may impose tighter limits for large
shapes. The tested dimension set is 1, 3, 127, 384, and 769.

## Larger result sets and exclusions

Request up to 1,024 candidates and collapse them using application metadata.
For example, keep the highest-scoring photo from each album until a page is full:

```rust
db.reserve_search(1024)?; // Optional setup: reserve scratch before serving queries.
let candidates = db.search(query, 1024)?;
let mut seen_albums = std::collections::HashSet::new();
let page: Vec<_> = candidates.into_iter()
    .filter(|n| seen_albums.insert(album_ids[n.id as usize]))
    .take(60)
    .collect();
```

The page can contain fewer than 60 albums if the candidates do not cover that
many distinct albums. A greedy similarity walk can exclude its visited IDs
directly, without reading back the full score array:

```rust
let next = db.search_excluding(query, 1, &visited_ids)?;
if let Some(neighbor) = next.first() {
    visited_ids.push(neighbor.id);
}
```

`search_excluding(query, k, excluded)` accepts unsorted, repeated insertion IDs.
Out-of-range IDs are errors. It returns `min(k, remaining_rows)` matches, or an
empty vector when every row is excluded. Exclusions apply to one query and
compose with the full k range and internal sharding. Both search methods run
selection on the GPU and read back at most k score/ID pairs.

Construction reserves single-query selection scratch for k=32. The first larger query grows
it to the next power of two, then reuses that capacity. `reserve_search(k)` lets
applications make that allocation during setup. Single-query searches compile no kernels.

## Batched queries

Use `search_batch` for up to **64 queries over the same corpus**, such as sixty
sampled album photos each requesting five neighbors:

```rust
db.reserve_batch(60, 5)?; // Optional: allocate and compile before serving requests.
let matches = db.search_batch(&queries, 5)?;
// queries is 60 * db.dimensions() FP32 components in row-major order.
// matches[i] contains the neighbors for query i.
```

`search_batch_excluding(&queries, k, &excluded_ids)` applies one shared exclusion
set to the batch. Both methods support k=1–1,024 and internal corpus sharding.
Exclusions may be unsorted and repeated, and affect only that call. Every query
is validated before GPU execution; incomplete rows, nonfinite or zero-norm
queries, more than 64 queries, and out-of-range excluded IDs are errors. Empty
batches return an empty vector; an empty index returns one empty result per
valid query. A one-query batch uses the single-query scan.

The [matrix kernel](kernels/batch_scan.loom) loads a tile of 64 corpus rows into workgroup memory and
reuses it across the queries. Corpus values expand from FP16 to FP32 in that
tile; normalized queries stay FP32, and accumulation is FP32. Its accumulation
order differs from the single-query scan, so very close scores can change rank.
Equal computed scores still prefer the lower insertion ID.

Scores are materialized for at most 262,144 corpus rows at a time. GPU selection
keeps each query's tile top-k, then merges it with that query's running top-k.
The full corpus × batch score matrix is never allocated, and only final results
are read back (2,400 bytes for sixty top-5 queries).

Batch workspace grows to accommodate the largest reserved query width and k,
and is reused. Widths round up to 8, 16, 32, or 64, with compiled kernels cached
per width. At width 64 the workspace uses approximately **66 MiB for k=5** or
**322 MiB for k=1,024**, plus query storage (96 KiB at dimension 384). Smaller
corpora need less. This storage is additional to the index and single-query
scratch; `batch_workspace_bytes()` reports the reserved buffer bytes. First use
can allocate and compile, so use `reserve_batch(query_count, k)` during setup
when first-request latency matters. `ScanConfig` tunes the single-query scan;
the matrix kernel has its own schedule.

On 6.9M × 384 generated rows, sixty top-5 queries took **165.9 ms batched**
versus **1,863 ms individually** (11.2×), with 66.11 MiB of workspace.
See the [measurement conditions and samples](results/README.md#sixty-query-batches).

## Full-score queries

`scores` and `scores_into` are supported query entry points for larger result
sets beyond 1,024 or custom host-side ranking.
They return the same cosine scores used by `search`, in insertion-ID order.
`scores` allocates a `Vec<f32>`; `scores_into` writes into an existing slice of
exactly `db.len()` entries so it can be reused across queries:

```rust
let mut scores = vec![0.0; db.len()]; // Allocate once per worker, outside its query loop.
db.scores_into(query, &mut scores)?;
```

Readback transfers four bytes per row and blocks until the output is ready.
`scores_into` allocates no host score array or GPU buffer, but full score transfer
and host selection still cost more than device top-k. Score readback always
returns all rows, regardless of exclusions supplied to previous searches.

## Kernel families

[`scan_family.loom`](kernels/scan_family.loom) uses Loom configuration, dependent
vector/view types, template providers, and explicit loop unrolling. Rust embeds
the authored source and supplies specialization values; it generates no kernel
source strings. Cosine and checksum providers share the same memory schedule.

The tuned default uses 128 threads, wave32, two rows per wave, and four adjacent
FP16 components per lane per load. At dimension 384 each lane reads 12 components
of each row, reuses query values, accumulates in FP32, and reduces within its
wave. Corpus values go directly into registers, with no LDS staging. Loom's
multidimensional views use 64-bit byte pointers with a 32-bit element index;
tests place unique matches on both sides of 4 GiB and at the end of 7.68 GB.
The 8,942,135 × 512 test covers the 8 GiB shard boundary and corpus tail.

[`select_family.loom`](kernels/select_family.loom) partitions the score array
into groups of 1,024. Each 256-thread group holds four candidates per thread,
repeatedly reduces to the maximum score and minimum matching ID, and invalidates
the winner. Loom workgroup reductions handle subgroup exchange, LDS publication,
and barriers. The same family selects from intermediate candidate pairs until
only the final top-k remains. Keeping each partition's top-k preserves the
global top-k. This path serves k=1–32. Invalid lanes use negative infinity and
a sentinel ID.

For k=33–1,024, [`sort_family.loom`](kernels/sort_family.loom) bitonic-sorts
1,024-row tiles in 8 KiB of workgroup memory and keeps each tile's top k.
Pairwise merges rank each candidate by binary-searching the other sorted list,
then retain the best k. Every round halves the number of lists, including at
k=1,024. Both selection paths order equal scores by ascending insertion ID.

Excluded IDs populate a reusable, imported host bitmap (one bit per row).
[`mask.loom`](kernels/mask.loom) marks excluded scores as negative infinity on
the GPU before selection. Search omits these entries from readback results.
The following scan overwrites the scores, so exclusions cannot leak into later
queries. Unmasked searches skip this dispatch entirely.

At 10M × 384, storage is 7.68 GB of vectors, 40 MB of inverse norms, 40 MB of
scores, a 1.25 MB exclusion bitmap, and about 5 MB of initial selection scratch
(about 160 MB at k=1,024). Scoring writes and initial selection reads
40 MB each, approximately 1% additional traffic relative to vectors. Corpus,
query, result, and scratch allocations are reused. Query/result host memory is
page-aligned and imported once into HRX, with explicit completion before host
access. Once selection capacity is reserved, searches allocate no new GPU buffers. Returned
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
# Larger result sets, with the first 100,000 IDs excluded:
cargo run --release --bin hrxdb-bench -- --k 1024 --exclude-first 100000 --output large-k.json
# Compare sixty top-5 queries with one shared-read batch:
cargo run --release --bin hrxdb-bench -- --rows 6900000 --batch 60 --k 5 --samples 10 --output batch.json
```

Defaults: 10M rows, 384 dimensions, k=10, three warmups and 30 measured queries.
`--k` accepts 1–1,024. `--exclude-first N` excludes IDs 0 through N−1 and must
leave at least one row. Full-search timings include bitmap preparation and
device masking. Scratch reservation happens before validation and timing.
`--batch B` compares individual queries with a tiled batch, alternating timing
order and checking results against individual GPU queries and CPU scores.
It records workspace bytes and counts rank differences within numerical
tolerance. The batch benchmark supports exclusions but cannot use `--sweep`.
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
scores. FP16 ingestion tests cover byte preservation, norm correction, invalid
rows, chunk boundaries, and reusable score readback. CPU tests check shard
layouts without allocating large corpora. Hardware tests compare sharded and
single-allocation results and exercise exclusions, growing walks, all-excluded
queries, large k, and odd merge tails. The 10M-row test allocates approximately
7.8 GB and checks unique matches around the 4 GiB boundary and at the allocation
tail. The faces-shape test allocates about 9.4 GB and crosses the compiler's
2^32-element boundary. Run hardware tests
separately from performance measurements to avoid GPU contention.

Batch tests cover query widths, row-major input validation, shared exclusions,
FP16 extremes and padded dimensions, exact ties, running selection across tiles
and shards, workspace reuse, and ownership across threads. The faces-shape test
also runs sixty queries across the 8 GiB shard boundary with bounded workspace.


The repository's [release guide](RELEASE.md) covers packaging and publication.
See [CHANGELOG.md](CHANGELOG.md) for version history. Contributions should include
appropriate CPU or opt-in GPU coverage; report bugs with the GPU target, driver,
crate version, and a minimal reproducer. Hardware tests do not run in ordinary CI.
