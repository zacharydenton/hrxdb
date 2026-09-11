# hrxdb

Shared GPU-resident vector storage, exact cosine search, and composable GPU top-k on [hrx-rs](https://github.com/zacharydenton/hrx-rs). It exhaustively
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
use hrxdb::{Corpus, Device};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
    ])?;
    let mut db = corpus.searcher()?;
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

For a corpus already stored as FP16, use `Corpus::build_fp16`. Each row is an `AsRef<[u8]>` containing exactly
`dimensions * 2` little-endian IEEE 754 binary16 bytes, without padding:

```rust
let corpus = Corpus::build_fp16(&device, dimensions, corpus_bytes.chunks_exact(dimensions * 2))?;
```

The bytes are preserved, zero padding is added, and inverse norms are computed
in one pass. Rows need not be normalized. This avoids allocating FP32 rows and
rounding them back to FP16. Reject incomplete trailing rows when reading a file:
`chunks_exact` omits any remainder. Both constructors require an accurately
sized iterator and upload in bounded chunks.

`Corpus` owns immutable storage and is `Clone + Send + Sync`. Cloning retains
those allocations without copying vectors. Each `corpus.searcher()` creates a
`Send` worker with its own ordered stream, kernels, and workspace; queries take
`&mut self`. Workers can run independently on different threads. Keep a corpus
clone as a snapshot, or replace the application's current corpus while existing
workers finish using the old one. The original `Device` may be dropped.

```rust
let mut first = corpus.searcher()?;
let mut second = corpus.searcher()?; // Shared vectors, independent scratch.
let snapshot = corpus.clone();      // No new GPU allocation.
```

`Searcher::new(corpus)` takes ownership of a handle. `Searcher::on_stream`
accepts an existing stream and `ScanConfig` for applications with an established
queue. `Searcher::build` and `build_fp16` are convenience constructors combining
storage and one worker; `Corpus::build` alone allocates no search workspace.
See the runnable [search example](examples/search.rs).

This redesign replaces `FlatIndex` and the borrowed `CorpusView`; there are no
compatibility aliases. `searcher.corpus()` now returns `&Corpus`, which callers
can clone to retain storage independently of the worker.

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
Shard capacity is rounded to 256 rows, including readable tail slack. This
permits 8,388,608 logical rows per shard at 512 dimensions or 11,184,640 at 384.
Both constructors automatically split larger corpora. For example, 8,942,135 ×
512 faces use two shards while retaining one index and global insertion IDs.
Scans write each shard's scores into its range of a shared score array; GPU
selection finds the best k across the entire corpus. `shard_count()` reports
the number of corpus allocations.
Available GPU memory and compiler resources may impose tighter limits for large
shapes. Hardware coverage includes dimensions 1, 3, 127, 129, 384, 512, and 769.

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

`search_into(query, k, &mut results)` and
`search_excluding_into(query, k, excluded, &mut results)` reuse the capacity of
a caller-owned `Vec<Neighbor>`. Their allocating counterparts are convenience
wrappers.

Searcher construction reserves single-query selection scratch for k=32. The first larger query grows
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

`search_batch_into` and `search_batch_excluding_into` accept a reusable
`Vec<Vec<Neighbor>>`, retaining capacities of rows that survive a batch resize.

The [matrix kernel](kernels/batch_scan.loom) loads a tile of 64 corpus rows into workgroup memory and
reuses it across the queries. Corpus values stay FP16 in workgroup memory and
expand to FP32 in registers; normalized queries and accumulation stay FP32. A
compiler scheduling fence bounds operand lifetimes to one column at a time.
The increasing-component accumulation order differs from the single-query scan,
so very close scores can change rank.
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

On 6,909,092 × 384 generated rows, sixty top-5 queries took **80.7 ms** with
the optimized batch scan versus **114.7 ms** with its predecessor (1.42×),
with identical returned IDs and scores. Workspace remains 66.11 MiB. See the
[interleaved comparison and samples](results/README.md#batch-scan-optimization);
the earlier batch-versus-individual measurement is retained separately.

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

## Custom GPU computations

`db.corpus()` returns `&Corpus` over the existing GPU allocations.
It performs no copy, allocation, compilation, or GPU work. Applications can bind
their own Loom kernels to these vectors, together with their own side arrays
and output buffers. Add `hrx = { package = "hrx-rs", version = "=0.4.0" }` to
the application's dependencies to use the matching runtime and compiler API.

```rust
let corpus = db.corpus();
let stream = corpus.stream()?; // A new stream on the index's device.
let compiler_options = hrx::loom::CompilerOptions {
    target: stream.target().clone(),
    ..Default::default()
};
for shard in corpus.shards() {
    let rows = shard.row_range(); // Global insertion IDs; exclusive end.
    let capacity = shard.capacity_rows(); // Readable rows, including tail slack.
    let vectors = shard.vectors(); // Borrowed hrx::View, local rows 0..capacity.
    let inverse_norms = shard.inverse_norms(); // Same local row numbering.
    // This caller-owned side array must also have enough readable tail space.
    let albums = album_buffer.try_slice(rows.start * 4, capacity * 4)?;
    // Bind vectors, inverse_norms, albums, and application-owned output.
    // Only local rows < rows.len() may contribute to the result.
}
```

`Corpus` exposes `stream`, `len`, `is_empty`, `dimensions`, `padded_dimensions`,
and `shards`. `CorpusShardView` exposes `row_range`, `capacity_rows`, `vectors`, and
`inverse_norms`.
Shard ranges cover all insertion IDs in order without gaps; an empty corpus
has no shards. Both bindings cover `capacity_rows()`, the logical shard length
rounded up to a multiple of 256. Each logical vector row contains little-endian
FP16 with zero dimension padding and a byte stride of `padded_dimensions() * 2`.
Each logical norm entry is one little-endian FP32 inverse norm. Multiply the dot
with a normalized query by this inverse norm for cosine similarity. These are
the stored values: `build` normalizes before FP16 conversion, whereas
`build_fp16` preserves its input values.

Rows between `row_range().len()` and `capacity_rows()` are readable slack with
**unspecified contents**, not indexed vectors. They have no insertion IDs.
A fixed 256-row kernel can read its final tile in full, but must mask slack by
local row index before reduction or selection. Do not rely on zero contents:
a zero score could beat every real negative cosine. Side-array loads must also
be in bounds; padding the ordinal array with a sentinel is one option. For other
tile sizes, check `rows.len().div_ceil(tile_rows) * tile_rows <= capacity_rows()`.
Each shard adds at most 255 rows of vector and norm storage; that capacity still
respects the 2^32-element limit. Built-in searches process only logical rows.

`corpus.stream()` creates an independent, owned stream on the original device,
including after the original `Device` is dropped or a corpus clone moves to
another thread. Configure the compiler with `stream.target()` and reuse the
stream for repeated work. The stream can outlive the corpus, but borrowed
bindings require a live corpus handle.

`db.stream()` borrows the searcher's **existing ordered stream**. Submit custom
producers, device searches, and consumers there to preserve execution order.
For separate streams, record a producer event and call `wait_event` on the
consumer stream. No accessor implicitly waits for work on another queue.

Corpus uploads have completed when construction returns. Callers own compilation,
submission, side data, scratch, and completion; borrowing a view does not wait
for asynchronous GPU work. **Treat both corpus bindings as read-only.** HRX's
binding type does not enforce this: writes are unsupported and invalidate the
index's data invariants. Normal HRX dispatch safety requirements still apply.

The runnable [album example](examples/album_scores.rs) binds an
[application-owned kernel](examples/album_scores.loom) and an album ordinal per
corpus row. It computes the best cosine for each `(query row, album)` directly
on the device, merges maxima across shards, and passes the score matrix to
`TopK`. Only the selected albums are read back. Its custom kernel reads full
256-row tiles and masks slack, including when an interior shard's side-array
slack overlaps the next shard's logical rows. Missing albums produce negative
infinity. The example favors clarity over throughput; album grouping and
reduction remain application code. It needs a corpus and an ordered stream,
and drops the original device before calling the operation.

```sh
HRX_OFFLINE=1 cargo run --locked --release --example album_scores
```

## Device inputs, outputs, and iterative retrieval

See the runnable [device search example](examples/device_search.rs).

`DeviceQueries` describes borrowed row-major FP32 input, including a row stride
in elements. `search_device` normalizes on the GPU and writes into reusable
`DeviceNeighbors`. The returned HRX event marks completion; submission performs
no result readback. Reserve anticipated shapes before latency-sensitive work:

```rust
use hrxdb::{DeviceExclusions, DeviceNeighbors, DeviceQueries};

let mut db = corpus.searcher()?;
db.reserve_device(1, 5)?;
let mut results = DeviceNeighbors::new(db.stream(), 1, 5)?;
let mut visited = DeviceExclusions::new(db.stream(), corpus.len())?;
// query_buffer contains dimensions FP32 values produced on db.stream().
let done = db.search_device(
    DeviceQueries::new(query_buffer.binding(), 1, corpus.dimensions(), corpus.dimensions())?,
    Some(visited.binding()),
    &mut results,
)?;
// Same-stream consumers can run immediately, without a host wait.
visited.insert_device(db.stream(), results.ids(), results.batch_size() * results.k())?;
// A separate stream must explicitly wait for the result producer.
let mut consumer = corpus.stream()?;
consumer.wait_event(&done)?;
let matches = results.read(&mut consumer)?;
```

`DeviceNeighbors` owns FP32 scores and u32 IDs in `[batch, k]` order, plus one
u32 count and status per query. These regions share one allocation and a single
host readback transfer. Slots after the valid count contain negative
infinity and `u32::MAX`. Status is 0 on success; a nonfinite or zero-norm device
query produces status 1 and zero matches. Host `read`/`read_into` reports invalid
queries as errors; kernels can inspect status directly. `read_into` retains its
host staging and surviving result-row capacities. GPU-only consumers allocate
no host readback staging. Do not overwrite inputs or outputs until prior uses
complete; views and mutable Rust borrows do not establish GPU completion.

Device normalization uses scaled FP32 reductions to handle extreme finite
magnitudes, including entirely subnormal rows. Its rounding can differ from
host FP64 normalization, and tiny relative components can underflow. Both paths
search exhaustively with FP32 scores; very close rankings can differ. Up to
64 queries and k=1–1,024 are supported. First use may allocate, compile, or wait
when replacing imported host workspace; fully reserved submissions do not wait
on the host.

`DeviceExclusions` retains one bit per insertion ID. `insert`/`remove` validate
host IDs and upload only those updates; `insert_device`/`remove_device` consume
u32 IDs without host transfers, ignoring out-of-range values including the
result sentinel. Atomic updates preserve duplicate IDs and shared bitmap words.
`clear` resets the set. Use the same stream, or events between streams. Device
search also accepts an application-owned bitmap with this layout.

## Standalone selection

`TopK` consumes a `ScoreBatch`: a contiguous row-major FP32 matrix with up to
64 queries and 2^30 candidate columns. It requires no corpus or cosine metric:

```rust
use hrxdb::{DeviceNeighbors, ScoreBatch, TopK};

let mut topk = TopK::new(&stream)?;
topk.reserve(&stream, query_count, album_count, 60)?;
let mut winners = DeviceNeighbors::new(&stream, query_count, 60)?;
// Run application scoring on this stream, writing score_buffer.
let done = topk.select(
    &mut stream,
    ScoreBatch::new(score_buffer.binding(), query_count, album_count)?,
    &mut winners,
)?;
```

IDs are score-column ordinals: album IDs in this example. Higher scores win;
equal scores prefer lower columns. NaN and negative infinity are absent;
positive infinity is valid. A row with fewer than k eligible values has a
smaller count. The scores remain in their original allocation. Large matrices
use bounded 262,144-column tiles and running GPU top-k; scratch does not grow
with the total column count beyond that tile size. Plans and buffers are reused.
Order a `TopK` worker's submissions on one stream, or establish completion
before reusing its scratch on another stream.

## Adopting resident vectors

`Corpus::from_device(&device, &mut stream, dimensions, shards)` consumes owned
FP16 `ResidentShard { vectors, rows }` allocations. This avoids copying a
resident corpus through host memory or into replacement vector buffers.

Each allocation uses `ceil(dimensions/128)*128` FP16 elements per row, with
capacity divisible by 256, enough room for its logical rows, and at most 2^32
FP16 elements. Logical rows must be finite and nonzero, with zero dimension
padding; tail row slack is unspecified. Shard order determines insertion IDs.
An empty shard list creates an empty corpus; individual shards cannot be empty.

The import kernel validates logical rows and computes inverse norms using FP32
accumulation. It reads back only a four-byte validation result and completes
the supplied stream before publishing the corpus. FP32 norm rounding can differ
from `build_fp16`'s host calculation. Producer writes on another stream require
an event dependency first. Ownership transfers even if validation fails.

## Memory accounting

`corpus.memory_usage()` separates logical FP16 vectors, dimension padding,
vector slack, logical FP32 norms, and norm slack. `db.memory_usage()` reports
that shared corpus separately from the worker's query, score, selection,
readback, exclusion, batch, and device-query allocations:

```rust
let shared_bytes = corpus.memory_usage().total(); // Count once across workers.
let worker_bytes = db.memory_usage().workspace.total();
```

Reports describe reserved buffer bytes, including page rounding of imported
host buffers. They exclude native allocation granularity, compiler/runtime
bookkeeping and staging, and caller-owned inputs/outputs. `DeviceNeighbors`,
`DeviceExclusions`, and `TopK` expose their own device-buffer byte totals.
Accounting does not synchronize or query the GPU.

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
access. Once selection capacity is reserved, searches allocate no new GPU buffers. Allocating convenience methods return fresh host vectors; `_into` methods
reuse caller-owned capacity.

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

Corpus-binding tests read and check every exported value, padding component, and
inverse norm for both ingestion paths. They run the album example on a separate
stream across unaligned shard boundaries, compare against built-in cosine
scores, and verify subsequent searches still return the same results. Empty
corpora expose no shards; borrowed bindings remain tied to a live corpus handle.
Tiled tests cover 201- and 55-row tails, exact tile boundaries, and positive
slack values that would otherwise outrank all-negative corpus scores. CPU tests
check rounded shard capacities against the address limit at every padded dimension.
Stream tests drop the original device, move the index to another thread, verify
device identity and independent streams, run a custom kernel, and use a returned
stream after the index itself has been dropped.

Device pipeline tests cover independent shared-storage workers, snapshot
ownership, zero-copy FP16 import and rejection of invalid rows, strided inputs,
widths through 64, extreme magnitudes, device status, cross-stream event
consumption, incremental GPU exclusions, and standalone selection across large
matrix tiles. They verify bounded workspace and repeated output reuse.

The repository's [release guide](RELEASE.md) covers packaging and publication.
See [CHANGELOG.md](CHANGELOG.md) for version history. Contributions should include
appropriate CPU or opt-in GPU coverage; report bugs with the GPU target, driver,
crate version, and a minimal reproducer. Hardware tests do not run in ordinary CI.
