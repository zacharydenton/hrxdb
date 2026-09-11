# Changelog

## Unreleased

- Add asynchronous `Corpus::gather_into` for ordered, duplicate-preserving
  gathers of named rows into compact normalized FP32 device matrices. Resolve
  shards internally, cache kernels across clones, and reuse stream-owned ID
  scratch without vector readback. Include a custom subset-comparison example.
- Expose `MAX_K`, `MAX_BATCH`, and corpus/shard `capacity_range()` accessors for
  consistent application limits and side-array extents, including interior slack.

- Reduce batch-scan register and shared-memory use with FP16 corpus staging
  and per-column scheduling fences. Preserve FP32 queries, accumulation order,
  and scores; add an interleaved predecessor comparison and narrow-tail checks.

- Replace `FlatIndex` and `CorpusView` with a cheaply cloned `Corpus` and
  independent `Searcher` workers. Corpus handles are `Send + Sync` and retain
  snapshots without copying vectors; each worker owns its stream and scratch.
- Expose immutable shard bindings, global row ranges, padded dimensions, and
  readable 256-row capacity. `Corpus::stream` creates an independent stream;
  `Searcher::stream` borrows the worker's ordered queue.
- Add asynchronous `DeviceQueries` → `DeviceNeighbors` search with GPU
  normalization, per-query validity/counts, and completion events. Add persistent
  `DeviceExclusions` with incremental host/device updates for iterative retrieval.
- Add standalone reusable `TopK` over application-defined batched GPU scores.
  Large matrices use bounded tiles and running selection; results remain on the
  device. Update the album example to compose custom scoring with GPU top-k.
- Add zero-copy `Corpus::from_device` adoption of owned FP16 allocations,
  validating logical rows and computing norms on the GPU before publication.
  Device normalization/import use FP32 reductions, with documented rounding
  differences from host ingestion and query normalization.
- Separate shared corpus memory from worker allocations in detailed accounting.
  Add reusable single/batch host result outputs and device-result readback.
- Support up to 64 batched queries with shared corpus reads and exclusions,
  bounded score tiles, cached width-specific kernels, and running device top-k.
  Add batch reservation, memory reporting, and comparative benchmarks.
- Automatically shard corpora at the 2^32-FP16-element allocation limit while
  preserving global insertion IDs. Add native FP16 byte ingestion, full-score
  queries with reusable readback, top-k through 1,024, and per-query exclusions.

## 0.1.0 — Initial release

- Immutable GPU-resident flat index for exhaustive cosine search on gfx1151.
- Normalized FP16 storage with FP32 accumulation and quantized-norm correction.
- Deterministic GPU top-k for k=1–32, ordered by score and insertion ID.
- `Send` index ownership with serialized queries and reusable host/device buffers.
- Authored Loom kernel families with 16 configurable scan schedules.
- Benchmark CLI, CPU reference checks, opt-in hardware tests, and saved 10M × 384 results.

IDs are insertion positions. External-ID storage, persistence, mutations,
metadata filtering, other distance metrics, and approximate search are not implemented.
