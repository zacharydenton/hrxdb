# Changelog

## 0.1.0 — Release candidate

First release of hrxdb: an embedded GPU-powered vector database for
unified-memory systems, built on HRX and Loom and targeting AMD Strix Halo
(`gfx1151`) on Linux x86_64.

- Exhaustive cosine search over FP16 vectors with FP32 accumulation and norm
  correction. Deterministic top-k from 1 to 1,024, insertion-position IDs,
  query-time exclusions, and reusable full-score and neighbor outputs.
- Batches of up to 64 queries share corpus reads, with FP16 shared-memory
  staging, bounded score tiles, and running GPU top-k.
- Cheap-clone `Corpus` handles share immutable `Send + Sync` storage. Independent
  `Searcher` workers own streams, cached kernels, and reusable workspaces.
- Automatic sharding at the 2^32-FP16-element allocation boundary, up to 2^30
  logical rows and 16,384 dimensions, subject to available memory.
- Bounded host FP32 ingestion, native FP16 byte ingestion, and validated
  zero-copy adoption of owned device buffers with `Corpus::from_device`.
- Read-only shard bindings expose row ranges, padded dimensions, inverse norms,
  and readable capacity for custom kernels. Public `MAX_K`, `MAX_BATCH`, and
  `capacity_range()` accessors make limits and side-array extents explicit.
- Asynchronous `Corpus::gather_into` writes ordered, duplicate-preserving
  subsets as compact normalized FP32 device matrices, resolving shards
  internally without vector readback.
- `DeviceQueries` and `DeviceNeighbors` support GPU query normalization,
  per-query validity/counts, device results, and event-ordered consumption.
  Persistent `DeviceExclusions` support incremental host and device updates.
- Standalone `TopK` ranks application-defined GPU score matrices with bounded
  workspace. Separate memory accounting covers shared storage and workers.
- Runnable search, album-scoring, device-pipeline, and subset-comparison
  examples; configurable scan schedules, a benchmark CLI, and recorded results.

Search is exact over the stored quantized representation. Device normalization
and import use FP32 reductions and can round differently from host paths.
Independent workers do not guarantee device overlap, priority, or latency
isolation. Persistence, application metadata, mutations, other distance metrics,
and approximate search are outside this release.

Requires Rust 1.88+ and the pinned `hrx-rs =0.4.0` native runtime/compiler for
GPU execution. See the [README](README.md) for platform prerequisites and the
[release guide](RELEASE.md) for qualification and publication status.
