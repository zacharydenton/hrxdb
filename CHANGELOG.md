# Changelog

## 0.3.1 — 2026-09-18

- Add `Corpus::build_in` and `build_fp16_in` for a caller's `ModelContext`, with
  retained context identity and shared native allocation/workspace budgeting.
- Add `load_resident_fp16_in` using the context's residency manager, with runtime
  isolation and allocation-owned charges. Context-built corpora reject prepared
  searches in another runtime; `Corpus::context` exposes their retained context.
- Preserve native shard bindings: context and budget sharing do not convert
  corpus storage to HRX runtime-tracked `BufferView`s.

## 0.3.0 — 2026-09-17

- Move to HRX 0.7 and its explicit inference-slot factory API. Search plans own
  their input/output tensors and return their executable graph together.
- Retain shared-context native search semantics, bounded slots and output
  leases. This is an API migration, not a new search-performance claim.

## 0.2.0 — 2026-09-16

- Move to HRX 0.6 and Rust 1.91. Accept compatible HRX patches while the lockfile
  records the qualified release; no local patches or compatibility aliases.
- Add context-bound prepared searches over owned device tensors, bounded private
  inference slots, producer dependencies and retained device results.
- Share corpus build reservations, persistent residency pins and per-worker
  workspace allocations with the application's HRX memory budget. Reject
  insufficient budgets before allocating, and retain charges with native owners.

## 0.1.2 — 2026-09-16

- Accept compatible HRX 0.5 releases instead of requiring an exact patch
  version. Qualify the lockfile against 0.5.1, including its empty ONNX tensor
  fix, so applications can share one runtime with their model libraries.

## 0.1.1 — 2026-09-16

- Require HRX 0.5 and use its shared compiler-target selection,
  specialization builders, and benchmark distributions.

## 0.1.0 — 2026-09-13

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
