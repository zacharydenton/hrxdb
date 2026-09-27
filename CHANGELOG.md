# Changelog

## Unreleased — cached native GPU memory, prepared search, live snapshots

- Upgrade to HRX 0.8.11 and the published `native-20260927-244cd3801b` bundle.
  Move protobuf to 3.7.2 and remove the obsolete onnx-protobuf dependency.
- Add `Searcher::profile_search` for explicit single/batch GPU timestamp
  replays, command geometry and arguments, transfers, and per-stage totals.
  Use the ordinary search commands and check bitwise result parity; normal
  searches remain uninstrumented.
- Add `Searcher::detailed_compilation_reports` and retain compiler/target
  identity, specialization, resource facts, wait reasons and related diagnostic
  locations. Keep library compilation at summary level; detailed collection
  is explicit and does not replace loaded kernels.
- Benchmark JSON collects detailed compiler reports by default, with
  `--compile-report summary` to omit detailed collection. `--profile` adds three
  diagnostic replays after ordinary measurement windows. Resource selection now
  consumes Loom evidence directly, without an `llvm-readobj` subprocess.

- Add `Corpus::append` and `append_fp16`, returning a snapshot with new rows at
  the next insertion IDs and sharing every existing allocation. Rows fill a
  geometric tail reserve guarded by an atomic claim, so repeated appends keep
  one tail shard. On 7M × 768, appending 50 rows takes 0.54 ms, and after 1,000
  appends search is within 0.1% of a fresh build.
- Add `Corpus::update` and `update_fp16`, replacing rows at existing IDs:
  in place with one scatter dispatch when no other handle shares the storage,
  otherwise copy-on-write of the touched 16,384-row pages and gaps shorter than
  one batch tile. Add `Corpus::compact` to restore the build layout on the device.
- Add `Searcher::set_corpus`, moving a worker to another snapshot and reusing
  its stream, kernels and workspace; 0.14 ms after an append at 7M rows.
- The last shard's single-query scan is specialized to its capacity and scores
  its reserve, whose scores are never selected; score and exclusion workspace
  and batch tiles are sized by `capacity_range()`. `memory_usage` counts each
  allocation once and reports append reserve and superseded rows as slack.
  Residency eviction also requires that no derived snapshot shares storage.
- Add `hrxdb-bench --append` for append, handover and update measurements.

- Require HRX 0.8.3, restoring GPU caching for corpus and search workspace.
- Publish direct host writes to query/exclusion buffers and acquire GPU writes
  before result readback. These transitions also cover batched searches.
- On gfx1151, five alternating process runs at 1M × 384, k=10 reduce full search
  from 4.1591 ms (0.8.2) to 3.4112 ms. Old HRX 0.7.0 measured 3.2765 ms;
  scan time is within 0.6% of that baseline. All timed result IDs/scores match.

- Reuse one prepared scan/mask/selection/readback graph per worker, keyed by
  effective k and exclusion mode; invalidate it when kernels or scratch change.
  Five fresh alternating runs measure 3.2674 ms versus 3.3988 ms without the
  graph and 3.2760 ms on HRX 0.7. All timed result IDs and scores match.

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
