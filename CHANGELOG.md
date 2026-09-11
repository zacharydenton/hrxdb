# Changelog

## Unreleased

- Automatically shard corpora into allocations of at most 2^32 FP16 elements,
  preserving global insertion IDs and device selection across the full index.
- Add `build_fp16` and `build_fp16_with_config` for little-endian FP16 byte rows,
  preserving values and computing inverse norms without FP32 row allocations.
- Support full-score queries for host selection through `scores` and the new
  `scores_into`, which reuses a caller-owned FP32 output buffer and avoids the
  temporary byte array and conversion copy.
- Extend device top-k to 1,024 with workgroup sorting and pairwise sorted merges.
  Keep the existing reduction path for k=1–32; reuse selection scratch and allow
  explicit preallocation with `reserve_search`.
- Add `search_excluding` for per-query excluded IDs, including duplicates,
  empty result sets, large k, and sharded corpora. A reusable bitmap masks
  scores on the GPU; only selected score/ID pairs are read back.
- Add `measure_excluding` and benchmark support for larger k and `--exclude-first`.

## 0.1.0 — Initial release

- Immutable GPU-resident flat index for exhaustive cosine search on gfx1151.
- Normalized FP16 storage with FP32 accumulation and quantized-norm correction.
- Deterministic GPU top-k for k=1–32, ordered by score and insertion ID.
- `Send` index ownership with serialized queries and reusable host/device buffers.
- Authored Loom kernel families with 16 configurable scan schedules.
- Benchmark CLI, CPU reference checks, opt-in hardware tests, and saved 10M × 384 results.

IDs are insertion positions. External-ID storage, persistence, mutations,
metadata filtering, other distance metrics, and approximate search are not implemented.
