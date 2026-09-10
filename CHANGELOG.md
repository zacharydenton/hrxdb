# Changelog

## 0.1.0 — Initial release

- Immutable GPU-resident flat index for exhaustive cosine search on gfx1151.
- Normalized FP16 storage with FP32 accumulation and quantized-norm correction.
- Deterministic GPU top-k for k=1–32, ordered by score and insertion ID.
- `Send` index ownership with serialized queries and reusable host/device buffers.
- Authored Loom kernel families with 16 configurable scan schedules.
- Benchmark CLI, CPU reference checks, opt-in hardware tests, and saved 10M × 384 results.

IDs are insertion positions. External-ID storage, persistence, mutations,
filtering, other distance metrics, and approximate search are not implemented.
