# Measured results

Local gfx1151, 2026-09-11. One query at a time, 10M × 384 FP16 corpus, cosine,
k=10, three warmups and 30 measured queries per configuration. No other hrxdb
GPU test or benchmark ran concurrently. Other system activity and clocks were
not controlled; results include host timing variation.

The final default is **128 threads, two rows per wave, four components per load**.

| Measurement | Final default |
|---|---:|
| Scoring median | 33.312 ms |
| Scoring p95 | 41.812 ms |
| Useful scoring throughput | **230.55 GB/s** |
| Full top-10 query median | **33.711 ms** |
| Full top-10 query p95 | 36.487 ms |
| Useful full-query throughput | 227.82 GB/s |
| Matching read-control median | 32.893 ms |
| Matching read-control throughput | 233.49 GB/s |
| Scoring / read-control throughput | **98.74%** |
| Scoring / 256 GB/s | 90.06% |

The 256 GB/s stretch target was not reached. The scan exceeded the acceptance
criterion of 90% of the matching read control. The scoring and search percentiles
come from separate invocations, so their p95 values need not be ordered. Raw
samples are retained rather than filtering timing outliers.

The [16-configuration sweep](10m-384-sweep.json) reached 235.44 GB/s and 33.105 ms
median full-query latency with eight rows per wave. Two rows per wave took
33.381 ms in that sweep, within 1%, and required **28 VGPRs instead of 64**. The
specified tie rule therefore selects two rows. ELF resource metadata was added
after the sweep; its recorded timings were not changed. The
[fresh default measurement](10m-384-default.json) includes resource collection
and full score-array selection validation in the benchmark itself.

The final benchmark compared top-k IDs and scores against CPU selection over
all 10M GPU scores. Sampled dot products included both sides of the 4 GiB boundary
and the corpus tail. All returned dot products were also checked against a CPU
reference using the same FP16 conversion, inverse norms, and normalized FP32
query. A separate test placed unique matches around that boundary and at row
9,999,999. The original-vs-quantized top-10 overlap was 100% on the generated
8,192-row subset for this run; this is not a claim about embedding-model recall.

## Large k and exclusions

The updated implementation was measured on the same local gfx1151 on
2026-09-11, using 10M × 384 generated rows, the default scan configuration,
three warmups, and ten changing queries per run. Runs were sequential, without
concurrent hrxdb hardware tests. These are host-completion measurements; system
activity and GPU clocks were not controlled.

| Query | Scan median | Full query median | Full query p95 |
|---|---:|---:|---:|
| Top-10 | 32.057 ms | 32.514 ms | 32.699 ms |
| Top-1,024 | 32.149 ms | 36.742 ms | 36.944 ms |
| Top-1,024, first 100,000 IDs excluded | 32.023 ms | 36.998 ms | 37.115 ms |

Top-1,024 adds about 4.2 ms relative to top-10 in these runs. Exclusions add
about 0.26 ms at k=1,024, including bitmap preparation and masking. Both paths
perform selection on the GPU: top-1,024 reads back 8 KiB of score/ID pairs,
versus 40 MB for a full score readback. Selection scratch was reserved before
timing; at k=1,024 it occupies about 160 MB and is retained for reuse.

The [benchmark records](large-k-and-exclusions.json) retain commands, timing
samples, result counts, and compiler reports. Large per-query neighbor lists
are omitted. Every run validated GPU selection against CPU top-k over all
scores after exclusions, and checked returned scores against the quantized
CPU reference. These timings describe generated data, not the photo library.

The hardware suite also passes the reported 8,942,135 × 512 faces shape with
two internal corpus allocations. It tests unique matches before, on, and after
the 8 GiB boundary and at the final row, including k=1,024 and excluded IDs.

## Sixty-query batches

The [60-query benchmark](6.9m-384-batch-60.json) uses 6.9M × 384 generated rows
(5.2992 GB of FP16 values), k=5, no exclusions, three warmups, and ten measured
batches on local gfx1151, 2026-09-11. Timing order alternates between sixty
individual searches and one batch. Compilation and workspace reservation are
excluded; query preparation, GPU selection, and final readback are included.
No other hrxdb GPU tests or benchmarks ran concurrently. System activity and
GPU clocks were not controlled.

| Measurement | Result |
|---|---:|
| Sixty individual searches, median | 1,863.143 ms |
| One sixty-query batch, median | **165.876 ms** |
| Batch p95 | 176.612 ms |
| Speedup over individual searches | **11.23×** |
| Additional batch workspace | **66.11 MiB** |
| Final score/ID readback | 2,400 bytes |
| Rank differences across 780 query comparisons, including warmups | 0 |
| Maximum returned-score error against CPU reference | 3.18 × 10⁻⁷ |

This is above the motivating estimate of roughly 60 ms. It is a measured
improvement from reusing corpus data across queries while preserving FP32
queries and accumulation. The SIMT matrix kernel expands each corpus tile
to FP32 in workgroup memory; it does not round queries to FP16. Different
accumulation order can reorder near-ties, even though this run matched every
individual-query result ID.

Only a 262,144-row score tile is materialized, followed by a running top-k
merge on the GPU. Workspace stays bounded as the corpus grows and is reported
by `batch_workspace_bytes()`. The record retains every timing sample and
compiler report, along with the reproduction command.

## Generated code

The [resource inventory](codegen.json) records all sweep variants. The default
scan uses **28 VGPRs, 20 SGPRs, zero private scratch, and zero LDS**. Every scan
variant in the sweep had zero private scratch. The default scan contains six
64-bit corpus loads per wave (two rows × three 128-component slices), shared
128-bit query loads, dual FP32 multiply-accumulates, and subgroup reductions.
Loads are issued ahead of their consumers with partial `vmcnt` waits. High-word
multiplication and carry-propagating pointer additions preserve addresses above
4 GiB.

Disassembly, generated with LLVM 22.1.8:

- [Cosine scan](scan.s)
- [Read control](read-control.s)
- [Initial top-k](select-first.s)
- [Candidate merge](select-merge.s)

The JSON records include content-addressed HRX artifact identifiers and the compiler
manifest. The `hrx-cache/kernels/` prefix represents the local HRX cache with its
user-specific parent path removed. Hashes and measurements are unchanged; the
checked-in Loom sources and specialization values reproduce the kernels.

## Verification

`cargo test --release`, `cargo test --release -- --ignored --test-threads=1`,
`cargo fmt --all -- --check`, and `cargo clippy --all-targets -- -D warnings`.
The current suite has fourteen opt-in GPU tests. Coverage includes all scan
configurations, padded dimensions, empty/small indexes, ties, negative scores,
every k=1–1,024 on exact synthetic scores, odd merge tails, exclusions and
growing walks, FP16 ingestion, cross-thread ownership, sharded/single-index
equivalence, the 7.68 GB allocation test, and the 8,942,135 × 512 faces shape.
Batch coverage includes widths through 64, FP16 extremes and padding, shared
exclusions, unaligned tiles and shards, running top-k, workspace reuse, and
sixty queries across the faces corpus's 8 GiB boundary.
Six CPU tests, Rust 1.88 compatibility, four doctests, warning-free Clippy
and rustdoc, and formatting checks passed as well.
