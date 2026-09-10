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
Hardware coverage includes all scan configurations, padded dimensions,
empty/small indexes, ties, negative scores, all k=1–32 values on exact synthetic
scores, three-level selection, and the 7.68 GB allocation boundary test.
