# Measured results

## WMMA comparison and threshold overflow fixes

The [2026-09-29 qualification](hrx-0.8.11-threshold-fixes.json) covers the
comparison harness fixes and parallel threshold overflow reduction on HRX
0.8.11. All four scan widths pass with the actual compiled workgroup sizes;
FP32/FP16 query comparisons use separate CPU references and record near-tie
ranking differences. Same-precision controls retain exact ID/score parity.
All 56 GPU tests pass, including capacity boundaries, masks, counter resets,
varied overflow scores, and three reduction levels at k=1,024.

For a synthetic 524,288-row, three-dimensional, two-query top-5 search whose
second tile entirely exceeds the threshold, the old serial fallback took
7.77–8.05 ms. Two later runs with the parallel fallback took 1.24–1.25 ms and
2.43–2.86 ms. These runs were sequential under changing external GPU load,
so they diagnose the bottleneck without establishing an isolated speedup.
The record retains both runs, the empty-candidate control, separate stage
timestamps, and compiler resource/instruction/spill counts. The new reduction
uses 20 VGPRs, 16 KiB LDS, and zero spills, reusing existing selection buffers.

## Batch score stores and real-query selection

Local gfx1151, 2026-09-28, HRX 0.8.11, bundle
`native-20260927-244cd3801b`, Rust 1.95.0-nightly, baseline `aa4e828`.
External GPU load changed throughout, including `krea2`'s `bench_runtime` and
`bench-before`. Our GPU work ran sequentially. Comparisons alternate arms on
one shared generated corpus with changing queries and three warmups. The final
comparisons use cached graphs in both arms. Absolute times are not comparable
between runs and these are not isolated latency claims.

Two changes are retained:

- At compiled widths 16/32/64, gather each query's four adjacent row scores
  into one vector store. Accumulation and normalization retain their arithmetic
  order. Scalar stores handle partial row groups without crossing query strides.
  Width 8 retains the baseline scan's executable code.
- Select and merge only real queries. A 33-query batch still scans at width 64
  but selects 33 query rows. Small-k kernels remain shared across counts within
  the compiled width; the change needs no extra compilation for those counts.

| Combined comparison | Queries / k | Samples | Baseline median | Candidate median | Ratio |
|---|---:|---:|---:|---:|---:|
| 6,909,092 × 384, run 1 | 60 / 5 | 31 | 139.339 ms | 133.252 ms | 1.046× |
| 6,909,092 × 384, run 2 | 60 / 5 | 31 | 69.769 ms | 62.348 ms | 1.119× |
| 6,909,092 × 384, run 3 | 60 / 5 | 31 | 70.273 ms | 61.534 ms | 1.142× |
| 6,909,092 × 384, run 4 | 60 / 5 | 31 | 149.745 ms | 152.750 ms | 0.980× |
| 1,000,000 × 384 | 33 / 5 | 31 | 9.206 ms | 7.895 ms | 1.166× |
| 1,000,000 × 384 | 16 / 5 | 31 | 5.219 ms | 5.140 ms | 1.015× |
| 1,000,000 × 768 | 60 / 5 | 31 | 16.903 ms | 16.786 ms | 1.007× |

The candidate won 100/124 pairs across the four large runs and all 31 pairs in
the final 33-query comparison, but the fourth large run's ratio of medians is
negative. The first two large runs preceded the width-8 fallback; their width-64
machine code is byte-identical to the final source. The record checks both that
identity and the final width-8 scan's identity with the baseline.

Controls quantify the uncertainty: identical kernels at 6.9M rows measured
140.281 versus 141.872 ms (0.989×). At 262,145 × 129, three queries, k=1,024,
the final width-8 baseline and candidate have identical code and work, yet
measured 4.203 versus 4.488 ms (0.937×). These controls do not correct other runs;
they limit what small differences under this load can establish. The changes
have useful wins, especially near padding boundaries, rather than a guaranteed
speedup across workloads or contention levels.

| Absolute evidence, 6.9M × 384 / 60 queries / k=5 | Baseline | Candidate |
|---|---:|---:|
| Selection workgroups | 436,928 | 409,620 |
| Running-merge workgroups | 1,664 | 1,560 |
| Scan workgroups | 107,955 | 107,955 |
| Commands / copies | 137 / 4 | 137 / 4 |
| Stores per thread on the full-row path | 16 × 32-bit | 4 × 128-bit |
| Bytes written per thread on that path | 64 | 64 |
| Width-64 VGPRs | 52 | 54 |
| Whole-kernel instructions / code bytes | 1,082 / 6,648 | 1,267 / 7,372 |
| Spill plans / spills / materialized reloads | 0 / 0 / 0 | 0 / 0 / 0 |
| Private bytes / LDS bytes | 0 / 12,544 | 0 / 12,544 |
| Modeled occupancy | 100% | 100% |

The static store count remains sixteen because the new code contains four
vector stores and twelve scalar tail stores. It does not measure dynamically
executed stores or memory traffic. Width-16/32 candidates use 72/54 VGPRs with
zero spills and unchanged 62%/93% modeled occupancy. Separate diagnostic
replays in large run 3 showed median scan intervals 54.770 → 48.592 ms and
selection 14.308 → 11.068 ms. Those instrumented intervals include marker/barrier
perturbation and cannot be subtracted from ordinary latency.

Selection-prefix dispatch alone measured 1.091× for 33 queries and 1.015× for
60 queries with unchanged scan code. Rejected experiments include dot groups
of 4/16/32, a higher-register vector-store epilogue, hoisting the full-row
branch, and batching norm loads. Universal vector stores produced adverse
narrow-batch measurements; subsequent controls exposed substantial noise, and
neither norm-load changes nor repeated timings established a reliable reason
to change the width-8 scan. Its scalar implementation is retained.

The [44-run record](hrx-0.8.11-batch-output.json) retains every host sample,
all stage intervals, workgroup/copy counts, actual compiler allocation/emission/
wait/resource summaries, artifact identities, executable-section hashes and
exploratory sources. Large repeated per-operation report trees and raw timestamp
arrays are omitted. LLVM 22.1.8 disassembly confirms the store widths.

```sh
HRX_OFFLINE=1 ROWS=6909092 DIM=384 BATCH=60 K=5 SAMPLES=31 \
  CACHED_GRAPHS=1 PADDED_SELECTION_BASELINE=1 \
  BASELINE_SOURCE=tests/fixtures/batch_scan_scalar_stores.loom OUTPUT=output.json \
  cargo test --locked --release --lib compare_batch_scan -- --ignored --nocapture
```

For selection alone, also set `CANDIDATE_SOURCE` to that baseline fixture. For
an identical-kernel control, use the same fixture in both arms and omit
`PADDED_SELECTION_BASELINE`. All result IDs and score bits matched. All 55 GPU
tests pass, with expanded tails 1/2/3/4/31/65/66/67, masks, shard boundaries,
FP16 extremes, real/padded query parity and cached snapshot changes. Profiling
tests assert actual selection/merge grid sizes, and kernel reuse is checked
across 33/34/60-query small-k batches.

## Prefetch and direct batch merge

Local gfx1151, 2026-09-28, HRX 0.8.11, bundle
`native-20260927-244cd3801b`, Rust 1.95.0-nightly, baseline `dd1e6b4`.
External load varied: `krea2` was active initially and later exited;
`bench-before` and `beam.smp` subsequently held render-device handles. Clocks
and competing work were uncontrolled. Our GPU runs were sequential. Each
comparison alternates arms with changing queries on one shared corpus after
three warmups. Compare within rows; these are not isolated latency claims.

Direct running merge reads separate score/ID lists and alternates output
buffers. Counting tiles across shard boundaries chooses the first destination
so the final list needs no extra copy. Each additional tile eliminates four
staging copies. For 6,909,092 × 384, 60 queries, k=5:

| Structural evidence | Staged baseline | Direct merge |
|---|---:|---:|
| Total commands | 241 | 137 |
| Copies | 108 | 4 |
| Scan / selection / running-merge dispatches | 27 / 80 / 26 | 27 / 80 / 26 |
| Merge scratch, reserved width 64 and k=8 | 8,192 bytes | 4,096 bytes |
| Running-merge VGPRs / SGPRs | 12 / 34 | 19 / 34 |
| Running-merge instructions / code bytes | 111 / 560 | 139 / 760 |
| Spills / private bytes / LDS bytes | 0 / 0 / 0 | 0 / 0 / 0 |
| Modeled occupancy | 100% | 100% |

The host supplies `ceil(log2(k+1))` search rounds when recording the graph:
three at k=5 instead of eleven. Both arms use cached graphs. Final-source
ordinary timings are effectively neutral:

| Corpus × dimensions | Queries / k | Samples | Staged median | Direct median | Ratio |
|---|---:|---:|---:|---:|---:|
| 6,909,092 × 384 | 60 / 5 | 31 | 74.158 ms | 74.273 ms | 0.998× |
| 262,145 × 129 | 3 / 1,024 | 31 | 1.845 ms | 1.836 ms | 1.005× |

Earlier direct-merge variants measured 0.6–2.1% gains in the large shape, but
those gains did not persist in the final comparison. Other exploratory shapes
were within 1% of baseline. Retain this change for fewer commands and half the
merge scratch, without claiming a latency improvement. Three separate diagnostic
replays follow each ordinary timing window; their copy intervals shrink but
include instrumentation overhead and cannot be subtracted from ordinary time.

Seven scan prefetch variants were rejected. At 1M × 384, 60 queries, k=5,
31 samples each, both arms use immediate submission and the baseline scan is
the eight-product implementation at `dd1e6b4`:

| Candidate | Speedup | VGPRs | Code bytes | LDS bytes | Modeled occupancy |
|---|---:|---:|---:|---:|---:|
| Baseline | — | 52 | 6,648 | 12,544 | 100% |
| Pipeline depth 2 | 1.009× | 73 | 11,588 | 12,544 | 100% |
| Pipeline depth 3 | 1.012× | 108 | 16,356 | 12,544 | 75% |
| Pipeline depth 4 | 0.994× | 120 | 20,848 | 12,544 | 75% |
| Depth 2 + unroll 2 | 0.985× | 96 | 16,516 | 12,544 | 100% |
| Depth 2 + recurrence scheduling | 1.002× | 115 | 21,192 | 12,544 | 75% |
| Double-buffered LDS | 0.998× | 52 | 6,748 | 25,088 | 62% |
| Double-buffered LDS + depth 2 | 0.973× | 75 | 11,832 | 25,088 | 62% |

All variants report zero spills. Depth 2 measured 1.013× at 6.9M rows, still a
weak gain against its register/code growth. Static wait/instruction counts
include pipeline prologue/drain code and are not measured dynamic stalls.
Production scan remains unchanged; exact score bits and IDs matched throughout.

The [experiment record](hrx-0.8.11-prefetch-merge.json) retains all host samples,
stage intervals, source/artifact identities, exploratory sources, final merge
compiler reports and detailed resource counters for rejected scan variants.
Earlier merge sources use two scalar launch arguments; the final source uses
three, so reproducing those exploratory variants requires their matching ABI.

```sh
HRX_OFFLINE=1 ROWS=6909092 DIM=384 BATCH=60 K=5 SAMPLES=31 OUTPUT=merge.json \
  cargo test --locked --release --lib compare_batch_merge -- --ignored --nocapture
```

The baseline kernel is preserved in `tests/fixtures/batch_merge_staged.loom`.
For prefetch comparisons, extract `baseline` and a candidate from the record's
`sources`, then pass `BASELINE_SOURCE` and `CANDIDATE_SOURCE` to
`compare_batch_scan`. All 55 GPU correctness tests pass, including every
k=1–1,024 against the staged kernel and an independent CPU merge with ties,
signed zeros, infinities, subnormals, padding and large IDs. Existing coverage
checks short tiles/shards, exclusions, cached replay and changing snapshots.
CPU tests, doctests, Clippy, rustdoc, Rust 1.91 and formatting checks pass.

## Selection and cached batch graphs

Local gfx1151, 2026-09-28, HRX 0.8.11, bundle
`native-20260927-244cd3801b`, Rust 1.95.0-nightly. The baseline is `c0f18ef`,
including the eight-product scan and previous selection implementation.
Another `krea2` GPU job remained active. Comparisons alternate arms on one
shared generated corpus, change queries each iteration, and retain all samples.
Three warmups exclude graph recording, compilation and allocation from ordinary
latency. Our GPU benchmarks and correctness tests ran sequentially.

Two changes are measured independently and together:

- Selection for k≤32 first finds each wave's score/ID winner, then reduces the
  eight winners in eight-lane clusters. Alternating shared-memory slots needs
  one barrier per rank: the next rank's barrier completes reads before the
  following rank reuses a slot. Ties still choose the lowest insertion ID.
- Host batch search caches one uninstrumented graph for its latest exact query
  count, effective k and exclusion mode. All 241 commands in the large top-5
  batch remain; graph reuse reduces repeated recording/submission work. Corpus
  replacement and workspace growth invalidate recorded bindings. First use of
  a shape records a graph even after `reserve_batch` has prepared its kernels.

| Combined comparison | Queries / k | Samples | Shipped median | New median | Speedup |
|---|---:|---:|---:|---:|---:|
| 6,909,092 × 384, first run | 60 / 5 | 31 | 73.271 ms | 64.858 ms | 1.130× |
| 6,909,092 × 384, repeat | 60 / 5 | 31 | 79.252 ms | 70.079 ms | 1.131× |
| 1,000,000 × 384 | 60 / 5 | 31 | 11.560 ms | 10.114 ms | 1.143× |
| 1,000,000 × 384 | 60 / 32 | 31 | 16.177 ms | 13.983 ms | 1.157× |
| 1,000,000 × 768 | 60 / 5 | 31 | 17.003 ms | 15.799 ms | 1.076× |
| 4,097 × 513 | 60 / 5 | 51 | 0.631 ms | 0.159 ms | 3.961× |

The candidate won all 62 pairs across the two large runs. These are gains under
concurrent load, not isolated latency claims. Compare within each row; absolute
latencies changed between runs. Every result ID and score bit matched, along
with all 5,974,272 final-tile scores in each large run.

Selection alone measured 1.020× end-to-end at 6.9M rows, top-5, and 1.077× at
1M rows, top-32. Separate instrumented selection intervals fell from a median
12.406 to 11.592 ms and 7.467 to 6.272 ms respectively. Top-1 was neutral at
1.001×. With both arms using the new selector, graph reuse alone measured
71.097 → 64.458 ms at 6.9M rows (1.103×). The k>32 sorted selector is unchanged;
graph reuse alone measured 0.559 → 0.100 ms for 4,097 × 513, three queries,
k=1,024. All these runs have their own samples in the record; their speedups
must not be multiplied as if collected under identical load.

| Static selector evidence | First pass, before → after | Merge pass, before → after |
|---|---:|---:|
| Emitted instructions | 234 → 210 | 244 → 221 |
| Code bytes | 1,352 → 1,256 | 1,404 → 1,312 |
| Workgroup barriers per rank | 2 → 1 | 2 → 1 |
| VGPRs / SGPRs | 16 / 28 → 16 / 28 | 17 / 26 → 17 / 26 |
| LDS bytes | 64 → 128 | 64 → 128 |
| Spills / private bytes | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 |
| Modeled occupancy | 100% → 100% | 100% → 100% |

Disassembly confirms the barrier counts. They and the compiler's wait reasons
are static evidence, not measured stall cycles. A first attempt that loaded all
eight winners into each thread's vector registers was neutral; double-buffering
that version halved barriers but regressed at top-32. Distributing the final
reduction over eight-lane clusters avoids that extra register/ALU work.

The [selection/replay record](hrx-0.8.11-selection-replay.json) retains raw host
samples, all per-stage replay samples, complete representative first/merge
compiler reports, instruction counts, source/artifact hashes, rejected sources,
and representative raw timestamp/command samples. Both arms' diagnostic replays
use newly recorded instrumented graphs; they isolate kernel differences but do
not measure the ordinary submission-path difference. Do not subtract their
intervals from ordinary host time.

```sh
# Combined change against c0f18ef's selection + immediate submission:
HRX_OFFLINE=1 ROWS=6909092 DIM=384 BATCH=60 K=5 SAMPLES=31 OUTPUT=combined.json \
  cargo test --locked --release --lib compare_batch_optimized -- --ignored --nocapture
# Selection only (both immediate), or replay only (identical current kernels):
HRX_OFFLINE=1 ROWS=1000000 K=32 SAMPLES=31 OUTPUT=selection.json \
  cargo test --locked --release --lib compare_batch_selection -- --ignored --nocapture
HRX_OFFLINE=1 ROWS=1000000 K=5 SAMPLES=31 OUTPUT=replay.json \
  cargo test --locked --release --lib compare_batch_replay -- --ignored --nocapture
```

Kernel comparisons accept `BASELINE_SOURCE` and `CANDIDATE_SOURCE`; selection
defaults to `tests/fixtures/select_two_reductions.loom` versus production.
The existing scan/tile comparisons keep immediate submission to isolate kernels.
All 54 GPU correctness tests pass. Added coverage compares raw selector output
against its predecessor and CPU ID ordering for signed zeros, NaNs, infinities,
subnormals, ties, short tails, multiple merge levels and large insertion-ID
offsets. Cached/uncached parity covers changed queries and exclusions, query
counts within one compiled width, k=5/10/33/1,024, workspace growth, all-excluded
queries, clamped k, and changed snapshots. CPU tests, doctests, Clippy, rustdoc,
Rust 1.91 checking, formatting and package verification also pass.

## Batch kernel tuning with HRX 0.8.11

Local gfx1151, 2026-09-27/28, HRX 0.8.11 and published bundle
`native-20260927-244cd3801b`, Rust 1.95.0-nightly. The baseline is the per-column
kernel at `abfcd5c`, preserved in `tests/fixtures/batch_scan_columns.loom`.
**Another GPU job was active throughout.** Each run alternated baseline and
candidate on one shared generated corpus, with changing queries, three warmups,
and k=5. No other hrxdb GPU work ran concurrently. These results demonstrate
improvement under that load; clocks and competing workload were not controlled.

| Corpus × dimensions | Queries | Samples | Baseline median | Eight-product median | Speedup |
|---|---:|---:|---:|---:|---:|
| 6,909,092 × 384, first run | 60 | 15 | 107.899 ms | 96.615 ms | 1.117× |
| 6,909,092 × 384, repeat | 60 | 31 | 87.680 ms | 76.281 ms | 1.149× |
| 1,000,000 × 384, first run | 60 | 25 | 13.310 ms | 10.778 ms | 1.235× |
| 1,000,000 × 384, final source | 60 | 31 | 12.022 ms | 10.186 ms | 1.180× |
| 1,000,000 × 768 | 60 | 25 | 22.763 ms | 19.002 ms | 1.198× |
| 1,000,000 × 384 | 8 | 31 | 6.417 ms | 5.558 ms | 1.155× |
| 1,000,000 × 384 | 16 | 31 | 6.318 ms | 5.912 ms | 1.069× |
| 1,000,000 × 384 | 32 | 31 | 7.935 ms | 7.134 ms | 1.112× |

The candidate won 15/15 and 30/31 ordinary timing pairs in the two large runs.
Compare arms within each row: absolute latency changed substantially between
runs. Complete 4,097-row × 513-dimensional score matrices also matched at
batches 3/8/16/32/64. Small-shape timing was noisy (repeat speedups 1.01–1.39×),
so it does not support a precise small-shape performance claim.

Detailed compiler reports and ISA inspection motivated two changes: express
eight ordered products as one dot so the compiler can reuse and pair
accumulators, and unroll exact-trip cooperative staging loops. Corpus LDS
storage remains FP16 and each FP32 accumulator still visits components in the
same order. The initial explicitly expanded implementation and final shaped
vector source produce byte-identical executables at width 64/dimension 384.

| Static evidence, width 64 | Baseline | Eight-product |
|---|---:|---:|
| Emitted instructions, whole kernel | 1,304 | 1,082 |
| Code bytes | 8,576 | 6,648 |
| Three-operand FMA instructions | 512 | 64 |
| Single / dual FMAC instructions | 0 / 0 | 90 / 179 |
| Arithmetic instructions for the same 512 terms | 512 | 333 |
| ALU delay instructions | 122 | 81 |
| LDS value-dependency waits (full / partial) | 64 (32 / 32) | 55 (14 / 41) |
| VGPRs / SGPRs | 29 / 26 | 52 / 26 |
| Spill plans / spills / materialized reloads | 0 / 0 / 0 | 0 / 0 / 0 |
| Private bytes / LDS bytes | 0 / 12,544 | 0 / 12,544 |
| Modeled occupancy | 100% | 100% |

Arithmetic counts describe the unrolled 32-component loop body. A dual FMAC
performs two terms; the mathematical work is unchanged. These are static
counts, not runtime counters or measured stall cycles. Unrolling staging
increases its static load count without increasing dynamic corpus traffic.
The width-8/16/32 variants use 104/68/49 VGPRs and zero spills; modeled occupancy
remains at the baseline's 37%/62%/93%. Reducing register count alone was therefore
not the right objective.

Three separate profiled replays followed every ordinary timing window. In the
large repeat, median summed `batch_scan` intervals fell from 69.217 to 54.229 ms,
while selection stayed at 13.367 versus 13.117 ms. Contention produced outliers
in both arms, including a candidate scan of 100.175 ms in the first large run;
all samples are retained. Instrumented intervals include marker/barrier overhead
and must not be subtracted from ordinary host latency.

Removing fences alone was neutral at 1M rows (0.999×); adding staging unrolling
gave 1.161× in its comparison. Two- and four-product grouping regressed in their
runs. Groups of 16 and 32 improved only 1.017× and 1.020× over eight in single
comparisons, with more registers; that was insufficient evidence to replace
eight. The [tuning record](hrx-0.8.11-batch-tuning.json) retains these experiments,
all raw host samples, stage samples, source/artifact hashes, absolute ISA counts,
and complete representative detailed compiler reports. Reproduce full reports
and raw per-command timestamps with:

```sh
HRX_OFFLINE=1 ROWS=6909092 DIM=384 BATCH=60 SAMPLES=31 OUTPUT=batch-tuning.json \
  cargo test --locked --release --lib compare_batch_scan -- --ignored --nocapture
```

`BASELINE_SOURCE` and `CANDIDATE_SOURCE` accept alternative Loom files. By default
the harness compares the preserved `abfcd5c` kernel with production. Both arms
load the detailed artifacts they report. Every neighbor ID and score bit must
agree, as must all materialized scores in the final tile (5,974,272 values for
the large shape). The optimized kernel passes all 52 GPU correctness tests,
including tails, FP16 extremes, ties, exclusions, sharding, device queries and
snapshot updates. CPU tests, doctests, Clippy, rustdoc, Rust 1.91 checking,
formatting and package verification also pass.

## HRX 0.8.11 profiling qualification

The [profiler qualification record](hrx-0.8.11-profiling.json) contains single and
60-query runs over 100K × 384 vectors, k=10, on 2026-09-27. It retains ordinary
samples, three separate GPU timestamp replays, command geometry and constants,
compiler resource facts, wait reasons, and selected detailed compiler sections.
The published bundle was used without runtime/compiler overrides. Another GPU
workload was active, so these are diagnostic qualification results, not an
isolated performance baseline. All profiled results matched ordinary search.

See [profiling and compiler evidence](../docs/profiling.md) for reproduction and
interpretation. All 49 existing GPU tests and three new profiling/report tests
pass with HRX 0.8.11. The remaining sections below retain their original dates,
runtime versions, and measurement conditions.

## Original scan measurements

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
queries and accumulation. The matrix kernel used for that measurement expanded each corpus tile
to FP32 in workgroup memory; it does not round queries to FP16. Different
accumulation order can reorder near-ties, even though this run matched every
individual-query result ID.

Only a 262,144-row score tile is materialized, followed by a running top-k
merge on the GPU. Workspace stays bounded as the corpus grows and is reported
by `batch_workspace_bytes()`. The record retains every timing sample and
compiler report, along with the reproduction command.

## Batch scan optimization

The [interleaved predecessor comparison](6.9m-384-batch-scan-optimized.json)
uses 6,909,092 × 384 generated rows, sixty top-5 queries, three warmups, and
15 measured batches on local gfx1151, 2026-09-11. Both searchers share the same
corpus allocation. Timing order alternates; queries change between iterations.
Host timing includes query preparation, all scan/selection/merge dispatches, and
result readback. Compilation, reservation, and verification are excluded. No
other GPU test or benchmark ran concurrently; clocks and other system activity
were not controlled.

| Measurement | Previous scan | Optimized scan |
|---|---:|---:|
| Full sixty-query batch, median | 114.683 ms | **80.741 ms** |
| VGPRs | 100 | **29** |
| Workgroup memory | 16,896 bytes | **12,544 bytes** |
| Private scratch | 0 | 0 |

The measured speedup is **1.42×**. This compares two batch implementations on
the same corpus in one run; the older batch-versus-individual timings above
were collected separately and are not its baseline.

The improvement depends on shape. A separate [small-corpus comparison](4097-513-batch-scan-tradeoff.json)
used 4,097 rows, 513 logical dimensions, k=5, three warmups, and 51 measured
samples per run with the same alternating order. At batch 8, the median rose
from 0.1153 to 0.1208 ms; at batch 16, it rose from 0.1056 to 0.1141 ms.
These runs were about 5–8% slower, an absolute difference of 5.5–8.5 µs.
Clocks and other system activity were not controlled. The current kernel
accepts this small-shape tradeoff for the measured large-corpus improvement;
there is no shape-dependent fallback. Both runs retained bitwise-equal scores
and identical result IDs. Their commands and samples are included in the record.

That kernel kept the already-quantized corpus values in FP16 in
workgroup memory, expands them in registers, and bounds operand lifetimes with
`scf.schedule.fence` after each column. This compiler hint emits no instruction
or hardware barrier. Queries remain FP32, and the FP32 multiply-accumulate order
is unchanged. At width 64, the generated kernel uses 29 VGPRs versus 100 and
4,352 fewer bytes of workgroup memory. Corpus order and query interfaces are
unchanged.

All 5,400 returned neighbors (including warmups) had identical IDs and bitwise
identical scores. All 5,974,272 materialized scores in the final 93,348-row tile,
including padded query slots, also matched bitwise. Complete 4,097-row score
matrices with 513 logical dimensions matched at compiled widths 8, 16, 32, and
64. CPU-reference tests cover every width, FP16 extremes, dimension padding,
and final dispatches containing 1, 31, or 65 rows. Host/device queries, shared
exclusions, running top-k, and the 8,942,135 × 512 two-shard case pass as well.

Reproduce the comparison against the preserved test fixture:

```sh
BASELINE_SOURCE=tests/fixtures/batch_scan_baseline.loom \
  CANDIDATE_SOURCE=tests/fixtures/batch_scan_columns.loom \
  ROWS=6909092 BATCH=60 SAMPLES=15 OUTPUT=batch-scan.json \
  cargo test --release --lib compare_batch_scan -- --ignored --nocapture
```

`ROWS`, `DIM`, `BATCH` (2–64), `K`, and `SAMPLES` configure the opt-in benchmark.
It requires identical result IDs/scores and compares every final-tile score
outside timing. With at most 262,144 rows, that comparison covers the complete
score matrix. The record retains timing samples and both compiler reports,
with artifact paths normalized to cache-relative identifiers for sharing.

## Batch score tile sizes

The [production tile sweep](6.9m-384-batch-tile-sweep.json) measures the current
scan and selection pipeline on 6,909,092 × 384 rows, batch 60 (compiled width
64), k=5. Five tile sizes share one corpus and the same workspace allocation.
Each iteration changes queries and rotates variant order; three warmups precede
15 measured iterations. Timing includes host preparation, dispatches, selection,
running merges, and readback. Clocks and other system activity were not controlled.

| Tile rows | Score tile at width 64 | Median batch latency |
|---|---:|---:|
| 16,384 | 4 MiB | 108.358 ms |
| 32,768 | 8 MiB | 94.491 ms |
| 65,536 | 16 MiB | 88.046 ms |
| 131,072 | 32 MiB | 82.419 ms |
| 262,144 (production) | 64 MiB | **79.427 ms** |

All returned IDs and score bits agreed across variants. Smaller tiles did not
improve this workload, so the production tile size remains unchanged. Unlike
Sinkhorn's repeatedly reused Gibbs matrices, search consumes each score tile
once through selection; reducing tile size also increases dispatch and running
merge counts. This sweep measures the combined tradeoff, not isolated GPU stages.

```sh
ROWS=6909092 BATCH=60 SAMPLES=15 OUTPUT=tile-sweep.json \
  cargo test --release --lib compare_batch_tiles -- --ignored --nocapture
```

## Appends and updates

Local gfx1151, 2026-09-27, hrx-rs 0.8.7, `rustc 1.95.0-nightly`, default scan
schedule, k=10. Each run builds a corpus of generated 768-dimensional rows, then
appends 50 rows 1,000 times, moving one searcher to each new snapshot with
`set_corpus`. The appended corpus is compared with a separate build of the same
rows: 30 measured single queries and 60-query batches after three warmups,
alternating which corpus runs first. Every timed result over the appended corpus
was identical to the fresh build's, and full score arrays matched bitwise for
four queries. No other hrxdb workload ran; clocks were not controlled.

| | [7M × 768](7m-768-append.json) | [3M × 768](3m-768-append.json) |
|---|---:|---:|
| Base shards; after appends | 2; 3 | 1; 2 |
| Append 50 rows, median (p95) | 0.540 ms (0.573) | 0.536 ms (0.652) |
| First append (creates the writer stream) | 11.2 ms | 13.9 ms |
| `set_corpus` after an append, median | 0.139 ms | 0.110 ms |
| New searcher, median | 17.8 ms | 15.2 ms |
| Single query: appended vs fresh build | 47.38 vs 47.33 ms (+0.10%) | 20.92 vs 20.80 ms (+0.58%) |
| 60-query batch: appended vs fresh build | 144.89 vs 144.78 ms (+0.07%) | 65.95 vs 67.12 ms (−1.75%) |
| Copy-on-write update, 100 consecutive IDs | 9.1 ms, +25 MB | 9.8 ms, +25 MB |
| Copy-on-write update, 100 scattered IDs | 994 ms, +10.6 GB, 7 shards | 421 ms, +4.6 GB, 5 shards |
| Single / batch after the scattered update | +0.26% / +1.79% | +0.64% / +2.25% |
| In-place update, 100 scattered IDs | 0.885 ms | 0.594 ms |
| Compaction | 1,097 ms | 407 ms |

The 3M reference is a single shard; at 7M × 768 the 2^32-element shard limit
gives every layout at least two. Differences after appends are within the
run-to-run spread of these measurements (about ±2% for batches).

Appends stage rows in GPU-coherent host memory and copy them on one stream kept
per corpus lineage. On a new stream the first copy loads transfer kernels
(about 8 ms) and the stream itself costs about 2 ms, which the first append pays.
An append that fits the tail's reserve leaves the last shard's capacity, and so
its capacity-specialized scan kernel and score storage, unchanged; `set_corpus`
then compiles and allocates nothing.

An earlier variant copied only the touched 16,384-row pages on update. For 100
scattered IDs over 3M rows that produced 175 shards and slowed 60-query batches
from 66.3 to 111.1 ms (+68%; single queries +6.7%), because each shard adds a
batch tile's selection and merge. Updates now also copy untouched gaps shorter
than one 262,144-row tile, trading copy volume for bounded fragmentation.

Reproduce with:

```sh
HRX_OFFLINE=1 cargo run --locked --release --bin hrxdb-bench -- \
  --rows 7000000 --dimensions 768 --append 50 --appends 1000 --updates 100 \
  --samples 30 --output results/7m-768-append.json
```

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
