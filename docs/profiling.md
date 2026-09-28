# Profile search and explain compiler decisions

HRX 0.8.11 supplies device-clock timestamps and detailed Loom compilation
reports. hrxdb exposes both for exhaustive GPU search. Start with a benchmark
that records ordinary latency, then collects diagnostic evidence:

```sh
cargo run --locked --release --bin hrxdb-bench -- \
  --rows 100000 --dimensions 384 --samples 10 --profile --output single.json
cargo run --locked --release --bin hrxdb-bench -- \
  --rows 100000 --dimensions 384 --batch 60 --samples 10 --profile --output batch.json
```

No sibling checkout or LLVM executable is required. The published bundle is
`native-20260927-244cd3801b`. Provision it once, then use `HRX_OFFLINE=1` if desired.
The benchmark records the configured native manifest, any override variable
names, and the actual compiler identity. Overrides can replace loaded libraries;
the configured manifest alone is not proof of their identity.

## Three different kinds of evidence

| Evidence | What it measures | Interpretation |
|---|---|---|
| Ordinary host latency | Query preparation, GPU submission and completion, result readback | Primary end-to-end performance comparison; excludes setup and compilation. |
| Instrumented device replay | Timestamps around every scan, mask, selection, merge and copy command | Includes timestamp/barrier perturbation. Useful for locating costly stages, not as a component of the separately measured ordinary latency. |
| Compiler report | Emitted resources and compiler models of pressure, waits, scheduling and memory | Explains the compiled program. Occupancy and wait counts are not measured GPU utilization or stall time. |

`--profile` collects three diagnostic replays **after all ordinary measurement
windows**, including when using `--sweep`. Each replay must return exactly the
same neighbors as ordinary search on the same queries and exclusions. Batch
profiling records the same tile/selection/merge commands in a serial graph;
ordinary host batches reuse a separate graph without timestamp markers. Results retain
query order and the same precision and tie rules.

Each replay contains:

- Raw start/end ticks, the device clock frequency, device/queue/recorder identity,
  interval union, total span and gaps.
- Unique command labels, kernel symbols, grid/block sizes, binding extents,
  packed scalar argument bytes and copy sizes. Addresses and corpus contents
  are not recorded. Interpret constants using the kernel's declared types.
- Per-operation command counts and summed device intervals.
- Host time for graph preparation and a separate host interval enclosing that
  replay's launch, completion and timestamp harvest. Query encoding and neighbor
  decoding are outside the latter interval.

Compare a device span only with its **own** enclosing `replay_host_ms`. Do not
subtract it from the ordinary search median or call the difference CPU overhead.
Copies include candidate staging as well as final result transfers. `stages`
groups repeated entry symbols; `commands` retains individual dispatches and
arguments when finer interpretation is needed.

`--profile` is opt-in. Ordinary searches allocate no timestamp storage and emit
no markers. Profiling failures, including unavailable native timestamp support,
return errors rather than falling back to host timing. Native timestamp storage
is diagnostic overhead outside the searcher's workspace budget. Profile graphs
are dropped after completion; normal query buffers/workspace remain reusable.

The native timestamp API does not expose hardware performance counters or shader
instruction traces. HRX-system's separate HAL profiling tools have a broader
surface; hrx-rs 0.8.11 does not expose those services through this search API.

## Use the Rust API

```rust,no_run
use hrxdb::{Corpus, Device};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    let mut searcher = corpus.searcher()?;
    let queries = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // Two row-major queries.
    searcher.reserve_batch(2, 1)?;
    let result = searcher.profile_search(&queries, 1, &[])?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    let compiler = searcher.detailed_compilation_reports()?;
    println!("{}", serde_json::to_string_pretty(&compiler)?);
    Ok(())
}
```

`profile_search` accepts zero through 64 queries, k=1–1,024, and shared insertion
ID exclusions. One query uses the single-query path. Invalid input follows the
ordinary search contract. `execution` is `None` when an empty batch/corpus or
all-excluded corpus requires no GPU commands; this does not invent zero-duration
GPU measurements.

## Inspect compiler evidence

Library construction still requests summary reports. `compilation_reports()`
returns these immediately. `detailed_compilation_reports()` recompiles the same
sources, specialization and target into separate cached analysis artifacts; it
does not replace loaded kernels. Hardware tests verify byte-identical summary
and detailed executables across the tested single/batch kernel families.

The benchmark collects details by default, outside ordinary timing. Choose
`--compile-report summary` to avoid the extra compilation and larger JSON. Each
compilation retains the original versioned report, its compiler digest, target,
processor mode, configuration, structured resources, wait reasons, and full
successful diagnostics with related source locations. Compiler errors also
retain HRX's structured diagnostics.

Resource facts come directly from Loom: registers, LDS, private storage, spills,
code bytes and modeled occupancy. Optional bank-service evidence distinguishes
modeled, unmodeled and structurally conflicted packets. Missing evidence stays
`null`, including bank fields the published compiler did not populate for these
kernels. Do not turn missing facts into zero conflicts. Wait reasons distinguish
explicit/planned actions and full/partial drains where reported.

Single-query reports live under `.trials[]`; batch reports use top-level
`.compiler` and `.profile`. Append/update mode attaches `reference_compiler` and
`reference_search_profile` for the fresh reference corpus, after mutation
measurements. It does not claim to timestamp the mutations themselves.

```sh
# Per-stage instrumented distributions for a single-query trial:
jq '.trials[0].profile.stages' single.json
# Batch dispatches, geometry and raw device intervals:
jq '.profile.samples[0] | {commands, device}' batch.json
# Resources and wait explanations, preserving unknown values:
jq '.compiler[] | {symbol, configuration, resources, wait_reasons}' batch.json
# Extract one canonical Loom report (not the surrounding hrxdb record):
jq '.compiler[] | select(.symbol == "batch_scan") | .report' batch.json > batch-scan.report.json
```

If the matching Loom tools are installed, the extracted document can be read by
`loom-compile-report show batch-scan.report.json` and
`loom-compile-report suggest batch-scan.report.json`. Use the tool revision that
matches the producing compiler; schema version zero is not a cross-version
compatibility guarantee. Suggestions propose experiments, not proven speedups.
Compiler/target identity and specialization must be held fixed or explicitly
accounted for when comparing reports.

## Qualification and kernel tuning

The 0.8.11 integration passes 49 existing GPU correctness tests and three new
profiling/report tests. Coverage includes bitwise result parity, k=10/33/1,024,
batches 1/3/8/60, exclusions, replay, multi-tile merges, append/update snapshots,
empty/invalid input, timestamp ordering and compiler artifact equality. CPU,
Rust 1.91 and documentation checks require no hardware.

The [recorded profiler qualification](../results/hrx-0.8.11-profiling.json)
retains commands, resource summaries, wait reasons and raw timestamp samples.
Another GPU workload was active during those runs, so the latencies are
**contention-affected diagnostic evidence**, not an isolated performance claim.
Full compiler documents can be regenerated with the recorded commands; the
checked-in record omits those large repeated trees.

For that integration's recorded 384-dimensional workload, the single-query scan uses 31 VGPRs
and no LDS or private storage. The width-64 batch scan uses 29 VGPRs and 12,544
bytes of LDS. Both report zero spills and 100% modeled occupancy; neither shows
a resource residency cliff that would by itself justify reducing registers.
The batch report attributes 64 static LDS waits to value dependencies and one
to a barrier. Those counts identify code to inspect, not elapsed stall cycles.

The subsequent [batch optimization measurements](../results/README.md#batch-kernel-tuning-with-hrx-0811)
use those details to replace per-column scheduling fences with groups of eight
ordered products, and unroll the cooperative staging loops. Inspecting
`loom/src/loom/target/arch/amdgpu/lower/dot.c` explained the key distinction:
the incoming accumulator uses a three-operand FMA, while subsequent products
inside one dot can use tied FMACs, which the scheduler can pair. FP32 accumulation
still follows increasing component order; no partial-sum reassociation is used.

At width 64, the same 512 arithmetic terms now require 333 FMA/FMAC instructions
per 32-component loop body, and code shrinks from 8,576 to 6,648 bytes. LDS
dependency waits fall from 64 to 55 (full drains from 32 to 14). VGPRs rise to
52 while reported spills, spill stores/reloads and private bytes remain zero;
modeled occupancy stays 100%. These are static compiler/ISA counts. In
particular, unrolling raises static staging-load counts without increasing
runtime corpus traffic. Instrumented replays localize the gain to `batch_scan`;
selection costs remain similar except when interrupted by the concurrent job.

The comparison harness accepts `BASELINE_SOURCE` and `CANDIDATE_SOURCE`, checks
IDs and score bits, and retains detailed reports from the actual loaded kernels
plus three separate interleaved profiled replays. Its default baseline is the
preserved per-column kernel from `abfcd5c`. All runs used another active GPU job;
the records support improvements under that load, not isolated latency claims.

Research in HRX-system at `244cd3801b` also covered affine-address fusion,
loop-carried accumulator reuse, phased scheduling, explicit read-ahead and
workgroup staging. The updated compiler applies its native optimizations to
existing kernels automatically. Explicit pipelining remains further work:
staging loops need their memory/recurrence contracts preserved, with oracle
checks and controlled comparisons for each candidate. Removing fences alone
did not improve the measured 1M-row case; instruction counts and latency both
matter when choosing a rewrite.

The next [selection and graph-reuse comparison](../results/README.md#selection-and-cached-batch-graphs)
cuts selection barriers from two to one per rank using wave winners,
double-buffered LDS and eight-lane clustered reductions. First-pass selection
shrinks from 234 to 210 emitted instructions with the same 16 VGPRs and no
spills. LDS grows from 64 to 128 bytes without changing modeled occupancy.
The separate replay-only comparison holds kernels fixed and measures ordinary
immediate submission against cached uninstrumented graphs. Its instrumented
replays use fresh graphs in both arms, so their stage intervals cannot quantify
that submission saving. The combined change passes 54 GPU correctness tests.

Upstream references:

- [Compile evidence and its interpretation](https://github.com/ROCm/hrx-system/blob/244cd3801b/loom/docs/src/workflows/compile-reports.md)
- [Detailed report queries](https://github.com/ROCm/hrx-system/blob/244cd3801b/loom/docs/src/workflows/compile-report-queries.md)
- [Read-ahead and workgroup staging](https://github.com/ROCm/hrx-system/blob/244cd3801b/loom/docs/src/workflows/tune-loop-schedules.md)
- [Instrumented replay boundaries](https://github.com/ROCm/hrx-system/blob/244cd3801b/loom/docs/src/workflows/benchmark.md)
