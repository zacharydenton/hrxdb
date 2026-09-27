# NPU-only search exploration

The experimental `npu_search_probe` runs exhaustive FP16 scoring, top-k selection,
and candidate merging on Strix Halo NPU5 without opening the GPU. It establishes
correctness and a path to a backend; it is not production NPU support. The public
`Corpus`, `Searcher`, `TopK`, and device bindings remain GPU-backed.

The current scalar implementation is much slower than GPU search and cannot
prepare the planned 100K/1M-row workloads because it exhausts native contexts.
Those failures are recorded, not reported as completed benchmarks.

## Run it

Requires Linux, access to `/dev/accel/accel0`, and the published HRX 0.8.11 bundle
`native-20260927-244cd3801b`. First use provisions the verified bundle; subsequent
runs can use `HRX_OFFLINE=1`. No local HRX/compiler overrides are needed.

```sh
cargo run --locked --release --features npu --example npu_search_probe -- --primitives
cargo run --locked --release --features npu --example npu_search_probe -- --rows 1033 --dimensions 129 --k 10
cargo run --locked --release --features npu --example npu_search_probe -- --rows 4096 --dimensions 768 --compare-gpu --output comparison.json
cargo test --locked --release --features npu --test npu -- --ignored --test-threads=1
python3 scripts/qualify_npu.py
```

The `npu` Cargo feature forwards to `hrx/npu`. It enables the example and tests;
it does not select a new public search backend. `--compare-gpu` explicitly opens
a separate GPU reference using the same quantized rows. Without that option,
ingestion, normalization, validation, and orchestration run on the CPU; all
scoring and selection execute on the NPU. `--fp16` rounds generated input directly
to FP16 instead of normalizing before quantization.

Supported probe shapes are dimensions 1–1,024, batches 1–64, and k=1–1,024.
Batches execute queries sequentially; this is not a matrix-accelerated batch path.
The internal test helper also handles empty corpora/batches and exclusions.
`--max-seconds N` bounds the search phase and exits between completed chunks;
it never kills an in-flight native submission. Preparation is outside this limit.
The qualification script records every case and exits nonzero if any case fails
or reaches its time limit. With the current context limit, that is expected.

## Representation and execution

Host encoding follows hrxdb's FP32/FP16 ingestion rules, including inverse norms
computed from the quantized values and zero dimension padding. NPU scoring widens
FP16 exactly and performs an ascending FP32 dot product, followed by the stored
inverse norm. It does not convert values to BF16/BFP16. Query normalization uses
the existing host FP64-norm/FP32-query convention.

Each NPU core scores eight rows per DMA record. The query uses a zero-stride
record dimension, so DMA repeats the same query without duplicating it on the
host. Metadata holds inverse norms and insertion IDs; a negative inverse masks
an exclusion or padding. One prepared scoring program covers up to 1,024 rows.
Its three input DMA channels require a two-column native context. Selection and
merge use one-column contexts.

Selection inserts candidates into a sorted bounded list. Merge combines that
list with the running top-k on the NPU. Ordering is descending score, then
ascending unsigned insertion ID. Score/candidate scratch is bounded by one
chunk and k, independent of corpus size. Only validation reads all scores;
normal search reads final neighbors. CPU sorting is an independent test oracle,
not a retrieval fallback.

All buffers use `MemoryPlacement::NpuLocal`; guarded host mapping publishes
inputs and acquires outputs. Do not replace this with `HostVisible`, `Shared`,
`ModelContext::allocate`, or `ModelContext::compiler`: those paths can initialize
the GPU. HRX graph copy/fill operations also currently require GPU-visible
backing. The prototype uses CPU mapped writes only during preparation/query setup.

## What passed

On 2026-09-27, the published bundle passed:

- All 63,488 finite FP16 bit patterns, including signed zeros and subnormals,
  widened exactly. FP32 addition, multiplication, comparison/selection matched
  host result bits across two replays with changed operands, including tiny
  FP32 operands.
- Both ingestion paths, dimensions 1, 3, 129, 384, 769, and 1,024, unaligned tails,
  multi-chunk merges, changed queries, exclusions and resetting exclusions.
- Every batch width 1–64; k=1, 10, 32, and 1,024; negative-score ties and ID ordering;
  all-excluded, empty, invalid, and extreme finite input cases.
- Top-k exactly matches CPU sorting of the NPU-produced scores. Scores meet the
  existing absolute `3e-6` tolerance against a FP64 reference over stored FP16
  values. GPU score bits are not required to match because reductions differ.
- Native allocation budget enforcement, rollback/release, and safe resume after
  an exploration deadline. Caller-owned host input copies are outside the native
  allocation budget.
- A multi-chunk search with GPU device nodes unavailable. HRX traces contain only
  NPU lane 3; counters show no GPU graphs, copy streams, shared imports, or copies.

Reproduce device isolation with bubblewrap after building the example:

```sh
bwrap --bind / / --dev /dev --dev-bind /dev/accel /dev/accel -- \
  /bin/sh -c 'test ! -e /dev/kfd && test ! -e /dev/dri && exec \
  target/release/examples/npu_search_probe --rows 1033 --dimensions 129 --samples 1 --warmups 0'
```

Primitive and isolation evidence is in
[`npu-capabilities.json`](../results/npu-capabilities.json). Full sample arrays,
commands, and failed large-shape attempts are in
[`npu-qualification.json`](../results/npu-qualification.json).

## Measurements and blockers

Three warmups and ten measured searches, k=10, generated rows, host wall time
with completion, excluding preparation and ingestion:

| Rows × dimensions | Queries | NPU median | GPU median | Maximum score error |
|---|---:|---:|---:|---:|
| 1,033 × 129 | 1 | 34.10 ms | 0.0348 ms | 1.20e-7 |
| 4,096 × 768 | 1 | 391.10 ms | 0.0494 ms | 1.50e-7 |
| 256 × 384 | 60 | 761.56 ms | 0.5867 ms | 1.50e-7 |

All diagnostic top-k IDs matched the GPU for these generated queries; this is
not a guarantee for near ties. The NPU spent approximately 32.84, 386.41, and
734.01 ms respectively in scoring. This scalar FP32 baseline does not use the
NPU matrix engine. The correctness result does not establish competitive speed.

Two separate native limitations were reproduced:

1. **Offset admission.** The generated XDNA image contracts permit external
   binding offset zero only. Binding a subview starting at byte 64 fails with
   `XDNA binding 0 offset 64 is outside [0, 0]`. The prototype gives each corpus
   chunk independent backing. Reproduce with
   `npu_search_probe --primitives --binding-offset 64` (expected failure).
2. **Prepared contexts.** HRX 0.8.11 prepares a private native context for each NPU
   graph node. This prototype has three nodes per corpus chunk. Preparation of
   the seventeenth program failed on this host with `[context create] libamdf
   ERRNO status 0x00000016`, while preparing selection for row 5,120 after five
   complete chunks. All eight requested combinations of 100K/1M rows,
   384/768 dimensions, and 1/60 queries failed here, before execution. This is an
   observed resource limit of this design/runtime/driver combination, not a
   universal maximum NPU corpus size. `--rows 6144 --dimensions 384` is a small
   reproducer. No latency is inferred for the failed cases.

The minimum next step is bounded program/context reuse or a fused streaming
pipeline with a fixed number of native programs. Raising the row limit or
adding more prepared graphs does not solve this. Even after the context issue,
FP16-preserving scoring needs a much faster implementation. Any matrix path
must independently meet the same numerical contract; replacing FP16 with the
bundled BF16/BFP16 GEMM would change that contract.

## Roadmap to a production backend

Keep the prototype isolated until context scaling and performance are resolved.
The following order avoids committing public APIs to an unqualified design:

| Area | Current evidence | Required production work |
|---|---|---|
| Execution | NPU-only primitives and search work | Bound contexts across chunks and workers; expose safe reuse/rebinding or stream records through fused pipelines; preserve retirement and budgets. |
| Storage | Separate immutable NPU chunks | Introduce backend-owned allocations and neutral row/segment metadata; support dimensions through 16,384 and large addressing; qualify subrange contracts. |
| Scoring/selection | Exact FP16 widening, scalar FP32 scores, bounded NPU top-k | Optimize/vectorize while preserving stored values; distribute work across cores/columns; fuse local top-k with scoring; rerun large-shape qualification. |
| Snapshots/mutations | GPU implementation only | Reuse append claims and copy-on-write page semantics with NPU copy/scatter kernels; test held snapshots, tail growth, compaction, and rollback. |
| Gather/exclusions | Host-set exclusions in probe | Implement NPU gather, persistent device exclusion updates, and capacity masking; avoid GPU copy/fill helpers. |
| Device APIs | Blocking host-driven prototype | Add NPU-native query/result bindings, normalization/status kernels, completion handles, and explicit device selection without GPU initialization. |
| Prepared search | Fixed private experimental graphs | Make context/compiler/placement explicit; retain slots, immutable snapshots and producer dependencies; qualify asynchronous replay and thread handoff. |
| Residency | Native budget and release tests pass | Account for contexts, programs, scratch and shared snapshot allocations; add eviction, failure-injection, and concurrency qualification. |

Public GPU APIs and default behavior should remain compatible. An eventual NPU
backend must be explicitly selected and report unsupported capabilities rather
than silently falling back to GPU or CPU scoring/selection. Enabling the Cargo
feature alone must not require an NPU on GPU-only applications.
