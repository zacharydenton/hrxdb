# Corpus storage and shared contexts

IDs are zero-based insertion positions. Keep external IDs and metadata in
parallel application-owned storage. `Corpus::build` accepts a sized iterator
of FP32 rows, normalizes and quantizes them, and uploads in bounded chunks.
`Corpus::build_fp16` preserves supplied little-endian FP16 values and computes
their inverse norms. Finite, nonzero rows are required.

`corpus.save(path)` writes a versioned native file containing the padded FP16
rows, FP32 inverse norms, and shard counts and capacities. Reload it with
`Corpus::load_in(&context, path)`: 4 MiB reads overlap the preceding upload,
without decoding floats or recomputing norms. Loading checks the file layout
and length, and trusts the saved payload. Save to a temporary file and rename
it to publish a replacement atomically; `save` itself overwrites its path.

When an encoder already has an HRX `ModelContext`, use
`Corpus::build_fp16_in(&context, dimensions, rows)` (or `build_in` for FP32).
The corpus retains that context and uses its GPU and memory budget for native
storage, upload staging and subsequent workspaces. `corpus.context()` returns
the retained context. Prepare search with that context to submit encoder
tensors directly. Existing device-based constructors remain available.

## Choose your operation

| Task | API |
|---|---|
| Search one query or a batch | `Searcher::search`, `search_batch` |
| Reuse host result capacity | `search_into`, `search_batch_into`, `scores_into` |
| Share one corpus across workers | `Corpus::clone`, `Corpus::searcher` |
| Save and reload native storage | `Corpus::save`, `Corpus::load_in` |
| Add rows to a live corpus | `Corpus::append`, `append_fp16`, `Searcher::set_corpus` |
| Replace rows at existing IDs | `Corpus::update`, `update_fp16`, `Corpus::compact` |
| Gather selected rows for a custom kernel | `Corpus::gather_into` |
| Bind resident vectors without a copy | `Corpus::shards` |
| Adopt owned FP16 device allocations | `Corpus::from_device` |
| Search with GPU-produced queries | `Searcher::search_device`, `DeviceQueries` |
| Rank application-defined GPU scores | `TopK`, `ScoreBatch` |
| Keep results and exclusions on the GPU | `DeviceNeighbors`, `DeviceExclusions` |
| Account for shared storage and workspace | `Corpus::memory_usage`, `Searcher::memory_usage` |
| Reserve loader peaks and pin a shared corpus | `Corpus::build_memory`, `Corpus::load_resident_fp16` |
| Load and cache within an encoder's context | `Corpus::build_fp16_in`, `Corpus::load_resident_fp16_in` |

The optional residency API uses an explicit HRX `ResidencyManager`. It reserves
corpus storage and bounded conversion/upload staging before allocating, rolls
back failed loads, and shares storage for the same immutable artifact/device/
shape key. A lease, `lease.pin()`, exported corpus clone or live searcher prevents
eviction. Attach `manager.budget()` with `Corpus::with_workspace_budget` to charge
subsequent searcher, batch and custom scoring allocations on corpus streams to
the same ceiling. Coordinated `prepare_search` uses its context's budget for
private workers. Caller mappings and compiler/driver memory remain separate;
this is not a process-wide memory ceiling.

`Corpus::load_resident_fp16_in(&context, artifact, dimensions, rows)` uses the
live residency manager attached to the context's memory budget. Its cache keys
include runtime identity, and individual allocations carry their charges, so
corpus bytes are not reserved twice. Native shard bindings retain their existing
stream-ordering contract; they are not coordinated `BufferView`s. See the
[shared-context example](../examples/context_search.rs).

`Corpus` is `Clone + Send + Sync`; clones share immutable allocations. Each
`Searcher` owns its stream and workspace.

A corpus is a snapshot. `corpus.append(rows)` returns a new snapshot with the
rows at IDs `len()..len() + n`, sharing every existing allocation: rows go into
a reserve at the end of the last shard, which grows geometrically, so repeated
small appends keep one tail shard and search as fast as a fresh build.
`corpus.update(ids, rows)` replaces vectors at existing IDs. A handle nothing
else shares is rewritten in place; otherwise the touched pages are copied on the
device, so older snapshots and in-flight searches never see a change.
`searcher.set_corpus(next)` moves a worker to the new snapshot, reusing its
stream, kernels and workspace. See the [guide](../docs/guide.md#appending-and-updating-rows)
for placement, costs and the slack contract. Independent workers allow independent
submission, but do not guarantee overlapping execution, higher throughput,
priority, or latency isolation on a saturated GPU. Use batching to share corpus
reads across queries. Use events to order work between streams.

Public `MAX_BATCH` and `MAX_K` expose the supported query and result ceilings.
Logical dimensions are 1–16,384; corpora support up to 2^30 rows, subject to
available memory. Internal shards contain at most 2^32 FP16 elements and include
readable row slack for custom tiled kernels. Only logical rows participate in
search; custom kernels must mask slack.

See the [API and execution guide](../docs/guide.md) for buffer layouts,
normalization, exclusions, memory limits, and synchronization contracts.
See the [API reference](https://docs.rs/hrxdb/0.3.1/hrxdb/), or generate it locally
with `cargo doc --locked --no-deps --open`.
