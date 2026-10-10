# Custom scoring pipelines

```text
resident corpus → gather named subsets → custom scoring kernel → GPU top-k
                                                                    ↓
                                                      selected IDs and scores
```

`gather_into` writes compact normalized FP32 rows, preserving requested order
and duplicates while resolving shards internally. `TopK` consumes your own
row-major score matrix and returns score-column IDs. Album grouping, custom
reductions, and metadata stay in application code.

Runnable examples:

- [Subset comparisons](../examples/subset_scores.rs): gather two blocks, score
  them on the GPU, and read back only the winners.
- [Album ranking](../examples/album_scores.rs): bind corpus vectors and an album
  side array, reduce scores by album, then select top-k.
- [Device search](../examples/device_search.rs): queue queries and update visited
  IDs on the GPU, with event-ordered result consumption.

The custom scoring kernels illustrate interoperability; they are not tuned GEMMs.
Applications passing HRX types across this API must resolve the same `hrx-rs`
crate instance as hrxdb. Check the resolved dependency in `Cargo.lock`.
