//! Shared ownership and borrowed bindings for immutable GPU vector storage.
use crate::{Device, Result, ScanConfig, Searcher, Shard};
use hrx::{Stream, View};
use std::{ops::Range, sync::Arc};

/// Cheaply cloned, immutable GPU-resident FP16 corpus. Clones share allocations.
///
/// A corpus is Send + Sync. Independent searchers own their own streams and
/// workspace. Replacing a snapshot leaves old clones and in-flight users valid.
/// Logical rows are insertion-ordered, zero-padded FP16 with FP32 inverse norms.
/// Treat exported bindings as read-only; HRX does not enforce write protection.
/// Callers own synchronization for their custom kernels. Construction completes
/// uploads before returning. Borrowed bindings may not outlive the corpus.
///
/// ```compile_fail
/// # use hrxdb::Corpus;
/// fn dangling(corpus: Corpus) -> hrx::View<'static> {
///     corpus.shards().next().unwrap().vectors()
/// }
/// ```
#[derive(Clone)]
pub struct Corpus {
    pub(crate) inner: Arc<CorpusStorage>,
    pub(crate) workspace_budget: Option<hrx::residency::MemoryBudget>,
}
pub(crate) struct CorpusStorage {
    pub(crate) context: Option<Arc<CorpusContext>>,
    pub(crate) device: Device,
    pub(crate) device_id: usize,
    pub(crate) shards: Vec<Shard>,
    pub(crate) dimensions: usize,
    pub(crate) padded: usize,
    pub(crate) count: usize,
    pub(crate) gather: std::sync::Mutex<Option<hrx::Kernel>>,
}

// HRX exposes domain equality, not a numeric ID. Weak entries give residency
// keys stable identity across ModelContext clones without retaining runtimes.
pub(crate) struct CorpusContext {
    pub(crate) model: hrx::inference::ModelContext,
    pub(crate) id: u64,
}
impl CorpusContext {
    pub(crate) fn shared(model: &hrx::inference::ModelContext) -> Arc<Self> {
        use std::sync::{Mutex, Weak};
        static DOMAINS: Mutex<(u64, Vec<Weak<CorpusContext>>)> = Mutex::new((0, Vec::new()));
        let mut domains = DOMAINS.lock().unwrap_or_else(|e| e.into_inner());
        domains.1.retain(|domain| domain.strong_count() != 0);
        for domain in domains.1.iter().filter_map(Weak::upgrade) {
            if domain.model.runtime().same_domain(model.runtime()) {
                return domain;
            }
        }
        domains.0 = domains
            .0
            .checked_add(1)
            .expect("corpus context IDs exhausted");
        let domain = Arc::new(Self {
            model: model.clone(),
            id: domains.0,
        });
        domains.1.push(Arc::downgrade(&domain));
        domain
    }
}
/// Borrowed bindings for a contiguous range of insertion IDs. Bindings include
/// readable slack; only rows in `row_range()` are corpus data.
#[derive(Clone, Copy, Debug)]
pub struct CorpusShardView<'a> {
    vectors: View<'a>,
    inverse_norms: View<'a>,
    start: usize,
    count: usize,
    capacity: usize,
}
impl Corpus {
    /// Context retained by [`Self::build_in`] or [`Self::build_fp16_in`].
    /// Device-based constructors return `None`. Clones retain the same runtime.
    /// Corpus shards remain native buffers; coordinated tensor access uses
    /// [`Self::prepare_search`] and HRX's scoped GPU handoff.
    pub fn context(&self) -> Option<&hrx::inference::ModelContext> {
        self.inner.context.as_ref().map(|context| &context.model)
    }

    /// Create an independent stream on this corpus's device.
    pub fn stream(&self) -> Result<Stream> {
        let stream = self.inner.device.stream()?;
        Ok(match &self.workspace_budget {
            Some(budget) => stream.with_memory_budget(budget.clone()),
            None => stream,
        })
    }
    /// Charge allocations on subsequent corpus streams, searchers and custom
    /// scoring workspaces to this ceiling. Existing shared corpus storage and
    /// already-created workers are unchanged. Clones retain the budget policy.
    pub fn with_workspace_budget(mut self, budget: hrx::residency::MemoryBudget) -> Self {
        self.workspace_budget = Some(budget);
        self
    }
    /// Create an independent search workspace sharing this corpus's allocations.
    pub fn searcher(&self) -> Result<Searcher> {
        Searcher::new(self.clone())
    }
    /// Create a searcher with a chosen single-query scan schedule.
    pub fn searcher_with_config(&self, config: ScanConfig) -> Result<Searcher> {
        Searcher::with_config(self.clone(), config)
    }
    /// Number of logical vectors.
    pub fn len(&self) -> usize {
        self.inner.count
    }
    /// Whether this corpus has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Logical vector dimension.
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions
    }
    /// Row stride in FP16 elements, including zero padding to a multiple of 128.
    pub fn padded_dimensions(&self) -> usize {
        self.inner.padded
    }
    /// Global row-coordinate extent needed by side arrays covering every shard's
    /// readable capacity. Includes slack, which has no insertion IDs. Empty
    /// corpora return `0..0`. This is the maximum shard end, not the sum of their
    /// capacities: interior slack can overlap later shards' logical rows.
    pub fn capacity_range(&self) -> Range<usize> {
        0..self
            .inner
            .shards
            .iter()
            .map(|s| s.start + s.capacity)
            .max()
            .unwrap_or(0)
    }
    /// Nonempty storage shards in insertion order, partitioning `0..len()`.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = CorpusShardView<'_>> {
        self.inner.shards.iter().map(|s| CorpusShardView {
            vectors: s.data.binding(),
            inverse_norms: s.norms.binding(),
            start: s.start,
            count: s.count,
            capacity: s.capacity,
        })
    }
}
impl Searcher {
    /// Shared immutable storage. Clone this handle to retain a snapshot or build
    /// another searcher without copying vectors.
    pub fn corpus(&self) -> &Corpus {
        &self.corpus
    }
    /// This searcher's ordered execution stream. Use it to submit producers or
    /// consumers around device search. Do not replace or synchronize other queues
    /// implicitly; use HRX events for cross-stream dependencies.
    pub fn stream(&mut self) -> &mut Stream {
        &mut self.stream
    }
}
impl<'a> CorpusShardView<'a> {
    /// Global insertion IDs, with an exclusive end.
    pub fn row_range(&self) -> Range<usize> {
        self.start..self.start + self.count
    }
    /// Readable rows in both bindings, a multiple of 256 and at least the logical
    /// count. Extra rows have unspecified contents and must be masked before
    /// reduction/selection. Caller side arrays must also cover their own reads.
    /// Capacity times padded dimensions is at most 2^32 FP16 elements.
    pub fn capacity_rows(&self) -> usize {
        self.capacity
    }
    /// Global row coordinates covered by this shard's readable bindings.
    /// Extends [`Self::row_range`] by tail slack and may overlap later shards.
    /// Only `row_range()` describes insertion IDs; mask other rows in kernels.
    pub fn capacity_range(&self) -> Range<usize> {
        self.start..self.start + self.capacity
    }
    /// Read-only little-endian FP16, row-major, including readable row slack.
    /// Logical rows have zero dimension padding. Length is capacity*stride*2 bytes.
    pub fn vectors(&self) -> View<'a> {
        self.vectors
    }
    /// Read-only little-endian FP32 inverse norms, one per capacity row. Only
    /// logical rows have defined values. Multiply a unit-query dot by this norm.
    pub fn inverse_norms(&self) -> View<'a> {
        self.inverse_norms
    }
}
#[cfg(test)]
#[path = "../examples/support/album_scores.rs"]
mod album_example;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Result, ScanConfig, encode, encode_fp16};
    use half::f16;

    #[test]
    #[ignore = "requires gfx1151"]
    fn custom_streams_survive_device_drop_and_index_move() -> Result<()> {
        for direct_fp16 in [false, true] {
            let (mut index, expected_device) = {
                // Exercise a non-default device when a second supported GPU is
                // available; always check native identity, not just its target.
                let device = match Device::open(1) {
                    Ok(device) if device.target().as_str() == "gfx1151" => device,
                    _ => Device::open(0)?,
                };
                let index = if direct_fp16 {
                    let rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].map(|row| {
                        row.into_iter()
                            .flat_map(|v| f16::from_f32(v).to_le_bytes())
                            .collect::<Vec<_>>()
                    });
                    Searcher::build_fp16(&device, 3, rows)?
                } else {
                    Searcher::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?
                };
                let identity = index.stream.device_id();
                (index, identity)
            }; // The original Device is gone before either public API is used.
            std::thread::spawn(move || -> Result<()> {
                let mut first = index.corpus().stream()?;
                let second = index.corpus().stream()?;
                assert_eq!(first.device_id(), expected_device);
                assert_eq!(second.device_id(), expected_device);
                assert_eq!(first.target(), index.stream.target());
                assert_ne!(first.id(), second.id());
                assert_ne!(first.id(), index.stream.id());
                assert_ne!(second.id(), index.stream.id());

                let binding = index.corpus().shards().next().unwrap().vectors();
                let mut bytes = [0; 2];
                first.read_blocking(binding.slice(0, 2)?, &mut bytes)?;
                assert_eq!(bytes, f16::ONE.to_le_bytes());
                drop(second);
                let scores =
                    album_example::best_by_album(&index.corpus, &[0, 1], 2, &[1.0, 0.0, 0.0])?;
                assert_eq!(scores, [vec![1.0, 0.0]]);
                assert_eq!(index.search(&[1.0, 0.0, 0.0], 1)?[0].id, 0);
                drop(index);

                // The returned stream is owned and usable after the index
                // drops; no borrowed corpus binding is used past that point.
                let buffer = first.allocate(4)?;
                first.upload(buffer.binding(), &[1, 2, 3, 4])?;
                let mut readback = [0; 4];
                first.read_blocking(buffer.binding(), &mut readback)?;
                assert_eq!(readback, [1, 2, 3, 4]);
                Ok(())
            })
            .join()
            .expect("custom kernel worker panicked")?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn custom_kernel_reads_shards_and_merges_album_scores() -> Result<()> {
        let device = Device::open(0)?;
        for d in [3usize, 129, 769] {
            let padded = d.div_ceil(128) * 128;
            let rows: Vec<Vec<f32>> = (0..263)
                .map(|i| {
                    (0..d)
                        .map(|j| ((i * 37 + j * 13) % 103) as f32 - 49.25)
                        .collect()
                })
                .collect();
            let ordinals: Vec<u32> = (0..rows.len())
                .map(|i| ((i * 7 + i / 129) % 5) as u32)
                .collect();
            let queries: Vec<f32> = (0..3).flat_map(|q| rows[q * 131].iter().copied()).collect();
            for direct_fp16 in [false, true] {
                let mut index = if direct_fp16 {
                    let bytes: Vec<Vec<u8>> = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                                .collect()
                        })
                        .collect();
                    Searcher::build_encoded(
                        &device,
                        d,
                        &bytes,
                        ScanConfig::default(),
                        129 * padded,
                        |r, d, p, bytes| encode_fp16(r, d, p, bytes),
                    )?
                } else {
                    Searcher::build_encoded(
                        &device,
                        d,
                        &rows,
                        ScanConfig::default(),
                        129 * padded,
                        |r, d, p, bytes| encode(r, d, p, bytes),
                    )?
                };
                let before = index.search(&queries[..d], 5)?;
                let corpus = index.corpus();
                assert_eq!(corpus.len(), rows.len());
                assert!(!corpus.is_empty());
                assert_eq!(corpus.dimensions(), d);
                assert_eq!(corpus.padded_dimensions(), padded);
                assert_eq!(corpus.shards().len(), 3);
                assert_eq!(
                    corpus.shards().map(|s| s.row_range()).collect::<Vec<_>>(),
                    [0..129, 129..258, 258..263]
                );

                // Bindings remain usable after the temporary corpus/shard
                // descriptors disappear, and work on a separate caller stream.
                let binding = index.corpus().shards().next().unwrap().vectors();
                let mut reader = device.stream()?;
                reader.read_blocking(binding.slice(0, 2)?, &mut [0; 2])?;
                for shard in corpus.shards() {
                    let range = shard.row_range();
                    assert_eq!(shard.capacity_rows(), range.len().div_ceil(256) * 256);
                    assert_eq!(shard.vectors().len(), shard.capacity_rows() * padded * 2);
                    assert_eq!(shard.inverse_norms().len(), shard.capacity_rows() * 4);
                    let mut vectors = vec![0; shard.vectors().len()];
                    let mut norms = vec![0; shard.inverse_norms().len()];
                    reader.read_blocking(shard.vectors(), &mut vectors)?;
                    reader.read_blocking(shard.inverse_norms(), &mut norms)?;
                    for (local, global) in range.enumerate() {
                        let source = &rows[global];
                        let length = source
                            .iter()
                            .map(|&v| (v as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let stored = &vectors[local * padded * 2..(local + 1) * padded * 2];
                        let mut sum = 0.0;
                        for (j, bytes) in stored.as_chunks::<2>().0.iter().enumerate() {
                            let actual = f16::from_le_bytes(*bytes);
                            let expected = if j >= d {
                                f16::ZERO
                            } else if direct_fp16 {
                                f16::from_f32(source[j])
                            } else {
                                f16::from_f64(source[j] as f64 / length)
                            };
                            assert_eq!(actual.to_bits(), expected.to_bits());
                            sum += actual.to_f64().powi(2);
                        }
                        let inverse =
                            f32::from_le_bytes(norms[local * 4..local * 4 + 4].try_into().unwrap());
                        assert_eq!(inverse, (1.0 / sum.sqrt()) as f32);
                    }
                }

                // Album 5 is absent; all other groups cross shard boundaries.
                // Maxima must persist when later shards contain worse matches.
                let actual = album_example::best_by_album(corpus, &ordinals, 6, &queries)?;
                for (q, query) in queries.chunks_exact(d).enumerate() {
                    let mut expected = [f32::NEG_INFINITY; 6];
                    for (score, &album) in index.scores(query)?.iter().zip(&ordinals) {
                        expected[album as usize] = expected[album as usize].max(*score);
                    }
                    for (&got, &want) in actual[q].iter().zip(&expected) {
                        assert!(
                            got == want || (got - want).abs() < 3e-6,
                            "d={d} direct_fp16={direct_fp16} query={q}: {got} vs {want}"
                        );
                    }
                }
                assert_eq!(index.search(&queries[..d], 5)?, before);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn full_tiles_mask_slack_and_keep_the_last_real_row() -> Result<()> {
        let device = Device::open(0)?;
        // Include both reported tail sizes, exact tiles, and a one-row tail.
        for n in [1, 255, 256, 257, 256 + 201, 2 * 257 + 55] {
            let mut rows = vec![[-1.0, 0.0, 0.0]; n];
            rows[n - 1] = [-0.6, -0.8, 0.0]; // Unique best, still negative.
            let mut index = Searcher::build_encoded(
                &device,
                3,
                &rows,
                ScanConfig::default(),
                257 * 128,
                |r, d, p, bytes| encode(r.as_ref(), d, p, bytes),
            )?;
            assert_eq!(index.len(), n);
            // Test-only writes to slack make an unmasked row beat every real
            // score. The public API promises no particular slack contents.
            let mut writer = device.stream()?;
            for shard in &index.corpus.inner.shards {
                let extra = shard.capacity - shard.count;
                if extra == 0 {
                    continue;
                }
                let mut positive = vec![0; extra * 128 * 2];
                for row in positive.as_chunks_mut::<256>().0 {
                    row[..2].copy_from_slice(&f16::ONE.to_le_bytes());
                }
                let inverses: Vec<_> = (0..extra).flat_map(|_| 1.0f32.to_le_bytes()).collect();
                writer.upload(
                    shard
                        .data
                        .try_slice(shard.count * 128 * 2, positive.len())?,
                    &positive,
                )?;
                writer.upload(
                    shard.norms.try_slice(shard.count * 4, inverses.len())?,
                    &inverses,
                )?;
            }
            writer.synchronize()?;
            let query = [1.0, 0.0, 0.0];
            let scores = index.scores(&query)?;
            assert_eq!(scores.len(), n);
            let expected = scores[n - 1];
            assert!(expected < 0.0);
            let grouped = album_example::best_by_album(&index.corpus, &vec![0; n], 2, &query)?;
            assert!(
                (grouped[0][0] - expected).abs() < 3e-6,
                "n={n}: {grouped:?}"
            );
            assert_eq!(grouped[0][1], f32::NEG_INFINITY);
            let matches = index.search(&query, 1024)?;
            assert_eq!(matches.len(), n);
            assert_eq!(matches[0].id as usize, n - 1);
            assert!(
                matches
                    .iter()
                    .all(|m| (m.id as usize) < n && m.similarity < 0.0)
            );
            let batch = index.search_batch(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1024)?;
            for result in batch {
                assert_eq!(result.len(), n);
                assert_eq!(result[0].id as usize, n - 1);
                assert!(
                    result
                        .iter()
                        .all(|m| (m.id as usize) < n && m.similarity < 0.0)
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn empty_corpus_exposes_no_shards() -> Result<()> {
        let device = Device::open(0)?;
        let index = Searcher::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
        let corpus = index.corpus();
        assert_eq!(corpus.len(), 0);
        assert!(corpus.is_empty());
        assert_eq!(corpus.dimensions(), 3);
        assert_eq!(corpus.padded_dimensions(), 128);
        assert_eq!(corpus.shards().len(), 0);
        assert!(corpus.shards().next().is_none());
        assert_eq!(corpus.stream()?.device_id(), index.stream.device_id());
        assert_eq!(
            album_example::best_by_album(corpus, &[], 2, &[1.0, 0.0, 0.0])?,
            vec![vec![f32::NEG_INFINITY; 2]]
        );
        Ok(())
    }
}
