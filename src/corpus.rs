//! Borrowed access to the resident corpus for application-owned GPU kernels.
use crate::{FlatIndex, Result};
use hrx::{Stream, View};
use std::ops::Range;

/// A borrowed, zero-copy view of the vectors owned by a [`FlatIndex`].
///
/// Rows retain their insertion order across contiguous, nonoverlapping shards.
/// The stored representation is row-major little-endian FP16, with zero padding
/// to [`padded_dimensions`](Self::padded_dimensions). Each row has an FP32
/// inverse norm computed from its stored FP16 values. [`FlatIndex::build`]
/// normalizes before quantization; [`FlatIndex::build_fp16`] preserves the
/// supplied FP16 values. Neither accessor reconstructs the original FP32 input.
/// Shard bindings include readable slack after their logical rows; see
/// [`CorpusShardView::capacity_rows`]. Slack is not indexed corpus data.
///
/// # Execution and access contract
///
/// Corpus uploads complete before construction returns. Create a caller-owned
/// stream on the corpus's device with [`Self::stream`] or [`FlatIndex::stream`].
/// The original construction-time [`crate::Device`] handle need not be retained.
/// The caller owns kernel compilation, side arrays, output buffers, and
/// execution/completion; acquiring a view performs no allocation or GPU work.
/// This view does not expose or synchronize the index's search workspace.
///
/// Treat vector and inverse-norm bindings as **read-only**. HRX's [`View`] type
/// does not enforce this restriction: uploading, filling, or dispatching writes
/// to them is unsupported and invalidates the index's data invariants. The
/// ordinary safety requirements of [`hrx::Stream::dispatch`] still apply.
///
/// Bindings borrow the index. Rust lifetimes do not wait for asynchronous GPU
/// work; callers must manage completion using HRX. The custom-kernel example
/// completes its stream before returning its results.
///
/// A binding cannot outlive the index that owns it:
///
/// ```compile_fail
/// fn escape(index: hrxdb::FlatIndex) -> hrx::View<'static> {
///     index.corpus().shards().next().unwrap().vectors()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct CorpusView<'a> {
    index: &'a FlatIndex,
}

/// Borrowed bindings for one contiguous range of corpus insertion IDs.
///
/// Both bindings start at local row zero. Use [`row_range`](Self::row_range)
/// to align application side arrays or convert local rows to global IDs.
/// Shard boundaries are storage details and may split application groups.
/// See [`CorpusView`] for the read-only and execution contract.
#[derive(Clone, Copy, Debug)]
pub struct CorpusShardView<'a> {
    vectors: View<'a>,
    inverse_norms: View<'a>,
    start: usize,
    count: usize,
    capacity: usize,
}

impl FlatIndex {
    /// Create an independent HRX stream on this index's device.
    ///
    /// The index retains the device selection supplied at construction, so the
    /// original [`crate::Device`] handle may already have been dropped. Use the
    /// returned stream's [`target`](Stream::target) when configuring a compiler.
    ///
    /// Every call creates a new stream. Reuse it for repeated custom operations;
    /// it owns its resources and can outlive the index. Corpus bindings still
    /// borrow the index. This does not expose the internal search stream or
    /// workspace, synchronize searches, or copy corpus data. The caller owns
    /// ordering and completion of work submitted to the returned stream.
    ///
    /// # Errors
    /// Returns an error if HRX cannot create a stream on the selected device.
    ///
    /// ```no_run
    /// use hrxdb::{Device, FlatIndex};
    /// # fn main() -> hrxdb::Result<()> {
    /// let index = {
    ///     let device = Device::open(0)?;
    ///     FlatIndex::build(&device, 3, [[1.0, 2.0, 3.0]])?
    /// };
    /// let mut stream = index.stream()?;
    /// let compiler_options = hrx::loom::CompilerOptions {
    ///     target: stream.target().clone(),
    ///     ..Default::default()
    /// };
    /// // Compile and bind a custom kernel to `index.corpus()` on this stream.
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn stream(&self) -> Result<Stream> {
        self.device.stream()
    }

    /// Borrow the resident FP16 corpus for custom GPU computations.
    ///
    /// This allocates no storage and performs no copy, compilation, or GPU work.
    /// See [`CorpusView`] for layout, normalization, and execution requirements.
    ///
    /// ```no_run
    /// use hrxdb::{Device, FlatIndex};
    /// # fn main() -> hrxdb::Result<()> {
    /// let device = Device::open(0)?;
    /// let index = FlatIndex::build(&device, 3, [[1.0, 2.0, 3.0]])?;
    /// let corpus = index.corpus();
    /// for shard in corpus.shards() {
    ///     let rows = shard.row_range();
    ///     assert!(shard.capacity_rows() >= rows.len());
    ///     assert_eq!(shard.vectors().len(), shard.capacity_rows() * corpus.padded_dimensions() * 2);
    ///     assert_eq!(shard.inverse_norms().len(), shard.capacity_rows() * 4);
    ///     // Bind these views to a custom kernel on `corpus.stream()?`.
    ///     // Side arrays use the global insertion IDs in `rows`.
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn corpus(&self) -> CorpusView<'_> {
        CorpusView { index: self }
    }
}

impl<'a> CorpusView<'a> {
    /// Create an independent HRX stream on the device that owns this corpus.
    ///
    /// Equivalent to [`FlatIndex::stream`]. The returned stream is owned, does
    /// not borrow this view or the index, and provides the compiler target via
    /// [`Stream::target`]. Reuse it across custom operations when possible.
    ///
    /// # Errors
    /// Returns an error if HRX cannot create a stream on the selected device.
    pub fn stream(&self) -> Result<Stream> {
        self.index.stream()
    }

    /// Total number of indexed vectors across all shards.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the corpus has no rows (and no exposed shards).
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Logical vector length supplied at construction.
    pub fn dimensions(&self) -> usize {
        self.index.dimensions()
    }

    /// Stored row stride in FP16 elements, including zero padding.
    ///
    /// Multiply by two for the row stride in bytes. Padding rounds the logical
    /// dimension up to a multiple of 128.
    pub fn padded_dimensions(&self) -> usize {
        self.index.padded_dimensions()
    }

    /// Iterate over nonempty shards in increasing insertion-ID order.
    ///
    /// The ranges partition `0..self.len()` without gaps. An empty corpus yields
    /// no shards. Iteration creates borrowed descriptors without allocating.
    /// Bindings borrow the index, so they can outlive this descriptor/iterator.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = CorpusShardView<'a>> + 'a {
        let index = self.index;
        index.shards[..index.shard_count()]
            .iter()
            .map(move |shard| CorpusShardView {
                vectors: shard.data.binding(),
                inverse_norms: shard.norms.binding(),
                start: shard.start,
                count: shard.count,
                capacity: shard.capacity,
            })
    }
}

impl<'a> CorpusShardView<'a> {
    /// Global insertion IDs covered by this shard, with an exclusive end.
    pub fn row_range(&self) -> Range<usize> {
        self.start..self.start + self.count
    }

    /// Number of rows that may be read from both corpus bindings.
    ///
    /// At least [`row_range`](Self::row_range)`().len()`, rounded up to a
    /// multiple of 256. Vector and inverse-norm bindings include this entire
    /// capacity, and the vector capacity still fits within 2^32 FP16 elements.
    ///
    /// Rows at or beyond the logical row count are **not corpus data**. Their
    /// contents are unspecified: mask them by local row index before reduction
    /// or selection, even if they happen to contain zeros. They have no global
    /// insertion IDs. Caller-owned side arrays must also cover every row a
    /// kernel reads, or guard those loads separately.
    ///
    /// Tiles up to 256 rows whose size divides 256 can read the rounded-up final
    /// tile in full. For other tile sizes, check that the tile's exclusive end
    /// is no greater than this capacity; larger tiles are not guaranteed to fit.
    pub fn capacity_rows(&self) -> usize {
        self.capacity
    }

    /// Read-only row-major FP16 vector binding, starting at local row zero.
    ///
    /// Its byte length is `capacity_rows() * corpus.padded_dimensions() * 2`.
    /// Logical rows contain little-endian IEEE binary16 with zero dimension
    /// padding. Extra rows have unspecified contents; see [`Self::capacity_rows`].
    /// HRX does not enforce read-only access; see [`CorpusView`]'s contract.
    pub fn vectors(&self) -> View<'a> {
        self.vectors
    }

    /// Read-only FP32 inverse-norm binding, starting at local row zero.
    ///
    /// Contains `capacity_rows()` little-endian FP32 values; only the first
    /// `row_range().len()` describe corpus rows. Extra values are unspecified.
    /// For logical local row `r`, multiply its FP32 dot product with a unit
    /// query by element `r`
    /// to obtain cosine similarity over the stored vector. For an unnormalized
    /// query, also divide by the query's norm. Accumulation order can affect
    /// rounding relative to built-in search; scores are not clamped.
    /// HRX does not enforce read-only access; see [`CorpusView`]'s contract.
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
                    FlatIndex::build_fp16(&device, 3, rows)?
                } else {
                    FlatIndex::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?
                };
                let identity = index.stream.device_id();
                (index, identity)
            }; // The original Device is gone before either public API is used.
            std::thread::spawn(move || -> Result<()> {
                let mut first = index.stream()?;
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
                    album_example::best_by_album(index.corpus(), &[0, 1], 2, &[1.0, 0.0, 0.0])?;
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
                    FlatIndex::build_encoded(
                        &device,
                        d,
                        &bytes,
                        ScanConfig::default(),
                        129 * padded,
                        |r, d, p, bytes| encode_fp16(r, d, p, bytes),
                    )?
                } else {
                    FlatIndex::build_encoded(
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
            let mut index = FlatIndex::build_encoded(
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
            for shard in &index.shards {
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
            let grouped = album_example::best_by_album(index.corpus(), &vec![0; n], 2, &query)?;
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
        let index = FlatIndex::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
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
